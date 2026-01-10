// Main simulation orchestration - simplified after refactoring
// All logic has been moved to dedicated modules

// Re-export everything from modules for backward compatibility
export * from './core';
export * from './models';
export * from './data';
export * from './clients';
export * from './server';
export * from './clustering';
export * from './assignment';
export * from './results';

// Import specific items needed for runFederatedRound
import type { FederatedState, ServerStatus, RoundMetrics, ModelWeights, ClusterMetrics, ClientRoundMetrics, ClientState } from './core/types';
import { getRng, setSeed, getSeed } from './core/random';
import { clusterModelStore, clientTestDataStore, mlpWeightsStore, clientDataStore, setMnistTrainData, setMnistTestData, resetStores } from './core/stores';
import { initializeMLPWeightsWithRng, flattenWeights, unflattenWeights, MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE } from './models/mlp';
import { loadMNISTTrain, loadMNISTTest } from './data/mnist';
import { simulateClientTraining, selectClients, createClient } from './clients/training';
import { aggregationMethods } from './server/aggregation';
import { evaluateOnTestSet, evaluateClusterModel, computeWeightsSnapshot } from './server/evaluation';
import { clusterClientModels, computeSilhouetteScore } from './clustering';
import { computeProbabilisticAssignments } from './assignment/probabilistic';

// Preload MNIST data
export const preloadMNIST = async (): Promise<void> => {
  const promises: Promise<void>[] = [];
  promises.push(loadMNISTTrain().then(data => { setMnistTrainData(data); }));
  promises.push(loadMNISTTest().then(data => { setMnistTestData(data); }));
  await Promise.all(promises);
};

// Initialize random model weights with seeded RNG
export const initializeModel = (architecture: string): ModelWeights => {
  const rng = getRng();
  const mlpWeights = initializeMLPWeightsWithRng(
    () => rng.next(),
    MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE
  );
  mlpWeightsStore.set('global', mlpWeights);
  resetStores();
  
  const flat = flattenWeights(mlpWeights);
  return {
    layers: flat.layers,
    bias: flat.bias,
    version: 0,
  };
};

// Run a single federated round
export const runFederatedRound = async (
  state: FederatedState,
  onStateUpdate: (state: Partial<FederatedState>) => void,
  onClientUpdate: (clientId: string, update: Partial<ClientState>) => void,
  onServerStatusUpdate: (status: ServerStatus) => void
): Promise<RoundMetrics> => {
  const { serverConfig, clients, globalModel, currentRound } = state;
  
  if (!globalModel) {
    throw new Error('Global model not initialized');
  }

  // Sync client aggregation method
  for (const client of clients) {
    client.clientAggregationMethod = serverConfig.clientAggregationMethod || 'none';
  }

  const selectedClients = selectClients(clients, serverConfig.clientsPerRound);
  
  if (selectedClients.length < serverConfig.minClientsRequired) {
    throw new Error(`Not enough clients available. Required: ${serverConfig.minClientsRequired}, Available: ${selectedClients.length}`);
  }

  const clientResults: { weights: ModelWeights; dataSize: number }[] = [];
  const clientMetricsForRound: ClientRoundMetrics[] = [];
  const participatingIds = selectedClients.map(c => c.id);

  // Phase 1: Server sends model
  onServerStatusUpdate('sending');
  for (const client of selectedClients) {
    onClientUpdate(client.id, { status: 'receiving', progress: 0 });
    await new Promise(resolve => setTimeout(resolve, 300));
  }

  // Phase 2: Clients train
  onServerStatusUpdate('waiting');
  const trainingPromises = selectedClients.map(async (client) => {
    onClientUpdate(client.id, { status: 'training', progress: 0 });
    const modelToSend = clusterModelStore.get(client.id) || globalModel;
    const result = await simulateClientTraining(
      client,
      modelToSend,
      (progress) => onClientUpdate(client.id, { progress }),
      (status) => onClientUpdate(client.id, { status })
    );
    onClientUpdate(client.id, {
      status: 'sending',
      progress: 100,
      localLoss: result.loss,
      localAccuracy: result.accuracy,
      localTestAccuracy: result.testAccuracy,
      lastLocalModel: client.lastLocalModel,
    });
    return { result, client };
  });

  const trainedClients = await Promise.all(trainingPromises);

  // Phase 3: Receive models
  onServerStatusUpdate('receiving');
  for (const { result, client } of trainedClients) {
    await new Promise(resolve => setTimeout(resolve, 200));
    clientResults.push({ weights: result.weights, dataSize: client.dataSize });
    clientMetricsForRound.push({
      clientId: client.id,
      clientName: client.name,
      loss: result.loss,
      accuracy: result.accuracy,
      testAccuracy: result.testAccuracy,
    });
    onClientUpdate(client.id, {
      status: 'completed',
      lastUpdate: Date.now(),
      roundsParticipated: client.roundsParticipated + 1,
    });
  }

  // Phase 4: Clustering and aggregation
  let silhouetteAvgForRound: number | undefined;
  let clusterMetricsForRound: ClusterMetrics[] = [];
  let distanceMatrixForRound: number[][] | undefined;
  let clustersForRound: string[][] | undefined;

  try {
    const clientResultsWithIds = trainedClients.map(({ result, client }) => ({ 
      id: client.id, 
      weights: result.weights, 
      dataSize: client.dataSize 
    }));
    
    const clustering = clusterClientModels(clientResultsWithIds);
    distanceMatrixForRound = clustering.distanceMatrix;
    clustersForRound = clustering.clusters;

    // Compute silhouette
    const idToIndex = new Map<string, number>();
    clientResultsWithIds.forEach((c, i) => idToIndex.set(c.id, i));
    silhouetteAvgForRound = computeSilhouetteScore(distanceMatrixForRound, clustersForRound, idToIndex);

    // Build cluster-averaged models
    if (clustersForRound && clustersForRound.length > 0) {
      const clientMap = new Map(clientResultsWithIds.map(c => [c.id, c]));
      
      for (let clusterIdx = 0; clusterIdx < clustersForRound.length; clusterIdx++) {
        const grp = clustersForRound[clusterIdx];
        const entries = grp.map(id => clientMap.get(id)).filter(Boolean) as typeof clientResultsWithIds;
        if (entries.length === 0) continue;

        const sumLayers: number[][] = entries[0].weights.layers.map(l => new Array(l.length).fill(0));
        const sumBias: number[] = new Array(entries[0].weights.bias.length).fill(0);
        let totalData = 0;

        for (const e of entries) {
          totalData += e.dataSize;
          for (let li = 0; li < e.weights.layers.length; li++) {
            for (let k = 0; k < e.weights.layers[li].length; k++) {
              sumLayers[li][k] += e.weights.layers[li][k] * e.dataSize;
            }
          }
          for (let b = 0; b < e.weights.bias.length; b++) {
            sumBias[b] += e.weights.bias[b] * e.dataSize;
          }
        }

        const averagedModel: ModelWeights = {
          layers: sumLayers.map(l => l.map(v => v / totalData)),
          bias: sumBias.map(v => v / totalData),
          version: entries[0].weights.version,
        };

        for (const e of entries) clusterModelStore.set(e.id, averagedModel);

        const clusterAccuracy = evaluateClusterModel(grp, averagedModel, clientTestDataStore);
        clusterMetricsForRound.push({ clusterId: clusterIdx, accuracy: clusterAccuracy, clientIds: grp });
      }
    }
  } catch (err) {
    console.warn('Clustering failed:', err);
  }

  onServerStatusUpdate('evaluating');
  const aggregationFn = aggregationMethods[serverConfig.aggregationMethod]?.fn || aggregationMethods.fedavg.fn;
  const aggregationStart = Date.now();
  const newGlobalModel = aggregationFn(clientResults);
  const aggregationTime = Date.now() - aggregationStart;

  const testMetrics = evaluateOnTestSet(newGlobalModel);
  onServerStatusUpdate('completed');

  setTimeout(() => {
    for (const client of selectedClients) {
      onClientUpdate(client.id, { status: 'idle', progress: 0 });
    }
  }, 500);

  const roundMetrics: RoundMetrics = {
    round: currentRound,
    globalLoss: testMetrics.loss,
    globalAccuracy: testMetrics.accuracy,
    participatingClients: participatingIds,
    aggregationTime,
    timestamp: Date.now(),
    weightsSnapshot: computeWeightsSnapshot(newGlobalModel),
    distanceMatrix: distanceMatrixForRound,
    clusters: clustersForRound,
    silhouetteAvg: silhouetteAvgForRound,
    clusterMetrics: clusterMetricsForRound.length > 0 ? clusterMetricsForRound : undefined,
    clientMetrics: clientMetricsForRound.length > 0 ? clientMetricsForRound : undefined,
  };

  onStateUpdate({
    globalModel: newGlobalModel,
    currentRound: currentRound + 1,
    roundHistory: [...state.roundHistory, roundMetrics],
  });

  return roundMetrics;
};

// Re-export for backward compatibility
export { createClient, setSeed, getSeed, getRng };
export { getClientModels, setClientModels } from './core/stores';
