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
import { applyAssignment } from './assignment';

import {
  pca3D_single
} from './models/mlp';

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
/**
 * Run a single federated round.
 * @param state FederatedState
 * @param onStateUpdate Callback for state update
 * @param onClientUpdate Callback for client update
 * @param onServerStatusUpdate Callback for server status update
 * @param clustersForRound Clusters from previous round (optional)
 * @returns [RoundMetrics, clustersForRound]
 */
export const runFederatedRound = async (
  // Dictionnaire pour stocker le modèle envoyé à chaque client
  state: FederatedState,
  onStateUpdate: (state: Partial<FederatedState>) => void,
  onClientUpdate: (clientId: string, update: Partial<ClientState>) => void,
  onServerStatusUpdate: (status: ServerStatus) => void,
  clustersForRound?: string[][]
): Promise<[RoundMetrics, string[][] | undefined]> => {
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
  let modelsSentToClients = {};
  const trainingPromises = selectedClients.map(async (client) => {
    onClientUpdate(client.id, { status: 'training', progress: 0 });
    // Fallback: if clustersForRound is not initialized, use only the global model
    let clusterModels: ModelWeights[] | undefined = undefined;
    let clusterAssignments: Record<string, number> | undefined = undefined;
    let clusterClientIds: string[][] | undefined = undefined;
    if (typeof clustersForRound !== 'undefined' && clustersForRound.length > 0) {
      clusterModels = clustersForRound.map((grp, idx) => {
        const firstClientId = grp[0];
        return clusterModelStore.get(firstClientId) || globalModel;
      });
      clusterAssignments = {};
      clustersForRound.forEach((grp, idx) => {
        grp.forEach(cid => { clusterAssignments![cid] = idx; });
      });
      clusterClientIds = clustersForRound;
    }

    const modelAssignmentMethod = serverConfig.modelAssignmentMethod || '1NN';

    // Modèle envoyé au client par le serveur
    const modelToSend = applyAssignment(
      modelAssignmentMethod,
      client,
      {
        globalModel,
        clusterModels,
        clusterAssignments,
        clusterClientIds,
        selectedClients,
        round: currentRound,
        distanceMetric: serverConfig.distanceMetric,
      }
    );
    // Stocker dans le dictionnaire pour ce client
    modelsSentToClients[client.id] = modelToSend;

    // Affichage PCA 3D du modèle envoyé au client
    try {
      if (client.id === "client-0") {
        const { vectorizeModel, pca3D_single } = await import('./models/mlp');
        const modelToShow = modelsSentToClients[client.id];
        const vec = vectorizeModel(unflattenWeights(modelToShow, MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE));
        const pca3d = pca3D_single(vec);
        console.log(`PCA3D du modèle envoyé au client ${client.id}:`, pca3d);
      }
    } catch (e) {
      console.warn('Erreur PCA3D:', e);
    }

    const result = await simulateClientTraining(
      client,
      modelToSend,
      (progress) => onClientUpdate(client.id, { progress }),
      (status) => onClientUpdate(client.id, { status }),
      currentRound,
      globalModel
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

  /*
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
  */

  // Phase 3: Receive models
  onServerStatusUpdate('receiving');
  for (let i = 0; i < trainedClients.length; i++) {
    const { result, client } = trainedClients[i];
    let weightsToUse = result.weights;

    // @debug HARDCODE DE STRATEGIE GRAVITY
    await new Promise(resolve => setTimeout(resolve, 200));
    // Gravity: dès le round 5, utiliser le modèle de cluster envoyé par le serveur (sans fine-tuning)
    if (currentRound >= 3 && client.id === "client-0" && currentRound < 10) {
      console.log("je suis le client " + client.id);
      // Affecter le learning rate à 0.05 pour ce client
      client.learningRate = 0.01;
      // Affecter le nombre d'epochs à 6 pour ce client
      client.localEpochs = 6;
      // On récupère le modèle envoyé au client depuis le dictionnaire
      if (modelsSentToClients[client.id]) {
        weightsToUse = modelsSentToClients[client.id];
        console.log("Le client retourne le modèle reçu sans fine-tuning");
      } else {
        // Fallback: on ne peut pas retrouver le modèle envoyé, on garde le modèle local
        console.log("Le client retourne le modèle fine-tuné");
        weightsToUse = result.weights;
      }
    }
    else {
      console.log("Le client retourne le modèle fine-tuné");
      client.localEpochs = 3;
      weightsToUse = result.weights;
    }

    trainedClients[i].result.weights = weightsToUse;

    clientResults.push({ weights: weightsToUse, dataSize: client.dataSize });
    clientMetricsForRound.push({
      clientId: client.id,
      clientName: client.name,
      loss: result.loss,
      accuracy: result.accuracy,
      testAccuracy: result.testAccuracy,
      gradientNorm: result.gradientNorm,
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

  try {
    const clientResultsWithIds = trainedClients.map(({ result, client }) => ({
      id: client.id,
      weights: result.weights,
      dataSize: client.dataSize
    }));

    const clustering = clusterClientModels(
      clientResultsWithIds,
      serverConfig.distanceMetric,
      serverConfig.clusteringMethod || 'louvain',
      serverConfig.kmeansNumClusters
    );
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

  return [roundMetrics, clustersForRound];
};

// Re-export for backward compatibility
export { createClient, setSeed, getSeed, getRng };
export { getClientModels, setClientModels } from './core/stores';
