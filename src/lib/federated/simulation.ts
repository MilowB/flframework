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
import { initializeMLPWeightsWithRng, flattenWeights, unflattenWeights, MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE, computeCosineSimilarity, computeModelDelta } from './models/mlp';
import { loadMNISTTrain, loadMNISTTest } from './data/mnist';
import { simulateClientTraining, selectClients, createClient } from './clients/training';
import { aggregationMethods } from './server/aggregation';
import { evaluateOnTestSet, evaluateClusterModel, computeWeightsSnapshot } from './server/evaluation';
import { clusterClientModels, computeSilhouetteScore } from './clustering';
import { applyAssignment, recordClientCosineSimilarity, detectCosineSimilarityDrop, findMostApproachedCluster, resetCosineSimilarityStores, getUnHappyClients, resetUnHappyClients, type AlexandreContext } from './assignment';
import { applyByzantineAttack } from './attacks';

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
  resetCosineSimilarityStores();

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

// Compute element-wise median of a list of gradient vectors
const computeMedianGradient = (gradients: number[][]): number[] => {
  if (gradients.length === 0) return [];
  if (gradients.length === 1) return gradients[0];
  const dim = gradients[0].length;
  const median = new Array(dim);
  for (let d = 0; d < dim; d++) {
    const values = gradients.map(g => g[d]).sort((a, b) => a - b);
    const mid = Math.floor(values.length / 2);
    median[d] = values.length % 2 === 0
      ? (values[mid - 1] + values[mid]) / 2
      : values[mid];
  }
  return median;
};

// DEBUG flag for RNG state tracking - set to true to diagnose reproducibility issues
const DEBUG_RNG_STATE = false;

// Helper to sample RNG state without consuming it (for debugging)
const sampleRngState = (label: string) => {
  if (!DEBUG_RNG_STATE) return;
  const rng = getRng();
  // Sample the next value to see current state (this does consume one value)
  const sample = rng.next();
  console.log(`[RNG Debug] ${label}: next value = ${sample.toFixed(8)}`);
};

export const runFederatedRound = async (
  // Dictionnaire pour stocker le modèle envoyé à chaque client
  state: FederatedState,
  onStateUpdate: (state: Partial<FederatedState>) => void,
  onClientUpdate: (clientId: string, update: Partial<ClientState>) => void,
  onServerStatusUpdate: (status: ServerStatus) => void,
  clustersForRound?: string[][]
): Promise<[RoundMetrics, string[][] | undefined]> => {
  const { serverConfig, clients, globalModel, currentRound } = state;

  console.log(`--- NOUVEAU ROUND ${currentRound} ---`);

  if (DEBUG_RNG_STATE) {
    console.log(`\n=== Round ${currentRound} START ===`);
    console.log(`[RNG Debug] Agreement matrix enabled: ${serverConfig.useAgreementMatrix}`);
    sampleRngState(`Round ${currentRound} - Before client selection`);
  }

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
  const participatingIds = selectedClients
    .map(c => c.id)
    .sort((a, b) => {
      // Extract numeric index from client ID (e.g., "client-5" -> 5)
      const numA = parseInt(a.split('-')[1] || '0', 10);
      const numB = parseInt(b.split('-')[1] || '0', 10);
      return numA - numB;
    });

  // Phase 1: Server sends model
  onServerStatusUpdate('sending');
  for (const client of selectedClients) {
    onClientUpdate(client.id, { status: 'receiving', progress: 0 });
    await new Promise(resolve => setTimeout(resolve, 300));
  }

  // Pre-Phase 2: Detect cosine similarity drops and compute immediate reassignments
  // This happens BEFORE sending models so clients receive the correct cluster model
  const immediateReassignments: Map<string, number> = new Map();
  let alexandreContextForRound: AlexandreContext | undefined;
  let previousClusterModels: Map<number, ModelWeights> = new Map();
  
  // Build alexandreContext from PREVIOUS round (N-1) data before Phase 2
  if (serverConfig.modelAssignmentMethod === 'Alexandre' && currentRound >= 1) {
    const prevRoundMetrics = state.roundHistory[currentRound - 1];
    if (prevRoundMetrics?.clusters && prevRoundMetrics.clientMetrics) {
      const prevClusters = prevRoundMetrics.clusters;
      const prevClientMetrics = prevRoundMetrics.clientMetrics;
      
      // Extract data from previous round
      const alexandreGradientNorms: Record<string, number> = {};
      const alexandreCosineSims: Record<string, number> = {};
      const alexandreClientGradients: Record<string, number[]> = {};
      
      for (const cm of prevClientMetrics) {
        alexandreGradientNorms[cm.clientId] = cm.gradientNorm || 0;
        // Use previous-round cosine similarity (avg similarity with cluster members)
        alexandreCosineSims[cm.clientId] = cm.clusterCosineSimilarity ?? 0;
        // Note: client gradients not stored in RoundMetrics, will be empty
      }
      
      const clusterAssignments: Record<string, number> = {};
      prevClusters.forEach((members, idx) => {
        members.forEach(clientId => { clusterAssignments[clientId] = idx; });
      });
      
      const alexandreClusterModels = prevClusters.map((_, idx) =>
        clusterModelStore.get(`cluster-${idx}`) || globalModel
      );
      
      // Compute cluster gradient norms using N-1 and N-2 cluster models
      const clusterGradientNorms: Record<number, number> = {};
      if (currentRound >= 2) {
        const prevPrevRoundMetrics = state.roundHistory[currentRound - 2];
        if (prevRoundMetrics.clusterMetrics && prevPrevRoundMetrics?.clusterMetrics) {
          for (let c = 0; c < prevClusters.length; c++) {
            const clusterN1 = prevRoundMetrics.clusterMetrics.find(cm => cm.clusterId === c);
            const clusterN2 = prevPrevRoundMetrics.clusterMetrics.find(cm => cm.clusterId === c);
            if (clusterN1?.weights && clusterN2?.weights) {
              const delta = computeModelDelta(clusterN1.weights, clusterN2.weights);
              clusterGradientNorms[c] = Math.sqrt(delta.reduce((s, v) => s + v * v, 0));
            } else {
              clusterGradientNorms[c] = 0;
            }
          }
        }
      }
      
      alexandreContextForRound = {
        gradientNorms: alexandreGradientNorms,
        cosineSimilarities: alexandreCosineSims,
        clusterGradientNorms,
        clusterMedianGradients: {},
        clientGradients: alexandreClientGradients,
        clusterAssignments,
        distanceMatrix: prevRoundMetrics.distanceMatrix,
        previousDistanceMatrix: currentRound >= 2 ? state.roundHistory[currentRound - 2]?.distanceMatrix : undefined,
        clusters: prevClusters,
        participatingClients: prevRoundMetrics.participatingClients,
        clusterModels: alexandreClusterModels,
        globalModel,
      };
    }
  }
  
  if (serverConfig.modelAssignmentMethod === 'CosineSimilarity' && currentRound >= 2) {
    // Get distance matrices from previous rounds
    const currentRoundMetrics = state.roundHistory[currentRound - 1]; // Round N-1 (last completed)
    const previousRoundMetrics = state.roundHistory[currentRound - 2]; // Round N-2
    
    const currentDistanceMatrix = currentRoundMetrics?.distanceMatrix;
    const previousDistanceMatrix = previousRoundMetrics?.distanceMatrix;
    const previousClusters = currentRoundMetrics?.clusters;
    const previousParticipants = currentRoundMetrics?.participatingClients;
    
    if (currentDistanceMatrix && previousDistanceMatrix && previousClusters && previousParticipants) {
      for (const client of selectedClients) {
        // Check if this client had a significant drop at round N-1
        if (detectCosineSimilarityDrop(client.id, currentRound)) {
          console.log(`[Cosine Similarity] Client ${client.id} detected drop at round ${currentRound - 1}`);
          
          // Find the cluster the client moved closest to
          const result = findMostApproachedCluster(
            client.id,
            currentDistanceMatrix,
            previousDistanceMatrix,
            previousClusters,
            previousParticipants
          );
          
          if (result && result.clusterId >= 0) {
            immediateReassignments.set(client.id, result.clusterId);
            console.log(`[Cosine Similarity] Client ${client.id} will receive model from cluster ${result.clusterId} (avg change: ${result.avgDistanceChange.toFixed(4)}, approached clients: ${result.approachedClients.join(', ')})`);
          }
        }
      }
    }
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

    // Check for immediate reassignment (cosine similarity drop detected)
    const reassignedCluster = immediateReassignments.get(client.id);
    let modelToSend: ModelWeights;
    
    if (reassignedCluster !== undefined) {
      // Client was detected with a drop - send the model from the target cluster
      const targetClusterModel = clusterModelStore.get(`cluster-${reassignedCluster}`);
      if (targetClusterModel) {
        modelToSend = targetClusterModel;
        console.log(`[Cosine Similarity] Client ${client.id} receiving model from cluster ${reassignedCluster} (immediate reassignment)`);
      } else {
        // Fallback to normal assignment if cluster model not found
        modelToSend = applyAssignment(
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
            alexandreContext: alexandreContextForRound,
          }
        );
      }
    } else {
      // Normal model assignment
      modelToSend = applyAssignment(
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
          alexandreContext: alexandreContextForRound,
        }
      );
    }
    
    // Stocker dans le dictionnaire pour ce client
    modelsSentToClients[client.id] = modelToSend;

    // Affichage PCA 3D du modèle envoyé au client
    try {
      if (client.id === "client-0") {
        const { vectorizeModel, pca3D_single } = await import('./models/mlp');
        const modelToShow = modelsSentToClients[client.id];
        const vec = vectorizeModel(unflattenWeights(modelToShow, MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE));
        const pca3d = pca3D_single(vec);
        //console.log(`PCA3D du modèle envoyé au client ${client.id}:`, pca3d);
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
      globalModel,
      serverConfig.modelArchitecture
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
  const clientResultsWithClientId: { weights: ModelWeights; dataSize: number; clientId: string }[] = [];
  for (let i = 0; i < trainedClients.length; i++) {
    const { result, client } = trainedClients[i];
    let weightsToUse = result.weights;

    trainedClients[i].result.weights = weightsToUse;

    clientResults.push({ weights: weightsToUse, dataSize: client.dataSize });
    clientResultsWithClientId.push({ weights: weightsToUse, dataSize: client.dataSize, clientId: client.id });
    clientMetricsForRound.push({
      clientId: client.id,
      clientName: client.name,
      loss: result.loss,
      accuracy: result.accuracy,
      testAccuracy: result.testAccuracy,
      gradientNorm: result.gradientNorm,
      weights: weightsToUse, // Store weights for visualization
    });
    onClientUpdate(client.id, {
      status: 'completed',
      lastUpdate: Date.now(),
      roundsParticipated: client.roundsParticipated + 1,
    });
  }

  // Phase 3.5: Apply Byzantine attack (poison Byzantine client weights before aggregation)
  const byzantineCount = serverConfig.byzantineCount ?? 0;
  if (byzantineCount > 0) {
    // Select the first N clients as Byzantine (deterministic, sorted by index)
    const sortedIds = participatingIds.slice().sort((a, b) => {
      const numA = parseInt(a.split('-')[1] || '0', 10);
      const numB = parseInt(b.split('-')[1] || '0', 10);
      return numA - numB;
    });
    const byzantineClientIds = sortedIds.slice(0, Math.min(byzantineCount, sortedIds.length));

    const attackedResults = applyByzantineAttack(
      clientResultsWithClientId,
      globalModel,
      {
        byzantineCount,
        attackMethod: serverConfig.byzantineAttackMethod || 'local-model-poisoning',
      },
      byzantineClientIds,
      currentRound,
      serverConfig.totalRounds
    );

    // Update clientResults with poisoned weights
    for (let i = 0; i < clientResults.length; i++) {
      const attacked = attackedResults.find(r => r.clientId === trainedClients[i].client.id);
      if (attacked) {
        clientResults[i] = { weights: attacked.weights, dataSize: attacked.dataSize };
        // Also update trainedClients for clustering phase
        trainedClients[i].result.weights = attacked.weights;
      }
    }
  }

  // Phase 4: Clustering and aggregation
  let silhouetteAvgForRound: number | undefined;
  let clusterMetricsForRound: ClusterMetrics[] = [];
  let distanceMatrixForRound: number[][] | undefined;
  let agreementMatrixForRound: number[][] | undefined;

  try {
    const clientResultsWithIds = trainedClients
      .map(({ result, client }) => ({
        id: client.id,
        weights: result.weights,
        dataSize: client.dataSize
      }))
      .sort((a, b) => {
        // Extract numeric index from client ID (e.g., "client-5" -> 5)
        const numA = parseInt(a.id.split('-')[1] || '0', 10);
        const numB = parseInt(b.id.split('-')[1] || '0', 10);
        return numA - numB;
      }); // Sort by numeric client index for consistent ordering

    const clustering = clusterClientModels(
      clientResultsWithIds,
      serverConfig.distanceMetric,
      serverConfig.clusteringMethod || 'louvain',
      serverConfig.kmeansNumClusters,
      serverConfig.useAgreementMatrix,
      serverConfig.spectralNumClusters
    );
    distanceMatrixForRound = clustering.distanceMatrix;
    clustersForRound = clustering.clusters;
    agreementMatrixForRound = clustering.agreementMatrix;

    if (DEBUG_RNG_STATE) {
      sampleRngState(`Round ${currentRound} - After clustering (agreement=${serverConfig.useAgreementMatrix})`);
      console.log(`[RNG Debug] Clusters: ${JSON.stringify(clustersForRound)}`);
    }

    // Compute silhouette
    const idToIndex = new Map<string, number>();
    clientResultsWithIds.forEach((c, i) => idToIndex.set(c.id, i));
    silhouetteAvgForRound = computeSilhouetteScore(distanceMatrixForRound, clustersForRound, idToIndex);

    // Build cluster-averaged models
    if (clustersForRound && clustersForRound.length > 0) {
      const clientMap = new Map(clientResultsWithIds.map(c => [c.id, c]));

      // Snapshot cluster models from previous round (N-1) before overwriting
      for (let clusterIdx = 0; clusterIdx < clustersForRound.length; clusterIdx++) {
        const prev = clusterModelStore.get(`cluster-${clusterIdx}`);
        if (prev) previousClusterModels.set(clusterIdx, prev);
      }

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

        // Store the cluster model with cluster-X key
        clusterModelStore.set(`cluster-${clusterIdx}`, averagedModel);
        // Also store for each client in the cluster
        for (const e of entries) clusterModelStore.set(e.id, averagedModel);

        const clusterAccuracy = evaluateClusterModel(grp, averagedModel, clientTestDataStore);
        clusterMetricsForRound.push({
          clusterId: clusterIdx,
          accuracy: clusterAccuracy,
          clientIds: grp,
          weights: averagedModel // Store weights for visualization
        });
      }
    }

    // Calculate cluster cosine similarity for each client
    // C = average cosine similarity of delta(client) with delta(other cluster members)
    // where delta = model_current - model_N-1 (after fine-tuning)
    if (clustersForRound && clustersForRound.length > 0 && currentRound > 0) {
      const clientMap = new Map(clientResultsWithIds.map(c => [c.id, c]));
      const selectedClientsMap = new Map(selectedClients.map(c => [c.id, c]));
      
      for (let i = 0; i < clientMetricsForRound.length; i++) {
        const clientMetric = clientMetricsForRound[i];
        const clientId = clientMetric.clientId;
        const client = selectedClientsMap.get(clientId);
        const clientResult = clientMap.get(clientId);
        
        if (!client || !clientResult) continue;
        
        // Get the client's local model from N-1 (localModelHistory[1] is N-1)
        const previousModel = client.localModelHistory?.[1];
        if (!previousModel) continue;
        
        // Find which cluster this client belongs to
        let clusterIdx = -1;
        let clusterMembers: string[] = [];
        for (let c = 0; c < clustersForRound.length; c++) {
          if (clustersForRound[c].includes(clientId)) {
            clusterIdx = c;
            clusterMembers = clustersForRound[c];
            break;
          }
        }
        
        if (clusterIdx === -1 || clusterMembers.length <= 1) continue;
        
        // Compute this client's delta
        const clientDelta = computeModelDelta(clientResult.weights, previousModel);
        
        // Compute cosine similarity with each other cluster member
        let totalSim = 0;
        let count = 0;
        
        for (const memberId of clusterMembers) {
          if (memberId === clientId) continue;
          
          const memberClient = selectedClientsMap.get(memberId);
          const memberResult = clientMap.get(memberId);
          if (!memberClient || !memberResult) continue;
          
          // Get member's previous model (N-1)
          const memberPreviousModel = memberClient.localModelHistory?.[1];
          if (!memberPreviousModel) continue;
          
          // Compute member's delta
          const memberDelta = computeModelDelta(memberResult.weights, memberPreviousModel);
          
          // Compute cosine similarity
          const similarity = computeCosineSimilarity(clientDelta, memberDelta);
          totalSim += similarity;
          count++;
        }
        
        if (count > 0) {
          const avgSimilarity = totalSim / count;
          clientMetricsForRound[i].clusterCosineSimilarity = avgSimilarity;
          
          // Record cosine similarity for Cosine Similarity assignment strategy
          recordClientCosineSimilarity(clientId, avgSimilarity, currentRound);
        }
      }
      
      // Note: Reassignment detection is now done BEFORE training (Pre-Phase 2)
      // so clients receive the correct model immediately at the current round
    }

    // Build alexandreContext: cosine similarity of each client with median gradient of its cluster
    if (clustersForRound && clustersForRound.length > 0) {
      const clientMap = new Map(clientResultsWithIds.map(c => [c.id, c]));
      const selectedClientsMap = new Map(selectedClients.map(c => [c.id, c]));

      // Step 1: collect gradient vectors per cluster
      const clusterGradients: Map<number, { clientId: string; delta: number[] }[]> = new Map();
      for (let c = 0; c < clustersForRound.length; c++) clusterGradients.set(c, []);
      const alexandreClientGradients: Record<string, number[]> = {};

      for (const { result, client } of trainedClients) {
        const prevModel = client.localModelHistory?.[1];
        if (!prevModel) continue;
        const delta = computeModelDelta(result.weights, prevModel);
        alexandreClientGradients[client.id] = delta;
        for (let c = 0; c < clustersForRound.length; c++) {
          if (clustersForRound[c].includes(client.id)) {
            clusterGradients.get(c)!.push({ clientId: client.id, delta });
            break;
          }
        }
      }

      // Step 2: compute median gradient per cluster
      const clusterMedians: Map<number, number[]> = new Map();
      for (const [c, entries] of clusterGradients) {
        if (entries.length > 0) {
          clusterMedians.set(c, computeMedianGradient(entries.map(e => e.delta)));
        }
      }

      // Step 3: compute cosine similarity of each client with its cluster median
      const alexandreCosineSims: Record<string, number> = {};
      const alexandreGradientNorms: Record<string, number> = {};

      for (const { result, client } of trainedClients) {
        const prevModel = client.localModelHistory?.[1];
        if (!prevModel) {
          alexandreCosineSims[client.id] = 0;
          alexandreGradientNorms[client.id] = 0;
          continue;
        }
        const delta = computeModelDelta(result.weights, prevModel);
        // gradient norm (L2)
        const norm = Math.sqrt(delta.reduce((s, v) => s + v * v, 0));
        alexandreGradientNorms[client.id] = norm;

        // find cluster index
        let clientClusterIdx = -1;
        for (let c = 0; c < clustersForRound.length; c++) {
          if (clustersForRound[c].includes(client.id)) { clientClusterIdx = c; break; }
        }
        const median = clusterMedians.get(clientClusterIdx);
        if (!median || median.length === 0) {
          alexandreCosineSims[client.id] = 0;
          continue;
        }
        alexandreCosineSims[client.id] = computeCosineSimilarity(delta, median);
      }

      // Build cluster models array aligned with clustersForRound indices
      const alexandreClusterModels = clustersForRound.map((_, idx) =>
        clusterModelStore.get(`cluster-${idx}`) || globalModel
      );

      // Compute gradient norms of cluster centroïds (delta between round N-1 and N centroid)
      const clusterGradientNorms: Record<number, number> = {};
      for (let c = 0; c < clustersForRound.length; c++) {
        const prevCentroid = previousClusterModels.get(c);
        const newCentroid = clusterModelStore.get(`cluster-${c}`);
        if (prevCentroid && newCentroid) {
          const delta = computeModelDelta(newCentroid, prevCentroid);
          clusterGradientNorms[c] = Math.sqrt(delta.reduce((s, v) => s + v * v, 0));
        } else {
          clusterGradientNorms[c] = 0;
        }
      }

      const alexandreClusterMedianGradients: Record<number, number[]> = {};
      for (const [c, median] of clusterMedians) {
        alexandreClusterMedianGradients[c] = median;
      }

      alexandreContextForRound = {
        gradientNorms: alexandreGradientNorms,
        cosineSimilarities: alexandreCosineSims,
        clusterGradientNorms,
        clusterMedianGradients: alexandreClusterMedianGradients,
        clientGradients: alexandreClientGradients,
        clusterAssignments: Object.fromEntries(
          clustersForRound.flatMap((members, idx) => members.map(id => [id, idx]))
        ),
        distanceMatrix: distanceMatrixForRound,
        previousDistanceMatrix: currentRound >= 1 ? state.roundHistory[currentRound - 1]?.distanceMatrix : undefined,
        clusters: clustersForRound,
        participatingClients: participatingIds,
        clusterModels: alexandreClusterModels,
        globalModel,
      };
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
    agreementMatrix: agreementMatrixForRound,
    clusters: clustersForRound,
    silhouetteAvg: silhouetteAvgForRound,
    clusterMetrics: clusterMetricsForRound.length > 0 ? clusterMetricsForRound : undefined,
    clientMetrics: clientMetricsForRound.length > 0 ? clientMetricsForRound : undefined,
    globalModelWeights: {
      layers: newGlobalModel.layers,
      bias: newGlobalModel.bias,
      version: newGlobalModel.version,
    },
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
