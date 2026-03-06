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
import { applyAssignment, recordClientCosineSimilarity, detectCosineSimilarityDrop, findMostApproachedCluster, resetCosineSimilarityStores, getUnHappyClients, resetUnHappyClients, type AlexandreContext, type AssignmentMethod } from './assignment';
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

/**
 * Initialize Byzantine objective models on first round
 */
const initializeByzantineObjectivesIfNeeded = async (
  currentRound: number,
  byzantineCount: number,
  clients: ClientState[],
  globalModel: ModelWeights
): Promise<void> => {
  if (currentRound === 0 && byzantineCount > 0) {
    const { initializeByzantineObjective } = await import('./attacks');
    const sortedIds = clients
      .map(c => c.id)
      .sort((a, b) => {
        const numA = parseInt(a.split('-')[1] || '0', 10);
        const numB = parseInt(b.split('-')[1] || '0', 10);
        return numA - numB;
      });
    const byzantineClientIds = sortedIds.slice(0, Math.min(byzantineCount, sortedIds.length));
    initializeByzantineObjective(byzantineClientIds, globalModel);
  }
};

/**
 * Identify Byzantine clients for current round
 */
const identifyByzantineClients = (
  selectedClients: ClientState[],
  byzantineCount: number
): Set<string> => {
  const sortedIds = selectedClients
    .map(c => c.id)
    .sort((a, b) => {
      const numA = parseInt(a.split('-')[1] || '0', 10);
      const numB = parseInt(b.split('-')[1] || '0', 10);
      return numA - numB;
    });
  return new Set(sortedIds.slice(0, Math.min(byzantineCount, sortedIds.length)));
};

/**
 * Build Alexandre context from previous round data
 */
const buildAlexandreContext = (
  state: FederatedState,
  currentRound: number,
  globalModel: ModelWeights
): AlexandreContext | undefined => {
  if (currentRound < 1) return undefined;

  const prevRoundMetrics = state.roundHistory[currentRound - 1];
  if (!prevRoundMetrics?.clusters || !prevRoundMetrics.clientMetrics) return undefined;

  const prevClusters = prevRoundMetrics.clusters;
  const prevClientMetrics = prevRoundMetrics.clientMetrics;

  const alexandreGradientNorms: Record<string, number> = {};
  const alexandreCosineSims: Record<string, number> = {};
  const alexandreClientGradients: Record<string, number[]> = {};

  for (const cm of prevClientMetrics) {
    alexandreGradientNorms[cm.clientId] = cm.gradientNorm || 0;
    alexandreCosineSims[cm.clientId] = cm.clusterCosineSimilarity ?? 0;
  }

  const clusterAssignments: Record<string, number> = {};
  prevClusters.forEach((members, idx) => {
    members.forEach(clientId => { clusterAssignments[clientId] = idx; });
  });

  const alexandreClusterModels = prevClusters.map((_, idx) =>
    clusterModelStore.get(`cluster-${idx}`) || globalModel
  );

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

  return {
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
};

/**
 * Detect cosine similarity drops and compute immediate reassignments
 */
const detectCosineSimilarityReassignments = (
  state: FederatedState,
  currentRound: number,
  selectedClients: ClientState[],
  assignmentMethod: string
): Map<string, number> => {
  const immediateReassignments: Map<string, number> = new Map();

  if (assignmentMethod !== 'CosineSimilarity' || currentRound < 2) {
    return immediateReassignments;
  }

  const currentRoundMetrics = state.roundHistory[currentRound - 1];
  const previousRoundMetrics = state.roundHistory[currentRound - 2];

  const currentDistanceMatrix = currentRoundMetrics?.distanceMatrix;
  const previousDistanceMatrix = previousRoundMetrics?.distanceMatrix;
  const previousClusters = currentRoundMetrics?.clusters;
  const previousParticipants = currentRoundMetrics?.participatingClients;

  if (!currentDistanceMatrix || !previousDistanceMatrix || !previousClusters || !previousParticipants) {
    return immediateReassignments;
  }

  for (const client of selectedClients) {
    if (detectCosineSimilarityDrop(client.id, currentRound)) {
      console.log(`[Cosine Similarity] Client ${client.id} detected drop at round ${currentRound - 1}`);

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

  return immediateReassignments;
};

/**
 * Determine which model to send to a client
 */
const determineModelToSend = (
  client: ClientState,
  globalModel: ModelWeights,
  clustersForRound: string[][] | undefined,
  modelAssignmentMethod: string,
  immediateReassignments: Map<string, number>,
  selectedClients: ClientState[],
  currentRound: number,
  distanceMetric: string,
  alexandreContext: AlexandreContext | undefined
): ModelWeights => {
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

  const reassignedCluster = immediateReassignments.get(client.id);
  if (reassignedCluster !== undefined) {
    const targetClusterModel = clusterModelStore.get(`cluster-${reassignedCluster}`);
    if (targetClusterModel) {
      console.log(`[Cosine Similarity] Client ${client.id} receiving model from cluster ${reassignedCluster} (immediate reassignment)`);
      return targetClusterModel;
    }
  }

  return applyAssignment(
    modelAssignmentMethod as AssignmentMethod,
    client,
    {
      globalModel,
      clusterModels,
      clusterAssignments,
      clusterClientIds,
      selectedClients,
      round: currentRound,
      distanceMetric: distanceMetric as 'l1' | 'l2' | 'cosine',
      alexandreContext,
    }
  );
};

/**
 * Execute client training phase
 */
const executeClientTraining = async (
  selectedClients: ClientState[],
  globalModel: ModelWeights,
  clustersForRound: string[][] | undefined,
  serverConfig: any,
  currentRound: number,
  byzantineClientIds: Set<string>,
  poisoningEpsilon: number,
  immediateReassignments: Map<string, number>,
  alexandreContext: AlexandreContext | undefined,
  onClientUpdate: (clientId: string, update: Partial<ClientState>) => void,
  onServerStatusUpdate: (status: ServerStatus) => void
): Promise<Array<{ result: any; client: ClientState }>> => {
  onServerStatusUpdate('sending');
  for (const client of selectedClients) {
    onClientUpdate(client.id, { status: 'receiving', progress: 0 });
    await new Promise(resolve => setTimeout(resolve, 300));
  }

  onServerStatusUpdate('waiting');
  const modelsSentToClients: Record<string, ModelWeights> = {};

  const trainingPromises = selectedClients.map(async (client) => {
    onClientUpdate(client.id, { status: 'training', progress: 0 });

    const modelToSend = determineModelToSend(
      client,
      globalModel,
      clustersForRound,
      serverConfig.modelAssignmentMethod || '1NN',
      immediateReassignments,
      selectedClients,
      currentRound,
      serverConfig.distanceMetric,
      alexandreContext
    );

    modelsSentToClients[client.id] = modelToSend;

    // Debug PCA 3D (optional)
    try {
      if (client.id === "client-0") {
        const { vectorizeModel, pca3D_single } = await import('./models/mlp');
        const vec = vectorizeModel(unflattenWeights(modelToSend, MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE));
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
      serverConfig.modelArchitecture,
      byzantineClientIds.has(client.id),
      poisoningEpsilon
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

  return Promise.all(trainingPromises);
};

/**
 * Collect client results and metrics
 */
const collectClientResults = async (
  trainedClients: Array<{ result: any; client: ClientState }>,
  onClientUpdate: (clientId: string, update: Partial<ClientState>) => void,
  onServerStatusUpdate: (status: ServerStatus) => void
): Promise<{
  clientResults: Array<{ weights: ModelWeights; dataSize: number }>;
  clientResultsWithClientId: Array<{ weights: ModelWeights; dataSize: number; clientId: string }>;
  clientMetricsForRound: ClientRoundMetrics[];
}> => {
  onServerStatusUpdate('receiving');

  const clientResults: Array<{ weights: ModelWeights; dataSize: number }> = [];
  const clientResultsWithClientId: Array<{ weights: ModelWeights; dataSize: number; clientId: string }> = [];
  const clientMetricsForRound: ClientRoundMetrics[] = [];

  for (const { result, client } of trainedClients) {
    const weightsToUse = result.weights;

    clientResults.push({ weights: weightsToUse, dataSize: client.dataSize });
    clientResultsWithClientId.push({ weights: weightsToUse, dataSize: client.dataSize, clientId: client.id });
    clientMetricsForRound.push({
      clientId: client.id,
      clientName: client.name,
      loss: result.loss,
      accuracy: result.accuracy,
      testAccuracy: result.testAccuracy,
      gradientNorm: result.gradientNorm,
      weights: weightsToUse,
    });

    onClientUpdate(client.id, {
      status: 'completed',
      lastUpdate: Date.now(),
      roundsParticipated: client.roundsParticipated + 1,
    });
  }

  return { clientResults, clientResultsWithClientId, clientMetricsForRound };
};

/**
 * Apply Byzantine attack if active
 */
const applyByzantineAttackIfActive = (
  serverConfig: any,
  currentRound: number,
  participatingIds: string[],
  clientResultsWithClientId: Array<{ weights: ModelWeights; dataSize: number; clientId: string }>,
  clientResults: Array<{ weights: ModelWeights; dataSize: number }>,
  trainedClients: Array<{ result: any; client: ClientState }>,
  clientMetricsForRound: ClientRoundMetrics[],
  globalModel: ModelWeights
): void => {
  const byzantineCount = serverConfig.byzantineCount ?? 0;
  const byzantineIntervals = serverConfig.byzantineActiveIntervals;
  const isByzantineActive = byzantineCount > 0 && (
    !byzantineIntervals || byzantineIntervals.length === 0 ||
    byzantineIntervals.some(iv => currentRound >= iv.start && currentRound <= iv.end)
  );

  if (!isByzantineActive) return;

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

  for (let i = 0; i < clientResults.length; i++) {
    const attacked = attackedResults.find(r => r.clientId === trainedClients[i].client.id);
    if (attacked) {
      clientResults[i] = { weights: attacked.weights, dataSize: attacked.dataSize };
      trainedClients[i].result.weights = attacked.weights;
      const metric = clientMetricsForRound.find(m => m.clientId === attacked.clientId);
      if (metric) {
        metric.weights = attacked.weights;
      }
    }
  }
};

/**
 * Perform clustering and compute cluster metrics
 */
const performClustering = (
  trainedClients: Array<{ result: any; client: ClientState }>,
  serverConfig: any,
  globalModel: ModelWeights,
  currentRound: number
): {
  clustersForRound: string[][] | undefined;
  distanceMatrixForRound: number[][] | undefined;
  agreementMatrixForRound: number[][] | undefined;
  silhouetteAvgForRound: number | undefined;
  clusterMetricsForRound: ClusterMetrics[];
  previousClusterModels: Map<number, ModelWeights>;
} => {
  let clustersForRound: string[][] | undefined = undefined;
  let distanceMatrixForRound: number[][] | undefined = undefined;
  let agreementMatrixForRound: number[][] | undefined = undefined;
  let silhouetteAvgForRound: number | undefined = undefined;
  let clusterMetricsForRound: ClusterMetrics[] = [];
  const previousClusterModels: Map<number, ModelWeights> = new Map();

  try {
    const clientResultsWithIds = trainedClients
      .map(({ result, client }) => ({
        id: client.id,
        weights: result.weights,
        dataSize: client.dataSize
      }))
      .sort((a, b) => {
        const numA = parseInt(a.id.split('-')[1] || '0', 10);
        const numB = parseInt(b.id.split('-')[1] || '0', 10);
        return numA - numB;
      });

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

    const idToIndex = new Map<string, number>();
    clientResultsWithIds.forEach((c, i) => idToIndex.set(c.id, i));
    silhouetteAvgForRound = computeSilhouetteScore(distanceMatrixForRound, clustersForRound, idToIndex);

    if (clustersForRound && clustersForRound.length > 0) {
      const clientMap = new Map(clientResultsWithIds.map(c => [c.id, c]));

      for (let clusterIdx = 0; clusterIdx < clustersForRound.length; clusterIdx++) {
        const prev = clusterModelStore.get(`cluster-${clusterIdx}`);
        if (prev) previousClusterModels.set(clusterIdx, prev);
      }

      for (let clusterIdx = 0; clusterIdx < clustersForRound.length; clusterIdx++) {
        const grp = clustersForRound[clusterIdx];
        const entries = grp.map(id => clientMap.get(id)).filter(Boolean) as typeof clientResultsWithIds;
        if (entries.length === 0) continue;

        console.log(`[Cluster ${clusterIdx}] Clients:`, grp);

        entries.forEach((e) => {
          const sample = e.weights.layers[0].slice(0, 5);
          console.log(`  Client ${e.id}: [${sample.map(v => v.toFixed(4)).join(', ')}...] dataSize=${e.dataSize}`);
        });

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

        const centroidSample = averagedModel.layers[0].slice(0, 5);
        console.log(`  Centroid: [${centroidSample.map(v => v.toFixed(4)).join(', ')}...]`);

        clusterModelStore.set(`cluster-${clusterIdx}`, averagedModel);
        for (const e of entries) clusterModelStore.set(e.id, averagedModel);

        const clusterAccuracy = evaluateClusterModel(grp, averagedModel, clientTestDataStore);
        clusterMetricsForRound.push({
          clusterId: clusterIdx,
          accuracy: clusterAccuracy,
          clientIds: grp,
          weights: averagedModel
        });
      }
    }
  } catch (err) {
    console.warn('Clustering failed:', err);
  }

  return {
    clustersForRound,
    distanceMatrixForRound,
    agreementMatrixForRound,
    silhouetteAvgForRound,
    clusterMetricsForRound,
    previousClusterModels
  };
};

/**
 * Compute cluster cosine similarities for clients
 */
const computeClusterCosineSimilarities = (
  clientMetricsForRound: ClientRoundMetrics[],
  trainedClients: Array<{ result: any; client: ClientState }>,
  selectedClients: ClientState[],
  clustersForRound: string[][] | undefined,
  currentRound: number
): void => {
  if (!clustersForRound || clustersForRound.length === 0 || currentRound === 0) return;

  const clientMap = new Map(trainedClients.map(({ result, client }) => [client.id, { result, client }]));
  const selectedClientsMap = new Map(selectedClients.map(c => [c.id, c]));

  for (let i = 0; i < clientMetricsForRound.length; i++) {
    const clientMetric = clientMetricsForRound[i];
    const clientId = clientMetric.clientId;
    const client = selectedClientsMap.get(clientId);
    const clientData = clientMap.get(clientId);

    if (!client || !clientData) continue;

    const previousModel = client.localModelHistory?.[1];
    if (!previousModel) continue;

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

    const clientDelta = computeModelDelta(clientData.result.weights, previousModel);

    let totalSim = 0;
    let count = 0;

    for (const memberId of clusterMembers) {
      if (memberId === clientId) continue;

      const memberClient = selectedClientsMap.get(memberId);
      const memberData = clientMap.get(memberId);
      if (!memberClient || !memberData) continue;

      const memberPreviousModel = memberClient.localModelHistory?.[1];
      if (!memberPreviousModel) continue;

      const memberDelta = computeModelDelta(memberData.result.weights, memberPreviousModel);
      const similarity = computeCosineSimilarity(clientDelta, memberDelta);
      totalSim += similarity;
      count++;
    }

    if (count > 0) {
      const avgSimilarity = totalSim / count;
      clientMetricsForRound[i].clusterCosineSimilarity = avgSimilarity;
      recordClientCosineSimilarity(clientId, avgSimilarity, currentRound);
    }
  }
};

/**
 * Update Alexandre context with cluster median gradients
 */
const updateAlexandreContextWithMedians = (
  trainedClients: Array<{ result: any; client: ClientState }>,
  clustersForRound: string[][] | undefined,
  globalModel: ModelWeights,
  previousClusterModels: Map<number, ModelWeights>
): AlexandreContext | undefined => {
  if (!clustersForRound || clustersForRound.length === 0) return undefined;

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

  const clusterMedians: Map<number, number[]> = new Map();
  for (const [c, entries] of clusterGradients) {
    if (entries.length > 0) {
      clusterMedians.set(c, computeMedianGradient(entries.map(e => e.delta)));
    }
  }

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
    const norm = Math.sqrt(delta.reduce((s, v) => s + v * v, 0));
    alexandreGradientNorms[client.id] = norm;

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

  const alexandreClusterModels = clustersForRound.map((_, idx) =>
    clusterModelStore.get(`cluster-${idx}`) || globalModel
  );

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

  return {
    gradientNorms: alexandreGradientNorms,
    cosineSimilarities: alexandreCosineSims,
    clusterGradientNorms,
    clusterMedianGradients: alexandreClusterMedianGradients,
    clientGradients: alexandreClientGradients,
    clusterAssignments: Object.fromEntries(
      clustersForRound.flatMap((members, idx) => members.map(id => [id, idx]))
    ),
    distanceMatrix: undefined,
    previousDistanceMatrix: undefined,
    clusters: clustersForRound,
    participatingClients: [],
    clusterModels: alexandreClusterModels,
    globalModel,
  };
};

export const runFederatedRound = async (
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

  // Initialize Byzantine objectives on round 0
  await initializeByzantineObjectivesIfNeeded(currentRound, serverConfig.byzantineCount ?? 0, clients, globalModel);

  // Sync client aggregation method
  for (const client of clients) {
    client.clientAggregationMethod = serverConfig.clientAggregationMethod || 'none';
  }

  const selectedClients = selectClients(clients, serverConfig.clientsPerRound);

  if (selectedClients.length < serverConfig.minClientsRequired) {
    throw new Error(`Not enough clients available. Required: ${serverConfig.minClientsRequired}, Available: ${selectedClients.length}`);
  }

  const byzantineClientIds = identifyByzantineClients(selectedClients, serverConfig.byzantineCount ?? 0);
  const poisoningEpsilon = 0.1;

  const participatingIds = selectedClients
    .map(c => c.id)
    .sort((a, b) => {
      const numA = parseInt(a.split('-')[1] || '0', 10);
      const numB = parseInt(b.split('-')[1] || '0', 10);
      return numA - numB;
    });

  // Build Alexandre context from previous round
  let alexandreContextForRound = serverConfig.modelAssignmentMethod === 'Alexandre'
    ? buildAlexandreContext(state, currentRound, globalModel)
    : undefined;

  // Detect cosine similarity drops and compute immediate reassignments
  const immediateReassignments = detectCosineSimilarityReassignments(
    state,
    currentRound,
    selectedClients,
    serverConfig.modelAssignmentMethod || '1NN'
  );

  // Execute client training
  const trainedClients = await executeClientTraining(
    selectedClients,
    globalModel,
    clustersForRound,
    serverConfig,
    currentRound,
    byzantineClientIds,
    poisoningEpsilon,
    immediateReassignments,
    alexandreContextForRound,
    onClientUpdate,
    onServerStatusUpdate
  );

  // Collect client results
  const { clientResults, clientResultsWithClientId, clientMetricsForRound } = await collectClientResults(
    trainedClients,
    onClientUpdate,
    onServerStatusUpdate
  );

  // Apply Byzantine attack if active
  applyByzantineAttackIfActive(
    serverConfig,
    currentRound,
    participatingIds,
    clientResultsWithClientId,
    clientResults,
    trainedClients,
    clientMetricsForRound,
    globalModel
  );

  // Perform clustering
  const {
    clustersForRound: updatedClusters,
    distanceMatrixForRound,
    agreementMatrixForRound,
    silhouetteAvgForRound,
    clusterMetricsForRound,
    previousClusterModels
  } = performClustering(trainedClients, serverConfig, globalModel, currentRound);
  clustersForRound = updatedClusters;

  // Compute cluster cosine similarities
  computeClusterCosineSimilarities(
    clientMetricsForRound,
    trainedClients,
    selectedClients,
    clustersForRound,
    currentRound
  );

  // Update Alexandre context with median gradients
  alexandreContextForRound = updateAlexandreContextWithMedians(
    trainedClients,
    clustersForRound,
    globalModel,
    previousClusterModels
  ) || alexandreContextForRound;

  // Global model aggregation and evaluation
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
