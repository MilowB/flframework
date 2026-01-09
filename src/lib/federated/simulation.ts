import { ModelWeights, ClientState, RoundMetrics, FederatedState, WeightsSnapshot, ServerStatus, ClusterMetrics, ClientRoundMetrics } from './types';
import { aggregationMethods } from './aggregations';
import { 
  initializeMLPWeights, 
  initializeMLPWeightsWithRng,
  flattenWeights, 
  unflattenWeights, 
  trainEpoch,
  trainEpochWithRng,
  computeAccuracy,
  cloneWeights,
  MLPWeights,
  MNIST_INPUT_SIZE,
  MNIST_HIDDEN_SIZE,
  MNIST_OUTPUT_SIZE,
  logClientModelDifferences
} from './mlp';
import { loadMNISTTrain, loadMNISTTest, getClientDataSubset, oneHot, MNISTData } from './mnist';
import { forward, computeLoss, computeAccuracy as computeMLPAccuracy } from './mlp';

// ============= Seeded Random Number Generator =============
// Xorshift32 PRNG for deterministic randomness
export class SeededRandom {
  private state: number;
  
  constructor(seed: number) {
    this.state = seed >>> 0 || 1;
  }
  
  // Returns a number in [0, 1)
  next(): number {
    this.state ^= this.state << 13;
    this.state ^= this.state >>> 17;
    this.state ^= this.state << 5;
    return (this.state >>> 0) / 4294967296;
  }
  
  // Returns an integer in [0, max)
  nextInt(max: number): number {
    return Math.floor(this.next() * max);
  }
  
  // Shuffle an array in place
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.nextInt(i + 1);
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
}

// Global RNG instance, reset when simulation starts
let globalRng: SeededRandom = new SeededRandom(42);
let currentSeed: number = 42;

export const setSeed = (seed: number): void => {
  currentSeed = seed;
  globalRng = new SeededRandom(seed);
};

export const getRng = (): SeededRandom => globalRng;
export const getSeed = (): number => currentSeed;

// --- Clustering utilities: compute L2 distances and perform community detection
// We'll convert distances to similarity weights and run a Louvain-like
// community detection with a small refinement step (Leiden-like behavior).

const vectorizeModel = (m: ModelWeights): number[] => {
  const vec: number[] = [];
  for (const layer of m.layers) {
    for (const v of layer) vec.push(v);
  }
  for (const b of m.bias) vec.push(b);
  return vec;
};

const l2Distance = (a: number[], b: number[]): number => {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
};

const computeDistanceMatrix = (models: ModelWeights[]): number[][] => {
  const n = models.length;
  const vecs = models.map(vectorizeModel);
  const D: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = l2Distance(vecs[i], vecs[j]);
      D[i][j] = d;
      D[j][i] = d;
    }
  }
  return D;
};

const distancesToAdjacency = (D: number[][]): number[][] => {
  // Convert distance to similarity weight. Use an RBF-like transform:
  // weight = exp(-d / meanDist). This ensures closer models have larger weight.
  const n = D.length;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      sum += D[i][j];
      count++;
    }
  }
  const mean = count > 0 ? sum / count : 1;
  const sigma = mean || 1;
  const A: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const w = Math.exp(-D[i][j] / sigma);
      A[i][j] = w;
    }
  }
  return A;
};

// Simple Louvain implementation (suitable for small N). Returns community id per node.
const louvainPartition = (A: number[][]): number[] => {
  const n = A.length;
  if (n === 0) return [];

  // total weight (each edge counted once)
  let m = 0;
  const k = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      k[i] += A[i][j];
    }
    m += k[i];
  }
  m = m / 2; // undirected

  if (m === 0) {
    // no edges / all-zero adjacency: return each node in its own community
    const single: number[] = [];
    for (let i = 0; i < n; i++) single.push(i);
    return single;
  }

  // initial communities: each node its own
  const community = new Array(n);
  for (let i = 0; i < n; i++) community[i] = i;

  // compute sum_tot per community
  const sumTot = k.slice();

  let improvement = true;
  const maxPasses = 10;
  let pass = 0;
  while (improvement && pass < maxPasses) {
    improvement = false;
    pass++;
    // iterate nodes in random order
    const nodes = Array.from({ length: n }, (_, i) => i).sort(() => Math.random() - 0.5);
    for (const i of nodes) {
      const oldC = community[i];
      // compute k_i_in for each neighboring community
      const neighCommWeights = new Map<number, number>();
      for (let j = 0; j < n; j++) {
        if (A[i][j] <= 0) continue;
        const c = community[j];
        neighCommWeights.set(c, (neighCommWeights.get(c) || 0) + A[i][j]);
      }

      // remove i temporarily
      sumTot[oldC] -= k[i];

      let bestC = oldC;
      let bestDelta = 0;
      let resolution = 2;
      for (const [c, k_i_in] of neighCommWeights.entries()) {
        const deltaQ = (k_i_in - resolution * (k[i] * sumTot[c]) / (2 * m)) / (2 * m);
        if (deltaQ > bestDelta) {
          bestDelta = deltaQ;
          bestC = c;
        }
      }

      // if best community is different, move
      if (bestC !== oldC) {
        community[i] = bestC;
        sumTot[bestC] += k[i];
        improvement = true;
      } else {
        // restore
        sumTot[oldC] += k[i];
      }
    }
  }

  // normalize community ids to contiguous labels
  const labelMap = new Map<number, number>();
  let nextLabel = 0;
  for (let i = 0; i < n; i++) {
    const c = community[i];
    if (!labelMap.has(c)) labelMap.set(c, nextLabel++);
  }
  for (let i = 0; i < n; i++) community[i] = labelMap.get(community[i])!;

  return community;
};

// Small refinement: ensure each community is reasonably internally connected by
// attempting to move weakly attached nodes to better communities (Leiden-like).
const refinePartition = (A: number[][], partition: number[]): number[] => {
  const n = A.length;
  const k = new Array(n).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) k[i] += A[i][j];

  // compute community internal degree
  const commNodes = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const c = partition[i];
    if (!commNodes.has(c)) commNodes.set(c, []);
    commNodes.get(c)!.push(i);
  }

  // For each node, if its strongest neighbor community is different and stronger
  // than its current internal connection, move it.
  for (let i = 0; i < n; i++) {
    const current = partition[i];
    let bestC = current;
    let bestWeight = 0;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const c = partition[j];
      const w = A[i][j];
      if (w > bestWeight) {
        bestWeight = w;
        bestC = c;
      }
    }
    // move if best community gives strictly more connections
    if (bestC !== current) {
      // compute current internal weight
      let internal = 0;
      for (const v of (commNodes.get(current) || [])) internal += A[i][v];
      // compute candidate internal weight
      let candidate = 0;
      for (const v of (commNodes.get(bestC) || [])) candidate += A[i][v];
      if (candidate > internal) {
        partition[i] = bestC;
        // update maps
        commNodes.get(current)!.splice(commNodes.get(current)!.indexOf(i), 1);
        if (!commNodes.has(bestC)) commNodes.set(bestC, []);
        commNodes.get(bestC)!.push(i);
      }
    }
  }

  // normalize labels
  const labelMap = new Map<number, number>();
  let next = 0;
  for (let i = 0; i < n; i++) {
    const c = partition[i];
    if (!labelMap.has(c)) labelMap.set(c, next++);
  }
  for (let i = 0; i < n; i++) partition[i] = labelMap.get(partition[i])!;

  return partition;
};

export const clusterClientModels = (clientResults: { id?: string; weights: ModelWeights; dataSize: number }[]) => {
  const models = clientResults.map(c => c.weights);
  const ids = clientResults.map((_, i) => i);
  const D = computeDistanceMatrix(models);
  if (models.length === 0) return { distanceMatrix: D, clusters: [] as string[][] };

  const A = distancesToAdjacency(D);
  const partition = louvainPartition(A);
  const refined = refinePartition(A, partition.slice());

  // build clusters of client ids (use original client ids strings if available)
  const clustersMap = new Map<number, string[]>();
  for (let i = 0; i < refined.length; i++) {
    const c = refined[i];
    if (!clustersMap.has(c)) clustersMap.set(c, []);
    const id = clientResults[i] && clientResults[i].id ? clientResults[i].id : `client-${i}`;
    clustersMap.get(c)!.push(id);
  }

  const clusters: string[][] = Array.from(clustersMap.values());
  return { distanceMatrix: D, clusters };
};

// Compute weights statistics for visualization
const computeWeightsSnapshot = (model: ModelWeights): WeightsSnapshot => {
  const W1 = model.layers[0];
  const W2 = model.layers[1];
  const b1 = model.bias.slice(0, MNIST_HIDDEN_SIZE);
  const b2 = model.bias.slice(MNIST_HIDDEN_SIZE);

  const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const std = (arr: number[]) => {
    const m = mean(arr);
    return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
  };

  return {
    W1Mean: mean(W1),
    W1Std: std(W1),
    W2Mean: mean(W2),
    W2Std: std(W2),
    b1Mean: mean(b1),
    b2Mean: mean(b2),
  };
};

// Store MLP weights and MNIST data
const mlpWeightsStore: Map<string, MLPWeights> = new Map();
const clientDataStore: Map<string, { inputs: number[][]; outputs: number[][] }> = new Map();
const clientTestDataStore: Map<string, { inputs: number[][]; outputs: number[][] }> = new Map();
let mnistTrainData: MNISTData | null = null;
let mnistTestData: MNISTData | null = null;
// Store per-client model to send (cluster-averaged). Keyed by client id.
export const clusterModelStore: Map<string, ModelWeights> = new Map();

// Getter for client models (used by save feature)
export const getClientModels = (): Map<string, ModelWeights> => {
  return new Map(clusterModelStore);
};

// Setter for client models (used by load feature)
export const setClientModels = (models: Map<string, ModelWeights>): void => {
  clusterModelStore.clear();
  models.forEach((value, key) => clusterModelStore.set(key, value));
};

// Preload MNIST data (train + test)
export const preloadMNIST = async (): Promise<void> => {
  const promises: Promise<void>[] = [];
  if (!mnistTrainData) {
    promises.push(loadMNISTTrain().then(data => { mnistTrainData = data; }));
  }
  if (!mnistTestData) {
    promises.push(loadMNISTTest().then(data => { mnistTestData = data; }));
  }
  await Promise.all(promises);
};

// Evaluate global model on test set
export const evaluateOnTestSet = (globalModel: { layers: number[][]; bias: number[] }): { loss: number; accuracy: number } => {
  if (!mnistTestData) {
    return { loss: 0, accuracy: 0 };
  }
  
  const mlpWeights = unflattenWeights(
    globalModel,
    MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE
  );
  
  let totalLoss = 0;
  let correct = 0;
  
  for (let i = 0; i < mnistTestData.images.length; i++) {
    const { output } = forward(mnistTestData.images[i], mlpWeights);
    const target = oneHot(mnistTestData.labels[i]);
    
    totalLoss += computeLoss(output, target);
    
    const predicted = output.indexOf(Math.max(...output));
    if (predicted === mnistTestData.labels[i]) {
      correct++;
    }
  }
  
  return {
    loss: totalLoss / mnistTestData.images.length,
    accuracy: correct / mnistTestData.images.length,
  };
};

// Initialize random model weights (using real MLP for MNIST) with seeded RNG
export const initializeModel = (architecture: string): ModelWeights => {
  const rng = getRng();
  const mlpWeights = initializeMLPWeightsWithRng(
    () => rng.next(),
    MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE
  );
  mlpWeightsStore.set('global', mlpWeights);
  // clear any stored cluster models when initializing
  clusterModelStore.clear();
  // Clear client data stores so they get regenerated with new seed
  clientDataStore.clear();
  clientTestDataStore.clear();
  
  const flat = flattenWeights(mlpWeights);
  return {
    layers: flat.layers,
    bias: flat.bias,
    version: 0,
  };
};

// Generate client names
const clientNames = [
  'Hospital Alpha', 'Clinic Beta', 'Lab Gamma', 'Center Delta',
  'Institute Epsilon', 'Facility Zeta', 'Station Eta', 'Node Theta',
  'Unit Iota', 'Branch Kappa', 'Site Lambda', 'Post Mu',
];

export const createClient = (index: number): ClientState => {
  const rng = getRng();
  return {
    id: `client-${index}`,
    name: clientNames[index % clientNames.length] || `Client ${index + 1}`,
    status: 'idle',
    progress: 0,
    localLoss: 0,
    localAccuracy: 0,
    localTestAccuracy: 0,
    dataSize: Math.floor(rng.next() * 400) + 200, // 200-600 MNIST samples per client
    lastUpdate: Date.now(),
    roundsParticipated: 0,
  };
};

// Generate client-specific test data that mimics training distribution
export const generateClientTestData = (clientId: string, trainDataSize: number): void => {
  if (clientTestDataStore.has(clientId) || !mnistTestData) return;
  
  // Use ~20% of training size for test, minimum 50 samples
  const testSize = Math.max(50, Math.floor(trainDataSize * 0.2));
  
  // Use same non-IID distribution logic but with test data (using global seed)
  const testSubset = getClientDataSubset(mnistTestData, clientId, testSize, true, getSeed());
  clientTestDataStore.set(clientId, testSubset);
};

// Evaluate local model on client's personalized test set
export const evaluateClientOnTestSet = (
  clientId: string,
  localMLP: MLPWeights
): number => {
  const testData = clientTestDataStore.get(clientId);
  if (!testData) return 0;
  
  return computeAccuracy(testData.inputs, testData.outputs, localMLP);
};

// Evaluate a cluster model on pooled test data from all clients in the cluster
export const evaluateClusterModel = (
  clusterClientIds: string[],
  clusterModel: ModelWeights
): number => {
  const mlpWeights = unflattenWeights(
    { layers: clusterModel.layers, bias: clusterModel.bias },
    MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE
  );
  
  // Pool test data from all clients in the cluster
  const allInputs: number[][] = [];
  const allOutputs: number[][] = [];
  
  for (const clientId of clusterClientIds) {
    const testData = clientTestDataStore.get(clientId);
    if (testData) {
      allInputs.push(...testData.inputs);
      allOutputs.push(...testData.outputs);
    }
  }
  
  if (allInputs.length === 0) return 0;
  
  return computeAccuracy(allInputs, allOutputs, mlpWeights);
};

// Real client training with MLP on MNIST data
export const simulateClientTraining = async (
  client: ClientState,
  globalModel: ModelWeights,
  onProgress: (progress: number) => void,
  onStatusUpdate?: (status: 'training' | 'evaluating') => void
): Promise<{ weights: ModelWeights; loss: number; accuracy: number; testAccuracy: number }> => {
  // Ensure MNIST is loaded
  if (!mnistTrainData) {
    mnistTrainData = await loadMNISTTrain();
  }

  // Convert global model to MLP weights
  let globalMLP = unflattenWeights(
    { layers: globalModel.layers, bias: globalModel.bias },
    MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE
  );

  // --- 50/50 aggregation client-side ---
  // Si la config demande une agrégation client et qu'un modèle local existe, on fait la moyenne (FedAvg)
  if (client.lastLocalModel && client.clientAggregationMethod === '50-50') {
    const prevMLP = unflattenWeights(
      { layers: client.lastLocalModel.layers, bias: client.lastLocalModel.bias },
      MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE
    );
    // Moyenne des poids W1
    for (let i = 0; i < globalMLP.W1.length; i++) {
      for (let j = 0; j < globalMLP.W1[i].length; j++) {
        globalMLP.W1[i][j] = (globalMLP.W1[i][j] + prevMLP.W1[i][j]) / 2;
        globalMLP.W1[i][j] = prevMLP.W1[i][j];
      }
    }
    // Moyenne des poids W2
    for (let i = 0; i < globalMLP.W2.length; i++) {
      for (let j = 0; j < globalMLP.W2[i].length; j++) {
        globalMLP.W2[i][j] = (globalMLP.W2[i][j] + prevMLP.W2[i][j]) / 2;
        globalMLP.W2[i][j] = prevMLP.W2[i][j];
      }
    }
    // Moyenne des biais
    for (let i = 0; i < globalMLP.b1.length; i++) {
      globalMLP.b1[i] = (globalMLP.b1[i] + prevMLP.b1[i]) / 2;
      globalMLP.b1[i] = prevMLP.b1[i];
    }
    for (let i = 0; i < globalMLP.b2.length; i++) {
      globalMLP.b2[i] = (globalMLP.b2[i] + prevMLP.b2[i]) / 2;
      globalMLP.b2[i] = prevMLP.b2[i];
    }
  }

  // Clone weights for local training
  const localMLP = cloneWeights(globalMLP);
  
  // Get or generate client-specific MNIST subset (non-IID) using global seed
  if (!clientDataStore.has(client.id)) {
    clientDataStore.set(client.id, getClientDataSubset(mnistTrainData, client.id, client.dataSize, true, getSeed()));
  }
  
  // Generate client-specific test data if not already done
  if (!clientTestDataStore.has(client.id) && mnistTestData) {
    generateClientTestData(client.id, client.dataSize);
  }
  
  const { inputs, outputs } = clientDataStore.get(client.id)!;
  
  // Training configuration for MNIST
  const localEpochs = 3;
  const learningRate = 0.01;
  
  let loss = 0;
  let accuracy = 0;
  
  // Real training loop with seeded RNG for shuffling
  const rng = getRng();
  console.log(`Avant entrainement du modèle client ${client.id}: ${localMLP.W1[0][0]}`);
  for (let epoch = 0; epoch < localEpochs; epoch++) {
    // Small delay to show progress
    await new Promise(resolve => setTimeout(resolve, 100));
    
    loss = trainEpochWithRng(inputs, outputs, localMLP, learningRate, () => rng.next());
    accuracy = computeAccuracy(inputs, outputs, localMLP);
    
    onProgress(((epoch + 1) / localEpochs) * 100);
  }
  console.log(`Après entrainement du modèle client ${client.id}: ${localMLP.W1[0][0]}`);
  
  // Evaluate on personalized test set
  onStatusUpdate?.('evaluating');
  await new Promise(resolve => setTimeout(resolve, 100));
  const testAccuracy = evaluateClientOnTestSet(client.id, localMLP);
  
  // Convert back to flat format
  const flat = flattenWeights(localMLP);
  
  // --- Sauvegarde du modèle local entraîné ---
  // On sauvegarde le modèle local dans le client (pour 50/50 ou autres usages)
  client.lastLocalModel = {
    layers: flat.layers,
    bias: flat.bias,
    version: globalModel.version,
  };

  return {
    weights: {
      layers: flat.layers,
      bias: flat.bias,
      version: globalModel.version,
    },
    loss,
    accuracy,
    testAccuracy,
  };
};

// Select clients for this round using global seeded RNG
export const selectClients = (
  clients: ClientState[],
  count: number
): ClientState[] => {
  const available = clients.filter(c => c.status === 'idle');
  // Shuffle using global RNG
  const rng = getRng();
  const shuffled = available.slice();
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = rng.nextInt(i + 1);
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled.slice(0, Math.min(count, shuffled.length));
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

  // Synchronise la méthode d'agrégation client pour tous les clients
  for (const client of clients) {
    client.clientAggregationMethod = serverConfig.clientAggregationMethod || 'none';
  }

  // Select clients for this round
  const selectedClients = selectClients(clients, serverConfig.clientsPerRound);
  
  if (selectedClients.length < serverConfig.minClientsRequired) {
    throw new Error(`Not enough clients available. Required: ${serverConfig.minClientsRequired}, Available: ${selectedClients.length}`);
  }

  const clientResults: { weights: ModelWeights; dataSize: number }[] = [];
  const clientMetricsForRound: ClientRoundMetrics[] = [];
  const participatingIds = selectedClients.map(c => c.id);

  // Phase 1: Server sends model to clients
  onServerStatusUpdate('sending');
  for (const client of selectedClients) {
    onClientUpdate(client.id, { status: 'receiving', progress: 0 });
    await new Promise(resolve => setTimeout(resolve, 200 + Math.random() * 300));
  }

  // Phase 2: Server waits for clients to train
  onServerStatusUpdate('waiting');
  const trainingPromises = selectedClients.map(async (client) => {
    onClientUpdate(client.id, { status: 'training', progress: 0 });
    // Send cluster-specific model if available, otherwise fall back to global model
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
    });

    return { result, client };
  });

  const trainedClients = await Promise.all(trainingPromises);

  // Phase 3: Server receives models from clients
  onServerStatusUpdate('receiving');
  for (const { result, client } of trainedClients) {
    await new Promise(resolve => setTimeout(resolve, 150 + Math.random() * 200));
    clientResults.push({
      weights: result.weights,
      dataSize: client.dataSize,
    });

    // Collect per-client metrics for this round
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

  // Phase 4: Aggregation and Evaluation
  // Compute distance matrix between client models and cluster them
  // declare silhouette outside the clustering try-block so it's available
  // when constructing round metrics (avoid ReferenceError from block scope)
  let silhouetteAvgForRound: number | undefined = undefined;
  let clusterMetricsForRound: ClusterMetrics[] = [];

  try {
    const clientResultsWithIds = trainedClients.map(({ result, client }) => ({ id: client.id, weights: result.weights, dataSize: client.dataSize }));
    const clustering = clusterClientModels(clientResultsWithIds as any);
    // attach to round metrics later
    // we'll store clustering.distanceMatrix and clustering.clusters
    var distanceMatrixForRound = clustering.distanceMatrix;
    var clustersForRound = clustering.clusters;
    // compute silhouette scores if distance matrix and clusters available
    try {
      const D = distanceMatrixForRound;
      const clusters = clustersForRound;
      if (D && clusters && clusters.length > 0) {
        // build id->index map based on clientResultsWithIds order
        const idToIndex = new Map<string, number>();
        for (let i = 0; i < clientResultsWithIds.length; i++) idToIndex.set(clientResultsWithIds[i].id, i);

        const clusterMeans: number[] = [];
        for (const grp of clusters) {
          const sList: number[] = [];
          for (const id of grp) {
            const i = idToIndex.get(id);
            if (i === undefined) continue;
            // a = average distance to other points in same cluster
            const others = grp.map(x => idToIndex.get(x)).filter((x): x is number => typeof x === 'number' && x !== i);
            let a = 0;
            if (others.length > 0) {
              let sum = 0;
              for (const j of others) sum += D[i][j];
              a = sum / others.length;
            } else {
              a = 0; // single-member cluster
            }

            // b = min average distance to points in other clusters
            let b = Infinity;
            for (const otherGrp of clusters) {
              if (otherGrp === grp) continue;
              const members = otherGrp.map(x => idToIndex.get(x)).filter((x): x is number => typeof x === 'number');
              if (members.length === 0) continue;
              let sum = 0;
              for (const j of members) sum += D[i][j];
              const avg = sum / members.length;
              if (avg < b) b = avg;
            }
            if (!isFinite(b)) b = a; // no other clusters

            const denom = Math.max(a, b);
            const s = denom > 0 ? (b - a) / denom : 0;
            sList.push(s);
          }
          // mean silhouette for this cluster
          const clusterMean = sList.length > 0 ? sList.reduce((u, v) => u + v, 0) / sList.length : 0;
          clusterMeans.push(clusterMean);
        }
        // average across clusters
        silhouetteAvgForRound = clusterMeans.length > 0 ? clusterMeans.reduce((u, v) => u + v, 0) / clusterMeans.length : undefined;
      }
    } catch (err) {
      console.warn('Failed to compute silhouette:', err);
      silhouetteAvgForRound = undefined;
    }
    
    // Build cluster-averaged models (FedAvg per cluster) and store per-client model
    // Also evaluate each cluster model
    try {
      const clientMap = new Map(clientResultsWithIds.map(c => [c.id, c] as [string, typeof c]));
      if (clustersForRound && clustersForRound.length > 0) {
        for (let clusterIdx = 0; clusterIdx < clustersForRound.length; clusterIdx++) {
          const grp = clustersForRound[clusterIdx];
          // gather valid entries for this cluster
          const entries = grp.map(id => clientMap.get(id)).filter(Boolean) as { id: string; weights: ModelWeights; dataSize: number }[];
          if (entries.length === 0) continue;

          // initialize sums
          const layerCount = entries[0].weights.layers.length;
          const sumLayers: number[][] = entries[0].weights.layers.map(layer => new Array(layer.length).fill(0));
          const sumBias: number[] = new Array(entries[0].weights.bias.length).fill(0);
          let totalData = 0;
          for (const e of entries) {
            const w = e.weights;
            const ds = e.dataSize || 1;
            totalData += ds;
            for (let li = 0; li < layerCount; li++) {
              const layer = w.layers[li];
              for (let k = 0; k < layer.length; k++) {
                sumLayers[li][k] += layer[k] * ds;
              }
            }
            for (let b = 0; b < w.bias.length; b++) sumBias[b] += w.bias[b] * ds;
          }

          const avgLayers = sumLayers.map(layer => layer.map(v => v / Math.max(1, totalData)));
          const avgBias = sumBias.map(v => v / Math.max(1, totalData));

          const averagedModel: ModelWeights = { layers: avgLayers, bias: avgBias, version: (entries[0].weights.version || 0) };

          // store averaged model for every member of this cluster
          for (const e of entries) {
            clusterModelStore.set(e.id, averagedModel);
          }

          // Evaluate cluster model on pooled test data
          const clusterAccuracy = evaluateClusterModel(grp, averagedModel);
          clusterMetricsForRound.push({
            clusterId: clusterIdx,
            accuracy: clusterAccuracy,
            clientIds: grp,
          });
        }
      }


      // Comparaison des modèles clients (log)
      const clientModelMap: Record<string, MLPWeights> = {};
      for (const c of clientResultsWithIds) {
        if (c.id && c.weights) {
          clientModelMap[c.id] = unflattenWeights(c.weights, MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE);
        }
      }
      logClientModelDifferences(clientModelMap);
    } catch (err) {
      console.warn('Failed to compute cluster-averaged models:', err);
    }
  } catch (err) {
    console.warn('Clustering failed:', err);
    distanceMatrixForRound = undefined;
    clustersForRound = undefined;
  }

  onServerStatusUpdate('evaluating');
  const aggregationFn = aggregationMethods[serverConfig.aggregationMethod]?.fn || aggregationMethods.fedavg.fn;
  const aggregationStart = Date.now();
  const newGlobalModel = aggregationFn(clientResults);
  const aggregationTime = Date.now() - aggregationStart;

  // Evaluate global model on test set
  const testMetrics = evaluateOnTestSet(newGlobalModel);

  // Server completed this round
  onServerStatusUpdate('completed');

  // Reset completed clients to idle
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
