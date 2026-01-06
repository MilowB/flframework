import { ModelWeights, ClientState, RoundMetrics, FederatedState, WeightsSnapshot } from './types';
import { aggregationMethods } from './aggregations';
import { 
  initializeMLPWeights, 
  flattenWeights, 
  unflattenWeights, 
  trainEpoch,
  computeAccuracy,
  cloneWeights,
  MLPWeights,
  MNIST_INPUT_SIZE,
  MNIST_HIDDEN_SIZE,
  MNIST_OUTPUT_SIZE
} from './mlp';
import { loadMNISTTrain, loadMNISTTest, getClientDataSubset, oneHot, MNISTData } from './mnist';
import { forward, computeLoss, computeAccuracy as computeMLPAccuracy } from './mlp';

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
let mnistTrainData: MNISTData | null = null;
let mnistTestData: MNISTData | null = null;

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

// Initialize random model weights (using real MLP for MNIST)
export const initializeModel = (architecture: string): ModelWeights => {
  const mlpWeights = initializeMLPWeights(MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE);
  mlpWeightsStore.set('global', mlpWeights);
  
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

export const createClient = (index: number): ClientState => ({
  id: `client-${index}`,
  name: clientNames[index % clientNames.length] || `Client ${index + 1}`,
  status: 'idle',
  progress: 0,
  localLoss: 0,
  localAccuracy: 0,
  dataSize: Math.floor(Math.random() * 400) + 200, // 200-600 MNIST samples per client
  lastUpdate: Date.now(),
  roundsParticipated: 0,
});

// Real client training with MLP on MNIST data
export const simulateClientTraining = async (
  client: ClientState,
  globalModel: ModelWeights,
  onProgress: (progress: number) => void
): Promise<{ weights: ModelWeights; loss: number; accuracy: number }> => {
  // Ensure MNIST is loaded
  if (!mnistTrainData) {
    mnistTrainData = await loadMNISTTrain();
  }
  
  // Convert global model to MLP weights
  const globalMLP = unflattenWeights(
    { layers: globalModel.layers, bias: globalModel.bias },
    MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE
  );
  
  // Clone weights for local training
  const localMLP = cloneWeights(globalMLP);
  
  // Get or generate client-specific MNIST subset (non-IID)
  if (!clientDataStore.has(client.id)) {
    clientDataStore.set(client.id, getClientDataSubset(mnistTrainData, client.id, client.dataSize, true));
  }
  const { inputs, outputs } = clientDataStore.get(client.id)!;
  
  // Training configuration for MNIST
  const localEpochs = 3;
  const learningRate = 0.01;
  
  let loss = 0;
  let accuracy = 0;
  
  // Real training loop
  for (let epoch = 0; epoch < localEpochs; epoch++) {
    // Small delay to show progress
    await new Promise(resolve => setTimeout(resolve, 100));
    
    loss = trainEpoch(inputs, outputs, localMLP, learningRate);
    accuracy = computeAccuracy(inputs, outputs, localMLP);
    
    onProgress(((epoch + 1) / localEpochs) * 100);
  }
  
  // Convert back to flat format
  const flat = flattenWeights(localMLP);
  
  return {
    weights: {
      layers: flat.layers,
      bias: flat.bias,
      version: globalModel.version,
    },
    loss,
    accuracy,
  };
};

// Select clients for this round
export const selectClients = (
  clients: ClientState[],
  count: number
): ClientState[] => {
  const available = clients.filter(c => c.status === 'idle');
  const shuffled = [...available].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(count, shuffled.length));
};

// Run a single federated round
export const runFederatedRound = async (
  state: FederatedState,
  onStateUpdate: (state: Partial<FederatedState>) => void,
  onClientUpdate: (clientId: string, update: Partial<ClientState>) => void
): Promise<RoundMetrics> => {
  const { serverConfig, clients, globalModel, currentRound } = state;
  
  if (!globalModel) {
    throw new Error('Global model not initialized');
  }

  // Select clients for this round
  const selectedClients = selectClients(clients, serverConfig.clientsPerRound);
  
  if (selectedClients.length < serverConfig.minClientsRequired) {
    throw new Error(`Not enough clients available. Required: ${serverConfig.minClientsRequired}, Available: ${selectedClients.length}`);
  }

  const clientResults: { weights: ModelWeights; dataSize: number }[] = [];
  const participatingIds = selectedClients.map(c => c.id);

  // Phase 1: Distribute model to clients
  for (const client of selectedClients) {
    onClientUpdate(client.id, { status: 'receiving', progress: 0 });
    await new Promise(resolve => setTimeout(resolve, 200 + Math.random() * 300));
  }

  // Phase 2: Local training
  const trainingPromises = selectedClients.map(async (client) => {
    onClientUpdate(client.id, { status: 'training', progress: 0 });
    
    const result = await simulateClientTraining(
      client,
      globalModel,
      (progress) => onClientUpdate(client.id, { progress })
    );

    onClientUpdate(client.id, {
      status: 'sending',
      progress: 100,
      localLoss: result.loss,
      localAccuracy: result.accuracy,
    });

    await new Promise(resolve => setTimeout(resolve, 150 + Math.random() * 200));

    clientResults.push({
      weights: result.weights,
      dataSize: client.dataSize,
    });

    onClientUpdate(client.id, {
      status: 'completed',
      lastUpdate: Date.now(),
      roundsParticipated: client.roundsParticipated + 1,
    });

    return result;
  });

  const results = await Promise.all(trainingPromises);

  // Phase 3: Aggregation
  const aggregationFn = aggregationMethods[serverConfig.aggregationMethod]?.fn || aggregationMethods.fedavg.fn;
  const aggregationStart = Date.now();
  const newGlobalModel = aggregationFn(clientResults);
  const aggregationTime = Date.now() - aggregationStart;

  // Evaluate global model on test set
  const testMetrics = evaluateOnTestSet(newGlobalModel);

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
  };

  onStateUpdate({
    globalModel: newGlobalModel,
    currentRound: currentRound + 1,
    roundHistory: [...state.roundHistory, roundMetrics],
  });

  return roundMetrics;
};
