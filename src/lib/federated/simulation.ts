import { ModelWeights, ClientState, RoundMetrics, FederatedState, WeightsSnapshot } from './types';
import { aggregationMethods } from './aggregations';
import { 
  initializeMLPWeights, 
  flattenWeights, 
  unflattenWeights, 
  generateClientData,
  trainEpoch,
  computeAccuracy,
  cloneWeights,
  MLPWeights
} from './mlp';

// Compute weights statistics for visualization
const computeWeightsSnapshot = (model: ModelWeights): WeightsSnapshot => {
  const W1 = model.layers[0];
  const W2 = model.layers[1];
  const b1 = model.bias.slice(0, 8);
  const b2 = model.bias.slice(8);

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

// MLP Configuration
const INPUT_SIZE = 2;
const HIDDEN_SIZE = 8;
const OUTPUT_SIZE = 1;

// Store MLP weights for each entity
const mlpWeightsStore: Map<string, MLPWeights> = new Map();
const clientDataStore: Map<string, { inputs: number[][]; outputs: number[][] }> = new Map();

// Initialize random model weights (using real MLP)
export const initializeModel = (architecture: string): ModelWeights => {
  const mlpWeights = initializeMLPWeights(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
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
  dataSize: Math.floor(Math.random() * 100) + 50, // Smaller dataset for XOR
  lastUpdate: Date.now(),
  roundsParticipated: 0,
});

// Real client training with MLP on XOR data
export const simulateClientTraining = async (
  client: ClientState,
  globalModel: ModelWeights,
  onProgress: (progress: number) => void
): Promise<{ weights: ModelWeights; loss: number; accuracy: number }> => {
  // Convert global model to MLP weights
  const globalMLP = unflattenWeights(
    { layers: globalModel.layers, bias: globalModel.bias },
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
  );
  
  // Clone weights for local training
  const localMLP = cloneWeights(globalMLP);
  
  // Get or generate client-specific data
  if (!clientDataStore.has(client.id)) {
    // Each client has different data (non-IID simulation)
    const noiseLevel = 0.05 + Math.random() * 0.1;
    clientDataStore.set(client.id, generateClientData(client.dataSize, noiseLevel));
  }
  const { inputs, outputs } = clientDataStore.get(client.id)!;
  
  // Training configuration
  const localEpochs = 5;
  const learningRate = 0.5;
  
  let loss = 0;
  let accuracy = 0;
  
  // Real training loop
  for (let epoch = 0; epoch < localEpochs; epoch++) {
    // Small delay to show progress
    await new Promise(resolve => setTimeout(resolve, 50));
    
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

  // Calculate global metrics
  const avgLoss = results.reduce((sum, r) => sum + r.loss, 0) / results.length;
  const avgAccuracy = results.reduce((sum, r) => sum + r.accuracy, 0) / results.length;

  // Reset completed clients to idle
  setTimeout(() => {
    for (const client of selectedClients) {
      onClientUpdate(client.id, { status: 'idle', progress: 0 });
    }
  }, 500);

  const roundMetrics: RoundMetrics = {
    round: currentRound,
    globalLoss: avgLoss,
    globalAccuracy: avgAccuracy,
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
