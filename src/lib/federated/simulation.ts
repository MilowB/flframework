import { ModelWeights, ClientState, RoundMetrics, ServerConfig, FederatedState } from './types';
import { aggregationMethods } from './aggregations';

// Initialize random model weights
export const initializeModel = (architecture: string): ModelWeights => {
  const architectures: Record<string, { layers: number[]; bias: number }> = {
    'mlp-small': { layers: [784, 128, 10], bias: 10 },
    'mlp-medium': { layers: [784, 256, 128, 10], bias: 10 },
    'mlp-large': { layers: [784, 512, 256, 128, 10], bias: 10 },
    'cnn-simple': { layers: [32, 64, 128, 10], bias: 10 },
    'resnet-mini': { layers: [64, 128, 256, 512, 10], bias: 10 },
  };

  const arch = architectures[architecture] || architectures['mlp-small'];
  
  return {
    layers: arch.layers.map(size => 
      Array.from({ length: size }, () => (Math.random() - 0.5) * 2)
    ),
    bias: Array.from({ length: arch.bias }, () => (Math.random() - 0.5) * 0.1),
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
  dataSize: Math.floor(Math.random() * 5000) + 1000,
  lastUpdate: Date.now(),
  roundsParticipated: 0,
});

// Simulate client training
export const simulateClientTraining = async (
  client: ClientState,
  globalModel: ModelWeights,
  onProgress: (progress: number) => void
): Promise<{ weights: ModelWeights; loss: number; accuracy: number }> => {
  const steps = 10;
  let loss = 2.5 - Math.random() * 0.5;
  let accuracy = 0.1 + Math.random() * 0.1;

  for (let i = 0; i < steps; i++) {
    await new Promise(resolve => setTimeout(resolve, 100 + Math.random() * 200));
    
    // Simulate training progress
    loss *= 0.85 + Math.random() * 0.1;
    accuracy += (1 - accuracy) * (0.08 + Math.random() * 0.05);
    
    onProgress(((i + 1) / steps) * 100);
  }

  // Create updated weights with small perturbations
  const newWeights: ModelWeights = {
    layers: globalModel.layers.map(layer =>
      layer.map(w => w + (Math.random() - 0.5) * 0.1)
    ),
    bias: globalModel.bias.map(b => b + (Math.random() - 0.5) * 0.05),
    version: globalModel.version,
  };

  return {
    weights: newWeights,
    loss: Math.max(0.01, loss),
    accuracy: Math.min(0.99, accuracy),
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
  };

  onStateUpdate({
    globalModel: newGlobalModel,
    currentRound: currentRound + 1,
    roundHistory: [...state.roundHistory, roundMetrics],
  });

  return roundMetrics;
};
