// Client training logic

import type { ClientState, ModelWeights } from '../core/types';
import type { MLPWeights } from '../models/mlp';
import { getRng, getSeed } from '../core/random';
import { 
  clientDataStore, 
  clientTestDataStore, 
  mnistTrainData, 
  mnistTestData,
  setMnistTrainData
} from '../core/stores';
import {
  unflattenWeights,
  flattenWeights,
  cloneWeights,
  trainEpochWithRng,
  computeAccuracy,
  vectorizeModel,
  pca3D_single,
  MNIST_INPUT_SIZE,
  MNIST_HIDDEN_SIZE,
  MNIST_OUTPUT_SIZE
} from '../models/mlp';
import { loadMNISTTrain, getClientDataSubset, oneHot } from '../data/mnist';
import { applyClientAggregation } from './aggregation';

// Generate client-specific test data that mimics training distribution
export const generateClientTestData = (clientId: string, trainDataSize: number): void => {
  if (clientTestDataStore.has(clientId) || !mnistTestData) return;
  
  const testSize = Math.max(50, Math.floor(trainDataSize * 0.2));
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

// Real client training with MLP on MNIST data
export const simulateClientTraining = async (
  client: ClientState,
  globalModel: ModelWeights,
  onProgress: (progress: number) => void,
  onStatusUpdate?: (status: 'training' | 'evaluating') => void
): Promise<{ weights: ModelWeights; loss: number; accuracy: number; testAccuracy: number }> => {
  // Ensure MNIST is loaded
  let trainData = mnistTrainData;
  if (!trainData) {
    trainData = await loadMNISTTrain();
    setMnistTrainData(trainData);
  }

  // Convert global model to MLP weights
  const receivedMLP = unflattenWeights(
    { layers: globalModel.layers, bias: globalModel.bias },
    MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE
  );

  // Apply client aggregation strategy (None or 50/50)
  const previousLocalMLP = client.lastLocalModel 
    ? unflattenWeights(
        { layers: client.lastLocalModel.layers, bias: client.lastLocalModel.bias },
        MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE
      )
    : null;
  
  const aggregationMethod = client.clientAggregationMethod || 'none';
  const startingModel = applyClientAggregation(aggregationMethod, receivedMLP, previousLocalMLP);

  // Clone weights for local training
  const localMLP = cloneWeights(startingModel);
  
  // Get or generate client-specific MNIST subset (non-IID) using global seed
  if (!clientDataStore.has(client.id)) {
    clientDataStore.set(client.id, getClientDataSubset(trainData, client.id, client.dataSize, true, getSeed()));
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
  for (let epoch = 0; epoch < localEpochs; epoch++) {
    await new Promise(resolve => setTimeout(resolve, 100));
    
    loss = trainEpochWithRng(inputs, outputs, localMLP, learningRate, () => rng.next());
    accuracy = computeAccuracy(inputs, outputs, localMLP);
    
    onProgress(((epoch + 1) / localEpochs) * 100);
  }
  
  // Evaluate on personalized test set
  onStatusUpdate?.('evaluating');
  await new Promise(resolve => setTimeout(resolve, 100));
  const testAccuracy = evaluateClientOnTestSet(client.id, localMLP);
  
  // Convert back to flat format
  const flat = flattenWeights(localMLP);
  
  // Save the trained local model for future 50/50 aggregation
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
  const rng = getRng();
  const shuffled = available.slice();
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = rng.nextInt(i + 1);
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled.slice(0, Math.min(count, shuffled.length));
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
    dataSize: Math.floor(rng.next() * 400) + 200,
    lastUpdate: Date.now(),
    roundsParticipated: 0,
  };
};
