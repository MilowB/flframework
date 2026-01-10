// Shared data stores for federated learning simulation
import type { ModelWeights } from './types';
import type { MLPWeights } from '../models/mlp';
import type { MNISTData } from '../data/mnist';

// Store MLP weights and MNIST data
export const mlpWeightsStore: Map<string, MLPWeights> = new Map();
export const clientDataStore: Map<string, { inputs: number[][]; outputs: number[][] }> = new Map();
export const clientTestDataStore: Map<string, { inputs: number[][]; outputs: number[][] }> = new Map();

// MNIST data caches
export let mnistTrainData: MNISTData | null = null;
export let mnistTestData: MNISTData | null = null;

export const setMnistTrainData = (data: MNISTData): void => {
  mnistTrainData = data;
};

export const setMnistTestData = (data: MNISTData): void => {
  mnistTestData = data;
};

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

// Reset all stores (for clean experiment restart)
export const resetStores = (): void => {
  mlpWeightsStore.clear();
  clientDataStore.clear();
  clientTestDataStore.clear();
  clusterModelStore.clear();
};
