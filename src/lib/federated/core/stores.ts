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

// Store sm history per client for Alexandre strategy (clientId -> sm[])
export const cosineHistoryStore: Map<string, number[]> = new Map();
export const gradientNormHistoryStore: Map<string, number[]> = new Map();

// Distribution config (set from UI before training starts)
export let distributionConfig: { type: '70-30' | 'dirichlet' | 'iid'; dirichletAlpha: number; muFraction: number } = { type: '70-30', dirichletAlpha: 0.5, muFraction: 40 };

export const setDistributionConfig = (type: '70-30' | 'dirichlet' | 'iid', dirichletAlpha: number = 0.5, muFraction: number = 40): void => {
  console.log(`[setDistributionConfig] Setting to type=${type}, dirichletAlpha=${dirichletAlpha}, muFraction=${muFraction}`);
  distributionConfig = { type, dirichletAlpha, muFraction };
};

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
  cosineHistoryStore.clear();
  gradientNormHistoryStore.clear();
};
