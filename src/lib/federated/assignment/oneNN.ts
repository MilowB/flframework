// Model Assignment Strategy: 1-Nearest Neighbor (1NN)
// Each client receives the cluster model from the cluster it belongs to

import type { ModelWeights } from '../core/types';
import { clusterModelStore } from '../core/stores';

// Get model to send to client using 1NN assignment
export const getModelFor1NN = (
  clientId: string,
  globalModel: ModelWeights
): ModelWeights => {
  // Return the cluster model if available, otherwise the global model
  return clusterModelStore.get(clientId) || globalModel;
};
