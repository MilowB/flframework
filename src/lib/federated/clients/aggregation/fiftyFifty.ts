// Client Aggregation Strategy: 50/50
// The client starts training from a 50/50 average between
// the received server/cluster model and its previous local model.

import type { MLPWeights } from '../../models/mlp';
import { cloneWeights } from '../../models/mlp';

export const applyFiftyFiftyAggregation = (
  receivedModel: MLPWeights,
  previousLocalModel: MLPWeights | null
): MLPWeights => {
  // If no previous local model exists, use received model directly
  if (!previousLocalModel) {
    return receivedModel;
  }
  
  // Create a copy to avoid mutating the original
  const result = cloneWeights(receivedModel);
  
  // Average W1 weights
  for (let i = 0; i < result.W1.length; i++) {
    for (let j = 0; j < result.W1[i].length; j++) {
      result.W1[i][j] = (result.W1[i][j] + previousLocalModel.W1[i][j]) / 2;
    }
  }
  
  // Average W2 weights
  for (let i = 0; i < result.W2.length; i++) {
    for (let j = 0; j < result.W2[i].length; j++) {
      result.W2[i][j] = (result.W2[i][j] + previousLocalModel.W2[i][j]) / 2;
    }
  }
  
  // Average biases
  for (let i = 0; i < result.b1.length; i++) {
    result.b1[i] = (result.b1[i] + previousLocalModel.b1[i]) / 2;
  }
  for (let i = 0; i < result.b2.length; i++) {
    result.b2[i] = (result.b2[i] + previousLocalModel.b2[i]) / 2;
  }
  
  return result;
};
