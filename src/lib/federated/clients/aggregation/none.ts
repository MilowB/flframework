// Client Aggregation Strategy: None
// The client starts training directly from the received server/cluster model
// without mixing with any previous local model.

import type { MLPWeights } from '../../models/mlp';

export const applyNoneAggregation = (
  receivedModel: MLPWeights,
  _previousLocalModel: MLPWeights | null
): MLPWeights => {
  // No aggregation - use received model directly
  return receivedModel;
};
