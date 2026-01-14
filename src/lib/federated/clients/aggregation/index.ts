// Client Aggregation Strategies
export { applyNoneAggregation } from './none';
export { applyFiftyFiftyAggregation } from './fiftyFifty';
export { applyGravityAggregation } from './gravity';

import type { MLPWeights } from '../../models/mlp';
import { applyNoneAggregation } from './none';
import { applyFiftyFiftyAggregation } from './fiftyFifty';
import { applyGravityAggregation } from './gravity';

export type ClientAggregationMethod = 'none' | '50-50' | 'gravity';

export const applyClientAggregation = (
  method: ClientAggregationMethod,
  receivedModel: MLPWeights,
  previousLocalModel: MLPWeights | null,
  localModelHistory?: Array<{
    layers: number[][];
    bias: number[];
    version: number;
  }>,
  receivedModelHistory?: Array<{
    layers: number[][];
    bias: number[];
    version: number;
  }>,
  currentRound?: number,
  clientId?: string
): MLPWeights => {
  switch (method) {
    case 'gravity':
      return applyGravityAggregation(receivedModel, previousLocalModel, localModelHistory, receivedModelHistory, 0.1, currentRound, clientId);
    case '50-50':
      return applyFiftyFiftyAggregation(receivedModel, previousLocalModel);
    case 'none':
    default:
      return applyNoneAggregation(receivedModel, previousLocalModel);
  }
};
