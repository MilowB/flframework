// Client Aggregation Strategies
export { applyNoneAggregation } from './none';
export { applyFiftyFiftyAggregation } from './fiftyFifty';

import type { MLPWeights } from '../../models/mlp';
import { applyNoneAggregation } from './none';
import { applyFiftyFiftyAggregation } from './fiftyFifty';

export type ClientAggregationMethod = 'none' | '50-50';

export const applyClientAggregation = (
  method: ClientAggregationMethod,
  receivedModel: MLPWeights,
  previousLocalModel: MLPWeights | null
): MLPWeights => {
  switch (method) {
    case '50-50':
      return applyFiftyFiftyAggregation(receivedModel, previousLocalModel);
    case 'none':
    default:
      return applyNoneAggregation(receivedModel, previousLocalModel);
  }
};
