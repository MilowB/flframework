// Server Aggregation Strategies
export { fedAvg } from './fedavg';
export { fedProx } from './fedprox';
export { simpleAverage } from './simple';
export { medianAggregation } from './median';

import type { AggregationFunction } from '../../core/types';
import { fedAvg } from './fedavg';
import { fedProx } from './fedprox';
import { simpleAverage } from './simple';
import { medianAggregation } from './median';

export const aggregationMethods: Record<string, { fn: AggregationFunction; name: string; description: string }> = {
  fedavg: {
    fn: fedAvg,
    name: 'FedAvg',
    description: 'Weighted average based on client data sizes',
  },
  fedprox: {
    fn: fedProx,
    name: 'FedProx',
    description: 'FedAvg with proximal regularization',
  },
  simple: {
    fn: simpleAverage,
    name: 'Simple Average',
    description: 'Equal weight for all clients',
  },
  median: {
    fn: medianAggregation,
    name: 'Median',
    description: 'Robust aggregation using median',
  },
};
