// Server Aggregation Strategy: FedProx
// Federated Proximal - Similar to FedAvg but with proximal term consideration
// The proximal term is applied during local training, not at aggregation time

import type { AggregationFunction } from '../../core/types';
import { fedAvg } from './fedavg';

export const fedProx: AggregationFunction = (clientWeights) => {
  // For simulation, FedProx behaves similarly to FedAvg at aggregation time
  // The proximal term is applied during local training
  return fedAvg(clientWeights);
};
