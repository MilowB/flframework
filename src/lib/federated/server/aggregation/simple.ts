// Server Aggregation Strategy: Simple Average
// Equal weight for all clients, regardless of data size

import type { ModelWeights, AggregationFunction } from '../../core/types';

export const simpleAverage: AggregationFunction = (clientWeights) => {
  if (clientWeights.length === 0) {
    throw new Error('No client weights to aggregate');
  }

  const numClients = clientWeights.length;
  const numLayers = clientWeights[0].weights.layers.length;
  const layerSizes = clientWeights[0].weights.layers.map(l => l.length);
  const biasSize = clientWeights[0].weights.bias.length;

  const aggregatedLayers: number[][] = layerSizes.map(size => new Array(size).fill(0));
  const aggregatedBias: number[] = new Array(biasSize).fill(0);

  for (const { weights } of clientWeights) {
    for (let l = 0; l < numLayers; l++) {
      for (let i = 0; i < weights.layers[l].length; i++) {
        aggregatedLayers[l][i] += weights.layers[l][i] / numClients;
      }
    }
    
    for (let i = 0; i < biasSize; i++) {
      aggregatedBias[i] += weights.bias[i] / numClients;
    }
  }

  return {
    layers: aggregatedLayers,
    bias: aggregatedBias,
    version: clientWeights[0].weights.version + 1,
  };
};
