// Server Aggregation Strategy: Median
// Robust aggregation using median - resistant to outliers and Byzantine attacks

import type { ModelWeights, AggregationFunction } from '../../core/types';

const median = (arr: number[]): number => {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

export const medianAggregation: AggregationFunction = (clientWeights) => {
  if (clientWeights.length === 0) {
    throw new Error('No client weights to aggregate');
  }

  const numLayers = clientWeights[0].weights.layers.length;
  const layerSizes = clientWeights[0].weights.layers.map(l => l.length);
  const biasSize = clientWeights[0].weights.bias.length;

  const aggregatedLayers: number[][] = layerSizes.map((size, l) => {
    return Array.from({ length: size }, (_, i) => 
      median(clientWeights.map(c => c.weights.layers[l][i]))
    );
  });

  const aggregatedBias: number[] = Array.from({ length: biasSize }, (_, i) =>
    median(clientWeights.map(c => c.weights.bias[i]))
  );

  return {
    layers: aggregatedLayers,
    bias: aggregatedBias,
    version: clientWeights[0].weights.version + 1,
  };
};
