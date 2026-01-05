import { ModelWeights, AggregationFunction } from './types';

/**
 * FedAvg - Federated Averaging
 * Weighted average of model weights based on client data sizes
 */
export const fedAvg: AggregationFunction = (clientWeights) => {
  if (clientWeights.length === 0) {
    throw new Error('No client weights to aggregate');
  }

  const totalDataSize = clientWeights.reduce((sum, c) => sum + c.dataSize, 0);
  
  const numLayers = clientWeights[0].weights.layers.length;
  const layerSizes = clientWeights[0].weights.layers.map(l => l.length);
  const biasSize = clientWeights[0].weights.bias.length;

  // Initialize aggregated weights
  const aggregatedLayers: number[][] = layerSizes.map(size => new Array(size).fill(0));
  const aggregatedBias: number[] = new Array(biasSize).fill(0);

  // Weighted sum
  for (const { weights, dataSize } of clientWeights) {
    const weight = dataSize / totalDataSize;
    
    for (let l = 0; l < numLayers; l++) {
      for (let i = 0; i < weights.layers[l].length; i++) {
        aggregatedLayers[l][i] += weights.layers[l][i] * weight;
      }
    }
    
    for (let i = 0; i < biasSize; i++) {
      aggregatedBias[i] += weights.bias[i] * weight;
    }
  }

  return {
    layers: aggregatedLayers,
    bias: aggregatedBias,
    version: clientWeights[0].weights.version + 1,
  };
};

/**
 * FedProx - Federated Proximal
 * Similar to FedAvg but with proximal term consideration
 */
export const fedProx: AggregationFunction = (clientWeights) => {
  // For simulation, FedProx behaves similarly to FedAvg at aggregation time
  // The proximal term is applied during local training
  return fedAvg(clientWeights);
};

/**
 * Simple Average - Equal weight for all clients
 */
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

/**
 * Median Aggregation - Robust to outliers
 */
export const medianAggregation: AggregationFunction = (clientWeights) => {
  if (clientWeights.length === 0) {
    throw new Error('No client weights to aggregate');
  }

  const median = (arr: number[]): number => {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  };

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
