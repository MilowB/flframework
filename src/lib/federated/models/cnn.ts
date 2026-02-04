// Configurable CNN (Convolutional Neural Network) for image classification
// Supports configurable number of conv layers, filter sizes, and fully connected layers

export interface ConvLayerConfig {
  filterCount: number;      // Number of filters (output channels)
  filterSize: number;       // Filter size (e.g., 3 for 3x3)
  stride: number;           // Stride for convolution
  padding: number;          // Padding size
  activation: 'relu' | 'leaky_relu' | 'sigmoid' | 'tanh';
  pooling?: {
    type: 'max' | 'avg';
    size: number;
    stride: number;
  };
}

export interface FCLayerConfig {
  neurons: number;
  activation: 'relu' | 'leaky_relu' | 'sigmoid' | 'tanh' | 'softmax';
  dropout?: number;         // Dropout rate (0-1), only used during training
}

export interface CNNConfig {
  inputShape: {
    width: number;
    height: number;
    channels: number;       // 1 for grayscale (MNIST), 3 for RGB
  };
  convLayers: ConvLayerConfig[];
  fcLayers: FCLayerConfig[];
  outputSize: number;       // Number of output classes
}

export interface CNNWeights {
  // Convolutional layers: filters[layer][outChannel][inChannel][row][col]
  convFilters: number[][][][][];
  convBiases: number[][];   // biases[layer][filterIndex]
  
  // Fully connected layers: weights[layer][inputNeuron][outputNeuron]
  fcWeights: number[][][];
  fcBiases: number[][];     // biases[layer][neuronIndex]
}

// Default MNIST CNN configuration
export const MNIST_CNN_CONFIG: CNNConfig = {
  inputShape: { width: 28, height: 28, channels: 1 },
  convLayers: [
    {
      filterCount: 32,
      filterSize: 3,
      stride: 1,
      padding: 1,
      activation: 'relu',
      pooling: { type: 'max', size: 2, stride: 2 }
    },
    {
      filterCount: 64,
      filterSize: 3,
      stride: 1,
      padding: 1,
      activation: 'relu',
      pooling: { type: 'max', size: 2, stride: 2 }
    }
  ],
  fcLayers: [
    { neurons: 128, activation: 'relu' },
  ],
  outputSize: 10
};

// Lightweight MNIST CNN for faster training
export const MNIST_CNN_LITE_CONFIG: CNNConfig = {
  inputShape: { width: 28, height: 28, channels: 1 },
  convLayers: [
    {
      filterCount: 16,
      filterSize: 3,
      stride: 1,
      padding: 1,
      activation: 'relu',
      pooling: { type: 'max', size: 2, stride: 2 }
    }
  ],
  fcLayers: [
    { neurons: 64, activation: 'relu' },
  ],
  outputSize: 10
};

// Calculate output size after a convolution layer
export const calcConvOutputSize = (
  inputSize: number,
  filterSize: number,
  stride: number,
  padding: number
): number => {
  return Math.floor((inputSize + 2 * padding - filterSize) / stride) + 1;
};

// Calculate output size after a pooling layer
export const calcPoolOutputSize = (
  inputSize: number,
  poolSize: number,
  stride: number
): number => {
  return Math.floor((inputSize - poolSize) / stride) + 1;
};

// Calculate the flattened size after all conv layers
export const calcFlattenedSize = (config: CNNConfig): number => {
  let width = config.inputShape.width;
  let height = config.inputShape.height;
  let channels = config.inputShape.channels;

  for (const layer of config.convLayers) {
    // After convolution
    width = calcConvOutputSize(width, layer.filterSize, layer.stride, layer.padding);
    height = calcConvOutputSize(height, layer.filterSize, layer.stride, layer.padding);
    channels = layer.filterCount;

    // After pooling
    if (layer.pooling) {
      width = calcPoolOutputSize(width, layer.pooling.size, layer.pooling.stride);
      height = calcPoolOutputSize(height, layer.pooling.size, layer.pooling.stride);
    }
  }

  return width * height * channels;
};

// Initialize CNN weights using He initialization for ReLU
export const initializeCNNWeights = (config: CNNConfig): CNNWeights => {
  const convFilters: number[][][][][] = [];
  const convBiases: number[][] = [];
  const fcWeights: number[][][] = [];
  const fcBiases: number[][] = [];

  let inChannels = config.inputShape.channels;

  // Initialize conv layers
  for (const layer of config.convLayers) {
    const filters: number[][][][] = [];
    const biases: number[] = [];

    // He initialization: stddev = sqrt(2 / fan_in)
    const fanIn = inChannels * layer.filterSize * layer.filterSize;
    const stddev = Math.sqrt(2 / fanIn);

    for (let f = 0; f < layer.filterCount; f++) {
      const filter: number[][][] = [];
      for (let c = 0; c < inChannels; c++) {
        const channel: number[][] = [];
        for (let i = 0; i < layer.filterSize; i++) {
          const row: number[] = [];
          for (let j = 0; j < layer.filterSize; j++) {
            row.push(gaussianRandom() * stddev);
          }
          channel.push(row);
        }
        filter.push(channel);
      }
      filters.push(filter);
      biases.push(0);
    }

    convFilters.push(filters);
    convBiases.push(biases);
    inChannels = layer.filterCount;
  }

  // Initialize FC layers
  let fcInputSize = calcFlattenedSize(config);

  for (let i = 0; i < config.fcLayers.length; i++) {
    const layer = config.fcLayers[i];
    const weights: number[][] = [];
    const biases: number[] = [];

    const fanIn = fcInputSize;
    const stddev = Math.sqrt(2 / fanIn);

    for (let j = 0; j < fcInputSize; j++) {
      const row: number[] = [];
      for (let k = 0; k < layer.neurons; k++) {
        row.push(gaussianRandom() * stddev);
      }
      weights.push(row);
    }

    for (let k = 0; k < layer.neurons; k++) {
      biases.push(0);
    }

    fcWeights.push(weights);
    fcBiases.push(biases);
    fcInputSize = layer.neurons;
  }

  // Output layer
  const outputWeights: number[][] = [];
  const outputBiases: number[] = [];
  const stddev = Math.sqrt(2 / fcInputSize);

  for (let j = 0; j < fcInputSize; j++) {
    const row: number[] = [];
    for (let k = 0; k < config.outputSize; k++) {
      row.push(gaussianRandom() * stddev);
    }
    outputWeights.push(row);
  }

  for (let k = 0; k < config.outputSize; k++) {
    outputBiases.push(0);
  }

  fcWeights.push(outputWeights);
  fcBiases.push(outputBiases);

  return { convFilters, convBiases, fcWeights, fcBiases };
};

// Gaussian random number generator (Box-Muller transform)
const gaussianRandom = (): number => {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

// Activation functions
const relu = (x: number): number => Math.max(0, x);
const leakyRelu = (x: number, alpha = 0.01): number => x > 0 ? x : alpha * x;
const sigmoid = (x: number): number => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
const tanh = (x: number): number => Math.tanh(x);

const applyActivation = (x: number, type: string): number => {
  switch (type) {
    case 'relu': return relu(x);
    case 'leaky_relu': return leakyRelu(x);
    case 'sigmoid': return sigmoid(x);
    case 'tanh': return tanh(x);
    default: return x;
  }
};

// Activation derivatives
const reluDerivative = (x: number): number => x > 0 ? 1 : 0;
const leakyReluDerivative = (x: number, alpha = 0.01): number => x > 0 ? 1 : alpha;
const sigmoidDerivative = (x: number): number => {
  const s = sigmoid(x);
  return s * (1 - s);
};
const tanhDerivative = (x: number): number => 1 - Math.tanh(x) ** 2;

const applyActivationDerivative = (x: number, type: string): number => {
  switch (type) {
    case 'relu': return reluDerivative(x);
    case 'leaky_relu': return leakyReluDerivative(x);
    case 'sigmoid': return sigmoidDerivative(x);
    case 'tanh': return tanhDerivative(x);
    default: return 1;
  }
};

// Softmax for output layer
const softmax = (x: number[]): number[] => {
  const maxVal = Math.max(...x);
  const exp = x.map(v => Math.exp(v - maxVal));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(v => v / sum);
};

// 2D Convolution operation
export const conv2d = (
  input: number[][][],  // [channels][height][width]
  filters: number[][][][],  // [outChannels][inChannels][filterH][filterW]
  biases: number[],
  stride: number,
  padding: number,
  activation: string
): { output: number[][][]; preActivation: number[][][] } => {
  const inChannels = input.length;
  const inHeight = input[0].length;
  const inWidth = input[0][0].length;
  const outChannels = filters.length;
  const filterSize = filters[0][0].length;

  const outHeight = calcConvOutputSize(inHeight, filterSize, stride, padding);
  const outWidth = calcConvOutputSize(inWidth, filterSize, stride, padding);

  const preActivation: number[][][] = [];
  const output: number[][][] = [];

  for (let oc = 0; oc < outChannels; oc++) {
    const preActChannel: number[][] = [];
    const outChannel: number[][] = [];

    for (let oh = 0; oh < outHeight; oh++) {
      const preActRow: number[] = [];
      const outRow: number[] = [];

      for (let ow = 0; ow < outWidth; ow++) {
        let sum = biases[oc];

        for (let ic = 0; ic < inChannels; ic++) {
          for (let fh = 0; fh < filterSize; fh++) {
            for (let fw = 0; fw < filterSize; fw++) {
              const ih = oh * stride - padding + fh;
              const iw = ow * stride - padding + fw;

              if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                sum += input[ic][ih][iw] * filters[oc][ic][fh][fw];
              }
            }
          }
        }

        preActRow.push(sum);
        outRow.push(applyActivation(sum, activation));
      }

      preActChannel.push(preActRow);
      outChannel.push(outRow);
    }

    preActivation.push(preActChannel);
    output.push(outChannel);
  }

  return { output, preActivation };
};

// Max pooling operation
export const maxPool2d = (
  input: number[][][],
  poolSize: number,
  stride: number
): { output: number[][][]; indices: number[][][][][] } => {
  const channels = input.length;
  const inHeight = input[0].length;
  const inWidth = input[0][0].length;

  const outHeight = calcPoolOutputSize(inHeight, poolSize, stride);
  const outWidth = calcPoolOutputSize(inWidth, poolSize, stride);

  const output: number[][][] = [];
  const indices: number[][][][][] = [];  // Store max indices for backprop [channel][h][w][2]

  for (let c = 0; c < channels; c++) {
    const outChannel: number[][] = [];
    const idxChannel: number[][][][] = [];

    for (let oh = 0; oh < outHeight; oh++) {
      const outRow: number[] = [];
      const idxRow: number[][][] = [];

      for (let ow = 0; ow < outWidth; ow++) {
        let maxVal = -Infinity;
        let maxIdx = [0, 0];

        for (let ph = 0; ph < poolSize; ph++) {
          for (let pw = 0; pw < poolSize; pw++) {
            const ih = oh * stride + ph;
            const iw = ow * stride + pw;

            if (ih < inHeight && iw < inWidth && input[c][ih][iw] > maxVal) {
              maxVal = input[c][ih][iw];
              maxIdx = [ih, iw];
            }
          }
        }

        outRow.push(maxVal);
        idxRow.push([maxIdx]);
      }

      outChannel.push(outRow);
      idxChannel.push(idxRow);
    }

    output.push(outChannel);
    indices.push(idxChannel);
  }

  return { output, indices };
};

// Average pooling operation
export const avgPool2d = (
  input: number[][][],
  poolSize: number,
  stride: number
): number[][][] => {
  const channels = input.length;
  const inHeight = input[0].length;
  const inWidth = input[0][0].length;

  const outHeight = calcPoolOutputSize(inHeight, poolSize, stride);
  const outWidth = calcPoolOutputSize(inWidth, poolSize, stride);

  const output: number[][][] = [];

  for (let c = 0; c < channels; c++) {
    const outChannel: number[][] = [];

    for (let oh = 0; oh < outHeight; oh++) {
      const outRow: number[] = [];

      for (let ow = 0; ow < outWidth; ow++) {
        let sum = 0;
        let count = 0;

        for (let ph = 0; ph < poolSize; ph++) {
          for (let pw = 0; pw < poolSize; pw++) {
            const ih = oh * stride + ph;
            const iw = ow * stride + pw;

            if (ih < inHeight && iw < inWidth) {
              sum += input[c][ih][iw];
              count++;
            }
          }
        }

        outRow.push(sum / count);
      }

      outChannel.push(outRow);
    }

    output.push(outChannel);
  }

  return output;
};

// Flatten 3D tensor to 1D array
export const flatten = (input: number[][][]): number[] => {
  const result: number[] = [];
  for (const channel of input) {
    for (const row of channel) {
      for (const val of row) {
        result.push(val);
      }
    }
  }
  return result;
};

// Reshape 1D array to image format
export const reshapeToImage = (
  input: number[],
  width: number,
  height: number,
  channels: number
): number[][][] => {
  const result: number[][][] = [];
  let idx = 0;

  for (let c = 0; c < channels; c++) {
    const channel: number[][] = [];
    for (let h = 0; h < height; h++) {
      const row: number[] = [];
      for (let w = 0; w < width; w++) {
        row.push(input[idx++]);
      }
      channel.push(row);
    }
    result.push(channel);
  }

  return result;
};

// Forward pass cache for backpropagation
export interface CNNForwardCache {
  convOutputs: number[][][][];       // Output after each conv layer [layer][channel][h][w]
  convPreActivations: number[][][][]; // Pre-activation values [layer][channel][h][w]
  poolOutputs: number[][][][];       // Output after each pooling [layer][channel][h][w]
  poolIndices: (number[][][][][] | null)[];  // Max pool indices [layer] or null if no pooling/avg pooling
  fcInputs: number[][];              // Input to each FC layer
  fcPreActivations: number[][];      // Pre-activation values for FC
  fcOutputs: number[][];             // Output of each FC layer
  flattened: number[];               // Flattened conv output
}

// CNN Forward pass
export const cnnForward = (
  input: number[],  // Flat input image (e.g., 784 for MNIST)
  weights: CNNWeights,
  config: CNNConfig
): { output: number[]; cache: CNNForwardCache } => {
  // Reshape input to image format
  let currentOutput = reshapeToImage(
    input,
    config.inputShape.width,
    config.inputShape.height,
    config.inputShape.channels
  );

  const cache: CNNForwardCache = {
    convOutputs: [],
    convPreActivations: [],
    poolOutputs: [],
    poolIndices: [],
    fcInputs: [],
    fcPreActivations: [],
    fcOutputs: [],
    flattened: []
  };

  // Process conv layers
  for (let i = 0; i < config.convLayers.length; i++) {
    const layer = config.convLayers[i];
    
    const { output: convOut, preActivation } = conv2d(
      currentOutput,
      weights.convFilters[i],
      weights.convBiases[i],
      layer.stride,
      layer.padding,
      layer.activation
    );

    cache.convPreActivations.push(preActivation);
    cache.convOutputs.push(convOut);
    currentOutput = convOut;

    // Apply pooling if configured
    if (layer.pooling) {
      if (layer.pooling.type === 'max') {
        const { output: poolOut, indices } = maxPool2d(
          currentOutput,
          layer.pooling.size,
          layer.pooling.stride
        );
        cache.poolOutputs.push(poolOut);
        cache.poolIndices.push(indices);
        currentOutput = poolOut;
      } else {
        const poolOut = avgPool2d(
          currentOutput,
          layer.pooling.size,
          layer.pooling.stride
        );
        cache.poolOutputs.push(poolOut);
        cache.poolIndices.push(null);
        currentOutput = poolOut;
      }
    } else {
      cache.poolOutputs.push(currentOutput);
      cache.poolIndices.push(null);
    }
  }

  // Flatten for FC layers
  const flattened = flatten(currentOutput);
  cache.flattened = flattened;

  // Process FC layers
  let fcInput = flattened;

  for (let i = 0; i < weights.fcWeights.length; i++) {
    cache.fcInputs.push([...fcInput]);

    const layerOutput: number[] = [];
    const preActivation: number[] = [];
    const isOutputLayer = i === weights.fcWeights.length - 1;
    const activation = isOutputLayer ? 'softmax' : config.fcLayers[i]?.activation || 'relu';

    // Matrix multiplication
    for (let j = 0; j < weights.fcWeights[i][0].length; j++) {
      let sum = weights.fcBiases[i][j];
      for (let k = 0; k < fcInput.length; k++) {
        sum += fcInput[k] * weights.fcWeights[i][k][j];
      }
      preActivation.push(sum);
    }

    cache.fcPreActivations.push(preActivation);

    // Apply activation
    if (activation === 'softmax') {
      layerOutput.push(...softmax(preActivation));
    } else {
      for (const val of preActivation) {
        layerOutput.push(applyActivation(val, activation));
      }
    }

    cache.fcOutputs.push(layerOutput);
    fcInput = layerOutput;
  }

  return { output: fcInput, cache };
};

// Cross-entropy loss
export const cnnComputeLoss = (output: number[], target: number[]): number => {
  let loss = 0;
  for (let i = 0; i < output.length; i++) {
    loss -= target[i] * Math.log(Math.max(output[i], 1e-15));
  }
  return loss;
};

// Compute accuracy
export const cnnComputeAccuracy = (
  inputs: number[][],
  targets: number[][],
  weights: CNNWeights,
  config: CNNConfig
): number => {
  let correct = 0;
  for (let i = 0; i < inputs.length; i++) {
    const { output } = cnnForward(inputs[i], weights, config);
    const predicted = output.indexOf(Math.max(...output));
    const actual = targets[i].indexOf(Math.max(...targets[i]));
    if (predicted === actual) correct++;
  }
  return correct / inputs.length;
};

// Clone CNN weights
export const cloneCNNWeights = (weights: CNNWeights): CNNWeights => {
  return {
    convFilters: weights.convFilters.map(layer =>
      layer.map(filter =>
        filter.map(channel =>
          channel.map(row => [...row])
        )
      )
    ),
    convBiases: weights.convBiases.map(layer => [...layer]),
    fcWeights: weights.fcWeights.map(layer =>
      layer.map(row => [...row])
    ),
    fcBiases: weights.fcBiases.map(layer => [...layer])
  };
};

// Flatten CNN weights to a 1D array (for distance calculations)
export const flattenCNNWeights = (weights: CNNWeights): number[] => {
  const result: number[] = [];

  for (const layer of weights.convFilters) {
    for (const filter of layer) {
      for (const channel of filter) {
        for (const row of channel) {
          result.push(...row);
        }
      }
    }
  }

  for (const layer of weights.convBiases) {
    result.push(...layer);
  }

  for (const layer of weights.fcWeights) {
    for (const row of layer) {
      result.push(...row);
    }
  }

  for (const layer of weights.fcBiases) {
    result.push(...layer);
  }

  return result;
};

// Count total parameters
export const countCNNParameters = (config: CNNConfig): number => {
  let total = 0;
  let inChannels = config.inputShape.channels;

  // Conv layers
  for (const layer of config.convLayers) {
    const filterParams = layer.filterCount * inChannels * layer.filterSize * layer.filterSize;
    const biasParams = layer.filterCount;
    total += filterParams + biasParams;
    inChannels = layer.filterCount;
  }

  // FC layers
  let fcInputSize = calcFlattenedSize(config);
  for (const layer of config.fcLayers) {
    total += fcInputSize * layer.neurons + layer.neurons;
    fcInputSize = layer.neurons;
  }

  // Output layer
  total += fcInputSize * config.outputSize + config.outputSize;

  return total;
};

// Print CNN architecture summary
export const printCNNSummary = (config: CNNConfig): void => {
  console.log('=== CNN Architecture Summary ===');
  console.log(`Input: ${config.inputShape.width}x${config.inputShape.height}x${config.inputShape.channels}`);

  let width = config.inputShape.width;
  let height = config.inputShape.height;
  let channels = config.inputShape.channels;

  for (let i = 0; i < config.convLayers.length; i++) {
    const layer = config.convLayers[i];
    width = calcConvOutputSize(width, layer.filterSize, layer.stride, layer.padding);
    height = calcConvOutputSize(height, layer.filterSize, layer.stride, layer.padding);
    channels = layer.filterCount;
    console.log(`Conv${i + 1}: ${width}x${height}x${channels} (${layer.filterSize}x${layer.filterSize}, ${layer.activation})`);

    if (layer.pooling) {
      width = calcPoolOutputSize(width, layer.pooling.size, layer.pooling.stride);
      height = calcPoolOutputSize(height, layer.pooling.size, layer.pooling.stride);
      console.log(`Pool${i + 1}: ${width}x${height}x${channels} (${layer.pooling.type} ${layer.pooling.size}x${layer.pooling.size})`);
    }
  }

  console.log(`Flatten: ${width * height * channels}`);

  for (let i = 0; i < config.fcLayers.length; i++) {
    console.log(`FC${i + 1}: ${config.fcLayers[i].neurons} (${config.fcLayers[i].activation})`);
  }

  console.log(`Output: ${config.outputSize} (softmax)`);
  console.log(`Total parameters: ${countCNNParameters(config).toLocaleString()}`);
};

// Simple training step (SGD)
export const cnnTrainStep = (
  input: number[],
  target: number[],
  weights: CNNWeights,
  config: CNNConfig,
  learningRate: number
): { loss: number; gradientNorm: number } => {
  const { output, cache } = cnnForward(input, weights, config);
  const loss = cnnComputeLoss(output, target);

  // Backpropagation through FC layers
  let delta = output.map((o, i) => o - target[i]);
  let gradientNormSquared = 0;

  // Backprop through FC layers (reverse order)
  for (let l = weights.fcWeights.length - 1; l >= 0; l--) {
    const input_l = cache.fcInputs[l];
    const newDelta: number[] = new Array(input_l.length).fill(0);

    for (let j = 0; j < weights.fcWeights[l][0].length; j++) {
      for (let i = 0; i < input_l.length; i++) {
        const grad = delta[j] * input_l[i];
        gradientNormSquared += grad * grad;
        weights.fcWeights[l][i][j] -= learningRate * grad;
        newDelta[i] += delta[j] * weights.fcWeights[l][i][j];
      }
      weights.fcBiases[l][j] -= learningRate * delta[j];
    }

    // Apply activation derivative if not output layer
    if (l > 0) {
      const activation = config.fcLayers[l - 1]?.activation || 'relu';
      for (let i = 0; i < newDelta.length; i++) {
        newDelta[i] *= applyActivationDerivative(cache.fcPreActivations[l - 1][i], activation);
      }
    }

    delta = newDelta;
  }

  // Note: Full conv backprop is complex; simplified version updates only FC layers
  // For production, implement full conv backprop or use a library

  return { loss, gradientNorm: Math.sqrt(gradientNormSquared) };
};

// Train one epoch
export const cnnTrainEpoch = (
  inputs: number[][],
  targets: number[][],
  weights: CNNWeights,
  config: CNNConfig,
  learningRate: number
): { avgLoss: number; avgGradientNorm: number } => {
  let totalLoss = 0;
  let totalGradientNorm = 0;

  // Shuffle indices
  const indices = Array.from({ length: inputs.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  for (const idx of indices) {
    const { loss, gradientNorm } = cnnTrainStep(
      inputs[idx],
      targets[idx],
      weights,
      config,
      learningRate
    );
    totalLoss += loss;
    totalGradientNorm += gradientNorm;
  }

  return {
    avgLoss: totalLoss / inputs.length,
    avgGradientNorm: totalGradientNorm / inputs.length
  };
};
