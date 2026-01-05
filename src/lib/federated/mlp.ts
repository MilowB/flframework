// Real MLP implementation for XOR problem

export interface MLPWeights {
  W1: number[][]; // Input to hidden weights [inputSize x hiddenSize]
  b1: number[];   // Hidden bias [hiddenSize]
  W2: number[][]; // Hidden to output weights [hiddenSize x outputSize]
  b2: number[];   // Output bias [outputSize]
}

export interface TrainingConfig {
  learningRate: number;
  epochs: number;
  batchSize: number;
}

// XOR Dataset
export const XOR_DATA = {
  inputs: [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ],
  outputs: [
    [0],
    [1],
    [1],
    [0],
  ],
};

// Generate synthetic XOR-like data with noise for each client
export const generateClientData = (
  numSamples: number,
  noiseLevel: number = 0.1
): { inputs: number[][]; outputs: number[][] } => {
  const inputs: number[][] = [];
  const outputs: number[][] = [];
  
  for (let i = 0; i < numSamples; i++) {
    // Random XOR-like data point
    const x1 = Math.random() > 0.5 ? 1 : 0;
    const x2 = Math.random() > 0.5 ? 1 : 0;
    const xorResult = x1 !== x2 ? 1 : 0;
    
    // Add noise to inputs
    inputs.push([
      x1 + (Math.random() - 0.5) * noiseLevel,
      x2 + (Math.random() - 0.5) * noiseLevel,
    ]);
    outputs.push([xorResult]);
  }
  
  return { inputs, outputs };
};

// Activation functions
const sigmoid = (x: number): number => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
const sigmoidDerivative = (x: number): number => x * (1 - x);

const relu = (x: number): number => Math.max(0, x);
const reluDerivative = (x: number): number => x > 0 ? 1 : 0;

// Initialize random weights
export const initializeMLPWeights = (
  inputSize: number = 2,
  hiddenSize: number = 8,
  outputSize: number = 1
): MLPWeights => {
  // Xavier initialization
  const xavierHidden = Math.sqrt(2 / (inputSize + hiddenSize));
  const xavierOutput = Math.sqrt(2 / (hiddenSize + outputSize));
  
  return {
    W1: Array.from({ length: inputSize }, () =>
      Array.from({ length: hiddenSize }, () => (Math.random() - 0.5) * 2 * xavierHidden)
    ),
    b1: Array.from({ length: hiddenSize }, () => 0),
    W2: Array.from({ length: hiddenSize }, () =>
      Array.from({ length: outputSize }, () => (Math.random() - 0.5) * 2 * xavierOutput)
    ),
    b2: Array.from({ length: outputSize }, () => 0),
  };
};

// Forward pass
export const forward = (
  input: number[],
  weights: MLPWeights
): { hidden: number[]; output: number[] } => {
  const { W1, b1, W2, b2 } = weights;
  
  // Hidden layer: relu(input @ W1 + b1)
  const hidden: number[] = [];
  for (let j = 0; j < W1[0].length; j++) {
    let sum = b1[j];
    for (let i = 0; i < input.length; i++) {
      sum += input[i] * W1[i][j];
    }
    hidden.push(relu(sum));
  }
  
  // Output layer: sigmoid(hidden @ W2 + b2)
  const output: number[] = [];
  for (let j = 0; j < W2[0].length; j++) {
    let sum = b2[j];
    for (let i = 0; i < hidden.length; i++) {
      sum += hidden[i] * W2[i][j];
    }
    output.push(sigmoid(sum));
  }
  
  return { hidden, output };
};

// Compute loss (Binary Cross Entropy)
export const computeLoss = (
  predicted: number[],
  target: number[]
): number => {
  let loss = 0;
  for (let i = 0; i < predicted.length; i++) {
    const p = Math.max(1e-7, Math.min(1 - 1e-7, predicted[i]));
    loss -= target[i] * Math.log(p) + (1 - target[i]) * Math.log(1 - p);
  }
  return loss / predicted.length;
};

// Compute accuracy
export const computeAccuracy = (
  inputs: number[][],
  outputs: number[][],
  weights: MLPWeights
): number => {
  let correct = 0;
  for (let i = 0; i < inputs.length; i++) {
    const { output } = forward(inputs[i], weights);
    const predicted = output[0] > 0.5 ? 1 : 0;
    if (predicted === Math.round(outputs[i][0])) {
      correct++;
    }
  }
  return correct / inputs.length;
};

// Backward pass and weight update
export const trainStep = (
  input: number[],
  target: number[],
  weights: MLPWeights,
  learningRate: number
): { loss: number; gradients: MLPWeights } => {
  const { W1, b1, W2, b2 } = weights;
  
  // Forward pass with intermediate values
  const hiddenPreActivation: number[] = [];
  const hidden: number[] = [];
  for (let j = 0; j < W1[0].length; j++) {
    let sum = b1[j];
    for (let i = 0; i < input.length; i++) {
      sum += input[i] * W1[i][j];
    }
    hiddenPreActivation.push(sum);
    hidden.push(relu(sum));
  }
  
  const output: number[] = [];
  for (let j = 0; j < W2[0].length; j++) {
    let sum = b2[j];
    for (let i = 0; i < hidden.length; i++) {
      sum += hidden[i] * W2[i][j];
    }
    output.push(sigmoid(sum));
  }
  
  // Compute loss
  const loss = computeLoss(output, target);
  
  // Backward pass
  // Output layer gradients
  const dOutput: number[] = output.map((o, i) => (o - target[i]) * sigmoidDerivative(o));
  
  // Gradients for W2 and b2
  const dW2: number[][] = W2.map((row, i) =>
    row.map((_, j) => hidden[i] * dOutput[j])
  );
  const db2: number[] = [...dOutput];
  
  // Hidden layer gradients
  const dHidden: number[] = hidden.map((h, i) => {
    let sum = 0;
    for (let j = 0; j < dOutput.length; j++) {
      sum += dOutput[j] * W2[i][j];
    }
    return sum * reluDerivative(hiddenPreActivation[i]);
  });
  
  // Gradients for W1 and b1
  const dW1: number[][] = W1.map((row, i) =>
    row.map((_, j) => input[i] * dHidden[j])
  );
  const db1: number[] = [...dHidden];
  
  // Update weights
  for (let i = 0; i < W1.length; i++) {
    for (let j = 0; j < W1[i].length; j++) {
      weights.W1[i][j] -= learningRate * dW1[i][j];
    }
  }
  for (let j = 0; j < b1.length; j++) {
    weights.b1[j] -= learningRate * db1[j];
  }
  for (let i = 0; i < W2.length; i++) {
    for (let j = 0; j < W2[i].length; j++) {
      weights.W2[i][j] -= learningRate * dW2[i][j];
    }
  }
  for (let j = 0; j < b2.length; j++) {
    weights.b2[j] -= learningRate * db2[j];
  }
  
  return { loss, gradients: { W1: dW1, b1: db1, W2: dW2, b2: db2 } };
};

// Train on a batch of data
export const trainEpoch = (
  inputs: number[][],
  outputs: number[][],
  weights: MLPWeights,
  learningRate: number
): number => {
  let totalLoss = 0;
  
  // Shuffle indices
  const indices = Array.from({ length: inputs.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  
  for (const idx of indices) {
    const { loss } = trainStep(inputs[idx], outputs[idx], weights, learningRate);
    totalLoss += loss;
  }
  
  return totalLoss / inputs.length;
};

// Clone weights (deep copy)
export const cloneWeights = (weights: MLPWeights): MLPWeights => ({
  W1: weights.W1.map(row => [...row]),
  b1: [...weights.b1],
  W2: weights.W2.map(row => [...row]),
  b2: [...weights.b2],
});

// Convert MLP weights to flat format for aggregation
export const flattenWeights = (weights: MLPWeights): { layers: number[][]; bias: number[] } => {
  return {
    layers: [
      weights.W1.flat(),
      weights.W2.flat(),
    ],
    bias: [...weights.b1, ...weights.b2],
  };
};

// Convert flat format back to MLP weights
export const unflattenWeights = (
  flat: { layers: number[][]; bias: number[] },
  inputSize: number = 2,
  hiddenSize: number = 8,
  outputSize: number = 1
): MLPWeights => {
  // Reconstruct W1 [inputSize x hiddenSize]
  const W1: number[][] = [];
  for (let i = 0; i < inputSize; i++) {
    W1.push(flat.layers[0].slice(i * hiddenSize, (i + 1) * hiddenSize));
  }
  
  // Reconstruct W2 [hiddenSize x outputSize]
  const W2: number[][] = [];
  for (let i = 0; i < hiddenSize; i++) {
    W2.push(flat.layers[1].slice(i * outputSize, (i + 1) * outputSize));
  }
  
  return {
    W1,
    b1: flat.bias.slice(0, hiddenSize),
    W2,
    b2: flat.bias.slice(hiddenSize),
  };
};
