// Real MLP implementation for MNIST classification

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

// MNIST configuration
export const MNIST_INPUT_SIZE = 784;  // 28x28 pixels
export const MNIST_HIDDEN_SIZE = 128; // Hidden layer neurons
export const MNIST_OUTPUT_SIZE = 10;  // 10 digit classes

// Activation functions
const relu = (x: number): number => Math.max(0, x);
const reluDerivative = (x: number): number => x > 0 ? 1 : 0;

// Softmax for multi-class classification
const softmax = (logits: number[]): number[] => {
  const maxLogit = Math.max(...logits);
  const expScores = logits.map(l => Math.exp(l - maxLogit));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  return expScores.map(e => e / sumExp);
};

// Initialize random weights with Xavier initialization
export const initializeMLPWeights = (
  inputSize: number = MNIST_INPUT_SIZE,
  hiddenSize: number = MNIST_HIDDEN_SIZE,
  outputSize: number = MNIST_OUTPUT_SIZE
): MLPWeights => {
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
): { hidden: number[]; hiddenPreAct: number[]; output: number[] } => {
  const { W1, b1, W2, b2 } = weights;
  
  // Hidden layer: relu(input @ W1 + b1)
  const hiddenPreAct: number[] = [];
  const hidden: number[] = [];
  for (let j = 0; j < W1[0].length; j++) {
    let sum = b1[j];
    for (let i = 0; i < input.length; i++) {
      sum += input[i] * W1[i][j];
    }
    hiddenPreAct.push(sum);
    hidden.push(relu(sum));
  }
  
  // Output layer: softmax(hidden @ W2 + b2)
  const logits: number[] = [];
  for (let j = 0; j < W2[0].length; j++) {
    let sum = b2[j];
    for (let i = 0; i < hidden.length; i++) {
      sum += hidden[i] * W2[i][j];
    }
    logits.push(sum);
  }
  const output = softmax(logits);
  
  return { hidden, hiddenPreAct, output };
};

// Compute loss (Cross-Entropy for multi-class)
export const computeLoss = (
  predicted: number[],
  target: number[]
): number => {
  let loss = 0;
  for (let i = 0; i < predicted.length; i++) {
    const p = Math.max(1e-7, Math.min(1 - 1e-7, predicted[i]));
    loss -= target[i] * Math.log(p);
  }
  return loss;
};

// Compute accuracy (argmax comparison)
export const computeAccuracy = (
  inputs: number[][],
  outputs: number[][],
  weights: MLPWeights
): number => {
  let correct = 0;
  for (let i = 0; i < inputs.length; i++) {
    const { output } = forward(inputs[i], weights);
    const predicted = output.indexOf(Math.max(...output));
    const actual = outputs[i].indexOf(Math.max(...outputs[i]));
    if (predicted === actual) {
      correct++;
    }
  }
  return correct / inputs.length;
};

// Backward pass and weight update (softmax + cross-entropy)
export const trainStep = (
  input: number[],
  target: number[],
  weights: MLPWeights,
  learningRate: number
): { loss: number } => {
  const { W1, W2 } = weights;
  
  // Forward pass
  const { hidden, hiddenPreAct, output } = forward(input, weights);
  
  // Compute loss
  const loss = computeLoss(output, target);
  
  // Backward pass
  // Output layer gradients: softmax + cross-entropy derivative = output - target
  const dOutput: number[] = output.map((o, i) => o - target[i]);
  
  // Gradients for W2 and b2
  const dW2: number[][] = W2.map((row, i) =>
    row.map((_, j) => hidden[i] * dOutput[j])
  );
  const db2: number[] = [...dOutput];
  
  // Hidden layer gradients
  const dHidden: number[] = hidden.map((_, i) => {
    let sum = 0;
    for (let j = 0; j < dOutput.length; j++) {
      sum += dOutput[j] * W2[i][j];
    }
    return sum * reluDerivative(hiddenPreAct[i]);
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
  for (let j = 0; j < weights.b1.length; j++) {
    weights.b1[j] -= learningRate * db1[j];
  }
  for (let i = 0; i < W2.length; i++) {
    for (let j = 0; j < W2[i].length; j++) {
      weights.W2[i][j] -= learningRate * dW2[i][j];
    }
  }
  for (let j = 0; j < weights.b2.length; j++) {
    weights.b2[j] -= learningRate * db2[j];
  }
  
  return { loss };
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
  inputSize: number = MNIST_INPUT_SIZE,
  hiddenSize: number = MNIST_HIDDEN_SIZE,
  outputSize: number = MNIST_OUTPUT_SIZE
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
