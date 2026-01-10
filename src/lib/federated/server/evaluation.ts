// Server evaluation logic

import type { ModelWeights, WeightsSnapshot } from '../core/types';
import { mnistTestData } from '../core/stores';
import {
  unflattenWeights,
  forward,
  computeLoss,
  computeAccuracy,
  MNIST_INPUT_SIZE,
  MNIST_HIDDEN_SIZE,
  MNIST_OUTPUT_SIZE
} from '../models/mlp';
import { oneHot } from '../data/mnist';

// Evaluate global model on test set
export const evaluateOnTestSet = (globalModel: { layers: number[][]; bias: number[] }): { loss: number; accuracy: number } => {
  if (!mnistTestData) {
    return { loss: 0, accuracy: 0 };
  }
  
  const mlpWeights = unflattenWeights(
    globalModel,
    MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE
  );
  
  let totalLoss = 0;
  let correct = 0;
  
  for (let i = 0; i < mnistTestData.images.length; i++) {
    const { output } = forward(mnistTestData.images[i], mlpWeights);
    const target = oneHot(mnistTestData.labels[i]);
    
    totalLoss += computeLoss(output, target);
    
    const predicted = output.indexOf(Math.max(...output));
    if (predicted === mnistTestData.labels[i]) {
      correct++;
    }
  }
  
  return {
    loss: totalLoss / mnistTestData.images.length,
    accuracy: correct / mnistTestData.images.length,
  };
};

// Evaluate a cluster model on pooled test data from all clients in the cluster
export const evaluateClusterModel = (
  clusterClientIds: string[],
  clusterModel: ModelWeights,
  clientTestDataStore: Map<string, { inputs: number[][]; outputs: number[][] }>
): number => {
  const mlpWeights = unflattenWeights(
    { layers: clusterModel.layers, bias: clusterModel.bias },
    MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE
  );
  
  const allInputs: number[][] = [];
  const allOutputs: number[][] = [];
  
  for (const clientId of clusterClientIds) {
    const testData = clientTestDataStore.get(clientId);
    if (testData) {
      allInputs.push(...testData.inputs);
      allOutputs.push(...testData.outputs);
    }
  }
  
  if (allInputs.length === 0) return 0;
  
  return computeAccuracy(allInputs, allOutputs, mlpWeights);
};

// Compute weights statistics for visualization
export const computeWeightsSnapshot = (model: ModelWeights): WeightsSnapshot => {
  const W1 = model.layers[0];
  const W2 = model.layers[1];
  const b1 = model.bias.slice(0, MNIST_HIDDEN_SIZE);
  const b2 = model.bias.slice(MNIST_HIDDEN_SIZE);

  const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const std = (arr: number[]) => {
    const m = mean(arr);
    return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
  };

  return {
    W1Mean: mean(W1),
    W1Std: std(W1),
    W2Mean: mean(W2),
    W2Std: std(W2),
    b1Mean: mean(b1),
    b2Mean: mean(b2),
  };
};
