// Local Model Poisoning Attack
// Coordinated attack where a centralized attacker controls c Byzantine clients.
// Goal: force the global model to evolve in the OPPOSITE direction of natural learning.

import type { ModelWeights } from '../core/types';

export interface LocalModelPoisoningConfig {
  /** Number of Byzantine clients */
  byzantineCount: number;
  /** IDs of Byzantine clients */
  byzantineClientIds: string[];
  /** Current round number (used to scale lambda) */
  currentRound: number;
  /** Total rounds (used to scale lambda) */
  totalRounds: number;
  /** Base lambda for deviation magnitude */
  baseLambda?: number;
}

/**
 * Estimate the natural evolution direction of the global model
 * by computing element-wise signs of the average honest update.
 * s[j] = +1 if parameter j increases on average, -1 otherwise.
 */
const estimateSignVector = (
  honestUpdates: { weights: ModelWeights; dataSize: number }[],
  globalModel: ModelWeights
): { layerSigns: number[][]; biasSigns: number[] } => {
  const numLayers = globalModel.layers.length;

  // Compute weighted average delta across honest clients
  const totalData = honestUpdates.reduce((s, c) => s + c.dataSize, 0);
  const avgDeltaLayers: number[][] = globalModel.layers.map(l => new Array(l.length).fill(0));
  const avgDeltaBias: number[] = new Array(globalModel.bias.length).fill(0);

  for (const { weights, dataSize } of honestUpdates) {
    const w = dataSize / totalData;
    for (let l = 0; l < numLayers; l++) {
      for (let i = 0; i < weights.layers[l].length; i++) {
        avgDeltaLayers[l][i] += (weights.layers[l][i] - globalModel.layers[l][i]) * w;
      }
    }
    for (let i = 0; i < weights.bias.length; i++) {
      avgDeltaBias[i] += (weights.bias[i] - globalModel.bias[i]) * w;
    }
  }

  // Extract signs
  const layerSigns = avgDeltaLayers.map(l => l.map(v => (v >= 0 ? 1 : -1)));
  const biasSigns = avgDeltaBias.map(v => (v >= 0 ? 1 : -1));

  return { layerSigns, biasSigns };
};

/**
 * Compute lambda that grows slowly over rounds to avoid early detection.
 * Bounded so the poisoned model stays close to the global model initially.
 */
const computeLambda = (
  currentRound: number,
  totalRounds: number,
  baseLambda: number
): number => {
  // Progressive scaling: lambda grows linearly from baseLambda * 0.1 to baseLambda
  const progress = Math.min(1, (currentRound + 1) / totalRounds);
  return baseLambda * (0.1 + 0.9 * progress);
};

/**
 * Estimate the range of honest client parameter values for stealth against
 * trimmed mean / median aggregations.
 */
const estimateHonestBounds = (
  honestUpdates: { weights: ModelWeights }[],
  layerIdx: number,
  paramIdx: number,
  isBias: boolean
): { min: number; max: number } => {
  const values = honestUpdates.map(c =>
    isBias ? c.weights.bias[paramIdx] : c.weights.layers[layerIdx][paramIdx]
  );
  return {
    min: Math.min(...values),
    max: Math.max(...values),
  };
};

/**
 * Apply Local Model Poisoning attack.
 * 
 * Takes all client results and replaces Byzantine client weights with
 * coordinated poisoned models that push the global model in the opposite
 * direction of natural learning.
 * 
 * @param allClientResults All client results (honest + Byzantine)
 * @param globalModel Current global model
 * @param config Attack configuration
 * @returns Modified client results with Byzantine weights replaced
 */
export const applyLocalModelPoisoning = (
  allClientResults: { weights: ModelWeights; dataSize: number; clientId: string }[],
  globalModel: ModelWeights,
  config: LocalModelPoisoningConfig
): { weights: ModelWeights; dataSize: number; clientId: string }[] => {
  const { byzantineClientIds, currentRound, totalRounds, baseLambda = 0.5 } = config;

  if (byzantineClientIds.length === 0) return allClientResults;

  const byzantineSet = new Set(byzantineClientIds);

  // Separate honest and Byzantine results
  const honestResults = allClientResults.filter(r => !byzantineSet.has(r.clientId));
  const byzantineResults = allClientResults.filter(r => byzantineSet.has(r.clientId));

  if (honestResults.length === 0) return allClientResults;

  // Step 1: Estimate natural direction
  const { layerSigns, biasSigns } = estimateSignVector(honestResults, globalModel);

  // Step 2: Compute adaptive lambda
  const lambda = computeLambda(currentRound, totalRounds, baseLambda);

  // Step 3: Build target malicious model: w' = w_global - lambda * s
  const maliciousLayers: number[][] = globalModel.layers.map((layer, l) =>
    layer.map((v, i) => v - lambda * layerSigns[l][i])
  );
  const maliciousBias: number[] = globalModel.bias.map((v, i) => v - lambda * biasSigns[i]);

  // Step 4: Generate coordinated Byzantine models with small perturbations
  const epsilon = 0.01; // Small perturbation to form a compact cluster
  const rng = () => (Math.random() - 0.5) * 2 * epsilon;

  const modifiedResults = allClientResults.map(result => {
    if (!byzantineSet.has(result.clientId)) return result;

    const isFirst = result.clientId === byzantineClientIds[0];

    // First Byzantine client gets exact malicious model, others get perturbed versions
    const poisonedLayers = maliciousLayers.map(layer =>
      layer.map(v => isFirst ? v : v + rng())
    );
    const poisonedBias = maliciousBias.map(v => isFirst ? v : v + rng());

    // Step 5: For robustness against trimmed mean/median, clamp values
    // just beyond honest bounds to influence the statistic
    const clampedLayers = poisonedLayers.map((layer, l) =>
      layer.map((v, i) => {
        const bounds = estimateHonestBounds(honestResults, l, i, false);
        const range = bounds.max - bounds.min;
        // Clamp to within 1.5x the range beyond honest bounds
        const margin = range * 0.5;
        return Math.max(bounds.min - margin, Math.min(bounds.max + margin, v));
      })
    );
    const clampedBias = poisonedBias.map((v, i) => {
      const bounds = estimateHonestBounds(honestResults, 0, i, true);
      const range = bounds.max - bounds.min;
      const margin = range * 0.5;
      return Math.max(bounds.min - margin, Math.min(bounds.max + margin, v));
    });

    console.log(`[Byzantine] Client ${result.clientId} poisoned (round ${currentRound}, λ=${lambda.toFixed(4)})`);

    return {
      ...result,
      weights: {
        layers: clampedLayers,
        bias: clampedBias,
        version: result.weights.version,
      },
    };
  });

  return modifiedResults;
};
