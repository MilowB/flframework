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

// Global store for objective models per Byzantine client (initialized once per experiment)
const byzantineObjectiveModels: Map<string, ModelWeights> = new Map();
let masterObjectiveModel: ModelWeights | null = null;

/**
 * Initialize objective models for all Byzantine clients (called once at experiment start)
 * Generates a single master objective, then adds noise to create per-client variants
 */
export const initializeByzantineObjective = (
  byzantineClientIds: string[],
  globalModel: ModelWeights,
  noiseScale: number = 0.5
): void => {
  if (masterObjectiveModel !== null) return; // Already initialized
  // Generate single master objective by adding noise to global model
  masterObjectiveModel = {
    layers: globalModel.layers.map(layer =>
      layer.map(v => v + (Math.random() - 0.5) * 2 * noiseScale)
    ),
    bias: globalModel.bias.map(b => b + (Math.random() - 0.5) * 2 * noiseScale),
    version: globalModel.version,
  };
  
  // Decline master objective for each Byzantine client (add small perturbations)
  const perturbationScale = noiseScale * 0.25; // Smaller perturbations for per-client variants
  for (const clientId of byzantineClientIds) {
    const clientObjective: ModelWeights = {
      layers: masterObjectiveModel.layers.map(layer =>
        layer.map(v => v + (Math.random() - 0.5) * 2 * perturbationScale)
      ),
      bias: masterObjectiveModel.bias.map(b => b + (Math.random() - 0.5) * 2 * perturbationScale),
      version: masterObjectiveModel.version,
    };
    byzantineObjectiveModels.set(clientId, clientObjective);
  }
  
  console.log(`[Byzantine] Initialized master objective and ${byzantineClientIds.length} per-client variants`);
};

/**
 * Reset all Byzantine objective models (call at experiment start)
 */
export const resetByzantineObjectives = (): void => {
  byzantineObjectiveModels.clear();
  masterObjectiveModel = null;
};

/**
 * Apply Local Model Poisoning attack to a single Byzantine client.
 * 
 * Called once per Byzantine client per round.
 * 
 * @param clientId Byzantine client ID
 * @param receivedModel Model received from server by the Byzantine client
 * @param epsilon Scaling factor for the poisoning (default 0.1)
 * @returns Poisoned model weights
 */
export const applyLocalModelPoisoning = (
  clientId: string,
  receivedModel: ModelWeights,
  epsilon: number = 0.1
): ModelWeights => {
  // Get the objective model for this client (must be initialized first)
  const objectiveModel = byzantineObjectiveModels.get(clientId);
  if (!objectiveModel) {
    console.warn(`[Byzantine] No objective model for ${clientId}, returning received model unchanged`);
    return receivedModel;
  }

  // Compute delta: objective - received
  const deltaLayers: number[][] = objectiveModel.layers.map((layer, l) =>
    layer.map((v, i) => v - receivedModel.layers[l][i])
  );
  const deltaBias: number[] = objectiveModel.bias.map((v, i) => v - receivedModel.bias[i]);

  // Add epsilon * delta to received model
  const poisonedLayers: number[][] = receivedModel.layers.map((layer, l) =>
    layer.map((v, i) => v + epsilon * deltaLayers[l][i])
  );
  const poisonedBias: number[] = receivedModel.bias.map((v, i) => v + epsilon * deltaBias[i]);

  console.log(`[Byzantine] Client ${clientId} poisoned (ε=${epsilon.toFixed(4)})`);

  return {
    layers: poisonedLayers,
    bias: poisonedBias,
    version: receivedModel.version,
  };
};
