// Byzantine Attack Strategies
export { applyLocalModelPoisoning, initializeByzantineObjective, resetByzantineObjectives } from './localModelPoisoning';
export type { LocalModelPoisoningConfig } from './localModelPoisoning';
export { applyHDPAToDataset, getOrCreateHDPAPoisonedDataset, resetHDPACache, DEFAULT_HDPA_CONFIG } from './hdpa';
export type { HDPAConfig } from './hdpa';

import type { ModelWeights } from '../core/types';
import { applyLocalModelPoisoning, initializeByzantineObjective } from './localModelPoisoning';

export type ByzantineAttackMethod = 'local-model-poisoning' | 'label-flipping' | 'gradient-scaling' | 'hdpa';

export interface ByzantineConfig {
  byzantineCount: number;
  attackMethod: ByzantineAttackMethod;
  epsilon?: number; // Scaling factor for poisoning (for local-model-poisoning)
}

/**
 * Apply the selected Byzantine attack to client results.
 * Called after Phase 3 (receiving models) and before Phase 4 (aggregation).
 */
export const applyByzantineAttack = (
  allClientResults: { weights: ModelWeights; dataSize: number; clientId: string }[],
  globalModel: ModelWeights,
  config: ByzantineConfig,
  byzantineClientIds: string[],
  currentRound: number,
  totalRounds: number
): { weights: ModelWeights; dataSize: number; clientId: string }[] => {
  if (config.byzantineCount === 0 || byzantineClientIds.length === 0) {
    return allClientResults;
  }

  switch (config.attackMethod) {
    case 'local-model-poisoning': {
      // For local-model-poisoning, apply per-client poisoning
      const epsilon = config.epsilon ?? 0.1;
      return allClientResults.map(result => {
        if (!byzantineClientIds.includes(result.clientId)) {
          return result;
        }
        // Apply poisoning to this Byzantine client
        const poisonedWeights = applyLocalModelPoisoning(result.clientId, result.weights, epsilon);
        return {
          ...result,
          weights: poisonedWeights,
        };
      });
    }
    case 'label-flipping':
      // TODO: implement label flipping attack
      console.warn('[Byzantine] Label flipping not yet implemented');
      return allClientResults;
    case 'gradient-scaling':
      // TODO: implement gradient scaling attack
      console.warn('[Byzantine] Gradient scaling not yet implemented');
      return allClientResults;
    default:
      return allClientResults;
  }
};
