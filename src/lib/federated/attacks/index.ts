// Byzantine Attack Strategies
export { applyLocalModelPoisoning } from './localModelPoisoning';
export type { LocalModelPoisoningConfig } from './localModelPoisoning';

import type { ModelWeights } from '../core/types';
import { applyLocalModelPoisoning } from './localModelPoisoning';

export type ByzantineAttackMethod = 'local-model-poisoning' | 'label-flipping' | 'gradient-scaling';

export interface ByzantineConfig {
  byzantineCount: number;
  attackMethod: ByzantineAttackMethod;
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
    case 'local-model-poisoning':
      return applyLocalModelPoisoning(allClientResults, globalModel, {
        byzantineCount: config.byzantineCount,
        byzantineClientIds,
        currentRound,
        totalRounds,
      });
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
