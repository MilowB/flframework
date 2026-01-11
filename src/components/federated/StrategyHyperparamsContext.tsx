import { createContext, useContext } from 'react';

export interface GravityHyperparams {
  gravitationConstant: number;
  clusterWeight: number;
  clientWeight: number;
  dynamicData: boolean;
  dynamicClient?: number;
  receiverClient?: number;
  changeRound?: number;
}

export interface StrategyHyperparamsContextType {
  gravity: GravityHyperparams;
  setGravity: (params: GravityHyperparams) => void;
  // Add other strategies here as needed
}

export const defaultGravity: GravityHyperparams = {
  gravitationConstant: 9.8,
  clusterWeight: 1e4,
  clientWeight: 10,
  dynamicData: false,
};

export const StrategyHyperparamsContext = createContext<StrategyHyperparamsContextType | undefined>(undefined);

export function useStrategyHyperparams() {
  const ctx = useContext(StrategyHyperparamsContext);
  if (!ctx) throw new Error('useStrategyHyperparams must be used within a StrategyHyperparamsProvider');
  return ctx;
}
