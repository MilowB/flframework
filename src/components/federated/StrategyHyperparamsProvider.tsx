import React, { useState } from 'react';
import {
  StrategyHyperparamsContext,
  defaultGravity,
  GravityHyperparams
} from './StrategyHyperparamsContext';

export const StrategyHyperparamsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [gravity, setGravity] = useState<GravityHyperparams>(defaultGravity);
  return (
    <StrategyHyperparamsContext.Provider value={{ gravity, setGravity }}>
      {children}
    </StrategyHyperparamsContext.Provider>
  );
};
