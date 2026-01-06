import { useState, useCallback, useRef, useEffect } from 'react';
import { FederatedState, ClientState, ServerConfig, ServerStatus } from '@/lib/federated/types';
import { initializeModel, createClient, runFederatedRound, preloadMNIST } from '@/lib/federated/simulation';

const defaultServerConfig: ServerConfig = {
  aggregationMethod: 'fedavg',
  clientsPerRound: 3,
  totalRounds: 10,
  minClientsRequired: 2,
  modelArchitecture: 'mlp-small',
};

export const useFederatedLearning = (initialClients: number = 5) => {
  const [state, setState] = useState<FederatedState>(() => ({
    isRunning: false,
    currentRound: 0,
    totalRounds: defaultServerConfig.totalRounds,
    clients: Array.from({ length: initialClients }, (_, i) => createClient(i)),
    serverConfig: defaultServerConfig,
    roundHistory: [],
    globalModel: null,
    serverStatus: 'idle' as ServerStatus,
  }));

  const [mnistLoaded, setMnistLoaded] = useState(false);
  const abortRef = useRef(false);

  // Preload MNIST on mount
  useEffect(() => {
    preloadMNIST().then(() => {
      setMnistLoaded(true);
      console.log('MNIST dataset ready');
    });
  }, []);

  const updateClient = useCallback((clientId: string, update: Partial<ClientState>) => {
    setState(prev => ({
      ...prev,
      clients: prev.clients.map(c =>
        c.id === clientId ? { ...c, ...update } : c
      ),
    }));
  }, []);

  const updateState = useCallback((update: Partial<FederatedState>) => {
    setState(prev => ({ ...prev, ...update }));
  }, []);

  const initializeTraining = useCallback(() => {
    const model = initializeModel(state.serverConfig.modelArchitecture);
    setState(prev => ({
      ...prev,
      globalModel: model,
      currentRound: 0,
      roundHistory: [],
      clients: prev.clients.map(c => ({
        ...c,
        status: 'idle' as const,
        progress: 0,
        localLoss: 0,
        localAccuracy: 0,
        roundsParticipated: 0,
      })),
    }));
  }, [state.serverConfig.modelArchitecture]);

  const updateServerStatus = useCallback((status: ServerStatus) => {
    setState(prev => ({ ...prev, serverStatus: status }));
  }, []);

  const startTraining = useCallback(async () => {
    if (!state.globalModel) {
      initializeTraining();
    }
    
    abortRef.current = false;
    setState(prev => ({ ...prev, isRunning: true }));

    const currentState = state.globalModel ? state : {
      ...state,
      globalModel: initializeModel(state.serverConfig.modelArchitecture),
    };

    for (let round = currentState.currentRound; round < currentState.serverConfig.totalRounds; round++) {
      if (abortRef.current) break;

      try {
        await runFederatedRound(
          { ...currentState, currentRound: round },
          updateState,
          updateClient,
          updateServerStatus
        );
        
        // Small delay between rounds
        setState(prev => ({ ...prev, serverStatus: 'idle' }));
        await new Promise(resolve => setTimeout(resolve, 500));
      } catch (error) {
        console.error('Round failed:', error);
        break;
      }
    }

    setState(prev => ({ ...prev, isRunning: false, serverStatus: 'idle' }));
  }, [state, initializeTraining, updateState, updateClient, updateServerStatus]);

  const stopTraining = useCallback(() => {
    abortRef.current = true;
    setState(prev => ({ ...prev, isRunning: false }));
  }, []);

  const resetTraining = useCallback(() => {
    abortRef.current = true;
    setState(prev => ({
      ...prev,
      isRunning: false,
      currentRound: 0,
      roundHistory: [],
      globalModel: null,
      serverStatus: 'idle' as ServerStatus,
      clients: prev.clients.map(c => ({
        ...c,
        status: 'idle' as const,
        progress: 0,
        localLoss: 0,
        localAccuracy: 0,
        roundsParticipated: 0,
      })),
    }));
  }, []);

  const setClientCount = useCallback((count: number) => {
    setState(prev => ({
      ...prev,
      clients: Array.from({ length: count }, (_, i) => 
        prev.clients[i] || createClient(i)
      ),
    }));
  }, []);

  const updateServerConfig = useCallback((config: Partial<ServerConfig>) => {
    setState(prev => ({
      ...prev,
      serverConfig: { ...prev.serverConfig, ...config },
      totalRounds: config.totalRounds ?? prev.totalRounds,
    }));
  }, []);

  return {
    state,
    mnistLoaded,
    startTraining,
    stopTraining,
    resetTraining,
    initializeTraining,
    setClientCount,
    updateServerConfig,
    updateClient,
  };
};
