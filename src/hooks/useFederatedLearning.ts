import { useState, useCallback, useRef, useEffect } from 'react';
import { FederatedState, ClientState, ServerConfig, ServerStatus, ModelWeights } from '@/lib/federated/types';
import { initializeModel, createClient, runFederatedRound, preloadMNIST, getClientModels, setClientModels, setSeed } from '@/lib/federated/simulation';
import { ExperimentData } from '@/lib/federated/experimentStorage';
// import { useStrategyHyperparams } from '@/components/federated/StrategyHyperparamsContext';
import { useToast } from '@/hooks/use-toast';

const defaultServerConfig: ServerConfig = {
  aggregationMethod: 'fedavg',
  clientsPerRound: 6,
  totalRounds: 5,
  minClientsRequired: 2,
  modelArchitecture: 'mlp-small',
  seed: 42,
};

// Utilitaire pour accès dynamique aux hyperparams dynamiques selon la stratégie sélectionnée
function getActiveDynamicData(hyperparamsByStrategy: Record<string, any>, selectedStrategy: string) {
  const params = hyperparamsByStrategy[selectedStrategy];
  if (
    params &&
    params.dynamicData &&
    typeof params.dynamicClient === 'number' &&
    typeof params.receiverClient === 'number' &&
    typeof params.changeRound === 'number'
  ) {
    return params;
  }
  return undefined;
}

export const useFederatedLearning = (initialClients: number = 5, gravity: any, none: any, fiftyFifty?: any, ...otherStrategies: any[]) => {
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

  // Keep a ref to the latest state so async loops use up-to-date values
  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const [mnistLoaded, setMnistLoaded] = useState(false);
  const abortRef = useRef(false);

  // Preload MNIST on mount
  useEffect(() => {
    preloadMNIST().then(() => {
      setMnistLoaded(true);
      console.log('MNIST dataset ready');
    });
  }, []);

  // gravity is now passed as an argument
  const { toast } = useToast();

  const updateClient = useCallback((clientId: string, update: Partial<ClientState>) => {
    setState(prev => ({
      ...prev,
      clients: prev.clients.map(c =>
        c.id === clientId
          ? { ...c, ...update, lastLocalModel: update.lastLocalModel ?? c.lastLocalModel }
          : c
      ),
    }));
  }, []);

  const updateState = useCallback((update: Partial<FederatedState>) => {
    setState(prev => ({ ...prev, ...update }));
  }, []);

  const initializeTraining = useCallback(() => {
    // Set the seed before initializing model and clients
    setSeed(state.serverConfig.seed ?? 42);
    const model = initializeModel(state.serverConfig.modelArchitecture);
    // Recreate clients with seeded dataSize
    const newClients = Array.from({ length: state.clients.length }, (_, i) => createClient(i));
    setState(prev => ({
      ...prev,
      globalModel: model,
      currentRound: 0,
      roundHistory: [],
      clients: newClients.map(c => ({
        ...c,
        status: 'idle' as const,
        progress: 0,
        localLoss: 0,
        localAccuracy: 0,
        roundsParticipated: 0,
      })),
    }));
  }, [state.serverConfig.modelArchitecture, state.serverConfig.seed, state.clients.length]);

  const updateServerStatus = useCallback((status: ServerStatus) => {
    setState(prev => ({ ...prev, serverStatus: status }));
  }, []);

  const startTraining = useCallback(async () => {
    if (!state.globalModel) {
      initializeTraining();
    }

    abortRef.current = false;
    setState(prev => ({ ...prev, isRunning: true }));

    // Use the latest state from ref so that roundHistory and other fields
    // updated by previous rounds are preserved.
    const startingState = stateRef.current.globalModel ? stateRef.current : {
      ...stateRef.current,
      globalModel: initializeModel(stateRef.current.serverConfig.modelArchitecture),
    };

    let clustersForRound: string[][] | undefined = undefined;

    // Pour éviter plusieurs transferts, on garde trace des transferts déjà faits (clé: clientId + round)
    const dynamicTransferDone = new Set<string>();

    for (let round = startingState.currentRound; round < startingState.serverConfig.totalRounds; round++) {
      if (abortRef.current) break;

      try {
        // Read the freshest state before each round
        const latest = stateRef.current;

        // --- Dynamic data transfer logic (extensible) ---
        const clientAggMethod = latest.serverConfig?.clientAggregationMethod || 'none';
        const hyperparamsByStrategy: Record<string, any> = {
          none,
          gravity,
          fiftyFifty,
          // Ajoutez ici d'autres stratégies si besoin, ex: ...otherStrategies
        };
        const dynamicParams = getActiveDynamicData(hyperparamsByStrategy, clientAggMethod);
        if (
          dynamicParams &&
          round === dynamicParams.changeRound
        ) {
          // Map client numbers to array indices (1-based UI, 0-based array)
          const dynamicIdx = dynamicParams.dynamicClient;
          const receiverIdx = dynamicParams.receiverClient;
          const clients = latest.clients;
          let errorMsg = '';
          if (
            dynamicIdx < 0 || dynamicIdx >= clients.length ||
            receiverIdx < 0 || receiverIdx >= clients.length ||
            dynamicIdx === receiverIdx
          ) {
            errorMsg = 'Numéro de client invalide.';
          } else {
            const receiver = clients[receiverIdx];
            const dynamic = clients[dynamicIdx];
            const transferKey = `${dynamic.id}|${round}`;
            if (!dynamicTransferDone.has(transferKey)) {
              // Data stores are keyed by client.id
              // Use clientDataStore and clientTestDataStore
              // Import here to avoid circular deps
              const { clientDataStore, clientTestDataStore } = await import('@/lib/federated/core/stores');
              const train = clientDataStore.get(receiver.id);
              const test = clientTestDataStore.get(receiver.id);
              if (!train || !test) {
                errorMsg = 'Données manquantes pour le client source.';
              } else {
                console.log(`Changement de dataset du client ${dynamic.id} en le client ${receiver.id}`);
                clientDataStore.set(dynamic.id, JSON.parse(JSON.stringify(train)));
                clientTestDataStore.set(dynamic.id, JSON.parse(JSON.stringify(test)));
                dynamicTransferDone.add(transferKey);
              }
            }
          }
          if (errorMsg) {
            toast({
              title: 'Erreur de transfert de données',
              description: errorMsg,
              variant: 'destructive',
            });
          }
        }

        // Ensure there are enough idle clients before starting the round.
        const minRequired = latest.serverConfig.minClientsRequired ?? 1;
        const waitStart = Date.now();
        const waitTimeout = 10000; // ms
        while (!abortRef.current) {
          const avail = (stateRef.current.clients || []).filter(c => c.status === 'idle').length;
          if (avail >= minRequired) break;
          if (Date.now() - waitStart > waitTimeout) {
            throw new Error(`Not enough clients available. Required: ${minRequired}, Available: ${avail}`);
          }
          // wait a bit and retry
          await new Promise(resolve => setTimeout(resolve, 200));
        }

        const [metrics, nextClusters] = await runFederatedRound(
          { ...stateRef.current, currentRound: round },
          updateState,
          updateClient,
          updateServerStatus,
          clustersForRound
        );
        clustersForRound = nextClusters;

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

  const loadExperiment = useCallback((data: ExperimentData) => {
    abortRef.current = true;
    
    // Restore the seed from the saved config
    setSeed(data.serverConfig.seed ?? 42);
    
    // Restore client models
    const clientModelsMap = new Map<string, ModelWeights>();
    data.clientModels.forEach(({ clientId, weights }) => {
      clientModelsMap.set(clientId, weights);
    });
    setClientModels(clientModelsMap);
    
    setState(prev => ({
      ...prev,
      isRunning: false,
      serverConfig: data.serverConfig,
      globalModel: data.globalModel,
      roundHistory: data.roundHistory,
      currentRound: data.roundHistory.length,
      totalRounds: data.serverConfig.totalRounds,
      serverStatus: 'idle' as ServerStatus,
    }));
  }, []);

  return {
    state,
    mnistLoaded,
    clientModels: getClientModels(),
    startTraining,
    stopTraining,
    resetTraining,
    initializeTraining,
    setClientCount,
    updateServerConfig,
    updateClient,
    loadExperiment,
  };
};
