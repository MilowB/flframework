// Experiment save/load utilities
import { FederatedState, RoundMetrics, ModelWeights } from './types';

export interface ExperimentData {
  version: string;
  savedAt: string;
  serverConfig: FederatedState['serverConfig'];
  globalModel: ModelWeights | null;
  roundHistory: RoundMetrics[];
  clientModels: { clientId: string; weights: ModelWeights }[];
}

export const saveExperiment = (state: FederatedState, clientModels: Map<string, ModelWeights>): void => {
  const data: ExperimentData = {
    version: '1.0',
    savedAt: new Date().toISOString(),
    serverConfig: state.serverConfig,
    globalModel: state.globalModel,
    roundHistory: state.roundHistory,
    clientModels: Array.from(clientModels.entries()).map(([clientId, weights]) => ({
      clientId,
      weights,
    })),
  };

  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = `federated-experiment-${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

export const loadExperiment = (file: File): Promise<ExperimentData> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target?.result as string) as ExperimentData;
        if (!data.version || !data.roundHistory) {
          throw new Error('Format de fichier invalide');
        }
        resolve(data);
      } catch (err) {
        reject(new Error('Impossible de lire le fichier d\'expérience'));
      }
    };
    reader.onerror = () => reject(new Error('Erreur de lecture du fichier'));
    reader.readAsText(file);
  });
};
