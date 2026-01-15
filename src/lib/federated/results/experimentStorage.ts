// Experiment save/load utilities
import type { FederatedState, RoundMetrics, ModelWeights } from '../core/types';
import type { Model3DPosition } from '../visualization/pca';

export interface ExperimentData {
  version: string;
  savedAt: string;
  serverConfig: FederatedState['serverConfig'];
  globalModel: ModelWeights | null;
  roundHistory: RoundMetrics[];
  clientModels: { clientId: string; weights: ModelWeights }[];
  visualizations3D?: {
    round: number;
    models: Model3DPosition[];
  }[];
}

// Generate timestamped filename
const generateFilename = (): string => {
  const now = new Date();
  const timestamp = now.toISOString()
    .replace(/[:.]/g, '-')
    .replace('T', '_')
    .slice(0, 19);
  return `federated-experiment-${timestamp}.json`;
};

export const saveExperiment = (
  state: FederatedState, 
  clientModels: Map<string, ModelWeights>,
  visualizations3D?: { round: number; models: Model3DPosition[] }[]
): void => {
  // Remove heavy weight data from round history for storage
  const cleanedRoundHistory = state.roundHistory.map(round => {
    const { clientMetrics, clusterMetrics, globalModelWeights, ...rest } = round;
    return {
      ...rest,
      // Keep metrics but remove weights to reduce size
      clientMetrics: clientMetrics?.map(({ weights, ...cm }) => cm),
      clusterMetrics: clusterMetrics?.map(({ weights, ...cm }) => cm),
      // Don't save globalModelWeights - too large and only needed for viz
    };
  });
  
  const data: ExperimentData = {
    version: '1.0',
    savedAt: new Date().toISOString(),
    serverConfig: state.serverConfig,
    globalModel: state.globalModel,
    roundHistory: cleanedRoundHistory as RoundMetrics[],
    clientModels: Array.from(clientModels.entries()).map(([clientId, weights]) => ({
      clientId,
      weights,
    })),
    visualizations3D,
  };

  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = generateFilename();
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
