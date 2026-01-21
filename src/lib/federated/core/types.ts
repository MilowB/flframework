// Core types for the Federated Learning Framework

export interface ModelWeights {
  layers: number[][];
  bias: number[];
  version: number;
}

export interface ClientConfig {
  id: string;
  name: string;
  dataSize: number;
  learningRate: number;
  localEpochs: number;
}

export interface ClientState {
  /**
   * Historique des 3 derniers modèles locaux après fine-tuning (N, N-1, N-2)
   */
  localModelHistory?: Array<{
    layers: number[][];
    bias: number[];
    version: number;
  }>;
  receivedModelHistory?: Array<{
    layers: number[][];
    bias: number[];
    version: number;
  }>;
  /**
   * Historique des normes de gradient des derniers rounds
   */
  gradientNormHistory?: number[];
  id: string;
  name: string;
  status: 'idle' | 'receiving' | 'training' | 'sending' | 'completed' | 'error' | 'evaluating';
  progress: number;
  localLoss: number;
  localAccuracy: number;
  localTestAccuracy: number;
  dataSize: number;
  lastUpdate: number;
  roundsParticipated: number;
  lastLocalModel?: ModelWeights;
  clientAggregationMethod?: 'none' | '50-50' | 'gravity';
  learningRate?: number;
  localEpochs?: number;
}

export interface ServerConfig {
  aggregationMethod: 'fedavg' | 'fedprox' | 'scaffold' | 'custom';
  clientAggregationMethod?: 'none' | '50-50' | 'gravity';
  modelAssignmentMethod?: '1NN' | 'Probabiliste';
  distanceMetric?: 'l1' | 'l2' | 'cosine';
  clusteringMethod?: 'louvain' | 'kmeans' | 'leiden' | 'spectral';
  kmeansNumClusters?: number; // Number of clusters for K-means (optional, auto-detect if not specified)
  spectralNumClusters?: number; // Number of clusters for Spectral clustering (optional, auto-detect if not specified)
  useAgreementMatrix?: boolean; // Whether to use agreement matrix for consensus clustering
  clientsPerRound: number;
  totalRounds: number;
  minClientsRequired: number;
  modelArchitecture: string;
  seed?: number;
  clientCount?: number;
}

export interface WeightsSnapshot {
  W1Mean: number;
  W1Std: number;
  W2Mean: number;
  W2Std: number;
  b1Mean: number;
  b2Mean: number;
}

export interface ClusterMetrics {
  clusterId: number;
  accuracy: number;
  clientIds: string[];
  weights?: ModelWeights; // Store cluster model weights for visualization
}

export interface ClientRoundMetrics {
  clientId: string;
  clientName: string;
  loss: number;
  accuracy: number;
  testAccuracy: number;
  gradientNorm?: number;
  weights?: ModelWeights; // Store client model weights for visualization
}

export interface RoundMetrics {
  round: number;
  globalLoss: number;
  globalAccuracy: number;
  participatingClients: string[];
  aggregationTime: number;
  timestamp: number;
  weightsSnapshot?: WeightsSnapshot;
  distanceMatrix?: number[][];
  agreementMatrix?: number[][]; // Agreement matrix from consensus clustering
  clusters?: string[][];
  silhouetteAvg?: number;
  clusterMetrics?: ClusterMetrics[];
  clientMetrics?: ClientRoundMetrics[];
  globalModelWeights?: ModelWeights; // Store global model weights for visualization
}

export type ServerStatus = 'idle' | 'sending' | 'waiting' | 'receiving' | 'evaluating' | 'completed';

export interface FederatedState {
  isRunning: boolean;
  currentRound: number;
  totalRounds: number;
  clients: ClientState[];
  serverConfig: ServerConfig;
  roundHistory: RoundMetrics[];
  globalModel: ModelWeights | null;
  serverStatus: ServerStatus;
  forceSpectralNextRound?: { numClusters: number }; // Force Spectral clustering for one round with specified number of clusters
}

export type AggregationFunction = (
  clientWeights: { weights: ModelWeights; dataSize: number }[]
) => ModelWeights;

export interface ClientBehavior {
  onReceiveModel: (model: ModelWeights) => Promise<void>;
  onTrain: (model: ModelWeights, epochs: number) => Promise<{ weights: ModelWeights; loss: number; accuracy: number }>;
  onSendModel: (weights: ModelWeights) => Promise<void>;
}
