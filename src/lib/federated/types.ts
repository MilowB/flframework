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
  id: string;
  name: string;
  status: 'idle' | 'receiving' | 'training' | 'sending' | 'completed' | 'error' | 'evaluating';
  progress: number;
  localLoss: number;
  localAccuracy: number;
  localTestAccuracy: number; // Accuracy on client's personalized test set
  dataSize: number;
  lastUpdate: number;
  roundsParticipated: number;
  lastLocalModel?: ModelWeights;
  clientAggregationMethod?: 'none' | '50-50';
}

export interface ServerConfig {
  aggregationMethod: 'fedavg' | 'fedprox' | 'scaffold' | 'custom';
  clientAggregationMethod?: 'none' | '50-50';
  clientsPerRound: number;
  totalRounds: number;
  minClientsRequired: number;
  modelArchitecture: string;
  seed?: number; // Ajout de la seed (défaut 42)
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
}

export interface RoundMetrics {
  round: number;
  globalLoss: number;
  globalAccuracy: number;
  participatingClients: string[];
  aggregationTime: number;
  timestamp: number;
  weightsSnapshot?: WeightsSnapshot;
  // Distance matrix (L2) between client models for this round (NxN)
  distanceMatrix?: number[][];
  // Clusters obtained from community detection (arrays of client ids)
  clusters?: string[][];
  // Average silhouette score across clusters for this round (range -1..1)
  silhouetteAvg?: number;
  // Accuracy of each cluster's averaged model on pooled test data
  clusterMetrics?: ClusterMetrics[];
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
}

export type AggregationFunction = (
  clientWeights: { weights: ModelWeights; dataSize: number }[]
) => ModelWeights;

export interface ClientBehavior {
  onReceiveModel: (model: ModelWeights) => Promise<void>;
  onTrain: (model: ModelWeights, epochs: number) => Promise<{ weights: ModelWeights; loss: number; accuracy: number }>;
  onSendModel: (weights: ModelWeights) => Promise<void>;
}
