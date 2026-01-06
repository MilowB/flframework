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
  status: 'idle' | 'receiving' | 'training' | 'sending' | 'completed' | 'error';
  progress: number;
  localLoss: number;
  localAccuracy: number;
  dataSize: number;
  lastUpdate: number;
  roundsParticipated: number;
}

export interface ServerConfig {
  aggregationMethod: 'fedavg' | 'fedprox' | 'scaffold' | 'custom';
  clientsPerRound: number;
  totalRounds: number;
  minClientsRequired: number;
  modelArchitecture: string;
}

export interface WeightsSnapshot {
  W1Mean: number;
  W1Std: number;
  W2Mean: number;
  W2Std: number;
  b1Mean: number;
  b2Mean: number;
}

export interface RoundMetrics {
  round: number;
  globalLoss: number;
  globalAccuracy: number;
  participatingClients: string[];
  aggregationTime: number;
  timestamp: number;
  weightsSnapshot?: WeightsSnapshot;
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
