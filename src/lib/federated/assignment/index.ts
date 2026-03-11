import type { ModelWeights, ClientState } from '../types';
import { getModelFor1NN } from './oneNN';
import { computeProbabilisticAssignments } from './probabilistic';
import { getModelForCosineSimilarity } from './cosineSimilarity';
import { getModelForAlexandre, type AlexandreContext } from './alexandre';

export type AssignmentMethod = '1NN' | '1NN-Embeddings' | 'Dynamic-1NN-Embeddings' | 'Probabiliste' | 'CosineSimilarity' | 'FedAvg' | 'Alexandre';

// Store for tracking unhappy clients (detected by Alexandre strategy when returning global model)
let unHappyClientsStore: Set<string> = new Set();
// Flag to ensure spectral clustering is only applied once after detection
let spectralClusteringAppliedOnce = false;

export const recordUnHappyClient = (clientId: string): void => {
  unHappyClientsStore.add(clientId);
};

export const getUnHappyClients = (): Set<string> => {
  return new Set(unHappyClientsStore);
};

export const resetUnHappyClients = (): void => {
  unHappyClientsStore.clear();
};

export const hasSpectralBeenApplied = (): boolean => {
  return spectralClusteringAppliedOnce;
};

export const markSpectralAsApplied = (): void => {
  spectralClusteringAppliedOnce = true;
};

export const resetSpectralAppliedFlag = (): void => {
  spectralClusteringAppliedOnce = false;
};

export interface AssignmentContext {
    globalModel: ModelWeights;
    clusterModels?: ModelWeights[];
    clusterAssignments?: Record<string, number>; // clientId -> idxCluster (for oneNN)
    clusterClientIds?: string[][]; // array of clusters (for probabilistic)
    selectedClients?: ClientState[];
    round?: number; // numéro du round fédéré
    distanceMetric?: 'l1' | 'l2' | 'cosine';
    alexandreContext?: AlexandreContext;
}

export const applyAssignment = (
    method: AssignmentMethod,
    client: ClientState,
    context: AssignmentContext
): ModelWeights => {
    // Le numéro du round est accessible ici :
    const round = context.round;
    switch (method) {
        case '1NN':
        case '1NN-Embeddings':
        case 'Dynamic-1NN-Embeddings':
            return getModelFor1NN(
                client.id,
                context.globalModel
            ) || context.globalModel;
        case 'FedAvg':
            return context.globalModel;
        case 'Probabiliste': {
            if (round <= 5) {
                if (!context.selectedClients || !context.clusterClientIds || !context.globalModel || !context.clusterModels) return context.globalModel;
                const assignments = computeProbabilisticAssignments(
                    context.selectedClients,
                    context.clusterClientIds,
                    context.globalModel,
                    context.distanceMetric || 'cosine'
                );
                const idx = assignments[client.id];
                if (typeof idx === 'number' && context.clusterModels[idx]) {
                    return context.clusterModels[idx];
                }
                return context.globalModel;
            }
            else {
                return getModelFor1NN(
                    client.id,
                    context.globalModel
                ) || context.globalModel;
            }
        }
        case 'CosineSimilarity':
            return getModelForCosineSimilarity(
                client.id,
                context.globalModel
            );
        case 'Alexandre': {
            if (!context.alexandreContext) return context.globalModel;
            return getModelForAlexandre(client.id, context.alexandreContext);
        }
        default:
            return context.globalModel;
    }
};
// Assignment module exports
export * from './oneNN';
export * from './probabilistic';
export * from './cosineSimilarity';
export * from './alexandre';
