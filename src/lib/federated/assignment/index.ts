import type { ModelWeights, ClientState } from '../types';
import { getModelFor1NN } from './oneNN';
import { computeProbabilisticAssignments } from './probabilistic';

export type AssignmentMethod = '1NN' | 'Probabiliste' | 'Gravity';

export interface AssignmentContext {
    globalModel: ModelWeights;
    clusterModels?: ModelWeights[];
    clusterAssignments?: Record<string, number>; // clientId -> idxCluster (for oneNN)
    clusterClientIds?: string[][]; // array of clusters (for probabilistic)
    selectedClients?: ClientState[];
    round?: number; // numéro du round fédéré
    distanceMetric?: 'l1' | 'l2' | 'cosine';
}

export const applyAssignment = (
    method: AssignmentMethod,
    client: ClientState,
    context: AssignmentContext
): ModelWeights => {
    // Le numéro du round est accessible ici :
    const round = context.round;
    console.log(`Stratégie ${method}`);
    switch (method) {
        case '1NN':
            return getModelFor1NN(
                client.id,
                context.globalModel
            ) || context.globalModel;
        case 'Probabiliste': {
            console.log("Stratégie Probabiliste", round !== undefined ? `(round ${round})` : '');
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
                    console.log("Retourne le modèle avec le plus de probabilité");
                    return context.clusterModels[idx];
                }
                console.log("Retourne modèle global");
                return context.globalModel;
            }
            else {
                console.log("Retourne 1NN");
                return getModelFor1NN(
                    client.id,
                    context.globalModel
                ) || context.globalModel;
            }
        }
        default:
            return context.globalModel;
    }
};
// Assignment module exports
export * from './oneNN';
export * from './probabilistic';
