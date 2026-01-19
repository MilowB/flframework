// Calcule la norme L2 d'un modèle MLP (tous les poids et biais)
function normMLPWeights(a: MLPWeights): number {
    let sum = 0;
    for (let i = 0; i < a.W1.length; i++) {
        for (let j = 0; j < a.W1[i].length; j++) {
            sum += a.W1[i][j] * a.W1[i][j];
        }
    }
    for (let i = 0; i < a.W2.length; i++) {
        for (let j = 0; j < a.W2[i].length; j++) {
            sum += a.W2[i][j] * a.W2[i][j];
        }
    }
    for (let i = 0; i < a.b1.length; i++) {
        sum += a.b1[i] * a.b1[i];
    }
    for (let i = 0; i < a.b2.length; i++) {
        sum += a.b2[i] * a.b2[i];
    }
    return Math.sqrt(sum);
}
// Soustrait deux modèles MLP (élément par élément)
function substractMLPWeights(a: MLPWeights, b: MLPWeights): MLPWeights {
    const result = cloneWeights(a);
    for (let i = 0; i < result.W1.length; i++) {
        for (let j = 0; j < result.W1[i].length; j++) {
            result.W1[i][j] = a.W1[i][j] - b.W1[i][j];
        }
    }
    for (let i = 0; i < result.W2.length; i++) {
        for (let j = 0; j < result.W2[i].length; j++) {
            result.W2[i][j] = a.W2[i][j] - b.W2[i][j];
        }
    }
    for (let i = 0; i < result.b1.length; i++) {
        result.b1[i] = a.b1[i] - b.b1[i];
    }
    for (let i = 0; i < result.b2.length; i++) {
        result.b2[i] = a.b2[i] - b.b2[i];
    }
    return result;
}
// Additionne deux modèles MLP (somme élément par élément)
function addMLPWeights(a: MLPWeights, b: MLPWeights): MLPWeights {
    const result = cloneWeights(a);
    for (let i = 0; i < result.W1.length; i++) {
        for (let j = 0; j < result.W1[i].length; j++) {
            result.W1[i][j] = a.W1[i][j] + b.W1[i][j];
        }
    }
    for (let i = 0; i < result.W2.length; i++) {
        for (let j = 0; j < result.W2[i].length; j++) {
            result.W2[i][j] = a.W2[i][j] + b.W2[i][j];
        }
    }
    for (let i = 0; i < result.b1.length; i++) {
        result.b1[i] = a.b1[i] + b.b1[i];
    }
    for (let i = 0; i < result.b2.length; i++) {
        result.b2[i] = a.b2[i] + b.b2[i];
    }
    return result;
}
// Client Aggregation Strategy: 50/50
// The client starts training from a 50/50 average between
// the received server/cluster model and its previous local model.


import type { MLPWeights } from '../../models/mlp';
import { cloneWeights } from '../../models/mlp';
import {
    unflattenWeights,
    MNIST_INPUT_SIZE,
    MNIST_HIDDEN_SIZE,
    MNIST_OUTPUT_SIZE
} from '../../models/mlp';


// Calcule la distance L2 entre deux modèles MLP
function l2DistanceMLP(a: MLPWeights, b: MLPWeights): number {
    let sum = 0;
    for (let i = 0; i < a.W1.length; i++) {
        for (let j = 0; j < a.W1[i].length; j++) {
            const d = a.W1[i][j] - b.W1[i][j];
            sum += d * d;
        }
    }
    for (let i = 0; i < a.W2.length; i++) {
        for (let j = 0; j < a.W2[i].length; j++) {
            const d = a.W2[i][j] - b.W2[i][j];
            sum += d * d;
        }
    }
    for (let i = 0; i < a.b1.length; i++) {
        const d = a.b1[i] - b.b1[i];
        sum += d * d;
    }
    for (let i = 0; i < a.b2.length; i++) {
        const d = a.b2[i] - b.b2[i];
        sum += d * d;
    }
    return Math.sqrt(sum);
}

/**
 * Stratégie d'agrégation Gravity pour client FL.
 * @param receivedModel Modèle reçu du serveur/cluster
 * @param previousLocalModel Dernier modèle local du client
 * @param localModelHistory Historique des modèles locaux aplatis (optionnel)
 * @param receivedModelHistory Historique des modèles reçus aplatis (optionnel)
 * @param k Facteur d'ajustement de la contre-force
 * @param currentRound Numéro du round courant (optionnel)
 * @param clientId Identifiant du client (optionnel)
 * @param globalModel Modèle global du serveur (moyenne sur tous les clients) (optionnel)
 */
export const applyGravityAggregation = (
    receivedModel: MLPWeights,
    previousLocalModel: MLPWeights | null,
    localModelHistory?: Array<{
        layers: number[][];
        bias: number[];
        version: number;
    }>,
    receivedModelHistory?: Array<{
        layers: number[][];
        bias: number[];
        version: number;
    }>,
    k: number = 0.1, // facteur d'ajustement de la contre-force
    currentRound?: number, // numéro de round courant (optionnel)
    clientId?: string, // identifiant du client (optionnel)
    globalModel?: MLPWeights, // modèle global du serveur (optionnel)
    gradientNormHistory?: number[] // historique des normes de gradient (optionnel)
): MLPWeights => {
    // Si pas de modèle local précédent, on retourne le modèle reçu
    if (!previousLocalModel) {
        return receivedModel;
    }

    // Si augmentation significative du gradient détectée, utiliser le modèle global
    if (detectGradientIncrease(gradientNormHistory) && globalModel) {
        console.log(`Client ${clientId}: Gradient increase detected, using global model`);
        receivedModel = globalModel;
    }

    let w = 1;
    
    // Création d'une copie pour ne pas muter l'original
    const result = cloneWeights(receivedModel);

    // Pondération W1
    for (let i = 0; i < result.W1.length; i++) {
        for (let j = 0; j < result.W1[i].length; j++) {
            result.W1[i][j] = w * receivedModel.W1[i][j] + (1 - w) * previousLocalModel.W1[i][j];
        }
    }
    // Pondération W2
    for (let i = 0; i < result.W2.length; i++) {
        for (let j = 0; j < result.W2[i].length; j++) {
            result.W2[i][j] = w * receivedModel.W2[i][j] + (1 - w) * previousLocalModel.W2[i][j];
        }
    }
    // Pondération b1
    for (let i = 0; i < result.b1.length; i++) {
        result.b1[i] = w * receivedModel.b1[i] + (1 - w) * previousLocalModel.b1[i];
    }
    // Pondération b2
    for (let i = 0; i < result.b2.length; i++) {
        result.b2[i] = w * receivedModel.b2[i] + (1 - w) * previousLocalModel.b2[i];
    }

    return result;
};

/**
 * Détecte si le gradient a augmenté de manière significative (>= 120% du round précédent).
 * @param gradientNormHistory Historique des normes de gradient (index 0 = plus récent)
 * @returns true si augmentation détectée, false sinon
 */
export const detectGradientIncrease = (
    gradientNormHistory: number[] | undefined
): boolean => {
    // Si pas d'historique ou moins de 2 entrées, pas d'augmentation détectée
    if (!gradientNormHistory || gradientNormHistory.length < 2) {
        return false;
    }

    const normN1 = gradientNormHistory[0]; // Most recent (N-1)
    const normN2 = gradientNormHistory[1]; // Second most recent (N-2)
    
    // Si la norme a augmenté de 20% ou plus
    return normN1 >= normN2 * 1.2;
};

/**
 * Calcule le nombre d'epochs ajusté en fonction de l'historique des normes de gradient.
 * Si la norme au round N-1 était >= 120% de la norme au round N-2, double les epochs.
 * @param gradientNormHistory Historique des normes de gradient (index 0 = plus récent)
 * @param baseEpochs Nombre d'epochs de base
 * @param clientId Identifiant du client pour le logging
 * @returns Le nombre d'epochs ajusté
 */
export const computeAdaptiveEpochs = (
    gradientNormHistory: number[] | undefined,
    baseEpochs: number,
    clientId: string
): number => {
    // Si pas d'historique ou moins de 2 entrées, retourner le nombre d'epochs de base
    if (!gradientNormHistory || gradientNormHistory.length < 2) {
        return baseEpochs;
    }

    const normN1 = gradientNormHistory[0]; // Most recent (N-1)
    const normN2 = gradientNormHistory[1]; // Second most recent (N-2)
    
    // Si la norme a augmenté de 20% ou plus, doubler les epochs
    if (normN1 >= normN2 * 1.2) {
        const adjustedEpochs = baseEpochs * 2;
        console.log(`Client ${clientId}: Gradient norm increased by ${((normN1/normN2 - 1) * 100).toFixed(1)}% (${normN2.toFixed(4)} → ${normN1.toFixed(4)}), doubling epochs to ${adjustedEpochs}`);
        return adjustedEpochs;
    }

    return baseEpochs;
};

