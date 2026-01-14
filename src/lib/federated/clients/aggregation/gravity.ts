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
    clientId?: string // identifiant du client (optionnel)
): MLPWeights => {
    // Si pas de modèle local précédent, on retourne le modèle reçu
    if (!previousLocalModel) {
        return receivedModel;
    }

    // Constantes physiques
    const G = 9.8;
    const m_centroid = 10e3;
    const m_client = 1;
    // Calcul de la distance L2 entre les deux modèles (N)
    const distance = l2DistanceMLP(receivedModel, previousLocalModel);
    // const sumWeightsNorm = normMLPWeights(sumWeights);
    let w = 1;
    if (distance > 0) {
        const epsilon = 1e-8;
        // Force gravitationnelle réelle (G=9.8, m1=100, m2=1)
        const F = G * m_centroid * m_client / (distance * distance + epsilon);
        // Force max pour distance nulle
        const Fmax = G * m_centroid * m_client;
        // Pondération normalisée entre 0 et 1
        //w = F / Fmax;
        //w = Math.max(0, Math.min(1, w));
    }

    if (clientId === "client-0") {
        if (currentRound < 3 || currentRound >= 10) {
            w = 1;
        }
        else {
            w = 0;
        }
    }
    else{
        w = 1;
    }

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
