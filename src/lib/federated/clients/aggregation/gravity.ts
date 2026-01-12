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
    k: number = 0.1 // facteur d'ajustement de la contre-force
): MLPWeights => {
    // Si pas de modèle local précédent, on retourne le modèle reçu
    if (!previousLocalModel) {
        return receivedModel;
    }

    // Constantes physiques
    const G = 9.8;
    const m_centroid = 10e3;
    const m_client = 10;
    // Calcul de la distance L2 entre les deux modèles (N)
    const distance = l2DistanceMLP(receivedModel, previousLocalModel);
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
        w = 1

        // --- Modulation par la "vitesse" d'éloignement (distance N-1) ---
        console.log(localModelHistory);
        console.log(receivedModelHistory);
        if (localModelHistory && receivedModelHistory &&
            localModelHistory.length > 1 && receivedModelHistory.length > 1 &&
            localModelHistory[1] && receivedModelHistory[1]) {
            const localN1 = unflattenWeights(localModelHistory[1], MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE);
            const receivedN1 = unflattenWeights(receivedModelHistory[1], MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE);
            // Somme des deux modèles
            const sumWeights = addMLPWeights(substractMLPWeights(localN1, receivedN1), substractMLPWeights(localN1, receivedModel));
            const sumWeightsNorm = normMLPWeights(sumWeights);
            const vb = substractMLPWeights(localN1, receivedModel);
            const sumVbNorm = normMLPWeights(vb);
            w = sumVbNorm / (sumVbNorm + sumWeightsNorm);
            console.log('Somme des modèles receivedN1 + receivedModel:', sumWeights);
            console.log('Norme (longueur) du vecteur sumWeights:', sumWeightsNorm);
            console.log(`Gravity aggregation: distance_N=${distance}, v_N-1=${vb}, w_final=${w}`);
        } else {
            console.log(`Gravity aggregation: distance_N=${distance}, pas d'historique N-1, w=${w}`);
        }
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
