import type { ModelWeights } from '../types';
import { computeCosineSimilarity } from '../models/mlp';
import { getModelFor1NN } from './oneNN';

export interface AlexandreContext {
  /** Gradient norms per client for current round (clientId -> norm) */
  gradientNorms: Record<string, number>;
  /** Cosine similarity per client with median gradient of its cluster (clientId -> similarity) */
  cosineSimilarities: Record<string, number>;
  /** L2 norm of each cluster centroid gradient (clusterIdx -> norm, delta between round N-1 and N centroid) */
  clusterGradientNorms: Record<number, number>;
  /** Cluster index per client (clientId -> clusterIdx) */
  clusterAssignments: Record<string, number>;
  /** Median gradient vector per cluster (clusterIdx -> vector) */
  clusterMedianGradients: Record<number, number[]>;
  /** Gradient vector per client (clientId -> vector) */
  clientGradients: Record<string, number[]>;
  /** Distance/similarity matrix between all participating clients (indices match participatingClients order) */
  distanceMatrix: number[][] | undefined;
  /** Ordered list of participating client IDs (matches distanceMatrix indices) */
  participatingClients: string[];
  /** Available cluster models */
  clusterModels: ModelWeights[];
  /** Global model as fallback */
  globalModel: ModelWeights;
}

/**
 * Assigns a model to a client based on gradient norms and cosine similarities.
 * 
 * @param clientId - The client to assign a model to
 * @param context - Contains gradientNorms and cosineSimilarities for all clients
 * @returns The model weights to send to this client
 */
export function getModelForAlexandre(
  clientId: string,
  context: AlexandreContext
): ModelWeights {
  const _gradientNorm = context.gradientNorms[clientId];
  const _cosineSimilarity = context.cosineSimilarities[clientId];
  const _clusterIdx = context.clusterAssignments[clientId];
  const _distanceMatrix = context.distanceMatrix;
  const _participatingClients = context.participatingClients;
  // Index of this client in the distance matrix
  const _clientMatrixIdx = _participatingClients.indexOf(clientId);

  // TODO: Implement assignment logic using gradientNorm and cosineSimilarity
  // Available data:
  //   - _gradientNorm: L2 norm of the client's gradient (delta weights between round N-1 and N)
  //   - _cosineSimilarity: cosine similarity of the client's delta weights with the median gradient of its cluster
  //   - _clusterIdx: index of the cluster this client currently belongs to
  //   - _distanceMatrix: server-computed distance matrix between all participating clients
  //   - _clientMatrixIdx: row/column index of this client in _distanceMatrix
  //   - context.clusterGradientNorms: L2 norm of each cluster centroid gradient (clusterIdx -> norm)
  //   - context.clusterModels: array of cluster model weights
  //   - context.globalModel: the global aggregated model
  //
  // Return the ModelWeights to assign to this client for the next round.

  /*
  mouvement V.e(itheta)
  => similarité du mouvement pour une paire (i,j): Sa=cosine_similarity(theta_i, theta_j)*(Vi/+Vj)/(2*max(V)) => valorise (proche de 1) les mouvements similaires (à amplitude suffisante)
  => non pénalisation des cas à mouvement faible Ss= 1 / (1 + exp(-Vm)) avec Vm=(Vi/+Vj)/2
  => plus globalement, on a une distance_mouvement comme Sm=alpha=Sa+(1-alpha)Ss => similaire si grand mouvement similaire OU similaire si mouvement moyen faible
  Distance: on a déjà la similarité des distances Sd
  Synthèse: SIm=Sd*Sm
  */

  // Basic safety checks
  if (_gradientNorm === undefined || _cosineSimilarity === undefined || _clusterIdx === undefined) {
    return context.globalModel;
  }

  const vi = _gradientNorm;
  const vj: number = context.clusterGradientNorms?.[_clusterIdx] ?? 0;
  const cosine = _cosineSimilarity;
  const clusterModels = context.clusterModels ?? [];
  console.log("-----------------");
  console.log("Client " + clientId);
  if (clusterModels.length === 0) return context.globalModel;
  const maxV = Math.max(vi, vj);
  console.log("cosine: " + cosine);
  console.log("vi: " + vi);
  console.log("vj: " + vj);

  if (cosine > 0) {
    // Sa = cosine_similarity * (Vi + Vj) / (2 * max(vi, vj, ...))
    const sa = cosine * (vi + vj) / (2 * maxV);
    console.log("sa: " + sa);

    // Ss = sigmoid of the (normalized) mean movement to avoid penalizing small movements
    const vm = (vi + vj) / (2 * maxV);
    console.log("vm: " + vm);
    const ss = Math.exp(-vm);
    console.log("ss: " + ss);

    // Combine Sa and Ss (alpha favors Sa)
    const alpha = 0.5;
    const sm = alpha * sa + (1 - alpha) * ss;

    // Decision threshold:
    // - sm > threshold: client wants to move → find the cluster with highest cosine similarity to this client
    // - sm ≤ threshold: client stays → return its current cluster model
    const threshold = 0.5;
    console.log("sm: " + sm);

    if (sm < threshold) {
      console.log("Client mécontent détecté, il va être réaffecté à un nouveau cluster.")
      console.log("Affectation: envoi du modèle global au client.");
      return context.globalModel;
    }

    // sm ≤ threshold or no better cluster found: stay with current cluster model
    if (_clusterIdx >= 0 && _clusterIdx < clusterModels.length) {
      return clusterModels[_clusterIdx];
    }
  }
  console.log("Retour 1NN");
  // Default: use 1NN (return the cluster model the client belongs to)
  return getModelFor1NN(clientId, context.globalModel);
}
