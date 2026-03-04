import type { ModelWeights } from '../types';
import { getModelFor1NN } from './oneNN';
import { findMostApproachedCluster } from './cosineSimilarity';
import { recordUnHappyClient } from './index';
import { cosineHistoryStore } from '../core/stores';
import { gradientNormHistoryStore } from '../core/stores';

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
  /** Previous round distance matrix used to compute movement between rounds */
  previousDistanceMatrix?: number[][];
  /** Cluster membership used for approached-cluster detection */
  clusters?: string[][];
  /** Ordered list of participating client IDs (matches distanceMatrix indices) */
  participatingClients: string[];
  /** Available cluster models */
  clusterModels: ModelWeights[];
  /** Global model as fallback */
  globalModel: ModelWeights;
  /** cosine score per client (clientId -> cosine) */
  cosine?: Record<string, number>;
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
  const _previousDistanceMatrix = context.previousDistanceMatrix;
  const _clusters = context.clusters;
  const _participatingClients = context.participatingClients;

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
  => plus globalement, on a une distance_mouvement comme cosine=alpha=Sa+(1-alpha)Ss => similaire si grand mouvement similaire OU similaire si mouvement moyen faible
  Distance: on a déjà la similarité des distances Sd
  Synthèse: SIm=Sd*cosine
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


  if (cosine > 0 && vi > 0 && vj > 0) {
    const memberCount: number = _clusters && _clusters[_clusterIdx]
      ? _clusters[_clusterIdx].length
      : _participatingClients.filter(id => context.clusterAssignments[id] === _clusterIdx).length;

    console.log(`Nombre de membres dans le cluster ${_clusterIdx}: ${memberCount}`);
    // Sa = cosine_similarity * (Vi + Vj) / (2 * max(vi, vj, ...))

    // Store cosine score for this client in context.cosine
    if (context.cosine) {
      context.cosine[clientId] = cosine;
    } else {
      context.cosine = { [clientId]: cosine };
    }

    // Initialize history store if needed
    if (!cosineHistoryStore.has(clientId)) {
      cosineHistoryStore.set(clientId, []);
    }
    // Initialize history store if needed
    if (!gradientNormHistoryStore.has(clientId)) {
      gradientNormHistoryStore.set(clientId, []);
    }

    // --- Z-score calculation for cosine (client time series) ---
    // Calculate z-score on PREVIOUS rounds only (don't include current cosine yet)
    const cosineClientHistory = cosineHistoryStore.get(clientId)!;
    const gradientNormHistory = gradientNormHistoryStore.get(clientId)!;
    let zCosineScore = 0.5;
    let zGradientNormScore = 0.5;
    let cosineMean = 0;
    let cosineStd = 0;
    let gradientNormMean = 0;
    let gradientNormStd = 0;
    if (cosineClientHistory.length > 1 && gradientNormHistory.length > 1) {
      cosineMean = cosineClientHistory.reduce((a, b) => a + b, 0) / cosineClientHistory.length;
      gradientNormMean = gradientNormHistory.reduce((a, b) => a + b, 0) / gradientNormHistory.length;
      cosineStd = Math.sqrt(cosineClientHistory.reduce((a, b) => a + Math.pow(b - cosineMean, 2), 0) / cosineClientHistory.length);
      gradientNormStd = Math.sqrt(gradientNormHistory.reduce((a, b) => a + Math.pow(b - gradientNormMean, 2), 0) / gradientNormHistory.length);
      if (cosineStd > 0) {
        zCosineScore = cosineMean - cosineStd;
      } else {
        zCosineScore = cosineMean;
      }
      if (gradientNormStd > 0) {
        zGradientNormScore = gradientNormMean - gradientNormStd;
      } else {
        zGradientNormScore = gradientNormMean;
      }
    }
    console.log(`Historique cosine du client ${clientId}: [${cosineClientHistory.map(v => v?.toFixed(4)).join(", ")}]`);
    console.log(`Historique gradient norm du client ${clientId}: [${gradientNormHistory.map(v => v?.toFixed(4)).join(", ")}]`);
    console.log(`zCosineScore (client ${clientId}): ${zCosineScore.toFixed(4)}, moyenne cosine: ${cosineMean.toFixed(4)}, std cosine: ${cosineStd.toFixed(4)}`);
    console.log(`zGradientNormScore (client ${clientId}): ${zGradientNormScore.toFixed(4)}, moyenne gradient norm: ${gradientNormMean.toFixed(4)}, std gradient norm: ${gradientNormStd.toFixed(4)}`);
    console.log(`cosine courant (${clientId}): ${cosine?.toFixed(4)}`);
    console.log(`gradient norm courant (${clientId}): ${_gradientNorm?.toFixed(4)}`);

    // Store current cosine in history AFTER z-score calculation
    cosineHistoryStore.get(clientId)!.push(cosine);
    gradientNormHistoryStore.get(clientId)!.push(_gradientNorm);

    if (cosine !== undefined && cosine < zCosineScore && _gradientNorm > zGradientNormScore) {
      console.log("Client mécontent détecté, il va être réaffecté à un nouveau cluster.")

      recordUnHappyClient(clientId);

      const approachedClusterResult =
        _distanceMatrix && _previousDistanceMatrix && _clusters
          ? findMostApproachedCluster(
            clientId,
            _distanceMatrix,
            _previousDistanceMatrix,
            _clusters,
            _participatingClients
          )
          : null;

      const bestCluster = approachedClusterResult?.clusterId;

      if (bestCluster !== undefined && bestCluster >= 0 && clusterModels[bestCluster]) {
        console.log(
          `Réaffectation Alexandre: ${clientId} vers cluster ${bestCluster} (avg change: ${approachedClusterResult?.avgDistanceChange.toFixed(4)})`
        );
        return clusterModels[bestCluster];
      }

      if (_clusterIdx >= 0 && _clusterIdx < clusterModels.length) {
        console.log(`Ce client reste dans son cluster`);
        return clusterModels[_clusterIdx];
      }

      console.log("Pas de meilleur cluster trouvé. 1NN appliqué.")
      return getModelFor1NN(clientId, context.globalModel);
    }

    // cosine ≤ zCosineScore or no better cluster found: stay with current cluster model
    if (_clusterIdx >= 0 && _clusterIdx < clusterModels.length) {
      return getModelFor1NN(clientId, context.globalModel);
    }
  }
  // Store cosine score for this client even if not calculated above (undefined)
  if (context.cosine && cosine !== undefined) {
    context.cosine[clientId] = cosine;
  }
  console.log("Retour 1NN");
  // Default: use 1NN (return the cluster model the client belongs to)
  return getModelFor1NN(clientId, context.globalModel);
}
