// Clustering module exports
export * from './louvain';
export * from './kmeans';
export * from './leiden';

import type { ModelWeights } from '../core/types';
import { modelWeightsToMLPWeights, vectorizeModel } from '../models/mlp';
import { computeDistance, distancesToAdjacency, louvainPartitionWithRng, refinePartitionWithRng } from './louvain';
import { kmeansClusteringWithRng, determineOptimalK } from './kmeans';
import { leidenPartitionWithRng } from './leiden';
import { clusterModelStore } from '../core/stores';
import { computeAgreementClustering } from '../server/agreement';
import { getSeed, SeededRandom } from '../core/random';

// DEBUG: Log clustering RNG usage
const DEBUG_CLUSTERING_RNG = false;

// Compute distance matrix between client models
// If referenceModel is provided, computes distances relative to that model (for gradient-based similarity)
export const computeDistanceMatrix = (
  models: { layers: number[][]; bias: number[] }[], 
  distanceMetric: 'l1' | 'l2' | 'cosine' = 'cosine',
  referenceModel?: { layers: number[][]; bias: number[] }
): number[][] => {
  const n = models.length;
  const vecs = models.map(m => {
    const mlp = modelWeightsToMLPWeights(m);
    return vectorizeModel(mlp);
  });
  
  // Vectorize reference model if provided
  let referenceVec: number[] | undefined;
  if (referenceModel) {
    const refMlp = modelWeightsToMLPWeights(referenceModel);
    referenceVec = vectorizeModel(refMlp);
  }
  
  const D: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = computeDistance(vecs[i], vecs[j], distanceMetric, referenceVec);
      D[i][j] = d;
      D[j][i] = d;
    }
  }
  return D;
};

// Cluster client models using specified clustering algorithm
export const clusterClientModels = (
  clientResults: { id?: string; weights: ModelWeights; dataSize: number }[],
  distanceMetric?: 'l1' | 'l2' | 'cosine',
  clusteringMethod: 'louvain' | 'kmeans' | 'leiden' = 'louvain',
  kmeansNumClusters?: number,
  useAgreementMatrix?: boolean,
  referenceModel?: ModelWeights
): { distanceMatrix: number[][]; clusters: string[][]; agreementMatrix?: number[][] } => {
  const validModels: { layers: number[][]; bias: number[] }[] = [];
  const ids: number[] = [];
  for (let i = 0; i < clientResults.length; i++) {
    const c = clientResults[i];
    try {
      modelWeightsToMLPWeights(c.weights); // Validate structure
      validModels.push(c.weights);
      ids.push(i);
    } catch (err) {
      console.warn('clusterClientModels: Skipping malformed ModelWeights for client', c.id, err);
    }
  }

  const D = computeDistanceMatrix(validModels, distanceMetric, referenceModel);
  if (validModels.length === 0) return { distanceMatrix: D, clusters: [] as string[][], agreementMatrix: undefined };

  // Get client IDs
  const clientIds = ids.map(i => 
    clientResults[i] && clientResults[i].id ? clientResults[i].id! : `client-${i}`
  );

  // Use agreement matrix clustering if enabled (only for Louvain/Leiden)
  if (useAgreementMatrix && (clusteringMethod === 'louvain' || clusteringMethod === 'leiden')) {
    const { agreementMatrix, clusters } = computeAgreementClustering(D, clientIds, clusteringMethod);
    return { distanceMatrix: D, clusters, agreementMatrix };
  }

  // Create an ISOLATED RNG for clustering to avoid affecting the global RNG
  // This ensures clustering doesn't change simulation reproducibility
  const isolatedSeed = getSeed() + 200000; // Different offset from agreement matrix
  const isolatedRng = new SeededRandom(isolatedSeed);
  
  if (DEBUG_CLUSTERING_RNG) {
    console.log(`[Clustering] Using isolated RNG with seed ${isolatedSeed}`);
  }

  let refined: number[];

  if (clusteringMethod === 'kmeans') {
    // K-means clustering
    const vecs = validModels.map(m => {
      const mlp = modelWeightsToMLPWeights(m);
      return vectorizeModel(mlp);
    });
    
    // Vectorize reference model if provided (for gradient-based cosine similarity)
    let referenceVec: number[] | undefined;
    if (referenceModel) {
      const refMlp = modelWeightsToMLPWeights(referenceModel);
      referenceVec = vectorizeModel(refMlp);
    }
    
    // Use specified k or determine optimal k automatically
    let k: number;
    if (kmeansNumClusters && kmeansNumClusters > 0) {
      k = Math.min(kmeansNumClusters, validModels.length);
    } else {
      k = Math.min(determineOptimalK(vecs, distanceMetric, 5), validModels.length);
    }
    // Pass reference vector to K-means for gradient-based cosine similarity
    refined = kmeansClusteringWithRng(vecs, k, distanceMetric, isolatedRng, 100, referenceVec);
  } else if (clusteringMethod === 'leiden') {
    // Leiden clustering
    const A = distancesToAdjacency(D);
    refined = leidenPartitionWithRng(A, isolatedRng);
  } else {
    // Louvain clustering (default)
    const A = distancesToAdjacency(D);
    //const partition = louvainPartitionWithRng(A, isolatedRng);
    const partition = louvainPartitionWithRng(A, isolatedRng);
    refined = refinePartitionWithRng(A, partition.slice(), isolatedRng);
  }

  if (DEBUG_CLUSTERING_RNG) {
    console.log(`[Clustering] Completed, global RNG unchanged`);
  }

  // Build clusters of client ids
  const clustersMap = new Map<number, string[]>();
  const clusterMembers: { [key: number]: { idxs: number[] } } = {};
  for (let i = 0; i < refined.length; i++) {
    const c = refined[i];
    if (!clustersMap.has(c)) clustersMap.set(c, []);
    const id = clientResults[ids[i]] && clientResults[ids[i]].id ? clientResults[ids[i]].id : `client-${ids[i]}`;
    clustersMap.get(c)!.push(id);
    if (!clusterMembers[c]) clusterMembers[c] = { idxs: [] };
    clusterMembers[c].idxs.push(i);
  }

  // Sort community IDs to ensure consistent ordering
  const sortedCommunities = Array.from(clustersMap.keys()).sort((a, b) => a - b);
  const clusters: string[][] = [];
  
  sortedCommunities.forEach((communityId) => {
    // Add cluster members in order
    clusters.push(clustersMap.get(communityId)!);
  });
  return { distanceMatrix: D, clusters, agreementMatrix: undefined };
};
