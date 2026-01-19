// Clustering module exports
export * from './louvain';
export * from './kmeans';
export * from './leiden';

import type { ModelWeights } from '../core/types';
import { modelWeightsToMLPWeights, vectorizeModel } from '../models/mlp';
import { computeDistance, distancesToAdjacency, louvainPartition, refinePartition } from './louvain';
import { kmeansClustering, determineOptimalK } from './kmeans';
import { leidenPartition } from './leiden';
import { clusterModelStore } from '../core/stores';
import { computeAgreementClustering } from '../server/agreement';

// Compute distance matrix between client models
export const computeDistanceMatrix = (models: { layers: number[][]; bias: number[] }[], distanceMetric: 'l1' | 'l2' | 'cosine' = 'cosine'): number[][] => {
  const n = models.length;
  const vecs = models.map(m => {
    const mlp = modelWeightsToMLPWeights(m);
    return vectorizeModel(mlp);
  });
  
  const D: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = computeDistance(vecs[i], vecs[j], distanceMetric);
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
  useAgreementMatrix?: boolean
) => {
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

  const D = computeDistanceMatrix(validModels, distanceMetric);
  if (validModels.length === 0) return { distanceMatrix: D, clusters: [] as string[][] };

  // Get client IDs
  const clientIds = ids.map(i => 
    clientResults[i] && clientResults[i].id ? clientResults[i].id! : `client-${i}`
  );

  // Use agreement matrix clustering if enabled (only for Louvain/Leiden)
  if (useAgreementMatrix && (clusteringMethod === 'louvain' || clusteringMethod === 'leiden')) {
    const { clusters } = computeAgreementClustering(D, clientIds, clusteringMethod);
    return { distanceMatrix: D, clusters };
  }

  let refined: number[];

  if (clusteringMethod === 'kmeans') {
    // K-means clustering
    const vecs = validModels.map(m => {
      const mlp = modelWeightsToMLPWeights(m);
      return vectorizeModel(mlp);
    });
    
    // Use specified k or determine optimal k automatically
    let k: number;
    if (kmeansNumClusters && kmeansNumClusters > 0) {
      k = Math.min(kmeansNumClusters, validModels.length);
    } else {
      k = Math.min(determineOptimalK(vecs, distanceMetric, 5), validModels.length);
    }
    refined = kmeansClustering(vecs, k, distanceMetric);
  } else if (clusteringMethod === 'leiden') {
    // Leiden clustering
    const A = distancesToAdjacency(D);
    refined = leidenPartition(A);
  } else {
    // Louvain clustering (default)
    const A = distancesToAdjacency(D);
    const partition = louvainPartition(A);
    refined = refinePartition(A, partition.slice());
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
  return { distanceMatrix: D, clusters };
};
