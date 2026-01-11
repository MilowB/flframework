// Clustering module exports
export * from './louvain';

import type { ModelWeights } from '../core/types';
import { modelWeightsToMLPWeights, vectorizeModel } from '../models/mlp';
import { l2Distance, distancesToAdjacency, louvainPartition, refinePartition } from './louvain';
import { clusterModelStore } from '../core/stores';

// Compute distance matrix between client models
export const computeDistanceMatrix = (models: { layers: number[][]; bias: number[] }[]): number[][] => {
  const n = models.length;
  const vecs = models.map(m => {
    const mlp = modelWeightsToMLPWeights(m);
    return vectorizeModel(mlp);
  });
  
  const D: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = l2Distance(vecs[i], vecs[j]);
      D[i][j] = d;
      D[j][i] = d;
    }
  }
  return D;
};

// Cluster client models using Louvain community detection
export const clusterClientModels = (clientResults: { id?: string; weights: ModelWeights; dataSize: number }[]) => {
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

  const D = computeDistanceMatrix(validModels);
  if (validModels.length === 0) return { distanceMatrix: D, clusters: [] as string[][] };

  const A = distancesToAdjacency(D);
  const partition = louvainPartition(A);
  const refined = refinePartition(A, partition.slice());

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

  // Stocke le modèle moyen de chaque cluster dans clusterModelStore avec une clé unique
  let clusterIdx = 0;
  for (const [c, members] of Object.entries(clusterMembers)) {
    const idxs = members.idxs;
    if (idxs.length === 0) continue;
    // Moyenne des modèles du cluster
    const sumLayers = validModels[idxs[0]].layers.map(l => new Array(l.length).fill(0));
    const sumBias = new Array(validModels[idxs[0]].bias.length).fill(0);
    for (const i of idxs) {
      for (let li = 0; li < validModels[i].layers.length; li++) {
        for (let k = 0; k < validModels[i].layers[li].length; k++) {
          sumLayers[li][k] += validModels[i].layers[li][k];
        }
      }
      for (let b = 0; b < validModels[i].bias.length; b++) {
        sumBias[b] += validModels[i].bias[b];
      }
    }
    const averagedModel: ModelWeights = {
      layers: sumLayers.map(l => l.map(v => v / idxs.length)),
      bias: sumBias.map(v => v / idxs.length),
      version: 0,
    };
    clusterModelStore.set(`cluster-${clusterIdx}`, averagedModel);
    clusterIdx++;
  }

  const clusters: string[][] = Array.from(clustersMap.values());
  return { distanceMatrix: D, clusters };
};
