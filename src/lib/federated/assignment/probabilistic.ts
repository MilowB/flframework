// Model Assignment Strategy: Probabilistic
// Each client is assigned to a cluster with probability inversely proportional to distance

import type { ModelWeights, ClientState } from '../core/types';
import { getRng } from '../core/random';
import { clusterModelStore } from '../core/stores';
import { unflattenWeights, vectorizeModel, MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE } from '../models/mlp';
import { l2Distance } from '../clustering/louvain';

const EPSILON = 1e-3;

// Compute probabilistic cluster assignments for all selected clients
export const computeProbabilisticAssignments = (
  selectedClients: ClientState[],
  clusterClientIds: string[][],
  globalModel: ModelWeights
): Record<string, number> => {
  const assignments: Record<string, number> = {};
  const rng = getRng();
  
  for (const client of selectedClients) {
    const clientModel = clusterModelStore.get(client.id) || globalModel;
    const clientVec = vectorizeModel(unflattenWeights(clientModel, MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE));
    
    // Compute distances to each cluster
    const clusterDistances = clusterClientIds.map((grp) => {
      if (!grp.length) return Infinity;
      let sum = 0;
      for (const cid of grp) {
        const cModel = clusterModelStore.get(cid) || globalModel;
        const cVec = vectorizeModel(unflattenWeights(cModel, MNIST_INPUT_SIZE, MNIST_HIDDEN_SIZE, MNIST_OUTPUT_SIZE));
        sum += l2Distance(clientVec, cVec);
      }
      return sum / grp.length;
    });
    
    // Check for clusters at zero distance (within epsilon)
    const zeroIdxs = clusterDistances
      .map((d, i) => (isFinite(d) && Math.abs(d) < EPSILON ? i : -1))
      .filter(i => i !== -1);
    
    let probs: number[];
    if (zeroIdxs.length > 0) {
      // Distribute probability only among zero-distance clusters
      probs = clusterDistances.map((_, i) => zeroIdxs.includes(i) ? 1 / zeroIdxs.length : 0);
    } else {
      // Inverse distance weighting
      const sumDist = clusterDistances.reduce((a, b) => a + (isFinite(b) ? b : 0), 0);
      probs = clusterDistances.map(d => {
        if (!isFinite(d) || sumDist === 0) return 1 / clusterDistances.length;
        return 1 - (d / sumDist);
      });
      probs = probs.map(p => Math.max(0, p));
      const total = probs.reduce((a, b) => a + b, 0);
      if (total > 0) probs = probs.map(p => p / total);
      else probs = Array(clusterDistances.length).fill(1 / clusterDistances.length);
    }
    
    // Sample cluster based on probabilities
    const r = rng.next();
    let acc = 0;
    let chosen = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r <= acc) { chosen = i; break; }
    }
    
    assignments[client.id] = chosen;
  }
  
  return assignments;
};
