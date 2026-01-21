// Louvain Community Detection Algorithm for Clustered Federated Learning
// Implements modularity optimization with Leiden-like refinement

import { getRng, SeededRandom } from '../core/random';

// Compute L1 (Manhattan) distance between two vectors
export const l1Distance = (a: number[], b: number[]): number => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum;
};

// Compute L2 (Euclidean) distance between two vectors
export const l2DistanceEuclidean = (a: number[], b: number[]): number => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
};

// Compute cosine similarity between two vectors
export const cosineSimilarity = (a: number[], b: number[]): number => {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  if (denominator === 0) return 0;
  return dotProduct / denominator;
};

// Compute distance using specified metric
// If reference is provided, computes distance between (a - reference) and (b - reference)
export const computeDistance = (a: number[], b: number[], metric: 'l1' | 'l2' | 'cosine' = 'cosine', reference?: number[]): number => {
  // If reference is provided, compute relative vectors
  let vecA = a;
  let vecB = b;
  
  if (reference && reference.length === a.length) {
    vecA = a.map((val, i) => val - reference[i]);
    vecB = b.map((val, i) => val - reference[i]);
  }
  
  switch (metric) {
    case 'l1':
      return l1Distance(vecA, vecB);
    case 'l2':
      return l2DistanceEuclidean(vecA, vecB);
    case 'cosine':
    default:
      const similarity = cosineSimilarity(vecA, vecB);
      return 1 - similarity;
  }
};

// Compute cosine distance (1 - cosine similarity) between two vectors
// Kept for backward compatibility
export const l2Distance = (a: number[], b: number[]): number => {
  const similarity = cosineSimilarity(a, b);
  return 1 - similarity;
};

// Convert distance matrix to adjacency (similarity) matrix using RBF kernel
export const distancesToAdjacency = (D: number[][]): number[][] => {
  const n = D.length;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      sum += D[i][j];
      count++;
    }
  }
  const mean = count > 0 ? sum / count : 1;
  const sigma = mean || 1;
  
  const A: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const w = Math.exp(-D[i][j] / sigma);
      A[i][j] = w;
    }
  }
  return A;
};

// Simple Louvain implementation for small N
export const louvainPartition = (A: number[][]): number[] => {
  const n = A.length;
  if (n === 0) return [];

  // Total weight (each edge counted once)
  let m = 0;
  const k = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      k[i] += A[i][j];
    }
    m += k[i];
  }
  m = m / 2;

  if (m === 0) {
    // No edges - each node in its own community
    const single: number[] = [];
    for (let i = 0; i < n; i++) single.push(i);
    return single;
  }

  // Initial communities: each node its own
  const community = new Array(n);
  for (let i = 0; i < n; i++) community[i] = i;

  // Compute sum_tot per community
  const sumTot = k.slice();

  let improvement = true;
  const maxPasses = 10;
  let pass = 0;
  
  const rng = getRng();
  
  while (improvement && pass < maxPasses) {
    improvement = false;
    pass++;
    
    // Iterate nodes in random order (using seeded RNG)
    const nodes = Array.from({ length: n }, (_, i) => i);
    rng.shuffle(nodes);
    
    for (const i of nodes) {
      const oldC = community[i];
      
      // Compute k_i_in for each neighboring community
      const neighCommWeights = new Map<number, number>();
      for (let j = 0; j < n; j++) {
        if (A[i][j] <= 0) continue;
        const c = community[j];
        neighCommWeights.set(c, (neighCommWeights.get(c) || 0) + A[i][j]);
      }

      // Remove i temporarily
      sumTot[oldC] -= k[i];

      let bestC = oldC;
      let bestDelta = 0;
      // Faible valeur = grand cluster et inversement
      const resolution = 2;
      
      for (const [c, k_i_in] of neighCommWeights.entries()) {
        const deltaQ = (k_i_in - resolution * (k[i] * sumTot[c]) / (2 * m)) / (2 * m);
        if (deltaQ > bestDelta) {
          bestDelta = deltaQ;
          bestC = c;
        }
      }

      // Move if best community is different
      if (bestC !== oldC) {
        community[i] = bestC;
        sumTot[bestC] += k[i];
        improvement = true;
      } else {
        sumTot[oldC] += k[i];
      }
    }
  }

  // Normalize community ids to contiguous labels
  const labelMap = new Map<number, number>();
  let nextLabel = 0;
  for (let i = 0; i < n; i++) {
    const c = community[i];
    if (!labelMap.has(c)) labelMap.set(c, nextLabel++);
  }
  for (let i = 0; i < n; i++) community[i] = labelMap.get(community[i])!;

  return community;
};

// Small refinement: ensure each community is reasonably internally connected (Leiden-like)
export const refinePartition = (A: number[][], partition: number[]): number[] => {
  const n = A.length;
  const k = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      k[i] += A[i][j];
    }
  }

  // Compute community internal degree
  const commNodes = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const c = partition[i];
    if (!commNodes.has(c)) commNodes.set(c, []);
    commNodes.get(c)!.push(i);
  }

  // For each node, if its strongest neighbor community is different and stronger
  // than its current internal connection, move it
  for (let i = 0; i < n; i++) {
    const current = partition[i];
    let bestC = current;
    let bestWeight = 0;
    
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const c = partition[j];
      const w = A[i][j];
      if (w > bestWeight) {
        bestWeight = w;
        bestC = c;
      }
    }
    
    // Move if best community gives strictly more connections
    if (bestC !== current) {
      let internal = 0;
      for (const v of (commNodes.get(current) || [])) internal += A[i][v];
      
      let candidate = 0;
      for (const v of (commNodes.get(bestC) || [])) candidate += A[i][v];
      
      if (candidate > internal) {
        partition[i] = bestC;
        commNodes.get(current)!.splice(commNodes.get(current)!.indexOf(i), 1);
        if (!commNodes.has(bestC)) commNodes.set(bestC, []);
        commNodes.get(bestC)!.push(i);
      }
    }
  }

  // Normalize labels
  const labelMap = new Map<number, number>();
  let next = 0;
  for (let i = 0; i < n; i++) {
    const c = partition[i];
    if (!labelMap.has(c)) labelMap.set(c, next++);
  }
  for (let i = 0; i < n; i++) partition[i] = labelMap.get(partition[i])!;

  return partition;
};

// Compute silhouette score for clustering quality
export const computeSilhouetteScore = (
  D: number[][],
  clusters: string[][],
  idToIndex: Map<string, number>
): number | undefined => {
  if (!D || !clusters || clusters.length === 0) return undefined;

  const clusterMeans: number[] = [];
  
  for (const grp of clusters) {
    const sList: number[] = [];
    
    for (const id of grp) {
      const i = idToIndex.get(id);
      if (i === undefined) continue;
      
      // a = average distance to other points in same cluster
      const others = grp.map(x => idToIndex.get(x)).filter((x): x is number => typeof x === 'number' && x !== i);
      let a = 0;
      if (others.length > 0) {
        let sum = 0;
        for (const j of others) sum += D[i][j];
        a = sum / others.length;
      }

      // b = min average distance to points in other clusters
      let b = Infinity;
      for (const otherGrp of clusters) {
        if (otherGrp === grp) continue;
        const members = otherGrp.map(x => idToIndex.get(x)).filter((x): x is number => typeof x === 'number');
        if (members.length === 0) continue;
        let sum = 0;
        for (const j of members) sum += D[i][j];
        const avg = sum / members.length;
        if (avg < b) b = avg;
      }
      if (!isFinite(b)) b = a;

      const denom = Math.max(a, b);
      const s = denom > 0 ? (b - a) / denom : 0;
      sList.push(s);
    }
    
    const clusterMean = sList.length > 0 ? sList.reduce((u, v) => u + v, 0) / sList.length : 0;
    clusterMeans.push(clusterMean);
  }
  
  return clusterMeans.length > 0 ? clusterMeans.reduce((u, v) => u + v, 0) / clusterMeans.length : undefined;
};

// Version of louvainPartition that accepts an external RNG (for isolated randomness)
export const louvainPartitionWithRng = (A: number[][], rng: SeededRandom): number[] => {
  const n = A.length;
  if (n === 0) return [];

  // Total weight (each edge counted once)
  let m = 0;
  const k = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      k[i] += A[i][j];
    }
    m += k[i];
  }
  m = m / 2;

  if (m === 0) {
    const single: number[] = [];
    for (let i = 0; i < n; i++) single.push(i);
    return single;
  }

  const community = new Array(n);
  for (let i = 0; i < n; i++) community[i] = i;
  const sumTot = k.slice();

  let improvement = true;
  const maxPasses = 10;
  let pass = 0;

  while (improvement && pass < maxPasses) {
    improvement = false;
    pass++;

    const nodes = Array.from({ length: n }, (_, i) => i);
    rng.shuffle(nodes);

    for (const i of nodes) {
      const oldC = community[i];
      const neighCommWeights = new Map<number, number>();
      for (let j = 0; j < n; j++) {
        if (A[i][j] <= 0) continue;
        const c = community[j];
        neighCommWeights.set(c, (neighCommWeights.get(c) || 0) + A[i][j]);
      }

      sumTot[oldC] -= k[i];
      let bestC = oldC;
      let bestDelta = 0;
      const resolution = 2;

      for (const [c, k_i_in] of neighCommWeights.entries()) {
        const deltaQ = (k_i_in - resolution * (k[i] * sumTot[c]) / (2 * m)) / (2 * m);
        if (deltaQ > bestDelta) {
          bestDelta = deltaQ;
          bestC = c;
        }
      }

      if (bestC !== oldC) {
        community[i] = bestC;
        sumTot[bestC] += k[i];
        improvement = true;
      } else {
        sumTot[oldC] += k[i];
      }
    }
  }

  const labelMap = new Map<number, number>();
  let nextLabel = 0;
  for (let i = 0; i < n; i++) {
    const c = community[i];
    if (!labelMap.has(c)) labelMap.set(c, nextLabel++);
  }
  for (let i = 0; i < n; i++) community[i] = labelMap.get(community[i])!;

  return community;
};

// Version of refinePartition that accepts an external RNG (for isolated randomness)
// Note: Current refinePartition doesn't use RNG, but we add the parameter for consistency
export const refinePartitionWithRng = (A: number[][], partition: number[], _rng: SeededRandom): number[] => {
  const n = A.length;
  const k = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      k[i] += A[i][j];
    }
  }

  const commNodes = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const c = partition[i];
    if (!commNodes.has(c)) commNodes.set(c, []);
    commNodes.get(c)!.push(i);
  }

  for (let i = 0; i < n; i++) {
    const current = partition[i];
    let bestC = current;
    let bestWeight = 0;

    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const c = partition[j];
      const w = A[i][j];
      if (w > bestWeight) {
        bestWeight = w;
        bestC = c;
      }
    }

    if (bestC !== current) {
      let internal = 0;
      for (const v of (commNodes.get(current) || [])) internal += A[i][v];

      let candidate = 0;
      for (const v of (commNodes.get(bestC) || [])) candidate += A[i][v];

      if (candidate > internal) {
        partition[i] = bestC;
        commNodes.get(current)!.splice(commNodes.get(current)!.indexOf(i), 1);
        if (!commNodes.has(bestC)) commNodes.set(bestC, []);
        commNodes.get(bestC)!.push(i);
      }
    }
  }

  const labelMap = new Map<number, number>();
  let next = 0;
  for (let i = 0; i < n; i++) {
    const c = partition[i];
    if (!labelMap.has(c)) labelMap.set(c, next++);
  }
  for (let i = 0; i < n; i++) partition[i] = labelMap.get(partition[i])!;

  return partition;
};
