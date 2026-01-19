// Leiden Algorithm for Community Detection
// An improved version of the Louvain algorithm with guaranteed connectivity

import { getRng, SeededRandom } from '../core/random';

/**
 * Compute modularity of a partition
 * @param A Adjacency matrix (weighted)
 * @param partition Array mapping node index to community
 * @returns Modularity score
 */
const computeModularity = (A: number[][], partition: number[]): number => {
  const n = A.length;
  if (n === 0) return 0;

  // Total weight
  let m = 0;
  const k = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      k[i] += A[i][j];
    }
    m += k[i];
  }
  m = m / 2;
  if (m === 0) return 0;

  // Compute modularity
  let Q = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (partition[i] === partition[j]) {
        Q += A[i][j] - (k[i] * k[j]) / (2 * m);
      }
    }
  }
  return Q / (2 * m);
};

/**
 * Move nodes to optimize modularity (fast local move phase)
 * @param A Adjacency matrix
 * @param partition Current partition
 * @param k Node degrees
 * @param m Total edge weight
 * @returns Updated partition
 */
const fastLocalMove = (A: number[][], partition: number[], k: number[], m: number): number[] => {
  const n = A.length;
  const rng = getRng();
  let improved = true;
  let iterations = 0;
  const maxIterations = 100;

  while (improved && iterations < maxIterations) {
    improved = false;
    iterations++;

    // Shuffle node order for randomness
    const order = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng.next() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }

    for (const i of order) {
      const currentCommunity = partition[i];
      
      // Find neighboring communities
      const neighbors = new Set<number>();
      for (let j = 0; j < n; j++) {
        if (A[i][j] > 0) {
          neighbors.add(partition[j]);
        }
      }

      let bestCommunity = currentCommunity;
      let bestGain = 0;

      // Try moving to each neighboring community
      for (const targetCommunity of neighbors) {
        if (targetCommunity === currentCommunity) continue;

        // Compute modularity gain
        let gain = 0;
        
        // Weight to current community
        let wCurrent = 0;
        // Weight to target community
        let wTarget = 0;
        // Total weight in communities
        let kCurrent = 0;
        let kTarget = 0;

        for (let j = 0; j < n; j++) {
          if (partition[j] === currentCommunity) {
            wCurrent += A[i][j];
            kCurrent += k[j];
          }
          if (partition[j] === targetCommunity) {
            wTarget += A[i][j];
            kTarget += k[j];
          }
        }

        // Modularity change
        const deltaQ = (wTarget - wCurrent) / m - k[i] * (kTarget - kCurrent + k[i]) / (2 * m * m);
        
        if (deltaQ > bestGain) {
          bestGain = deltaQ;
          bestCommunity = targetCommunity;
        }
      }

      // Move node if improvement found
      if (bestCommunity !== currentCommunity && bestGain > 1e-10) {
        partition[i] = bestCommunity;
        improved = true;
      }
    }
  }

  return partition;
};

/**
 * Refine partition by splitting and merging communities
 * @param A Adjacency matrix
 * @param partition Current partition
 * @returns Refined partition
 */
const refinePartition = (A: number[][], partition: number[]): number[] => {
  const n = A.length;
  const rng = getRng();
  
  // Group nodes by community
  const communities = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const c = partition[i];
    if (!communities.has(c)) communities.set(c, []);
    communities.get(c)!.push(i);
  }

  // For each community, check if it should be split
  const newPartition = [...partition];
  let nextCommunityId = Math.max(...partition) + 1;

  for (const [communityId, nodes] of communities) {
    if (nodes.length <= 1) continue;

    // Check connectivity within community
    const subgraph: number[][] = Array.from({ length: nodes.length }, () => 
      new Array(nodes.length).fill(0)
    );
    
    for (let i = 0; i < nodes.length; i++) {
      for (let j = 0; j < nodes.length; j++) {
        subgraph[i][j] = A[nodes[i]][nodes[j]];
      }
    }

    // Check if community is well-connected
    let totalWeight = 0;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        totalWeight += subgraph[i][j];
      }
    }

    // If community has low internal connectivity, consider splitting
    const avgWeight = totalWeight / (nodes.length * (nodes.length - 1) / 2);
    if (avgWeight < 0.1 && nodes.length > 2) {
      // Simple split: assign nodes randomly to two subcommunities
      for (let i = 0; i < nodes.length; i++) {
        if (rng.next() > 0.5) {
          newPartition[nodes[i]] = nextCommunityId;
        }
      }
      nextCommunityId++;
    }
  }

  return newPartition;
};

/**
 * Leiden algorithm for community detection
 * @param A Adjacency (similarity) matrix
 * @param maxIterations Maximum number of iterations
 * @returns Partition array (community assignment for each node)
 */
export const leidenPartition = (A: number[][], maxIterations: number = 10): number[] => {
  const n = A.length;
  if (n === 0) return [];

  // Initialize: each node in its own community
  let partition = Array.from({ length: n }, (_, i) => i);

  // Compute total weight and degrees
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
    // No edges, return singleton partition
    return partition;
  }

  let previousModularity = computeModularity(A, partition);
  let iterations = 0;

  while (iterations < maxIterations) {
    iterations++;

    // Phase 1: Fast local move
    partition = fastLocalMove(A, partition, k, m);

    // Phase 2: Refinement (unique to Leiden)
    partition = refinePartition(A, partition);

    // Phase 3: Aggregate network (simplified - just relabel communities)
    const communityMap = new Map<number, number>();
    let nextId = 0;
    for (let i = 0; i < n; i++) {
      const c = partition[i];
      if (!communityMap.has(c)) {
        communityMap.set(c, nextId++);
      }
      partition[i] = communityMap.get(c)!;
    }

    // Check convergence
    const currentModularity = computeModularity(A, partition);
    if (Math.abs(currentModularity - previousModularity) < 1e-6) {
      break;
    }
    previousModularity = currentModularity;
  }

  // Relabel communities to be consecutive starting from 0
  const uniqueCommunities = Array.from(new Set(partition)).sort((a, b) => a - b);
  const labelMap = new Map<number, number>();
  uniqueCommunities.forEach((c, idx) => labelMap.set(c, idx));
  
  return partition.map(c => labelMap.get(c)!);
};

// Internal helper functions that accept RNG as parameter

const fastLocalMoveWithRng = (A: number[][], partition: number[], k: number[], m: number, rng: SeededRandom): number[] => {
  const n = A.length;
  let improved = true;
  let iterations = 0;
  const maxIterations = 100;

  while (improved && iterations < maxIterations) {
    improved = false;
    iterations++;

    const order = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng.next() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }

    for (const i of order) {
      const currentCommunity = partition[i];
      const neighbors = new Set<number>();
      for (let j = 0; j < n; j++) {
        if (A[i][j] > 0) neighbors.add(partition[j]);
      }

      let bestCommunity = currentCommunity;
      let bestGain = 0;

      for (const targetCommunity of neighbors) {
        if (targetCommunity === currentCommunity) continue;

        let wCurrent = 0;
        let wTarget = 0;
        let kCurrent = 0;
        let kTarget = 0;

        for (let j = 0; j < n; j++) {
          if (partition[j] === currentCommunity) {
            wCurrent += A[i][j];
            kCurrent += k[j];
          }
          if (partition[j] === targetCommunity) {
            wTarget += A[i][j];
            kTarget += k[j];
          }
        }

        const deltaQ = (wTarget - wCurrent) / m - k[i] * (kTarget - kCurrent + k[i]) / (2 * m * m);
        if (deltaQ > bestGain) {
          bestGain = deltaQ;
          bestCommunity = targetCommunity;
        }
      }

      if (bestCommunity !== currentCommunity && bestGain > 1e-10) {
        partition[i] = bestCommunity;
        improved = true;
      }
    }
  }
  return partition;
};

const refinePartitionWithRng = (A: number[][], partition: number[], rng: SeededRandom): number[] => {
  const n = A.length;
  const communities = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const c = partition[i];
    if (!communities.has(c)) communities.set(c, []);
    communities.get(c)!.push(i);
  }

  const newPartition = [...partition];
  let nextCommunityId = Math.max(...partition) + 1;

  for (const [, nodes] of communities) {
    if (nodes.length <= 1) continue;

    const subgraph: number[][] = Array.from({ length: nodes.length }, () =>
      new Array(nodes.length).fill(0)
    );

    for (let i = 0; i < nodes.length; i++) {
      for (let j = 0; j < nodes.length; j++) {
        subgraph[i][j] = A[nodes[i]][nodes[j]];
      }
    }

    let totalWeight = 0;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        totalWeight += subgraph[i][j];
      }
    }

    const avgWeight = totalWeight / (nodes.length * (nodes.length - 1) / 2);
    if (avgWeight < 0.1 && nodes.length > 2) {
      for (let i = 0; i < nodes.length; i++) {
        if (rng.next() > 0.5) {
          newPartition[nodes[i]] = nextCommunityId;
        }
      }
      nextCommunityId++;
    }
  }

  return newPartition;
};

/**
 * Leiden algorithm with external RNG (for isolated randomness)
 * @param A Adjacency (similarity) matrix
 * @param rng Seeded random number generator
 * @param maxIterations Maximum number of iterations
 * @returns Partition array (community assignment for each node)
 */
export const leidenPartitionWithRng = (A: number[][], rng: SeededRandom, maxIterations: number = 10): number[] => {
  const n = A.length;
  if (n === 0) return [];

  let partition = Array.from({ length: n }, (_, i) => i);

  let m = 0;
  const k = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      k[i] += A[i][j];
    }
    m += k[i];
  }
  m = m / 2;

  if (m === 0) return partition;

  let previousModularity = computeModularity(A, partition);
  let iterations = 0;

  while (iterations < maxIterations) {
    iterations++;
    partition = fastLocalMoveWithRng(A, partition, k, m, rng);
    partition = refinePartitionWithRng(A, partition, rng);

    const communityMap = new Map<number, number>();
    let nextId = 0;
    for (let i = 0; i < n; i++) {
      const c = partition[i];
      if (!communityMap.has(c)) {
        communityMap.set(c, nextId++);
      }
      partition[i] = communityMap.get(c)!;
    }

    const currentModularity = computeModularity(A, partition);
    if (Math.abs(currentModularity - previousModularity) < 1e-6) break;
    previousModularity = currentModularity;
  }

  const uniqueCommunities = Array.from(new Set(partition)).sort((a, b) => a - b);
  const labelMap = new Map<number, number>();
  uniqueCommunities.forEach((c, idx) => labelMap.set(c, idx));

  return partition.map(c => labelMap.get(c)!);
};
