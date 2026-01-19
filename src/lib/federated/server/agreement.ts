// Agreement Matrix for Consensus Clustering
// Runs multiple clustering iterations with varying resolution to find stable clusters

import { distancesToAdjacency } from '../clustering/louvain';
import { getRng, getSeed, SeededRandom } from '../core/random';

// DEBUG: Log RNG state consumption for agreement matrix
const DEBUG_RNG = true;

/**
 * Louvain partition with configurable resolution parameter
 * @param A Adjacency matrix (weighted)
 * @param resolution Resolution parameter (lower = larger clusters, higher = smaller clusters)
 * @param rngOverride Optional RNG to use (if not provided, uses global RNG)
 * @returns Partition array (community assignment for each node)
 */
export const louvainWithResolution = (A: number[][], resolution: number, rngOverride?: SeededRandom): number[] => {
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
  
  // Use provided RNG or fall back to global
  const rng = rngOverride || getRng();
  
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

/**
 * Leiden partition with configurable resolution parameter
 * @param A Adjacency matrix (weighted)
 * @param resolution Resolution parameter
 * @param maxIterations Maximum number of iterations
 * @param rngOverride Optional RNG to use (if not provided, uses global RNG)
 * @returns Partition array (community assignment for each node)
 */
export const leidenWithResolution = (A: number[][], resolution: number, maxIterations: number = 10, rngOverride?: SeededRandom): number[] => {
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
    return partition;
  }

  // Use provided RNG or fall back to global
  const rng = rngOverride || getRng();
  let previousModularity = computeModularityWithResolution(A, partition, resolution);
  let iterations = 0;

  while (iterations < maxIterations) {
    iterations++;

    // Phase 1: Fast local move with resolution
    partition = fastLocalMoveWithResolution(A, partition, k, m, resolution, rng);

    // Phase 2: Refinement
    partition = refinePartitionLeiden(A, partition, rng);

    // Phase 3: Normalize labels
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
    const currentModularity = computeModularityWithResolution(A, partition, resolution);
    if (Math.abs(currentModularity - previousModularity) < 1e-6) {
      break;
    }
    previousModularity = currentModularity;
  }

  // Relabel communities
  const uniqueCommunities = Array.from(new Set(partition)).sort((a, b) => a - b);
  const labelMap = new Map<number, number>();
  uniqueCommunities.forEach((c, idx) => labelMap.set(c, idx));
  
  return partition.map(c => labelMap.get(c)!);
};

/**
 * Compute modularity with resolution parameter
 */
const computeModularityWithResolution = (A: number[][], partition: number[], resolution: number): number => {
  const n = A.length;
  if (n === 0) return 0;

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

  let Q = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (partition[i] === partition[j]) {
        Q += A[i][j] - resolution * (k[i] * k[j]) / (2 * m);
      }
    }
  }
  return Q / (2 * m);
};

/**
 * Fast local move with resolution parameter for Leiden
 */
const fastLocalMoveWithResolution = (
  A: number[][], 
  partition: number[], 
  k: number[], 
  m: number, 
  resolution: number,
  rng: { next: () => number; shuffle: (arr: number[]) => void }
): number[] => {
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
        if (A[i][j] > 0) {
          neighbors.add(partition[j]);
        }
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

        const deltaQ = (wTarget - wCurrent) / m - resolution * k[i] * (kTarget - kCurrent + k[i]) / (2 * m * m);
        
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

/**
 * Refinement phase for Leiden
 */
const refinePartitionLeiden = (
  A: number[][], 
  partition: number[],
  rng: { next: () => number }
): number[] => {
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
 * Build agreement matrix by running multiple clusterings with varying resolutions
 * Uses an ISOLATED RNG to avoid affecting the global simulation RNG state.
 * This ensures that using agreement matrix vs not using it produces the same results
 * when clusterings are identical.
 * 
 * @param D Distance matrix between clients
 * @param clusteringMethod 'louvain' or 'leiden' (defaults to leiden)
 * @param numRuns Number of clustering runs (default: 20)
 * @param minResolution Minimum resolution value (default: 0.5)
 * @param maxResolution Maximum resolution value (default: 2.5)
 * @returns Agreement matrix where entry [i][j] is the count of times clients i and j were clustered together
 */
export const buildAgreementMatrix = (
  D: number[][],
  clusteringMethod: 'louvain' | 'leiden' = 'leiden',
  numRuns: number = 20,
  minResolution: number = 0.5,
  maxResolution: number = 2.5
): number[][] => {
  const n = D.length;
  if (n === 0) return [];

  // Convert distance matrix to adjacency matrix
  const A = distancesToAdjacency(D);

  // Initialize agreement matrix
  const agreement: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  // Create an ISOLATED RNG for agreement matrix computation
  // This ensures agreement matrix doesn't consume values from the global RNG
  // and thus doesn't affect subsequent simulation steps
  const isolatedSeed = getSeed() + 100000; // Use a derived seed to keep reproducibility
  const isolatedRng = new SeededRandom(isolatedSeed);
  
  if (DEBUG_RNG) {
    console.log(`[AgreementMatrix] Using isolated RNG with seed ${isolatedSeed} for ${numRuns} runs`);
  }

  // Run clustering multiple times with varying resolution
  for (let run = 0; run < numRuns; run++) {
    const resolution = minResolution + (maxResolution - minResolution) * (run / (numRuns - 1));
    
    let partition: number[];
    if (clusteringMethod === 'louvain') {
      partition = louvainWithResolution(A, resolution, isolatedRng);
    } else {
      partition = leidenWithResolution(A, resolution, 10, isolatedRng);
    }

    // Update agreement matrix: increment if two clients are in same cluster
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (partition[i] === partition[j]) {
          agreement[i][j]++;
        }
      }
    }
  }

  if (DEBUG_RNG) {
    console.log(`[AgreementMatrix] Completed ${numRuns} runs, global RNG unchanged`);
  }

  return agreement;
};

/**
 * Extract final clusters from agreement matrix using threshold
 * @param agreementMatrix Agreement matrix
 * @param numRuns Total number of runs used to build the agreement matrix
 * @param threshold Fraction of runs required for two clients to be clustered together (default: 0.6)
 * @returns Clusters as arrays of client indices
 */
export const extractClustersFromAgreement = (
  agreementMatrix: number[][],
  numRuns: number = 20,
  threshold: number = 0.6
): number[][] => {
  const n = agreementMatrix.length;
  if (n === 0) return [];

  const thresholdCount = numRuns * threshold;
  
  // Build connected components based on threshold
  const visited = new Array(n).fill(false);
  const clusters: number[][] = [];

  for (let i = 0; i < n; i++) {
    if (visited[i]) continue;
    
    // BFS to find all connected nodes
    const cluster: number[] = [];
    const queue: number[] = [i];
    visited[i] = true;

    while (queue.length > 0) {
      const current = queue.shift()!;
      cluster.push(current);

      for (let j = 0; j < n; j++) {
        if (!visited[j] && agreementMatrix[current][j] >= thresholdCount) {
          visited[j] = true;
          queue.push(j);
        }
      }
    }

    clusters.push(cluster);
  }

  return clusters;
};

/**
 * Compute agreement-based clustering
 * Runs multiple clusterings with varying resolution and groups clients that 
 * were together in the same cluster at least `threshold` fraction of the time
 * 
 * @param D Distance matrix between clients
 * @param clientIds Array of client IDs
 * @param clusteringMethod 'louvain' or 'leiden' (defaults to leiden if invalid method)
 * @param numRuns Number of clustering runs (default: 20)
 * @param threshold Fraction threshold for agreement (default: 0.6)
 * @returns Object containing agreement matrix and clusters
 */
export const computeAgreementClustering = (
  D: number[][],
  clientIds: string[],
  clusteringMethod: 'louvain' | 'kmeans' | 'leiden' = 'leiden',
  numRuns: number = 20,
  threshold: number = 0.6
): { agreementMatrix: number[][]; clusters: string[][] } => {
  // If method is kmeans or invalid, default to leiden
  const method = (clusteringMethod === 'louvain' || clusteringMethod === 'leiden') 
    ? clusteringMethod 
    : 'leiden';

  const agreementMatrix = buildAgreementMatrix(D, method, numRuns);
  const clusterIndices = extractClustersFromAgreement(agreementMatrix, numRuns, threshold);

  // Convert indices to client IDs
  const clusters: string[][] = clusterIndices.map(cluster => 
    cluster.map(idx => clientIds[idx] || `client-${idx}`)
  );

  return { agreementMatrix, clusters };
};
