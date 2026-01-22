// K-means clustering algorithm
import { getRng, SeededRandom } from '../core/random';
import { computeDistance } from './louvain';

/**
 * K-means clustering algorithm
 * @param vectors Array of feature vectors to cluster
 * @param k Number of clusters
 * @param distanceMetric Distance metric to use ('l1' | 'l2' | 'cosine')
 * @param maxIterations Maximum number of iterations
 * @returns Array of cluster assignments (one per vector)
 */
export const kmeansClustering = (
  vectors: number[][],
  k: number,
  distanceMetric: 'l1' | 'l2' | 'cosine' = 'cosine',
  maxIterations: number = 100
): number[] => {
  const n = vectors.length;
  if (n === 0) return [];
  if (k <= 0) k = 1;
  if (k >= n) k = n;

  const rng = getRng();
  
  // Initialize centroids using k-means++ method
  const centroids: number[][] = [];
  const firstIdx = Math.floor(rng.next() * n);
  centroids.push([...vectors[firstIdx]]);

  // K-means++ initialization
  while (centroids.length < k) {
    const distances: number[] = [];
    let sumDistances = 0;

    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      for (const centroid of centroids) {
        const dist = computeDistance(vectors[i], centroid, distanceMetric);
        minDist = Math.min(minDist, dist);
      }
      distances.push(minDist * minDist); // Squared distance for k-means++
      sumDistances += minDist * minDist;
    }

    // Select next centroid with probability proportional to squared distance
    let threshold = rng.next() * sumDistances;
    let selectedIdx = 0;
    for (let i = 0; i < n; i++) {
      threshold -= distances[i];
      if (threshold <= 0) {
        selectedIdx = i;
        break;
      }
    }
    centroids.push([...vectors[selectedIdx]]);
  }

  // Assignment array: which cluster each vector belongs to
  let assignments = new Array(n).fill(0);
  let hasConverged = false;
  let iterations = 0;

  while (!hasConverged && iterations < maxIterations) {
    iterations++;
    const newAssignments = new Array(n).fill(0);

    // Assignment step: assign each vector to nearest centroid
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      let bestCluster = 0;
      for (let c = 0; c < k; c++) {
        const dist = computeDistance(vectors[i], centroids[c], distanceMetric);
        if (dist < minDist) {
          minDist = dist;
          bestCluster = c;
        }
      }
      newAssignments[i] = bestCluster;
    }

    // Check convergence
    hasConverged = newAssignments.every((val, idx) => val === assignments[idx]);
    assignments = newAssignments;

    if (hasConverged) break;

    // Update step: recompute centroids
    for (let c = 0; c < k; c++) {
      const clusterMembers = assignments
        .map((cluster, idx) => (cluster === c ? idx : -1))
        .filter(idx => idx !== -1);

      if (clusterMembers.length === 0) {
        // Empty cluster: reinitialize with random point
        const randomIdx = Math.floor(rng.next() * n);
        centroids[c] = [...vectors[randomIdx]];
      } else {
        // Compute mean of cluster members
        const dim = vectors[0].length;
        const newCentroid = new Array(dim).fill(0);
        for (const idx of clusterMembers) {
          for (let d = 0; d < dim; d++) {
            newCentroid[d] += vectors[idx][d];
          }
        }
        for (let d = 0; d < dim; d++) {
          newCentroid[d] /= clusterMembers.length;
        }
        centroids[c] = newCentroid;
      }
    }
  }

  return assignments;
};

/**
 * Automatically determine optimal number of clusters using elbow method
 * @param vectors Array of feature vectors
 * @param distanceMetric Distance metric to use
 * @param maxK Maximum number of clusters to try
 * @returns Optimal k value
 */
export const determineOptimalK = (
  vectors: number[][],
  distanceMetric: 'l1' | 'l2' | 'cosine' = 'cosine',
  maxK: number = 5
): number => {
  const n = vectors.length;
  if (n <= 1) return 1;
  if (n <= maxK) maxK = n - 1;

  const inertias: number[] = [];

  for (let k = 1; k <= maxK; k++) {
    const assignments = kmeansClustering(vectors, k, distanceMetric, 50);
    
    // Compute centroids
    const centroids: number[][] = [];
    for (let c = 0; c < k; c++) {
      const clusterMembers = assignments
        .map((cluster, idx) => (cluster === c ? idx : -1))
        .filter(idx => idx !== -1);

      if (clusterMembers.length === 0) {
        centroids.push(new Array(vectors[0].length).fill(0));
      } else {
        const dim = vectors[0].length;
        const centroid = new Array(dim).fill(0);
        for (const idx of clusterMembers) {
          for (let d = 0; d < dim; d++) {
            centroid[d] += vectors[idx][d];
          }
        }
        for (let d = 0; d < dim; d++) {
          centroid[d] /= clusterMembers.length;
        }
        centroids.push(centroid);
      }
    }

    // Compute inertia (sum of squared distances to centroids)
    let inertia = 0;
    for (let i = 0; i < n; i++) {
      const cluster = assignments[i];
      const dist = computeDistance(vectors[i], centroids[cluster], distanceMetric);
      inertia += dist * dist;
    }
    inertias.push(inertia);
  }

  // Use elbow method: find point with maximum rate of change decrease
  if (inertias.length <= 2) return Math.min(2, n);

  let bestK = 2;
  let maxDiff = 0;
  for (let i = 1; i < inertias.length - 1; i++) {
    const diff1 = inertias[i - 1] - inertias[i];
    const diff2 = inertias[i] - inertias[i + 1];
    const changeDiff = diff1 - diff2;
    if (changeDiff > maxDiff) {
      maxDiff = changeDiff;
      bestK = i + 1;
    }
  }

  return bestK;
};

/**
 * K-means clustering with external RNG (for isolated randomness)
 * @param vectors Array of feature vectors to cluster
 * @param k Number of clusters
 * @param distanceMetric Distance metric to use ('l1' | 'l2' | 'cosine')
 * @param rng Seeded random number generator
 * @param maxIterations Maximum number of iterations
 * @returns Array of cluster assignments (one per vector)
 */
export const kmeansClusteringWithRng = (
  vectors: number[][],
  k: number,
  distanceMetric: 'l1' | 'l2' | 'cosine' = 'cosine',
  rng: SeededRandom,
  maxIterations: number = 100
): number[] => {
  const n = vectors.length;
  if (n === 0) return [];
  if (k <= 0) k = 1;
  if (k >= n) k = n;

  // Initialize centroids using k-means++ method
  const centroids: number[][] = [];
  const firstIdx = Math.floor(rng.next() * n);
  centroids.push([...vectors[firstIdx]]);

  // K-means++ initialization
  while (centroids.length < k) {
    const distances: number[] = [];
    let sumDistances = 0;

    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      for (const centroid of centroids) {
        const dist = computeDistance(vectors[i], centroid, distanceMetric);
        minDist = Math.min(minDist, dist);
      }
      distances.push(minDist * minDist);
      sumDistances += minDist * minDist;
    }

    let threshold = rng.next() * sumDistances;
    let selectedIdx = 0;
    for (let i = 0; i < n; i++) {
      threshold -= distances[i];
      if (threshold <= 0) {
        selectedIdx = i;
        break;
      }
    }
    centroids.push([...vectors[selectedIdx]]);
  }

  let assignments = new Array(n).fill(0);
  let hasConverged = false;
  let iterations = 0;

  while (!hasConverged && iterations < maxIterations) {
    iterations++;
    const newAssignments = new Array(n).fill(0);

    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      let bestCluster = 0;
      for (let c = 0; c < k; c++) {
        const dist = computeDistance(vectors[i], centroids[c], distanceMetric);
        if (dist < minDist) {
          minDist = dist;
          bestCluster = c;
        }
      }
      newAssignments[i] = bestCluster;
    }

    hasConverged = newAssignments.every((val, idx) => val === assignments[idx]);
    assignments = newAssignments;

    if (hasConverged) break;

    for (let c = 0; c < k; c++) {
      const clusterMembers = assignments
        .map((cluster, idx) => (cluster === c ? idx : -1))
        .filter(idx => idx !== -1);

      if (clusterMembers.length === 0) {
        const randomIdx = Math.floor(rng.next() * n);
        centroids[c] = [...vectors[randomIdx]];
      } else {
        const dim = vectors[0].length;
        const newCentroid = new Array(dim).fill(0);
        for (const idx of clusterMembers) {
          for (let d = 0; d < dim; d++) {
            newCentroid[d] += vectors[idx][d];
          }
        }
        for (let d = 0; d < dim; d++) {
          newCentroid[d] /= clusterMembers.length;
        }
        centroids[c] = newCentroid;
      }
    }
  }

  return assignments;
};
