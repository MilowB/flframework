// Spectral Clustering Algorithm
// Implements spectral clustering based on a similarity matrix

import { SeededRandom, getSeed } from '../core/random';

/**
 * Compute the degree matrix from a similarity matrix
 */
const computeDegreeMatrix = (S: number[][]): number[] => {
  const n = S.length;
  const D = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      D[i] += S[i][j];
    }
  }
  return D;
};

/**
 * Compute the normalized Laplacian matrix: L = I - D^(-1/2) * S * D^(-1/2)
 */
const computeNormalizedLaplacian = (S: number[][]): number[][] => {
  const n = S.length;
  const D = computeDegreeMatrix(S);
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        L[i][j] = D[i] > 0 ? 1 : 0;
      } else {
        const denom = Math.sqrt(D[i]) * Math.sqrt(D[j]);
        L[i][j] = denom > 0 ? -S[i][j] / denom : 0;
      }
    }
  }
  return L;
};

/**
 * Power iteration method to find the dominant eigenvector
 */
const powerIteration = (
  matrix: number[][],
  rng: SeededRandom,
  maxIter: number = 100,
  tolerance: number = 1e-6
): number[] => {
  const n = matrix.length;
  
  // Random initial vector
  let v = new Array(n);
  for (let i = 0; i < n; i++) {
    v[i] = rng.next() - 0.5;
  }
  
  // Normalize
  let norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
  v = v.map(x => x / norm);
  
  for (let iter = 0; iter < maxIter; iter++) {
    // Matrix-vector multiplication
    const newV = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        newV[i] += matrix[i][j] * v[j];
      }
    }
    
    // Normalize
    norm = Math.sqrt(newV.reduce((sum, x) => sum + x * x, 0));
    if (norm === 0) break;
    
    const normalizedV = newV.map(x => x / norm);
    
    // Check convergence
    let diff = 0;
    for (let i = 0; i < n; i++) {
      diff += Math.abs(normalizedV[i] - v[i]);
    }
    
    v = normalizedV;
    if (diff < tolerance) break;
  }
  
  return v;
};

/**
 * Deflate the matrix to find the next eigenvector
 */
const deflateMatrix = (
  matrix: number[][],
  eigenvector: number[],
  eigenvalue: number
): number[][] => {
  const n = matrix.length;
  const deflated: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      deflated[i][j] = matrix[i][j] - eigenvalue * eigenvector[i] * eigenvector[j];
    }
  }
  
  return deflated;
};

/**
 * Compute k smallest eigenvectors of the Laplacian using power iteration with deflation
 */
const computeSmallestEigenvectors = (
  L: number[][],
  k: number,
  rng: SeededRandom
): number[][] => {
  const n = L.length;
  
  // For finding smallest eigenvectors, we work with (maxEig * I - L)
  // First, estimate the maximum eigenvalue using power iteration
  let maxEig = 0;
  for (let i = 0; i < n; i++) {
    let rowSum = 0;
    for (let j = 0; j < n; j++) {
      rowSum += Math.abs(L[i][j]);
    }
    maxEig = Math.max(maxEig, rowSum);
  }
  maxEig += 1; // Safety margin
  
  // Compute shifted matrix: maxEig * I - L
  const shifted: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        shifted[i][j] = maxEig - L[i][j];
      } else {
        shifted[i][j] = -L[i][j];
      }
    }
  }
  
  const eigenvectors: number[][] = [];
  let currentMatrix = shifted;
  
  for (let i = 0; i < k && i < n; i++) {
    const eigenvector = powerIteration(currentMatrix, rng);
    eigenvectors.push(eigenvector);
    
    // Compute eigenvalue
    let numerator = 0;
    let denominator = 0;
    for (let j = 0; j < n; j++) {
      let av = 0;
      for (let l = 0; l < n; l++) {
        av += currentMatrix[j][l] * eigenvector[l];
      }
      numerator += eigenvector[j] * av;
      denominator += eigenvector[j] * eigenvector[j];
    }
    const eigenvalue = denominator > 0 ? numerator / denominator : 0;
    
    // Deflate for next iteration
    currentMatrix = deflateMatrix(currentMatrix, eigenvector, eigenvalue);
  }
  
  return eigenvectors;
};

/**
 * K-means clustering in the spectral embedding space
 */
const kmeansInEmbedding = (
  embeddings: number[][],
  k: number,
  rng: SeededRandom,
  maxIter: number = 100
): number[] => {
  const n = embeddings.length;
  if (n === 0) return [];
  
  const dim = embeddings[0].length;
  
  // Initialize centroids randomly (k-means++)
  const centroids: number[][] = [];
  const usedIndices = new Set<number>();
  
  // First centroid is random
  const firstIdx = Math.floor(rng.next() * n);
  centroids.push([...embeddings[firstIdx]]);
  usedIndices.add(firstIdx);
  
  // Remaining centroids using k-means++ initialization
  for (let c = 1; c < k && c < n; c++) {
    const distances: number[] = [];
    let totalDist = 0;
    
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      for (const centroid of centroids) {
        let dist = 0;
        for (let d = 0; d < dim; d++) {
          dist += (embeddings[i][d] - centroid[d]) ** 2;
        }
        minDist = Math.min(minDist, dist);
      }
      distances.push(minDist);
      totalDist += minDist;
    }
    
    // Sample proportional to distance squared
    let target = rng.next() * totalDist;
    let cumDist = 0;
    for (let i = 0; i < n; i++) {
      cumDist += distances[i];
      if (cumDist >= target && !usedIndices.has(i)) {
        centroids.push([...embeddings[i]]);
        usedIndices.add(i);
        break;
      }
    }
    
    // Fallback if no point was selected
    if (centroids.length <= c) {
      for (let i = 0; i < n; i++) {
        if (!usedIndices.has(i)) {
          centroids.push([...embeddings[i]]);
          usedIndices.add(i);
          break;
        }
      }
    }
  }
  
  // K-means iterations
  const assignments = new Array(n).fill(0);
  
  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;
    
    // Assign points to nearest centroid
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      let bestCluster = 0;
      
      for (let c = 0; c < centroids.length; c++) {
        let dist = 0;
        for (let d = 0; d < dim; d++) {
          dist += (embeddings[i][d] - centroids[c][d]) ** 2;
        }
        if (dist < minDist) {
          minDist = dist;
          bestCluster = c;
        }
      }
      
      if (assignments[i] !== bestCluster) {
        assignments[i] = bestCluster;
        changed = true;
      }
    }
    
    if (!changed) break;
    
    // Update centroids
    const counts = new Array(centroids.length).fill(0);
    for (let c = 0; c < centroids.length; c++) {
      for (let d = 0; d < dim; d++) {
        centroids[c][d] = 0;
      }
    }
    
    for (let i = 0; i < n; i++) {
      const c = assignments[i];
      counts[c]++;
      for (let d = 0; d < dim; d++) {
        centroids[c][d] += embeddings[i][d];
      }
    }
    
    for (let c = 0; c < centroids.length; c++) {
      if (counts[c] > 0) {
        for (let d = 0; d < dim; d++) {
          centroids[c][d] /= counts[c];
        }
      }
    }
  }
  
  return assignments;
};

/**
 * Determine optimal number of clusters using eigengap heuristic
 */
const determineOptimalK = (L: number[][], maxK: number, rng: SeededRandom): number => {
  const n = L.length;
  if (n <= 2) return n;
  
  // Compute eigenvalues of the Laplacian (approximated via power iteration)
  const eigenvalues: number[] = [];
  let currentMatrix = L;
  
  for (let i = 0; i < Math.min(maxK + 1, n); i++) {
    const eigenvector = powerIteration(currentMatrix, rng, 50);
    
    // Compute Rayleigh quotient as eigenvalue estimate
    let numerator = 0;
    let denominator = 0;
    for (let j = 0; j < n; j++) {
      let av = 0;
      for (let l = 0; l < n; l++) {
        av += L[j][l] * eigenvector[l];
      }
      numerator += eigenvector[j] * av;
      denominator += eigenvector[j] * eigenvector[j];
    }
    const eigenvalue = denominator > 0 ? numerator / denominator : 0;
    eigenvalues.push(Math.abs(eigenvalue));
    
    // Deflate
    currentMatrix = deflateMatrix(currentMatrix, eigenvector, eigenvalue);
  }
  
  // Sort eigenvalues and find largest gap
  eigenvalues.sort((a, b) => a - b);
  
  let maxGap = 0;
  let optimalK = 2;
  
  for (let i = 1; i < eigenvalues.length - 1 && i < maxK; i++) {
    const gap = eigenvalues[i + 1] - eigenvalues[i];
    if (gap > maxGap) {
      maxGap = gap;
      optimalK = i + 1;
    }
  }
  
  return Math.max(2, Math.min(optimalK, n - 1));
};

/**
 * Main spectral clustering function
 * @param S - Similarity matrix (n x n)
 * @param k - Number of clusters (optional, auto-detected if not specified)
 * @returns Array of cluster assignments for each node
 */
export const spectralClustering = (
  S: number[][],
  k?: number
): number[] => {
  const seed = getSeed();
  const rng = new SeededRandom(seed + 300000); // Isolated seed for spectral
  return spectralClusteringWithRng(S, rng, k);
};

/**
 * Spectral clustering with external RNG for reproducibility
 */
export const spectralClusteringWithRng = (
  S: number[][],
  rng: SeededRandom,
  k?: number
): number[] => {
  const n = S.length;
  if (n === 0) return [];
  if (n === 1) return [0];
  if (n === 2) return [0, 1];
  
  // Compute normalized Laplacian
  const L = computeNormalizedLaplacian(S);
  
  // Determine number of clusters
  const numClusters = k ?? determineOptimalK(L, Math.min(10, n - 1), rng);
  
  // Compute k smallest eigenvectors
  const eigenvectors = computeSmallestEigenvectors(L, numClusters, rng);
  
  // Build embedding matrix (n x k)
  const embeddings: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < eigenvectors.length; j++) {
      row.push(eigenvectors[j][i]);
    }
    
    // Normalize each row
    const norm = Math.sqrt(row.reduce((sum, x) => sum + x * x, 0));
    if (norm > 0) {
      for (let j = 0; j < row.length; j++) {
        row[j] /= norm;
      }
    }
    embeddings.push(row);
  }
  
  // Cluster in embedding space using k-means
  return kmeansInEmbedding(embeddings, numClusters, rng);
};
