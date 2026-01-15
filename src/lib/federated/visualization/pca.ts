// PCA-based 3D projection for model visualization
import { ModelWeights } from '../core/types';

/**
 * Flatten a ModelWeights object into a 1D array
 */
export function flattenModelWeights(model: ModelWeights): number[] {
  const flat: number[] = [];
  for (const layer of model.layers) {
    flat.push(...layer);
  }
  flat.push(...model.bias);
  return flat;
}

/**
 * Compute PCA projection vectors from a set of model vectors
 * Uses power iteration method for simplicity
 */
function computePCAProjection(vectors: number[][], numComponents: number = 3): number[][] {
  if (vectors.length === 0) return [];
  
  const dim = vectors[0].length;
  const n = vectors.length;
  
  // Compute mean
  const mean = new Array(dim).fill(0);
  for (const vec of vectors) {
    for (let i = 0; i < dim; i++) {
      mean[i] += vec[i] / n;
    }
  }
  
  // Center data
  const centered = vectors.map(vec => vec.map((v, i) => v - mean[i]));
  
  // Compute covariance matrix (approximated for high dimensions)
  // Use random projection for very high dimensional data
  const projectionVectors: number[][] = [];
  
  for (let comp = 0; comp < numComponents; comp++) {
    // Initialize random vector
    let v = new Array(dim).fill(0).map(() => Math.random() - 0.5);
    
    // Normalize
    let norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
    v = v.map(x => x / (norm || 1));
    
    // Power iteration (simplified)
    for (let iter = 0; iter < 10; iter++) {
      // Compute X^T * X * v
      const scores = centered.map(row => row.reduce((sum, x, i) => sum + x * v[i], 0));
      const newV = new Array(dim).fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < dim; j++) {
          newV[j] += centered[i][j] * scores[i];
        }
      }
      
      // Gram-Schmidt orthogonalization against previous components
      for (const prev of projectionVectors) {
        const dot = newV.reduce((sum, x, i) => sum + x * prev[i], 0);
        for (let i = 0; i < dim; i++) {
          newV[i] -= dot * prev[i];
        }
      }
      
      // Normalize
      norm = Math.sqrt(newV.reduce((sum, x) => sum + x * x, 0));
      v = newV.map(x => x / (norm || 1));
    }
    
    projectionVectors.push(v);
  }
  
  return projectionVectors;
}

/**
 * Project a single vector using pre-computed PCA vectors
 */
function projectVector(vec: number[], mean: number[], projVectors: number[][]): [number, number, number] {
  const centered = vec.map((v, i) => v - mean[i]);
  const coords: [number, number, number] = [0, 0, 0];
  for (let d = 0; d < 3; d++) {
    coords[d] = centered.reduce((sum, v, i) => sum + v * projVectors[d][i], 0);
  }
  return coords;
}

export interface Model3DPosition {
  id: string;
  name: string;
  type: 'client' | 'cluster' | 'global';
  position: [number, number, number];
  color: string;
}

export interface RoundSnapshot3D {
  round: number;
  phase: 'received' | 'aggregated' | 'before_send';
  positions: Model3DPosition[];
}

// Color palette for visualization
const CLIENT_COLORS = [
  '#8b5cf6', // purple
  '#06b6d4', // cyan
  '#22c55e', // green
  '#f59e0b', // orange
  '#ef4444', // red
  '#3b82f6', // blue
  '#ec4899', // pink
  '#eab308', // yellow
];

const CLUSTER_COLORS = [
  '#a855f7', // violet
  '#14b8a6', // teal
  '#84cc16', // lime
  '#fb923c', // orange-light
];

const GLOBAL_COLOR = '#ffffff';

/**
 * Compute 3D positions for all models at a specific point
 */
export function computeModelPositions(
  clientModels: { id: string; name: string; weights: ModelWeights }[],
  clusterModels: { id: string; weights: ModelWeights }[],
  globalModel: ModelWeights | null,
  phase: 'received' | 'aggregated' | 'before_send'
): Model3DPosition[] {
  // Collect all model vectors
  const allVectors: number[][] = [];
  const allMeta: { id: string; name: string; type: 'client' | 'cluster' | 'global' }[] = [];
  
  for (let i = 0; i < clientModels.length; i++) {
    allVectors.push(flattenModelWeights(clientModels[i].weights));
    allMeta.push({ 
      id: clientModels[i].id, 
      name: clientModels[i].name, 
      type: 'client' 
    });
  }
  
  for (let i = 0; i < clusterModels.length; i++) {
    allVectors.push(flattenModelWeights(clusterModels[i].weights));
    allMeta.push({ 
      id: clusterModels[i].id, 
      name: `Cluster ${i + 1}`, 
      type: 'cluster' 
    });
  }
  
  if (globalModel) {
    allVectors.push(flattenModelWeights(globalModel));
    allMeta.push({ id: 'global', name: 'Global', type: 'global' });
  }
  
  if (allVectors.length === 0) return [];
  
  // Compute PCA
  const dim = allVectors[0].length;
  const mean = new Array(dim).fill(0);
  for (const vec of allVectors) {
    for (let i = 0; i < dim; i++) {
      mean[i] += vec[i] / allVectors.length;
    }
  }
  
  const projVectors = computePCAProjection(allVectors, 3);
  if (projVectors.length < 3) {
    // Fallback if PCA fails
    return allMeta.map((meta, i) => ({
      ...meta,
      position: [i * 2, 0, 0] as [number, number, number],
      color: meta.type === 'global' ? GLOBAL_COLOR : 
             meta.type === 'cluster' ? CLUSTER_COLORS[i % CLUSTER_COLORS.length] :
             CLIENT_COLORS[i % CLIENT_COLORS.length]
    }));
  }
  
  // Project all vectors
  const positions = allVectors.map((vec, i) => ({
    ...allMeta[i],
    position: projectVector(vec, mean, projVectors),
    color: allMeta[i].type === 'global' ? GLOBAL_COLOR :
           allMeta[i].type === 'cluster' ? CLUSTER_COLORS[parseInt(allMeta[i].id.replace('cluster-', '')) % CLUSTER_COLORS.length] :
           CLIENT_COLORS[i % CLIENT_COLORS.length]
  }));
  
  // Normalize positions to reasonable range
  const allCoords = positions.flatMap(p => p.position);
  const maxAbs = Math.max(...allCoords.map(Math.abs), 1);
  const scale = 5 / maxAbs;
  
  return positions.map(p => ({
    ...p,
    position: p.position.map(c => c * scale) as [number, number, number]
  }));
}
