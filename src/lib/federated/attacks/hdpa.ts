// HDPA - Hyperdimensional Poisoning Attack
// Encodes images into hyperdimensional space, generates adversarial examples
// by shifting them toward a target class, then decodes back to image space.

import { SeededRandom } from '../core/random';

export interface HDPAConfig {
  /** Hypervector dimension (default 10000) */
  dimension: number;
  /** Attack strength λ for shifting toward target class (default 0.3) */
  attackStrength: number;
  /** Ratio of local data to poison (default 0.2) */
  poisonRatio: number;
  /** Target class for targeted attack, or -1 for untargeted (random) */
  targetClass: number;
  /** 'targeted' or 'untargeted' */
  attackType: 'targeted' | 'untargeted';
}

export const DEFAULT_HDPA_CONFIG: HDPAConfig = {
  dimension: 10000,
  attackStrength: 0.3,
  poisonRatio: 0.2,
  targetClass: -1,
  attackType: 'untargeted',
};

// Cached projection matrix per image size (shared across clients for consistency)
const projectionMatrixCache: Map<string, Float32Array> = new Map();

/**
 * Get or create a random projection matrix P ∈ R^(dimension × imageSize).
 * Stored as a flat Float32Array for performance.
 */
const getProjectionMatrix = (dimension: number, imageSize: number, seed: number = 42): Float32Array => {
  const key = `${dimension}_${imageSize}_${seed}`;
  let P = projectionMatrixCache.get(key);
  if (P) return P;

  const rng = new SeededRandom(seed);
  P = new Float32Array(dimension * imageSize);
  // Gaussian-like via Box-Muller (approximated with uniform pairs)
  for (let i = 0; i < P.length; i += 2) {
    const u1 = Math.max(1e-10, rng.next());
    const u2 = rng.next();
    const mag = Math.sqrt(-2 * Math.log(u1));
    P[i] = mag * Math.cos(2 * Math.PI * u2);
    if (i + 1 < P.length) {
      P[i + 1] = mag * Math.sin(2 * Math.PI * u2);
    }
  }
  // Scale by 1/sqrt(dimension) for numerical stability
  const scale = 1 / Math.sqrt(dimension);
  for (let i = 0; i < P.length; i++) P[i] *= scale;

  projectionMatrixCache.set(key, P);
  return P;
};

/**
 * Encode an image (flat array) into a bipolar hypervector hv ∈ {-1, +1}^dimension
 */
const encodeToHypervector = (
  image: number[],
  P: Float32Array,
  dimension: number
): Float32Array => {
  const imageSize = image.length;
  const hv = new Float32Array(dimension);
  for (let d = 0; d < dimension; d++) {
    let sum = 0;
    const offset = d * imageSize;
    for (let j = 0; j < imageSize; j++) {
      sum += P[offset + j] * image[j];
    }
    hv[d] = sum >= 0 ? 1 : -1;
  }
  return hv;
};

/**
 * Build class prototype hypervectors by summing & sign-normalizing
 * all images of each class.
 */
const buildClassHypervectors = (
  images: number[][],
  labels: number[][],   // one-hot
  P: Float32Array,
  dimension: number,
  numClasses: number
): Map<number, Float32Array> => {
  const classHVSums = new Map<number, Float32Array>();
  for (let c = 0; c < numClasses; c++) {
    classHVSums.set(c, new Float32Array(dimension));
  }

  for (let i = 0; i < images.length; i++) {
    const label = labels[i].indexOf(1); // one-hot → class index
    if (label < 0) continue;
    const hv = encodeToHypervector(images[i], P, dimension);
    const sum = classHVSums.get(label)!;
    for (let d = 0; d < dimension; d++) sum[d] += hv[d];
  }

  // Sign-normalise
  const classHVs = new Map<number, Float32Array>();
  for (let c = 0; c < numClasses; c++) {
    const sum = classHVSums.get(c)!;
    const hv = new Float32Array(dimension);
    for (let d = 0; d < dimension; d++) hv[d] = sum[d] >= 0 ? 1 : -1;
    classHVs.set(c, hv);
  }
  return classHVs;
};

/**
 * Generate an adversarial hypervector by shifting hv_x toward hv_target.
 *   hv_adv = sign(hv_x + λ * hv_target)
 */
const generateAdversarialHV = (
  hvX: Float32Array,
  hvTarget: Float32Array,
  attackStrength: number,
  dimension: number
): Float32Array => {
  const hv = new Float32Array(dimension);
  for (let d = 0; d < dimension; d++) {
    const val = hvX[d] + attackStrength * hvTarget[d];
    hv[d] = val >= 0 ? 1 : -1;
  }
  return hv;
};

/**
 * Compute pseudo-inverse decode: x_adv ≈ Pᵀ @ hv_adv  (since P is ~orthonormal scaled)
 * Then clip to [0, 1].
 */
const decodeHypervector = (
  hv: Float32Array,
  P: Float32Array,
  dimension: number,
  imageSize: number
): number[] => {
  const x = new Array<number>(imageSize).fill(0);
  for (let j = 0; j < imageSize; j++) {
    let sum = 0;
    for (let d = 0; d < dimension; d++) {
      sum += P[d * imageSize + j] * hv[d];
    }
    x[j] = sum;
  }
  // Normalise to [0,1]
  let min = Infinity, max = -Infinity;
  for (let j = 0; j < imageSize; j++) {
    if (x[j] < min) min = x[j];
    if (x[j] > max) max = x[j];
  }
  const range = max - min || 1;
  for (let j = 0; j < imageSize; j++) {
    x[j] = Math.max(0, Math.min(1, (x[j] - min) / range));
  }
  return x;
};

/**
 * Apply HDPA to a client's training data.
 * Returns a new dataset (inputs + outputs) with poisoned samples mixed in.
 */
export const applyHDPAToDataset = (
  inputs: number[][],
  outputs: number[][],
  config: HDPAConfig,
  clientRng: SeededRandom
): { inputs: number[][]; outputs: number[][] } => {
  const numClasses = outputs[0]?.length || 10;
  const imageSize = inputs[0]?.length || 784;
  const dimension = config.dimension;

  // Get projection matrix
  const P = getProjectionMatrix(dimension, imageSize, 7777);

  // Build class hypervectors from the client's own data
  const classHVs = buildClassHypervectors(inputs, outputs, P, dimension, numClasses);

  // Determine how many samples to poison
  const numPoison = Math.max(1, Math.floor(inputs.length * config.poisonRatio));

  // Select indices to poison
  const indices = Array.from({ length: inputs.length }, (_, i) => i);
  clientRng.shuffle(indices);
  const poisonIndices = new Set(indices.slice(0, numPoison));

  // Build poisoned dataset
  const newInputs = [...inputs];
  const newOutputs = [...outputs];

  for (const idx of poisonIndices) {
    const originalLabel = outputs[idx].indexOf(1);
    if (originalLabel < 0) continue;

    // Choose target class
    let targetClass: number;
    if (config.attackType === 'targeted' && config.targetClass >= 0 && config.targetClass < numClasses) {
      targetClass = config.targetClass;
    } else {
      // Untargeted: random different class
      do {
        targetClass = clientRng.nextInt(numClasses);
      } while (targetClass === originalLabel);
    }

    const hvTarget = classHVs.get(targetClass);
    if (!hvTarget) continue;

    // Encode → shift → decode
    const hvX = encodeToHypervector(inputs[idx], P, dimension);
    const hvAdv = generateAdversarialHV(hvX, hvTarget, config.attackStrength, dimension);
    const advImage = decodeHypervector(hvAdv, P, dimension, imageSize);

    // Replace with adversarial image + target label
    newInputs[idx] = advImage;
    const targetOneHot = new Array(numClasses).fill(0);
    targetOneHot[targetClass] = 1;
    newOutputs[idx] = targetOneHot;
  }

  return { inputs: newInputs, outputs: newOutputs };
};

/**
 * Reset projection matrix cache (call on experiment reset)
 */
export const resetHDPACache = (): void => {
  projectionMatrixCache.clear();
};
