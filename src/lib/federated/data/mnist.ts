// MNIST dataset loader from IDX ubyte files

export interface MNISTData {
  images: number[][];  // [numSamples x 784] normalized 0-1
  labels: number[];    // [numSamples] digit 0-9
}

// Parse IDX3 ubyte file (images)
const parseImages = (buffer: ArrayBuffer): number[][] => {
  const view = new DataView(buffer);
  
  const magic = view.getUint32(0, false);
  if (magic !== 2051) {
    throw new Error(`Invalid images file magic: ${magic}`);
  }
  
  const numImages = view.getUint32(4, false);
  const rows = view.getUint32(8, false);
  const cols = view.getUint32(12, false);
  const imageSize = rows * cols;
  
  const images: number[][] = [];
  const data = new Uint8Array(buffer, 16);
  
  for (let i = 0; i < numImages; i++) {
    const image: number[] = [];
    for (let j = 0; j < imageSize; j++) {
      image.push(data[i * imageSize + j] / 255);
    }
    images.push(image);
  }
  
  return images;
};

// Parse IDX1 ubyte file (labels)
const parseLabels = (buffer: ArrayBuffer): number[] => {
  const view = new DataView(buffer);
  
  const magic = view.getUint32(0, false);
  if (magic !== 2049) {
    throw new Error(`Invalid labels file magic: ${magic}`);
  }
  
  const numLabels = view.getUint32(4, false);
  const labels: number[] = [];
  const data = new Uint8Array(buffer, 8);
  
  for (let i = 0; i < numLabels; i++) {
    labels.push(data[i]);
  }
  
  return labels;
};

// Load MNIST from public folder
let trainDataCache: MNISTData | null = null;
let testDataCache: MNISTData | null = null;

export const loadMNISTTrain = async (): Promise<MNISTData> => {
  if (trainDataCache) return trainDataCache;
  
  const [imagesBuffer, labelsBuffer] = await Promise.all([
    fetch('/mnist/train-images.idx3-ubyte').then(r => r.arrayBuffer()),
    fetch('/mnist/train-labels.idx1-ubyte').then(r => r.arrayBuffer()),
  ]);
  
  trainDataCache = {
    images: parseImages(imagesBuffer),
    labels: parseLabels(labelsBuffer),
  };
  
  console.log(`MNIST train loaded: ${trainDataCache.images.length} samples`);
  return trainDataCache;
};

export const loadMNISTTest = async (): Promise<MNISTData> => {
  if (testDataCache) return testDataCache;
  
  const [imagesBuffer, labelsBuffer] = await Promise.all([
    fetch('/mnist/t10k-images.idx3-ubyte').then(r => r.arrayBuffer()),
    fetch('/mnist/t10k-labels.idx1-ubyte').then(r => r.arrayBuffer()),
  ]);
  
  testDataCache = {
    images: parseImages(imagesBuffer),
    labels: parseLabels(labelsBuffer),
  };
  
  console.log(`MNIST test loaded: ${testDataCache.images.length} samples`);
  return testDataCache;
};

// One-hot encode labels
export const oneHot = (label: number, numClasses: number = 10): number[] => {
  const encoded = new Array(numClasses).fill(0);
  encoded[label] = 1;
  return encoded;
};

// Map to store assigned primary label per client pair so assignments are stable
const pairLabelMap: Map<number, number> = new Map();

// Update pairLabelMap when transferring data from one client group to another
export const updatePairLabelMapForTransfer = (
  fromGroupIndex: number,
  toGroupIndex: number
): void => {
  const sourceLabel = pairLabelMap.get(fromGroupIndex);
  if (sourceLabel !== undefined) {
    pairLabelMap.set(toGroupIndex, sourceLabel);
    console.log(`[pairLabelMap] Updated: group ${toGroupIndex} now maps to label ${sourceLabel} (from group ${fromGroupIndex})`);
  }
};

// Reset pairLabelMap for fresh experiment
export const resetPairLabelMap = (): void => {
  pairLabelMap.clear();
  console.log('[pairLabelMap] Cleared for new experiment');
};

// Get the group index for a client (extracted to avoid duplication)
export const getClientGroupIndex = (
  clientId: string,
  distributionMode: 'pairs' | 'groups' = 'groups'
): number => {
  let clientIndex = 0;
  const m = clientId.match(/client-(\d+)/);
  if (m) clientIndex = Number(m[1]);
  else clientIndex = clientId.split('').reduce((a, c) => a * 31 + c.charCodeAt(0), 7) & 0xffffffff;

  if (distributionMode === 'groups') {
    if (clientIndex <= 2) return 0;
    else if (clientIndex <= 5) return 1;
    else return 2;
  } else {
    return Math.floor(clientIndex / 2);
  }
};

// Sample from a Dirichlet distribution using Gamma samples
const sampleDirichlet = (alpha: number[], seed: number): number[] => {
  // Marsaglia and Tsang's method for Gamma(alpha, 1)
  let s = seed >>> 0 || 1;
  const rnd = () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
  // Box-Muller for normal
  const randn = () => {
    const u1 = rnd();
    const u2 = rnd();
    return Math.sqrt(-2 * Math.log(u1 + 1e-30)) * Math.cos(2 * Math.PI * u2);
  };

  const sampleGamma = (a: number): number => {
    if (a < 1) {
      // Gamma(a) = Gamma(a+1) * U^(1/a)
      return sampleGamma(a + 1) * Math.pow(rnd() + 1e-30, 1 / a);
    }
    const d = a - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      let x: number, v: number;
      do {
        x = randn();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = rnd();
      if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
      if (Math.log(u + 1e-30) < 0.5 * x * x + d * (1 - v + Math.log(v + 1e-30))) return d * v;
    }
  };

  const samples = alpha.map(a => sampleGamma(a));
  const sum = samples.reduce((a, b) => a + b, 0);
  return samples.map(x => x / sum);
};

// Get a random subset of MNIST for a client (non-IID simulation)
export const getClientDataSubset = (
  data: MNISTData,
  clientId: string,
  numSamples: number,
  nonIID: boolean = true,
  seed: number = 42,
  distributionMode: 'pairs' | 'groups' = 'groups',
  dataType: 'train' | 'test' = 'train',
  distributionType: '70-30' | 'dirichlet' = '70-30',
  dirichletAlpha: number = 0.5
): { inputs: number[][]; outputs: number[][] } => {
  const inputs: number[][] = [];
  const outputs: number[][] = [];

  // Parse client index
  let clientIndex = 0;
  const m = clientId.match(/client-(\d+)/);
  if (m) clientIndex = Number(m[1]);
  else clientIndex = clientId.split('').reduce((a, c) => a * 31 + c.charCodeAt(0), 7) & 0xffffffff;

  const seededShuffle = <T,>(arr: T[], seed: number) => {
    const a = arr.slice();
    let s = seed >>> 0;
    const rnd = () => {
      s ^= s << 13;
      s ^= s >>> 17;
      s ^= s << 5;
      return (s >>> 0) / 4294967295;
    };
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const tmp = a[i]; a[i] = a[j]; a[j] = tmp;
    }
    return a;
  };

  if (!nonIID) {
    // IID: uniform random sampling
    const indices = Array.from({ length: data.labels.length }, (_, i) => i);
    const shuffled = seededShuffle(indices, seed);
    for (let i = 0; i < Math.min(numSamples, shuffled.length); i++) {
      const idx = shuffled[i];
      inputs.push(data.images[idx]);
      outputs.push(oneHot(data.labels[idx]));
    }
    return { inputs, outputs };
  }

  if (distributionType === 'dirichlet') {
    // Dirichlet non-IID distribution
    const numClasses = 10;
    // Each client gets a different Dirichlet sample using client-specific seed
    const alphaVec = new Array(numClasses).fill(dirichletAlpha);
    const proportions = sampleDirichlet(alphaVec, seed * 1000 + clientIndex + 1);

    // Build per-class index pools
    const classIndices: number[][] = Array.from({ length: numClasses }, () => []);
    for (let i = 0; i < data.labels.length; i++) {
      classIndices[data.labels[i]].push(i);
    }
    // Shuffle each class pool
    for (let c = 0; c < numClasses; c++) {
      classIndices[c] = seededShuffle(classIndices[c], seed + clientIndex + c);
    }

    // Determine how many samples per class
    const samplesPerClass = proportions.map(p => Math.max(1, Math.round(p * numSamples)));
    // Adjust total to match numSamples
    let total = samplesPerClass.reduce((a, b) => a + b, 0);
    while (total > numSamples) {
      const maxIdx = samplesPerClass.indexOf(Math.max(...samplesPerClass));
      samplesPerClass[maxIdx]--;
      total--;
    }
    while (total < numSamples) {
      const minIdx = samplesPerClass.indexOf(Math.min(...samplesPerClass));
      samplesPerClass[minIdx]++;
      total++;
    }

    // Collect samples
    const classPointers = new Array(numClasses).fill(0);
    for (let c = 0; c < numClasses; c++) {
      for (let i = 0; i < samplesPerClass[c]; i++) {
        const idx = classIndices[c][classPointers[c] % classIndices[c].length];
        classPointers[c]++;
        inputs.push(data.images[idx]);
        outputs.push(oneHot(data.labels[idx]));
      }
    }

    // Log distribution
    if (dataType === 'train') {
      const labelCounts: Record<number, number> = {};
      for (const output of outputs) {
        const label = output.indexOf(1);
        labelCounts[label] = (labelCounts[label] || 0) + 1;
      }
      const sortedLabels = Object.entries(labelCounts)
        .map(([label, count]) => ({ label: parseInt(label), count, percentage: (count / outputs.length) * 100 }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 4);
      const dist = sortedLabels.map(({ label, percentage }) => `${label}:${percentage.toFixed(1)}%`).join(', ');
      console.log(`${clientId} [dirichlet α=${dirichletAlpha}] {${dist}}`);
    }

    return { inputs, outputs };
  }

  // Original 70-30 non-IID distribution
  let groupIndex: number;
  if (distributionMode === 'groups') {
    if (clientIndex <= 2) groupIndex = 0;
    else if (clientIndex <= 5) groupIndex = 1;
    else groupIndex = 2;
  } else {
    groupIndex = Math.floor(clientIndex / 2);
  }

  const pairIndex = groupIndex;

  let primaryLabel: number;
  if (pairLabelMap.has(pairIndex)) {
    primaryLabel = pairLabelMap.get(pairIndex)!;
  } else {
    const used = new Set(Array.from(pairLabelMap.values()));
    let found = -1;
    for (let l = 0; l < 10; l++) {
      if (!used.has(l)) { found = l; break; }
    }
    if (found === -1) {
      found = pairIndex % 10;
      console.warn(`All labels already assigned to pairs; reusing label ${found} for pair ${pairIndex}`);
    }
    pairLabelMap.set(pairIndex, found);
    primaryLabel = found;
  }

  const primaryCount = Math.floor(numSamples * 0.7);
  const randomCount = numSamples - primaryCount;

  const primaryIndices = data.labels
    .map((label, idx) => label === primaryLabel ? idx : -1)
    .filter(idx => idx !== -1);

  const shuffledPrimary = seededShuffle(primaryIndices, clientIndex + 1 + seed);
  for (let i = 0; i < Math.min(primaryCount, shuffledPrimary.length); i++) {
    const idx = shuffledPrimary[i];
    inputs.push(data.images[idx]);
    outputs.push(oneHot(data.labels[idx]));
  }

  const allIndices = Array.from({ length: data.labels.length }, (_, i) => i);
  const shuffledAll = seededShuffle(allIndices, clientIndex + 12345 + seed);
  for (let i = 0; i < shuffledAll.length && inputs.length < numSamples; i++) {
    const idx = shuffledAll[i];
    inputs.push(data.images[idx]);
    outputs.push(oneHot(data.labels[idx]));
  }

  // Log distribution for this client
  if (dataType === 'train') {
    const labelCounts: Record<number, number> = {};
    for (const output of outputs) {
      const label = output.indexOf(1);
      labelCounts[label] = (labelCounts[label] || 0) + 1;
    }
    const sortedLabels = Object.entries(labelCounts)
      .map(([label, count]) => ({ label: parseInt(label), count, percentage: (count / outputs.length) * 100 }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 4);
    const dist = sortedLabels.map(({ label, percentage }) => `${label}:${percentage.toFixed(1)}%`).join(', ');
    console.log(`${clientId} [group ${pairIndex}→label ${primaryLabel}] {${dist}}`);
  }

  return { inputs, outputs };
};
