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

// Get a random subset of MNIST for a client (non-IID simulation)
export const getClientDataSubset = (
  data: MNISTData,
  clientId: string,
  numSamples: number,
  nonIID: boolean = true,
  seed: number = 42,
  distributionMode: 'pairs' | 'groups' = 'pairs'
): { inputs: number[][]; outputs: number[][] } => {
  const inputs: number[][] = [];
  const outputs: number[][] = [];
  
  if (nonIID) {
    // Non-IID: each client has 70% of its train data of ONE label
    let clientIndex = 0;
    const m = clientId.match(/client-(\d+)/);
    if (m) clientIndex = Number(m[1]);
    else clientIndex = clientId.split('').reduce((a, c) => a * 31 + c.charCodeAt(0), 7) & 0xffffffff;

    let groupIndex: number;
    if (distributionMode === 'groups') {
      // Distribution par groupes: (0,1,2), (3,4,5), (6,7,8,9)
      if (clientIndex <= 2) groupIndex = 0;
      else if (clientIndex <= 5) groupIndex = 1;
      else groupIndex = 2;
    } else {
      // Distribution par paires (comportement par défaut)
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
  } else {
    // IID: uniform random sampling
    const indices = Array.from({ length: data.labels.length }, (_, i) => i);
    const seededShuffleIID = <T,>(arr: T[], seed: number) => {
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
    const shuffledIID = seededShuffleIID(indices, seed);
    
    for (let i = 0; i < Math.min(numSamples, shuffledIID.length); i++) {
      const idx = shuffledIID[i];
      inputs.push(data.images[idx]);
      outputs.push(oneHot(data.labels[idx]));
    }
  }
  
  return { inputs, outputs };
};
