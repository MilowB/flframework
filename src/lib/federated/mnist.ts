// MNIST dataset loader from IDX ubyte files

export interface MNISTData {
  images: number[][];  // [numSamples x 784] normalized 0-1
  labels: number[];    // [numSamples] digit 0-9
}

// Parse IDX3 ubyte file (images)
const parseImages = (buffer: ArrayBuffer): number[][] => {
  const view = new DataView(buffer);
  
  // Check magic number
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
      // Normalize to [0, 1]
      image.push(data[i * imageSize + j] / 255);
    }
    images.push(image);
  }
  
  return images;
};

// Parse IDX1 ubyte file (labels)
const parseLabels = (buffer: ArrayBuffer): number[] => {
  const view = new DataView(buffer);
  
  // Check magic number
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
  nonIID: boolean = true
): { inputs: number[][]; outputs: number[][] } => {
  const inputs: number[][] = [];
  const outputs: number[][] = [];
  
  if (nonIID) {
    // Non-IID: new rule — each client has 70% of its train data of ONE label
    // that it shares with exactly one other client (paired clients).
    // Determine numeric client index (clients are named `client-N` in simulation).
    let clientIndex = 0;
    const m = clientId.match(/client-(\d+)/);
    if (m) clientIndex = Number(m[1]);
    else clientIndex = clientId.split('').reduce((a, c) => a * 31 + c.charCodeAt(0), 7) & 0xffffffff;

    // Pairing: partners are (0,1), (2,3), ... partner = clientIndex ^ 1
    const pairIndex = Math.floor(clientIndex / 2);

    // Ensure each pair gets a primary label. Try to assign labels 0..9 uniquely to pairs
    // until labels are exhausted. If there are more than 10 pairs, labels will be reused
    // (unavoidable given 10 digits); we keep assignments stable in `pairLabelMap`.
    let primaryLabel: number;
    if (pairLabelMap.has(pairIndex)) {
      primaryLabel = pairLabelMap.get(pairIndex)!;
    } else {
      // find an unused label
      const used = new Set(Array.from(pairLabelMap.values()));
      let found = -1;
      for (let l = 0; l < 10; l++) {
        if (!used.has(l)) { found = l; break; }
      }
      if (found === -1) {
        // all labels used; fallback to deterministic choice
        found = pairIndex % 10;
        console.warn(`All labels already assigned to pairs; reusing label ${found} for pair ${pairIndex}`);
      }
      pairLabelMap.set(pairIndex, found);
      primaryLabel = found;
    }

    const primaryDigits = [primaryLabel];

    // 70% from primary label, 30% random
    const primaryCount = Math.floor(numSamples * 0.7);
    const randomCount = numSamples - primaryCount;

    // deterministic pseudo-random shuffle using clientIndex as seed
    const seededShuffle = <T,>(arr: T[], seed: number) => {
      const a = arr.slice();
      let s = seed >>> 0;
      const rnd = () => {
        // xorshift32
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

    // Find indices for primary label
    const primaryIndices = data.labels
      .map((label, idx) => label === primaryLabel ? idx : -1)
      .filter(idx => idx !== -1);

    const shuffledPrimary = seededShuffle(primaryIndices, clientIndex + 1);
    for (let i = 0; i < Math.min(primaryCount, shuffledPrimary.length); i++) {
      const idx = shuffledPrimary[i];
      inputs.push(data.images[idx]);
      outputs.push(oneHot(data.labels[idx]));
    }

    // Random samples (avoid taking too many from primaryIndices again)
    const allIndices = Array.from({ length: data.labels.length }, (_, i) => i);
    const shuffledAll = seededShuffle(allIndices, clientIndex + 12345);
    for (let i = 0; i < shuffledAll.length && inputs.length < numSamples; i++) {
      const idx = shuffledAll[i];
      // allow repetition of primary label in the random tail as well, it's fine
      inputs.push(data.images[idx]);
      outputs.push(oneHot(data.labels[idx]));
    }
  } else {
    // IID: uniform random sampling
    const indices = Array.from({ length: data.labels.length }, (_, i) => i)
      .sort(() => Math.random() - 0.5);
    
    for (let i = 0; i < Math.min(numSamples, indices.length); i++) {
      const idx = indices[i];
      inputs.push(data.images[idx]);
      outputs.push(oneHot(data.labels[idx]));
    }
  }
  
  return { inputs, outputs };
};
