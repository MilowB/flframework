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
    // Non-IID: each client specializes in 2-3 digits
    const hash = clientId.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
    const primaryDigits = [hash % 10, (hash + 3) % 10, (hash + 7) % 10];
    
    // 70% from primary digits, 30% random
    const primaryCount = Math.floor(numSamples * 0.7);
    const randomCount = numSamples - primaryCount;
    
    // Find indices for primary digits
    const primaryIndices = data.labels
      .map((label, idx) => primaryDigits.includes(label) ? idx : -1)
      .filter(idx => idx !== -1);
    
    // Shuffle and take
    const shuffledPrimary = primaryIndices.sort(() => Math.random() - 0.5);
    for (let i = 0; i < Math.min(primaryCount, shuffledPrimary.length); i++) {
      const idx = shuffledPrimary[i];
      inputs.push(data.images[idx]);
      outputs.push(oneHot(data.labels[idx]));
    }
    
    // Random samples
    const randomIndices = Array.from({ length: data.labels.length }, (_, i) => i)
      .sort(() => Math.random() - 0.5);
    for (let i = 0; i < randomCount && inputs.length < numSamples; i++) {
      const idx = randomIndices[i];
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
