// Seeded Random Number Generator
// Xorshift32 PRNG for deterministic randomness

export class SeededRandom {
  private state: number;
  
  constructor(seed: number) {
    this.state = seed >>> 0 || 1;
  }
  
  // Returns a number in [0, 1)
  next(): number {
    this.state ^= this.state << 13;
    this.state ^= this.state >>> 17;
    this.state ^= this.state << 5;
    return (this.state >>> 0) / 4294967296;
  }
  
  // Returns an integer in [0, max)
  nextInt(max: number): number {
    return Math.floor(this.next() * max);
  }
  
  // Shuffle an array in place
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.nextInt(i + 1);
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
}

// Global RNG instance, reset when simulation starts
let globalRng: SeededRandom = new SeededRandom(42);
let currentSeed: number = 42;

export const setSeed = (seed: number): void => {
  currentSeed = seed;
  globalRng = new SeededRandom(seed);
};

export const getRng = (): SeededRandom => globalRng;
export const getSeed = (): number => currentSeed;

// Simple seeded PRNG (Mulberry32) - used for PCA projections
export function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return function() {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
