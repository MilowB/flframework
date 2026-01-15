declare module 'gif.js' {
  interface GIFOptions {
    workers?: number;
    quality?: number;
    width?: number;
    height?: number;
    workerScript?: string;
    background?: string;
    transparent?: string;
  }

  interface GIFFrameOptions {
    delay?: number;
    copy?: boolean;
  }

  export default class GIF {
    constructor(options: GIFOptions);
    addFrame(canvas: HTMLCanvasElement | CanvasImageSource, options?: GIFFrameOptions): void;
    on(event: 'finished', callback: (blob: Blob) => void): void;
    on(event: 'progress', callback: (progress: number) => void): void;
    render(): void;
    abort(): void;
  }
}
