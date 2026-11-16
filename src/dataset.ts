import fs from "node:fs";
import path from "node:path";
import { isSupportedImageFile, loadImageAsTensor } from "./image-loader";
import { oneHot } from "./softmax-cross-entropy";
import { Tensor } from "./tensor";

export type SplitName = "train" | "validation" | "test";

export interface Sample {
  path: string;
  label: number;
  className: string;
}

export interface DatasetSplit {
  name: SplitName;
  samples: Sample[];
}

export class ImageDataset {
  root: string;
  imageSize: number;
  classes: string[];
  private cache = new Map<string, Tensor>();

  constructor(root: string, imageSize = 32, classes?: string[]) {
    this.root = root;
    this.imageSize = imageSize;
    this.classes = classes ?? ImageDataset.discoverClasses(root);
  }

  static discoverClasses(root: string): string[] {
    const dir = path.join(root, "train");
    if (!fs.existsSync(dir)) return [];
    return fs
      .readdirSync(dir, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => d.name)
      .sort();
  }

  list(split: SplitName): DatasetSplit {
    const samples: Sample[] = [];
    this.classes.forEach((className, label) => {
      const dir = path.join(this.root, split, className);
      if (!fs.existsSync(dir)) return;
      for (const file of fs.readdirSync(dir).sort()) {
        if (!isSupportedImageFile(file)) continue;
        samples.push({ path: path.join(dir, file), label, className });
      }
    });
    return { name: split, samples };
  }

  async loadSample(
    sample: Sample,
    flip = false,
  ): Promise<{ image: Tensor; label: Tensor }> {
    let image = this.cache.get(sample.path);
    if (!image) {
      image = await loadImageAsTensor(sample.path, this.imageSize);
      this.cache.set(sample.path, image);
    }
    return {
      image: flip ? flipHorizontal(image) : image,
      label: oneHot(sample.label, this.classes.length),
    };
  }

  static shuffled(samples: Sample[], seed = 123): Sample[] {
    const arr = [...samples];
    let state = seed;
    for (let i = arr.length - 1; i > 0; i--) {
      state = (state * 1664525 + 1013904223) % 4294967296;
      const j = state % (i + 1);
      [arr[i], arr[j]] = [arr[j]!, arr[i]!];
    }
    return arr;
  }

  describe(): string {
    const lines = ["Dataset splits", "--------------"];
    const header = "split".padEnd(12) + this.classes.map((c) => c.padStart(12)).join("") + "total".padStart(8);
    lines.push(header);
    for (const split of ["train", "validation", "test"] as SplitName[]) {
      const s = this.list(split);
      const counts = this.classes.map(
        (c) => String(s.samples.filter((x) => x.className === c).length).padStart(12),
      );
      lines.push(split.padEnd(12) + counts.join("") + String(s.samples.length).padStart(8));
    }
    return lines.join("\n");
  }
}

export function flipHorizontal(image: Tensor): Tensor {
  const [c, h, w] = image.shape as [number, number, number];
  const out = new Float64Array(image.data.length);
  for (let ch = 0; ch < c; ch++) {
    for (let y = 0; y < h; y++) {
      const row = ch * h * w + y * w;
      for (let x = 0; x < w; x++) {
        out[row + x] = image.data[row + (w - 1 - x)]!;
      }
    }
  }
  return new Tensor(out, [c, h, w]);
}
