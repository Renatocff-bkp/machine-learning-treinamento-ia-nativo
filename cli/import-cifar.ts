import fs from "node:fs";
import path from "node:path";
import ps from "node:process";
import { PNG } from "pngjs";

type Source = "cifar10" | "cifar100";

interface ClassMapping {
  folder: string;
  source: Source;
  labels: string[];
}

const MAPPINGS: ClassMapping[] = [
  { folder: "cats", source: "cifar10", labels: ["cat"] },
  { folder: "dogs", source: "cifar10", labels: ["dog"] },
  { folder: "cars", source: "cifar10", labels: ["automobile"] },
  { folder: "trucks", source: "cifar10", labels: ["truck"] },
  { folder: "buses", source: "cifar100", labels: ["bus"] },
  { folder: "motorcycles", source: "cifar100", labels: ["motorcycle"] },
  { folder: "people", source: "cifar100", labels: ["baby", "boy", "girl", "man", "woman"] },
];

const IMAGE_BYTES = 32 * 32 * 3;
const PREFIX = "cifar_";

interface RawImage {
  label: number;
  pixels: Buffer;
}

const root = ps.cwd();
const rawDir = path.join(root, "data-raw");
const datasetDir = path.join(root, "dataset");

const [trainCount, valCount, testCount] = [
  Number(ps.argv[2] ?? 400),
  Number(ps.argv[3] ?? 100),
  Number(ps.argv[4] ?? 100),
];

function readNames(file: string): string[] {
  return fs
    .readFileSync(file, "utf-8")
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
}

function readRecords(files: string[], headerBytes: number, labelOffset: number): RawImage[] {
  const out: RawImage[] = [];
  const recordSize = headerBytes + IMAGE_BYTES;
  for (const file of files) {
    const bytes = fs.readFileSync(file);
    for (let pos = 0; pos + recordSize <= bytes.length; pos += recordSize) {
      out.push({
        label: bytes[pos + labelOffset]!,
        pixels: bytes.subarray(pos + headerBytes, pos + recordSize),
      });
    }
  }
  return out;
}

function loadCifar10() {
  const dir = path.join(rawDir, "cifar-10-batches-bin");
  const names = readNames(path.join(dir, "batches.meta.txt"));
  const train = readRecords(
    [1, 2, 3, 4, 5].map((i) => path.join(dir, `data_batch_${i}.bin`)),
    1,
    0,
  );
  const test = readRecords([path.join(dir, "test_batch.bin")], 1, 0);
  return { names, train, test };
}

function loadCifar100() {
  const dir = path.join(rawDir, "cifar-100-binary");
  const names = readNames(path.join(dir, "fine_label_names.txt"));
  const train = readRecords([path.join(dir, "train.bin")], 2, 1);
  const test = readRecords([path.join(dir, "test.bin")], 2, 1);
  return { names, train, test };
}

function pick(records: RawImage[], labelIds: number[], count: number, skip = 0): RawImage[] {
  const perLabel = labelIds.map((id) => records.filter((r) => r.label === id));
  const longest = Math.max(...perLabel.map((b) => b.length));
  const interleaved: RawImage[] = [];
  for (let i = 0; i < longest; i++) {
    for (const bucket of perLabel) {
      if (bucket[i]) interleaved.push(bucket[i]!);
    }
  }
  return interleaved.slice(skip, skip + count);
}

function savePng(pixels: Buffer, file: string) {
  const png = new PNG({ width: 32, height: 32 });
  for (let i = 0; i < 1024; i++) {
    png.data[i * 4] = pixels[i]!;
    png.data[i * 4 + 1] = pixels[1024 + i]!;
    png.data[i * 4 + 2] = pixels[2048 + i]!;
    png.data[i * 4 + 3] = 255;
  }
  fs.writeFileSync(file, PNG.sync.write(png));
}

function clearPrevious(dir: string) {
  if (!fs.existsSync(dir)) return;
  for (const f of fs.readdirSync(dir)) {
    if (f.startsWith(PREFIX)) fs.unlinkSync(path.join(dir, f));
  }
}

function writeSplit(split: string, folder: string, images: RawImage[]) {
  const dir = path.join(datasetDir, split, folder);
  fs.mkdirSync(dir, { recursive: true });
  clearPrevious(dir);
  images.forEach((img, i) => {
    savePng(img.pixels, path.join(dir, `${PREFIX}${String(i + 1).padStart(4, "0")}.png`));
  });
  return images.length;
}

const sources = { cifar10: loadCifar10(), cifar100: loadCifar100() };

console.log(`Importando CIFAR: train=${trainCount} validation=${valCount} test=${testCount} por classe`);
console.log("");

for (const mapping of MAPPINGS) {
  const src = sources[mapping.source];
  const ids = mapping.labels.map((name) => {
    const id = src.names.indexOf(name);
    if (id < 0) throw new Error(`Label "${name}" not found in ${mapping.source}`);
    return id;
  });

  const train = writeSplit("train", mapping.folder, pick(src.train, ids, trainCount));
  const val = writeSplit("validation", mapping.folder, pick(src.train, ids, valCount, trainCount));
  const test = writeSplit("test", mapping.folder, pick(src.test, ids, testCount));

  console.log(
    `${mapping.folder.padEnd(12)} <- ${mapping.source}:${mapping.labels.join("+").padEnd(28)} train=${train} val=${val} test=${test}`,
  );
}

console.log("");
console.log("Pronto! Imagens salvas em dataset/<split>/<classe>/");
