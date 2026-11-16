import { Adam } from "./adam";
import { ImageDataset, Sample } from "./dataset";
import { argmax } from "./softmax-cross-entropy";
import { TinyCNN } from "./tiny-cnn";
import { createRng } from "./utils";

export interface ValidationEpoch {
  epoch: number;
  trainLoss: number;
  trainAccuracy: number;
  valLoss: number;
  valAccuracy: number;
  seconds: number;
}

export interface EvaluationResult {
  loss: number;
  accuracy: number;
  confusion: number[][];
}

export function detectOverfitting(
  history: ValidationEpoch[],
  window = 3
) {
  if (history.length < (window + 1)) {
    return false
  }

  const recent = history.slice(-window)
  const earlier = history[history.length - window - 1]
  const last = recent[recent.length - 1]

  return last.trainLoss < earlier.trainLoss && last.valLoss > earlier.valLoss;
}

export async function evaluate(
  model: TinyCNN,
  dataset: ImageDataset,
  samples: Sample[],
): Promise<EvaluationResult> {
  const n = dataset.classes.length;
  const confusion = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  let totalLoss = 0;
  let correct = 0;

  for (const sample of samples) {
    const { image, label } = await dataset.loadSample(sample);
    const prediction = model.forward(image);
    totalLoss += model.lossFn.forward(prediction, label);
    const predictedClass = argmax(prediction);
    confusion[sample.label]![predictedClass]! += 1;
    if (predictedClass === sample.label) correct += 1;
  }

  return {
    loss: totalLoss / Math.max(1, samples.length),
    accuracy: correct / Math.max(1, samples.length),
    confusion,
  };
}

export function formatConfusion(classes: string[], confusion: number[][]): string {
  const width = Math.max(8, ...classes.map((c) => c.length + 1));
  const lines = [
    "real \\ previsto".padEnd(width + 4) + classes.map((c) => c.padStart(width)).join("") + "  acerto",
  ];
  confusion.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0);
    const pct = total ? ((row[i]! / total) * 100).toFixed(0) + "%" : "-";
    lines.push(
      classes[i]!.padEnd(width + 4) + row.map((v) => String(v).padStart(width)).join("") + pct.padStart(8),
    );
  });
  return lines.join("\n");
}

export async function trainCnn(
  model: TinyCNN,
  dataset: ImageDataset,
  trainSamples: Sample[],
  valSamples: Sample[],
  options: {
    epochs: number;
    learningRate?: number;
    seed?: number;
    augment?: boolean;
    patience?: number;
  },
): Promise<ValidationEpoch[]> {
  const optimizer = new Adam(options.learningRate ?? 0.001);
  const rng = createRng(options.seed ?? 7);
  const history: ValidationEpoch[] = [];
  const patience = options.patience ?? Infinity;
  let best = { valAccuracy: -1, valLoss: Infinity, epoch: 0, weights: [] as Float64Array[] };

  for (let epoch = 1; epoch <= options.epochs; epoch++) {
    const started = Date.now();
    const order = ImageDataset.shuffled(
      trainSamples,
      (options.seed ?? 0) + epoch,
    );

    for (const sample of order) {
      const flip = (options.augment ?? true) && rng() < 0.5;
      const { image, label } = await dataset.loadSample(sample, flip);
      const prediction = model.forward(image);
      model.backward(prediction, label);
      optimizer.step(model.parameters());
    }

    const train = await evaluate(model, dataset, trainSamples);
    const val = await evaluate(model, dataset, valSamples);

    const row: ValidationEpoch = {
      epoch,
      trainLoss: train.loss,
      trainAccuracy: train.accuracy,
      valLoss: val.loss,
      valAccuracy: val.accuracy,
      seconds: (Date.now() - started) / 1000,
    };
    history.push(row);

    const improved =
      val.accuracy > best.valAccuracy ||
      (val.accuracy === best.valAccuracy && val.loss < best.valLoss);
    if (improved) {
      best = {
        valAccuracy: val.accuracy,
        valLoss: val.loss,
        epoch,
        weights: model.parameters().map((p) => new Float64Array(p.data.data)),
      };
    }

    console.log(`Epoch ${epoch}/${options.epochs} (${row.seconds.toFixed(1)}s)${improved ? "  * melhor até agora" : ""}`);
    console.log(
      `  train loss=${row.trainLoss.toFixed(4)}  acc=${(row.trainAccuracy * 100).toFixed(1)}%`,
    );
    console.log(
      `  val   loss=${row.valLoss.toFixed(4)}  acc=${(row.valAccuracy * 100).toFixed(1)}%`,
    );

    if (detectOverfitting(history)) {
      console.log("  Model has overfitted!")
    }

    if (epoch - best.epoch >= patience) {
      console.log(`Sem melhora na validação há ${patience} épocas. Parando cedo.`);
      break;
    }
  }

  if (best.weights.length > 0) {
    model.parameters().forEach((p, i) => p.data.data.set(best.weights[i]!));
    console.log(
      `Pesos restaurados da época ${best.epoch} (val acc=${(best.valAccuracy * 100).toFixed(1)}%).`,
    );
  }

  return history;
}
