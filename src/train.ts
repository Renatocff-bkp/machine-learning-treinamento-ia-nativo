import { ImageDataset } from "./dataset";
import { loadImageAsTensor } from "./image-loader";
import { saveModel } from "./io";
import { argmax } from "./softmax-cross-entropy";
import { defaultCnnConfig, TinyCNN } from "./tiny-cnn";
import { evaluate, formatConfusion, trainCnn } from "./training-loop";

export async function trainClassifier(
  options: {
    datasetRoot: string,
    epochs?: number,
    learningRate?: number,
    imageSize?: number,
    modelPath?: string,
    patience?: number,
  }
) {
  const imageSize = options.imageSize ?? 32;
  const dataset = new ImageDataset(options.datasetRoot, imageSize);

  if (dataset.classes.length < 2) {
    throw new Error(`Need at least 2 class folders inside ${options.datasetRoot}/train`);
  }

  const train = dataset.list("train").samples;
  const val = dataset.list("validation").samples;

  const config = {
    ...defaultCnnConfig(dataset.classes.length),
    inputHeight: imageSize,
    inputWidth: imageSize,
  };
  const model = new TinyCNN(config);

  console.log(dataset.describe());
  console.log("");
  console.log(model.summary());
  console.log("");

  await trainCnn(model, dataset, train, val, {
    epochs: options.epochs ?? 20,
    learningRate: options.learningRate ?? 0.001,
    patience: options.patience,
  })

  const test = dataset.list("test").samples;
  if (test.length > 0) {
    const testStats = await evaluate(model, dataset, test);
    console.log("");
    console.log(`Test acc: ${(testStats.accuracy * 100).toFixed(1)}%  (acaso: ${(100 / dataset.classes.length).toFixed(1)}%)`)
    console.log("");
    console.log("Matriz de confusão (teste)");
    console.log(formatConfusion(dataset.classes, testStats.confusion));
  }

  if (options.modelPath) {
    saveModel(model, dataset.classes, options.modelPath);
    console.log("");
    console.log(`Modelo salvo em ${options.modelPath}`);
  }

  return model;
}

export async function predictImage(
  model: TinyCNN,
  classes: string[],
  imagePath: string,
) {
  const image = await loadImageAsTensor(imagePath, model.config.inputHeight);
  const probs = model.forward(image);
  const ranking = classes
    .map((name, i) => ({ name, probability: probs.data[i]! }))
    .sort((a, b) => b.probability - a.probability);

  return {
    label: classes[argmax(probs)]!,
    ranking,
  };
}
