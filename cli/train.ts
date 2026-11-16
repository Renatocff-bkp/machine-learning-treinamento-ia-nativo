import path from "node:path"
import ps from "node:process"
import { trainClassifier } from "../src/train"

const root = ps.cwd();
const epochs = Number(ps.argv[2] ?? 30);

trainClassifier({
  datasetRoot: path.join(root, "dataset"),
  modelPath: path.join(root, "models", "classificador.json"),
  epochs,
  learningRate: 0.001,
  imageSize: 32,
  patience: 6,
}).then(() => {
  console.log("Train has completed!!!")
});
