import fs from "node:fs"
import path from "node:path"
import { CnnConfig, TinyCNN } from "./tiny-cnn";

type TrainedModelArtifact = {
  config: CnnConfig;
  classes: string[];
  parameters: Record<string, number[]>;
}

export function saveModel(
  model: TinyCNN,
  classes: string[],
  filePath: string
) {
  const artifact: TrainedModelArtifact = {
    config: { ...model.config },
    classes,
    parameters: {}
  }

  for (const p of model.parameters()) {
    artifact.parameters[p.name] = [...p.data.data];
  }

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(artifact))
}

export function loadModel(filePath: string): { model: TinyCNN; classes: string[] } {
  const artifact = JSON.parse(
    fs.readFileSync(filePath, "utf-8"),
  ) as TrainedModelArtifact

  if (!artifact.classes || !artifact.config.numClasses) {
    throw new Error(`${filePath} was saved by the old binary model and is not compatible.`);
  }

  const model = new TinyCNN(artifact.config);

  for (const p of model.parameters()) {
    const values = artifact.parameters[p.name]

    if (!values) {
      throw new Error(`Missing parameter ${p.name} in ${filePath}`);
    }

    for (let i = 0; i < values.length; i++) {
      p.data.data[i] = values[i]
    }
  }

  return { model, classes: artifact.classes };
}
