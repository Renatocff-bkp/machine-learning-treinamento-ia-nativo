import ps from "node:process";
import { loadModel } from "../src/io";
import { predictImage } from "../src/train";

const imagePath = ps.argv[2];

if (!imagePath) {
  console.log("Uso: npm run predict -- <caminho-da-imagem>");
  ps.exit(1);
}

const { model, classes } = loadModel("models/classificador.json");

predictImage(model, classes, imagePath).then(result => {
  console.log(`Imagem: ${imagePath}`);
  console.log("");
  for (const { name, probability } of result.ranking) {
    const bar = "█".repeat(Math.round(probability * 30));
    console.log(`${name.padEnd(12)} ${(probability * 100).toFixed(1).padStart(5)}%  ${bar}`);
  }
  console.log("");
  console.log(`I have detected: ${result.label.toUpperCase()}`);
});
