import { oneHot } from "../src/softmax-cross-entropy";
import { Tensor } from "../src/tensor";
import { TinyCNN } from "../src/tiny-cnn";
import { createRng } from "../src/utils";

const EPSILON = 1e-5;
const rng = createRng(1);

const model = new TinyCNN({
  inChannels: 3,
  inputHeight: 8,
  inputWidth: 8,
  conv1Filters: 2,
  conv2Filters: 3,
  kernelSize: 3,
  denseUnits: 5,
  numClasses: 4,
  seed: 3,
});

const image = new Tensor(new Float64Array(3 * 8 * 8).map(() => rng()), [3, 8, 8]);
const target = oneHot(2, 4);

const loss = () => model.lossFn.forward(model.forward(image), target);

model.backward(model.forward(image), target);

let worst = 0;
let checked = 0;

for (const p of model.parameters()) {
  const analytic = new Float64Array(p.grad.data);
  for (let i = 0; i < p.data.data.length; i++) {
    const original = p.data.data[i]!;
    p.data.data[i] = original + EPSILON;
    const plus = loss();
    p.data.data[i] = original - EPSILON;
    const minus = loss();
    p.data.data[i] = original;

    const numeric = (plus - minus) / (2 * EPSILON);
    const diff = Math.abs(numeric - analytic[i]!) / Math.max(1e-8, Math.abs(numeric) + Math.abs(analytic[i]!));
    if (Math.abs(numeric) + Math.abs(analytic[i]!) > 1e-9) worst = Math.max(worst, diff);
    checked++;
  }
}

console.log(`Parâmetros verificados: ${checked}`);
console.log(`Maior diferença relativa: ${worst.toExponential(2)}`);
console.log(worst < 1e-4 ? "OK: backpropagation confere com a derivada numérica." : "FALHOU: gradientes divergem.");
