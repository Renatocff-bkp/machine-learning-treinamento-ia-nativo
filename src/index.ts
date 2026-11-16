import { Tensor } from "./tensor";
import { TinyCNN } from "./tiny-cnn";
import { defaultCnnConfig } from "./tiny-cnn";

const model = new TinyCNN(defaultCnnConfig());

console.log(model.summary());

const image = Tensor.zeros([3, 32, 32]); // RGB 32x32
const pred = model.forward(image, true);

for(const a of model.activations) {
    console.log(a.name, a.shape);
}