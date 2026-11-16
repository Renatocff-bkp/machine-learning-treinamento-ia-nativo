import { Dense } from "./dense";
import { ReLU, Sigmoid } from "./activation";
import { BinaryCrossEntropy } from "./bce";
import { Adam } from "./adam";
import { randomMatrix } from "./utils";
import { Tensor } from "./tensor";
import { Example, Parameter } from "./types";

type TrainableLayer = {
    forward: (input: Tensor) => Tensor;
    backward: (gradOutput: Tensor) => Tensor;
}

// MLP = Multi-Layer Perceptron
export class TrainableMLP {
    dense1: Dense;
    dense2: Dense;
    relu: ReLU;
    sigmoid: Sigmoid;
    layers: TrainableLayer[];
    lossFn: BinaryCrossEntropy;
    optimizer: Adam;
    lastLoss: number = 0;

    constructor(
        public inputSize: number,
        public hiddenSize: number,
        public seed: number = 42,
        public learningRate: number = 0.5
    ) {
        this.dense1 = new Dense(
            randomMatrix(inputSize, hiddenSize, seed),
            Tensor.zeros([hiddenSize])
        );
        this.dense2 = new Dense(
            randomMatrix(hiddenSize, 1, seed + 1),
            Tensor.zeros([1])
        );
        this.relu = new ReLU();
        this.sigmoid = new Sigmoid();
        this.layers = [this.dense1, this.relu, this.dense2, this.sigmoid];
        this.lossFn = new BinaryCrossEntropy();
        this.optimizer = new Adam(learningRate);
    }


    trainStep(examples: Example): number {
        const pred = this.forward(examples.inputs);
        this.backward(pred, examples.target);
        this.optimizer.step(this.parameters());
        return this.lastLoss;
    }

    forward(input: Tensor): Tensor {
        let output = input;
        for(const layer of this.layers) {
            output = layer.forward(output);
        }
        return output;
    }

    backward(pred: Tensor, target: Tensor): void {
        this.lastLoss = this.lossFn.forward(pred, target);
        let grad = this.lossFn.backward();

        for(let i = this.layers.length - 1; i >= 0; i--) {
            grad = this.layers[i].backward(grad);
        }
    }

    parameters(): Parameter[] {
        return [
            { name: "dense1.weights", data: this.dense1.weights, grad: this.dense1.gradWeights! },
            { name: "dense1.bias", data: this.dense1.bias, grad: this.dense1.gradBias! },
            { name: "dense2.weights", data: this.dense2.weights, grad: this.dense2.gradWeights! },
            { name: "dense2.bias", data: this.dense2.bias, grad: this.dense2.gradBias! },
        ]
    }
}