import { ReLU } from "./activation";
import { Conv2D } from "./convolve2d";
import { Dense } from "./dense";
import { MaxPool2x2 as MaxPool2D } from "./maxPool2xWindow";
import { SoftmaxCrossEntropy, softmax } from "./softmax-cross-entropy";
import { Tensor } from "./tensor";
import { Parameter } from "./types";
import { heUniform } from "./utils";

export interface CnnConfig {
    inChannels: number;
    inputHeight: number;
    inputWidth: number;
    conv1Filters: number;
    conv2Filters: number;
    kernelSize: number;
    denseUnits: number;
    numClasses: number;
    seed?: number;
}

export class TinyCNN {
    config: CnnConfig;
    conv1: Conv2D;
    relu1: ReLU;
    pool1: MaxPool2D;
    conv2: Conv2D;
    relu2: ReLU;
    pool2: MaxPool2D;
    dense1: Dense;
    relu3: ReLU;
    denseOut: Dense;
    lossFn: SoftmaxCrossEntropy;

    lastLoss = 0;
    activations: { name: string; shape: number[] }[] = [];

    private lastFlatShape: number[] | null = null;
    private denseInputSize: number;

    constructor(config: CnnConfig) {
        this.config = config;
        const seed = config.seed ?? 42;

        this.conv1 = new Conv2D(config.inChannels, config.conv1Filters, config.kernelSize, {
            padding: 1,
            seed,
        });
        this.relu1 = new ReLU();
        this.pool1 = new MaxPool2D(2, 2);

        this.conv2 = new Conv2D(config.conv1Filters, config.conv2Filters, config.kernelSize, {
            padding: 1,
            seed: seed + 1000,
        });
        this.relu2 = new ReLU();
        this.pool2 = new MaxPool2D(2, 2);

        // Com pad=1 e kernel 3×3, conv preserva H×W; cada pool divide por 2
        const h1 = Math.floor(config.inputHeight / 2);
        const w1 = Math.floor(config.inputWidth / 2);
        const h2 = Math.floor(h1 / 2);
        const w2 = Math.floor(w1 / 2);
        this.denseInputSize = config.conv2Filters * h2 * w2;

        this.dense1 = new Dense(
            heUniform(config.denseUnits, this.denseInputSize, this.denseInputSize, seed + 2000),
            Tensor.zeros([config.denseUnits]),
        );
        this.relu3 = new ReLU();
        this.denseOut = new Dense(
            heUniform(config.numClasses, config.denseUnits, config.denseUnits, seed + 3000).scale(0.5),
            Tensor.zeros([config.numClasses]),
        );
        this.lossFn = new SoftmaxCrossEntropy();
    }

    forward(image: Tensor, record = false): Tensor {
        if (record) this.activations = [];

        let x = this.conv1.forward(image);
        if (record) this.activations.push({ name: "conv1", shape: [...x.shape] });

        x = this.relu1.forward(x);
        if (record) this.activations.push({ name: "relu1", shape: [...x.shape] });

        x = this.pool1.forward(x);
        if (record) this.activations.push({ name: "pool1", shape: [...x.shape] });

        x = this.conv2.forward(x);
        if (record) this.activations.push({ name: "conv2", shape: [...x.shape] });

        x = this.relu2.forward(x);
        if (record) this.activations.push({ name: "relu2", shape: [...x.shape] });

        x = this.pool2.forward(x);
        if (record) this.activations.push({ name: "pool2", shape: [...x.shape] });

        this.lastFlatShape = [...x.shape];
        const flat = x.reshape([x.data.length]);
        if (record) this.activations.push({ name: "flatten", shape: [...flat.shape] });

        let h = this.dense1.forward(flat);
        if (record) this.activations.push({ name: "dense1", shape: [...h.shape] });

        h = this.relu3.forward(h);
        const logits = this.denseOut.forward(h);
        if (record) this.activations.push({ name: "logits", shape: [...logits.shape] });

        const probs = softmax(logits);
        if (record) this.activations.push({ name: "softmax", shape: [...probs.shape] });

        return probs;
    }

    backward(prediction: Tensor, target: Tensor): void {
        this.lastLoss = this.lossFn.forward(prediction, target);
        let grad = this.lossFn.backward();

        grad = this.denseOut.backward(grad);
        grad = this.relu3.backward(grad);
        grad = this.dense1.backward(grad);

        if (this.lastFlatShape === null) {
            throw new Error("forward() must be called before backward().");
        }
        grad = grad.reshape(this.lastFlatShape);

        grad = this.pool2.backward(grad);
        grad = this.relu2.backward(grad);
        grad = this.conv2.backward(grad);

        grad = this.pool1.backward(grad);
        grad = this.relu1.backward(grad);
        this.conv1.backward(grad);
    }

    parameters(): Parameter[] {
        const rename = (prefix: string, params: Parameter[]): Parameter[] =>
            params.map((p) => ({ ...p, name: `${prefix}.${p.name}` }));

        return [
            ...rename("conv1", this.conv1.parameters()),
            ...rename("conv2", this.conv2.parameters()),
            {
                name: "dense1.weights",
                data: this.dense1.weights,
                grad: this.dense1.gradWeights ?? Tensor.zeros(this.dense1.weights.shape),
            },
            {
                name: "dense1.bias",
                data: this.dense1.bias,
                grad: this.dense1.gradBias ?? Tensor.zeros(this.dense1.bias.shape),
            },
            {
                name: "denseOut.weights",
                data: this.denseOut.weights,
                grad: this.denseOut.gradWeights ?? Tensor.zeros(this.denseOut.weights.shape),
            },
            {
                name: "denseOut.bias",
                data: this.denseOut.bias,
                grad: this.denseOut.gradBias ?? Tensor.zeros(this.denseOut.bias.shape),
            },
        ];
    }

    countParameters(): number {
        return this.parameters().reduce((n, p) => n + p.data.data.length, 0);
    }

    summary(): string {
        const c = this.config;
        const lines = [
            "Model",
            "-----",
            `Input: ${c.inputHeight} × ${c.inputWidth} × ${c.inChannels}`,
            "Conv2D",
            ` filters: ${c.conv1Filters}`,
            ` kernel: ${c.kernelSize} × ${c.kernelSize}`,
            "ReLU",
            "MaxPool",
            " pool: 2 × 2",
            "Conv2D",
            ` filters: ${c.conv2Filters}`,
            ` kernel: ${c.kernelSize} × ${c.kernelSize}`,
            "ReLU",
            "MaxPool",
            " pool: 2 × 2",
            "Flatten",
            "Dense",
            ` units: ${c.denseUnits}`,
            "ReLU",
            "Dense",
            ` units: ${c.numClasses}`,
            "Softmax",
            `Parameters: ${this.countParameters().toLocaleString()}`,
        ];
        return lines.join("\n");
    }
}

export function defaultCnnConfig(numClasses = 7): CnnConfig {
    return {
        inChannels: 3,
        inputHeight: 32,
        inputWidth: 32,
        conv1Filters: 8,
        conv2Filters: 16,
        kernelSize: 3,
        denseUnits: 64,
        numClasses,
        seed: 42,
    };
}
