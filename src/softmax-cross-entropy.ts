import { Tensor } from "./tensor";

const EPS = 1e-12;

export function softmax(logits: Tensor): Tensor {
    let max = -Infinity;
    for (const v of logits.data) if (v > max) max = v;

    const exps = logits.data.map(v => Math.exp(v - max));
    let sum = 0;
    for (const e of exps) sum += e;

    return new Tensor(exps.map(e => e / sum), [...logits.shape]);
}

export function oneHot(index: number, numClasses: number): Tensor {
    const data = new Float64Array(numClasses);
    data[index] = 1;
    return new Tensor(data, [numClasses]);
}

export function argmax(tensor: Tensor): number {
    let best = 0;
    for (let i = 1; i < tensor.data.length; i++) {
        if (tensor.data[i] > tensor.data[best]) best = i;
    }
    return best;
}

export class SoftmaxCrossEntropy {
    private lastProbs?: Tensor;
    private lastTarget?: Tensor;

    forward(probs: Tensor, target: Tensor): number {
        this.lastProbs = probs;
        this.lastTarget = target;

        let loss = 0;
        for (let i = 0; i < probs.data.length; i++) {
            loss -= target.data[i] * Math.log(probs.data[i] + EPS);
        }

        return loss;
    }

    backward(): Tensor {
        if (!this.lastProbs || !this.lastTarget) {
            throw new Error("Forward method must be called before backward method.");
        }

        const target = this.lastTarget;
        return this.lastProbs.map((p, i) => p - target.data[i]);
    }
}
