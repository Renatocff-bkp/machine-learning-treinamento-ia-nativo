import { Tensor } from "./tensor";

export class ReLU {
    private lastInput?: Tensor;

    forward(input: Tensor): Tensor {
        this.lastInput = input;

        return input.map(value => Math.max(0, value));
    }

    backward(grad: Tensor): Tensor {
        return grad.map((g, i) => this.lastInput!.data[i] > 0 ? g : 0);
    }
}

export class Sigmoid {
    private lastOutput?: Tensor;

    forward(input: Tensor): Tensor {
        const output = input.map(value => 1 / (1 + Math.exp(-value)));
        this.lastOutput = output;

        return output;
    }

    backward(grad: Tensor): Tensor {
        return grad.map(
            (g, i) => {
                const s = this.lastOutput!.data[i];
                return g * s * (1 - s);
            }
        );
    }
}