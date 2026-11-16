import { sigmoid } from "./maths";

export class Neuron {
    constructor(
        public weights: number[],
        public bias: number,
    ){}

    weightedSum(inputs: number[]): number {
        let sum = this.bias;

        for(let i = 0; i < this.weights.length; i++) {
            sum += this.weights[i] * inputs[i];
        }

        return sum;
    }

    forward(inputs: number[], activation = sigmoid) {
        return activation(this.weightedSum(inputs));
    }
}