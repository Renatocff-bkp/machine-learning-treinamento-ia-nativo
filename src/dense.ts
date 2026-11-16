import { Tensor } from "./tensor";

export class Dense {
    lastInput?: Tensor;
    gradWeights?: Tensor;
    gradBias?: Tensor;

    constructor(
        public weights: Tensor,
        public bias: Tensor,
    ) {}

    forward(input: Tensor): Tensor {
        this.lastInput = input;

        // Cada neurônio de saída é a soma ponderada dos inputs ( todas as entradas )
        const output = new Float64Array(this.bias.data.length);

        for(let o = 0; o < output.length; o++) {
            let sum = this.bias.data[o];
        
            for(let i = 0; i < input.data.length; i++) {
                sum += input.data[i] * this.weights.data[o * input.data.length + i];
            }

            output[o] = sum;
        }

        return new Tensor(output, [output.length]);
    }

    backward(gradOutput: Tensor): Tensor {
        const input = this.lastInput!;
        const inFeatures = input.data.length;
        const outFeatures = gradOutput.data.length;

        this.gradWeights = Tensor.zeros(this.weights.shape);
        this.gradBias = Tensor.zeros(this.bias.shape);
        const gradInput = new Float64Array(inFeatures);

        for(let o = 0; o < outFeatures; o++) {
            this.gradBias.data[o] = gradOutput.data[o];

            for(let i = 0; i < inFeatures; i++) {
                const wIdx = o * inFeatures + i;
                this.gradWeights.data[wIdx] += input.data[i] * gradOutput.data[o];
                gradInput[i] += this.weights.data[wIdx] * gradOutput.data[o];
            }
        }

        return new Tensor(gradInput, [inFeatures]);
    }
}