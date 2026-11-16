import { BinaryCrossEntropy } from "./bce";
import { Neuron } from "./neuron";

type Output = {
    weightGrads: number[];
    biasGrad: number;
}

const EPSILON = 1e-5;

const bce = new BinaryCrossEntropy();

export function numericalGradientsForNeuron(
    neuron: Neuron,
    inputs: number[],
    target: number,
): Output {
    const lossFn = (n: Neuron) => {
        const prediction = n.forward(inputs);
        return bce.forward(prediction, target);
    }

    const weightGrads: number[] = [];
    const originalBias = neuron.bias;

    for(let i = 0; i < neuron.weights.length; i++) {
        const originalWeight = neuron.weights[i];

        neuron.weights[i] = originalWeight + EPSILON;
        const lossPlus = lossFn(neuron);
        
        neuron.weights[i] = originalWeight - EPSILON;
        const lossMinus = lossFn(neuron);

        weightGrads.push((lossPlus - lossMinus) / (2 * EPSILON));
    }

    neuron.bias = originalBias + EPSILON;
    const lossPlus = lossFn(neuron);

    neuron.bias = originalBias - EPSILON;
    const lossMinus = lossFn(neuron);

    const biasGrad = (lossPlus - lossMinus) / (2 * EPSILON);

    return {
        weightGrads,
        biasGrad,
    }
}