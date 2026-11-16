import { Neuron } from "./neuron";

type Example = {
    inputs: number[];
    target: number;
    name?: string;
}

export function accuracy(
    neuron: Neuron,
    examples: Example[],
    threshold: number = 0.5 // converter a previsão do neurônio para um valor binário (0 ou 1)
) {
    let correct = 0;

    for(const example of examples) {
        const guess = neuron.forward(example.inputs) >= threshold ? 1 : 0;
        if(guess === example.target) {
            correct++;
        }
    }

    return correct / examples.length;
}

export function sigmoid(x: number): number {
    // Sigmoid é uma função de ativação que transforma um valor em um valor entre 0 e 1
    return 1 / (1 + Math.exp(-x));
}

export function clamp(x: number, min: number, max: number): number {
    return Math.max(min, Math.min(x, max));
}