import { Neuron } from "./neuron";
import { BinaryCrossEntropy } from "./bce";

const bce = new BinaryCrossEntropy();

export function backPropagation(
    neuron: Neuron,
    inputs: number[],
    target: number,
) {
    const prediction = neuron.forward(inputs);
    const loss = bce.forward(prediction, target);

    //1. O quanto o loss vai mudar quando a previsão for alterada
    const dL_dPred = (prediction - target) / (prediction * (1 - prediction) + 1e-15);
        
    //2. Derivar a função de ativação (sigmoide) em relação a soma ponderada
    const dpred_dZ = prediction * (1 - prediction); //derivada da sigmoide em relação a soma ponderada
    
    //3. Combinar os dois para obter a derivada da loss em relação aos pesos e bias
    const dL_dZ = dL_dPred * dpred_dZ;
    
    //4. Gradiente para cada peso e bias
    const weightGrads = inputs.map((input) => dL_dZ * input);
    const biasGrad = dL_dZ;

    return {
        prediction,
        loss,
        weightGrads,
        biasGrad
    }
}