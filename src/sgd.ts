import { Parameter } from "./types";

export class SGD {
    constructor(
        public learningRate: number,
    ) {}

    step(params: Parameter[]) {
        for(const param of params) {
            if(param.grad == null) {
                throw new Error(`Gradient for parameter ${param.name} has no gradient.`);
            }
            
            for(let i = 0; i < param.data.data.length; i++) {
                param.data.data[i] = param.data.data[i] - this.learningRate * param.grad.data[i];
            }
        }
    }
}