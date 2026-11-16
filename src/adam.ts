import { Tensor } from "./tensor";
import { Parameter } from "./types";

// DAM (Adaptive Moment Estimation) is an optimization algorithm that uses the first and second moments of the gradient to adapt the learning rate.

export class Adam {
    private m: Map<string, Tensor> = new Map(); // m = média de movimento exponencial de gradientes ( momentum )
    private v: Map<string, Tensor> = new Map(); // v = média de movimento exponencial de gradientes quadrados ( momento quadrático )
    private t = 0; //contador de iterações

    constructor(
        private lr: number, // learning rate
        private beta1: number = 0.9, // decay rate for the first moment
        private beta2: number = 0.999, // decay rate for the second moment
        private eps: number = 1e-8, // small constant to avoid division by zero
    ){}

    step(parameters: Parameter[]){
       this.t++;

       for(const p of parameters){
        if(p.grad == null){
            throw new Error(`Gradient for parameter ${p.name} has no gradient.`);
        }

        // cria buffers M e V na primeira execução ( primeiro step ) seguindo o shapes dos parametros
        if(!this.m.has(p.name)){
            this.m.set(p.name, Tensor.zeros(p.data.shape));
            this.v.set(p.name, Tensor.zeros(p.data.shape));
        }

        // Atualiza o adaptativo: média movel exponencial do gradiente
        const m = this.m.get(p.name)!;
        const v = this.v.get(p.name)!;
        const g = p.grad;

        for(let i = 0; i < p.data.data.length; i++){
           const gi = g.data[i];

           m.data[i] = this.beta1 * m.data[i] + (1 - this.beta1) * gi;
           v.data[i] = this.beta2 * v.data[i] + (1 - this.beta2) * gi * gi;

           const mHat = m.data[i] / (1 - Math.pow(this.beta1, this.t));
           const vHat = v.data[i] / (1 - Math.pow(this.beta2, this.t));

           p.data.data[i] = p.data.data[i] - this.lr * mHat / (Math.sqrt(vHat) + this.eps);
        }
       }
    }
}