import { Tensor } from "./tensor";

export type Example ={
    inputs: Tensor; //feature vector
    target: Tensor; //ground truth
}

export type Parameter = {
    name: string;
    data: Tensor; // Valor do peso atual
    grad: Tensor; // O quanto o peso deve mudar
}

export type Nullable<T> = T | null;
