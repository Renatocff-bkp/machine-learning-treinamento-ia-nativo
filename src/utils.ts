import { Tensor } from "./tensor";

export function randomMatrix(rows: number, cols: number, seed: number): Tensor {
    let state = seed;
    const data: number[] = [];

    for(let i = 0; i < rows * cols; i++) {
        state = (state * 1664525 + 1013904223) % 4294967296;
        const u = state / 4294967296;
        data.push((u - 0.5) * 0.5);
    }

    return new Tensor(new Float64Array(data), [rows, cols]);
}

export function createRng(seed: number): () => number {
    let state = (Math.abs(Math.floor(seed)) * 2654435761) % 4294967296;

    const next = () => {
        state = (state * 1664525 + 1013904223) % 4294967296;
        return state / 4294967296;
    };

    for (let i = 0; i < 10; i++) next();

    return next;
}

export function heUniform(rows: number, cols: number, fanIn: number, seed: number): Tensor {
    const rng = createRng(seed);
    const limit = Math.sqrt(6 / fanIn);
    const data = new Float64Array(rows * cols);

    for (let i = 0; i < data.length; i++) {
        data[i] = (rng() * 2 - 1) * limit;
    }

    return new Tensor(data, [rows, cols]);
}
