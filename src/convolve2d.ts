import { Tensor } from "./tensor";
import { Parameter } from "./types";
import { heUniform } from "./utils";

export type Conv2DOptions = {
    stride?: number;
    padding?: number;
    seed?: number;
};

export class Conv2D {
    inChannels: number;
    outChannels: number;
    kernelSize: number;
    stride: number;
    padding: number;

    kernels: Tensor[][];
    biases: Tensor[];

    gradKernels: Tensor[][] | null = null;
    gradBiases: Tensor[] | null = null;
    private lastInput: Tensor | null = null;
    private lastOutShape: [number, number] = [0, 0];

    constructor(
        inChannels: number,
        outChannels: number,
        kernelSize: number,
        options: Conv2DOptions = {},
    ) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = options.stride ?? 1;
        this.padding = options.padding ?? 0;

        let seed = options.seed ?? 42;
        this.kernels = [];
        for (let o = 0; o < outChannels; o++) {
            const row: Tensor[] = [];
            for (let c = 0; c < inChannels; c++) {
                row.push(randomKernel(kernelSize, inChannels * kernelSize * kernelSize, seed++));
            }
            this.kernels.push(row);
        }

        this.biases = [];
        for (let o = 0; o < outChannels; o++) {
            this.biases.push(new Tensor(new Float64Array([0]), [1]));
        }
    }

    forward(input: Tensor): Tensor {
        if (input.shape.length !== 3) {
            throw new Error("Conv2D.forward expects shape [C, H, W].");
        }
        if (input.shape[0] !== this.inChannels) {
            throw new Error(
                `Expected ${this.inChannels} channels, got ${input.shape[0]}.`,
            );
        }

        this.lastInput = input;
        const [, h, w] = input.shape as [number, number, number];
        const k = this.kernelSize;
        const s = this.stride;
        const p = this.padding;
        const outH = Math.floor((h + 2 * p - k) / s) + 1;
        const outW = Math.floor((w + 2 * p - k) / s) + 1;
        this.lastOutShape = [outH, outW];

        const x = input.data;
        const out = new Float64Array(this.outChannels * outH * outW);

        for (let o = 0; o < this.outChannels; o++) {
            const outBase = o * outH * outW;
            out.fill(this.biases[o]!.data[0]!, outBase, outBase + outH * outW);

            for (let c = 0; c < this.inChannels; c++) {
                const kern = this.kernels[o]![c]!.data;
                const inBase = c * h * w;

                for (let i = 0; i < outH; i++) {
                    for (let j = 0; j < outW; j++) {
                        let sum = 0;
                        for (let a = 0; a < k; a++) {
                            const r = i * s + a - p;
                            if (r < 0 || r >= h) continue;
                            const rowBase = inBase + r * w;
                            for (let b = 0; b < k; b++) {
                                const q = j * s + b - p;
                                if (q < 0 || q >= w) continue;
                                sum += x[rowBase + q]! * kern[a * k + b]!;
                            }
                        }
                        out[outBase + i * outW + j]! += sum;
                    }
                }
            }
        }

        return new Tensor(out, [this.outChannels, outH, outW]);
    }

    backward(gradOutput: Tensor): Tensor {
        if (this.lastInput === null) {
            throw new Error("Must call forward() before backward().");
        }

        const input = this.lastInput;
        const [, h, w] = input.shape as [number, number, number];
        const [outH, outW] = this.lastOutShape;
        const k = this.kernelSize;
        const s = this.stride;
        const p = this.padding;
        const x = input.data;
        const g = gradOutput.data;
        const gradInput = new Float64Array(this.inChannels * h * w);

        this.gradKernels = [];
        this.gradBiases = [];

        for (let o = 0; o < this.outChannels; o++) {
            const outBase = o * outH * outW;

            let gb = 0;
            for (let n = 0; n < outH * outW; n++) gb += g[outBase + n]!;
            this.gradBiases.push(new Tensor(new Float64Array([gb]), [1]));

            const gradRow: Tensor[] = [];
            for (let c = 0; c < this.inChannels; c++) {
                const kern = this.kernels[o]![c]!.data;
                const gK = new Float64Array(k * k);
                const inBase = c * h * w;

                for (let i = 0; i < outH; i++) {
                    for (let j = 0; j < outW; j++) {
                        const go = g[outBase + i * outW + j]!;
                        if (go === 0) continue;
                        for (let a = 0; a < k; a++) {
                            const r = i * s + a - p;
                            if (r < 0 || r >= h) continue;
                            const rowBase = inBase + r * w;
                            for (let b = 0; b < k; b++) {
                                const q = j * s + b - p;
                                if (q < 0 || q >= w) continue;
                                gK[a * k + b]! += x[rowBase + q]! * go;
                                gradInput[rowBase + q]! += kern[a * k + b]! * go;
                            }
                        }
                    }
                }

                gradRow.push(new Tensor(gK, [k, k]));
            }
            this.gradKernels.push(gradRow);
        }

        return new Tensor(gradInput, [this.inChannels, h, w]);
    }

    parameters(): Parameter[] {
        const params: Parameter[] = [];

        for (let o = 0; o < this.outChannels; o++) {
            for (let c = 0; c < this.inChannels; c++) {
                const kernel = this.kernels[o]![c]!;
                params.push({
                    name: `conv.k[${o},${c}]`,
                    data: kernel,
                    grad: this.gradKernels?.[o]?.[c] ?? Tensor.zeros(kernel.shape),
                });
            }
            const bias = this.biases[o]!;
            params.push({
                name: `conv.b[${o}]`,
                data: bias,
                grad: this.gradBiases?.[o] ?? Tensor.zeros(bias.shape),
            });
        }

        return params;
    }
}

function randomKernel(size: number, fanIn: number, seed: number): Tensor {
    return heUniform(size, size, fanIn, seed);
}

function pad2d(image: Tensor, padding: number): Tensor {
    if (padding === 0) return image;
    if (padding < 0) {
        throw new Error(`pad2d expects padding >= 0, got ${padding}.`);
    }

    const [h, w] = image.shape as [number, number];
    const padded = Tensor.zeros([h + 2 * padding, w + 2 * padding]);

    for (let r = 0; r < h; r++) {
        for (let c = 0; c < w; c++) {
            padded.set(image.get(r, c), r + padding, c + padding);
        }
    }

    return padded;
}

export function convolve2d(
    image: Tensor,
    kernel: Tensor,
    options: { stride?: number; padding?: number } = {},
): Tensor {
    const stride = options.stride ?? 1;
    const padding = options.padding ?? 0;
    const [kh, kw] = kernel.shape as [number, number];
    const padded = pad2d(image, padding);
    const outH = Math.floor((padded.shape[0] - kh) / stride) + 1;
    const outW = Math.floor((padded.shape[1] - kw) / stride) + 1;
    const output = Tensor.zeros([outH, outW]);

    for (let i = 0; i < outH; i++) {
        for (let j = 0; j < outW; j++) {
            let sum = 0;

            for (let a = 0; a < kh; a++) {
                for (let b = 0; b < kw; b++) {
                    sum += padded.get(i * stride + a, j * stride + b) * kernel.get(a, b);
                }
            }

            output.set(sum, i, j);
        }
    }

    return output;
}
