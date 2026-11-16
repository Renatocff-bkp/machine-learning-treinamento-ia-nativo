import { Tensor } from "./tensor";
import { Nullable } from "./types";

export function maxPool2x2(
    window: number[][]
): number{
    let m = -Infinity;

    for(const row of window){
        for(const value of row){
            if(value > m){
                m = value;
            }
        }
    }

    return m;
}

export class MaxPool2x2 {
    private lastArgMax: Nullable<Int32Array> = null;
    private lastInputShape: Nullable<number[]> = null;

    constructor(
        private poolSize = 2,
        private stride?: number,
    ) {}

    forward(input: Tensor): Tensor {
      const [c, h, w] = input.shape as [number, number, number];
      const stride = this.stride ?? this.poolSize;
      const outH = Math.floor((h - this.poolSize) / stride) + 1;
      const outW = Math.floor((w - this.poolSize) / stride) + 1;
      const out = new Float64Array(c * outH * outW);
      const argMax = new Int32Array(out.length);
      const x = input.data;
      this.lastInputShape = [c, h, w];

      let idx = 0;
      for(let ch = 0; ch < c; ch++){
        for(let i = 0; i < outH; i++){
          for(let j = 0; j < outW; j++){
            let maxVal = -Infinity;
            let maxPos = 0;

            for(let a = 0; a < this.poolSize; a++){
                for(let b = 0; b < this.poolSize; b++){
                    const pos = ch * h * w + (i * stride + a) * w + (j * stride + b);
                    if(x[pos]! > maxVal){
                        maxVal = x[pos]!;
                        maxPos = pos;
                    }
                }
            }

            out[idx] = maxVal;
            argMax[idx] = maxPos;
            idx++;
          }
        }
      }

      this.lastArgMax = argMax;
      return new Tensor(out, [c, outH, outW]);
    }

    backward(gradOutPut: Tensor): Tensor{
        const gradInput = Tensor.zeros(this.lastInputShape!);
        const argMax = this.lastArgMax!;

        for(let idx = 0; idx < argMax.length; idx++){
            gradInput.data[argMax[idx]!]! += gradOutPut.data[idx]!;
        }

        return gradInput;
    }
}
