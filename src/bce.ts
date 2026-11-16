import { clamp } from "./maths";
import { Tensor } from "./tensor";

const EPS = 1e-7;

export class BinaryCrossEntropy {
    private lastPrediction?: Tensor;
    private lastTarget?: Tensor;

    forward(
        prediction: Tensor,
        target: Tensor,
    ) {
     this.lastPrediction = prediction;
     this.lastTarget = target;

     let total = 0;
     for(let i = 0; i < prediction.data.length; i++) {
        const p = clamp(prediction.data[i], EPS, 1 - EPS);
        const y = target.data[i];
        total += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
     }

      return total / prediction.data.length;
    }

    backward(): Tensor {
        if(this.lastPrediction === null || this.lastTarget === null) {
            throw new Error("Forward method must be called before backward method.");
        }
        const prediction = this.lastPrediction!;
        const target = this.lastTarget;
        const n = prediction?.data.length;

        return prediction.map((pRaw, i) => {
            const p = clamp(pRaw, EPS, 1 - EPS);
            const y = target?.data[i]!;
            return (p - y) / (p * (1 - p) * n!);
        });
    }
}