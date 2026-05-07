# Final Model Selection Decision

## Candidate Models

We trained and benchmarked three main fine-tuned DistilBERT checkpoints:

1. 4,000-example fine-tuned model
2. 8,000-example fine-tuned model
3. 8,000-example fine-tuned model with fixed seed 42

## Main Result

The 4,000-example fine-tuned model was selected as the final project baseline.

## Reason

The 4,000-example model gave the cleanest inference optimization result:

| Variant | Batch Size | Accuracy | Latency ms | Model Size MB |
|---|---:|---:|---:|---:|
| FP32 baseline | 8 | 0.8784 | 440.35 | 255.45 |
| Dynamic quantized | 8 | 0.8784 | 207.49 | 132.29 |
| Compiled FP32 | 8 | 0.8784 | 357.35 | 255.45 |

Dynamic quantization nearly halved latency and reduced model size while preserving accuracy at the main batch size.

## 8,000-Example Experiment

The 8,000-example models achieved similar FP32 accuracy but were less robust after dynamic quantization.

Unseeded 8,000 model:

- FP32 batch-8 accuracy: 0.8796
- Quantized batch-8 accuracy: 0.8349

Seeded 8,000 model:

- FP32 batch-8 accuracy: 0.8784
- Quantized batch-8 accuracy: 0.8589

## Conclusion

More training examples did not meaningfully improve FP32 accuracy in this setup, and the 8,000-example checkpoints were less stable under dynamic quantization. Therefore, the 4,000-example model is the best final model for demonstrating inference optimization tradeoffs.
