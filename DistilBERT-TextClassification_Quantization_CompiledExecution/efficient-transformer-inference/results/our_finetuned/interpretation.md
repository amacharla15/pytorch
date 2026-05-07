# Interpretation of Fine-Tuned DistilBERT Optimization Results

## Baseline

We fine-tuned DistilBERT on SST-2 using 4,000 training examples and evaluated it on the full 872-example validation split.

The fine-tuned FP32 baseline reached validation accuracy of 0.8784.

This FP32 model is the main baseline for the project.

## Dynamic Quantization

Dynamic quantization reduced the model size from about 255 MB to about 132 MB.

It also reduced latency across all tested batch sizes:

| Batch Size | FP32 Latency ms | Quantized Latency ms |
|---:|---:|---:|
| 1 | 56.36 | 28.61 |
| 8 | 440.35 | 207.49 |
| 16 | 779.31 | 415.60 |
| 32 | 1758.99 | 908.53 |

The quantized model kept similar accuracy while giving the strongest speed and size improvements.

## Compiled Execution

torch.compile preserved the same accuracy as the FP32 baseline.

However, it introduced first-call compile overhead:

| Batch Size | Compile First Call seconds |
|---:|---:|
| 1 | 80.92 |
| 8 | 9.92 |
| 16 | 2.35 |
| 32 | 3.34 |

Steady-state latency improved for some larger batch sizes, but the improvement was smaller than dynamic quantization.

## Main Conclusion

Dynamic quantization was the most effective optimization in this CPU-based experiment. It gave a much smaller model and nearly doubled inference throughput in several batch settings.

torch.compile was useful for studying compiled execution tradeoffs, but its first-call overhead means it is only attractive when the model will run many repeated inferences after compilation.

## Deployment Takeaway

For CPU inference with this DistilBERT sentiment classifier, dynamic quantization is the better practical optimization. For long-running services with repeated calls and stable input shapes, compiled execution may still be worth considering.
