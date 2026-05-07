# Experiment Comparison: 4,000 vs 8,000 Fine-Tuning Examples

## Summary

We fine-tuned DistilBERT on SST-2 using two training sizes: 4,000 examples and 8,000 examples. Both models were evaluated on the full 872-example validation split.

## Baseline Accuracy

| Training Examples | FP32 Validation Accuracy |
|---:|---:|
| 4,000 | 0.8784 |
| 8,000 | 0.8796 |

The 8,000-example model improved FP32 accuracy by only about 0.11 percentage points.

## Main Optimization Result: 4,000-example Model

| Variant | Batch Size | Accuracy | Latency ms | Throughput samples/sec | Model Size MB |
|---|---:|---:|---:|---:|---:|
| FP32 baseline | 8 | 0.8784 | 440.35 | 18.17 | 255.45 |
| Dynamic quantized | 8 | 0.8784 | 207.49 | 38.56 | 132.29 |
| Compiled FP32 | 8 | 0.8784 | 357.35 | 22.39 | 255.45 |

Dynamic quantization gave the strongest result for the 4,000-example model: it reduced model size and latency while preserving accuracy at batch size 8.

## Additional Experiment: 8,000-example Model

| Variant | Batch Size | Accuracy | Latency ms | Throughput samples/sec | Model Size MB |
|---|---:|---:|---:|---:|---:|
| FP32 baseline | 8 | 0.8796 | 505.76 | 15.82 | 255.45 |
| Dynamic quantized | 8 | 0.8349 | 228.68 | 34.98 | 132.29 |
| Compiled FP32 | 8 | 0.8796 | 411.03 | 19.46 | 255.45 |

The 8,000-example model slightly improved FP32 accuracy, but dynamic quantization caused a larger accuracy drop. This shows that optimization must be evaluated empirically instead of assuming every optimized model preserves accuracy.

## Final Decision

The 4,000-example fine-tuned model is the best main result for the project because it demonstrates the clearest inference optimization tradeoff:

- FP32 baseline accuracy is strong enough.
- Dynamic quantization nearly halves latency.
- Dynamic quantization reduces model size from about 255 MB to about 132 MB.
- Accuracy is preserved at the main batch size.
- torch.compile preserves accuracy but has first-call compile overhead.

The 8,000-example model is included as an additional experiment.
