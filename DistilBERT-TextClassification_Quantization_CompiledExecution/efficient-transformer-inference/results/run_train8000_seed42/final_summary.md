# Final Benchmark Summary

Benchmark device: CPU

## Results

| Variant | Batch Size | Accuracy | Latency ms | Throughput samples/sec | Model Size MB | Compile First Call sec |
|---|---:|---:|---:|---:|---:|---:|
| baseline_fp32 | 1 | 0.8784 | 105.7857 | 9.4531 | 255.4521 | 0.0000 |
| baseline_fp32 | 8 | 0.8784 | 423.6985 | 18.8813 | 255.4521 | 0.0000 |
| baseline_fp32 | 16 | 0.8784 | 786.0494 | 20.3550 | 255.4521 | 0.0000 |
| baseline_fp32 | 32 | 0.8784 | 1585.5513 | 20.1823 | 255.4521 | 0.0000 |
| dynamic_quantized_int8 | 1 | 0.8727 | 28.4266 | 35.1783 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 8 | 0.8589 | 195.4667 | 40.9277 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 16 | 0.8589 | 387.6030 | 41.2794 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 32 | 0.8589 | 886.0685 | 36.1146 | 132.2892 | 0.0000 |

## Notes

- Baseline FP32 is the original fine-tuned DistilBERT sentiment model.
- Dynamic quantization is CPU-focused and applied to Linear layers.
- Compiled execution reports the first compiled call separately from steady-state latency.
- All variants use the same validation split, max length, batch sizes, warmup steps, and measured steps.
