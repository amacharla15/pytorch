# Final Benchmark Summary

Benchmark device: CPU

## Results

| Variant | Batch Size | Accuracy | Latency ms | Throughput samples/sec | Model Size MB | Compile First Call sec |
|---|---:|---:|---:|---:|---:|---:|
| baseline_fp32 | 8 | 0.9048 | 435.2838 | 18.3788 | 255.4521 | 0.0000 |
| dynamic_quantized_int8 | 8 | 0.8968 | 238.4011 | 33.5569 | 132.2892 | 0.0000 |
| compiled_fp32 | 8 | 0.9048 | 394.0482 | 20.3021 | 255.4540 | 81.5165 |

## Notes

- Baseline FP32 is the original fine-tuned DistilBERT sentiment model.
- Dynamic quantization is CPU-focused and applied to Linear layers.
- Compiled execution reports the first compiled call separately from steady-state latency.
- All variants use the same validation split, max length, batch sizes, warmup steps, and measured steps.
