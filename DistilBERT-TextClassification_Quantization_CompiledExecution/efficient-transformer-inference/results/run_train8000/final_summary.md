# Final Benchmark Summary

Benchmark device: CPU

## Results

| Variant | Batch Size | Accuracy | Latency ms | Throughput samples/sec | Model Size MB | Compile First Call sec |
|---|---:|---:|---:|---:|---:|---:|
| baseline_fp32 | 1 | 0.8796 | 65.8257 | 15.1916 | 255.4521 | 0.0000 |
| baseline_fp32 | 8 | 0.8796 | 505.7588 | 15.8178 | 255.4521 | 0.0000 |
| baseline_fp32 | 16 | 0.8796 | 960.0355 | 16.6661 | 255.4521 | 0.0000 |
| baseline_fp32 | 32 | 0.8796 | 1837.7051 | 17.4130 | 255.4521 | 0.0000 |
| dynamic_quantized_int8 | 1 | 0.8452 | 30.2849 | 33.0197 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 8 | 0.8349 | 228.6829 | 34.9829 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 16 | 0.8360 | 470.5488 | 34.0028 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 32 | 0.8326 | 1005.7655 | 31.8166 | 132.2892 | 0.0000 |
| compiled_fp32 | 1 | 0.8796 | 59.4256 | 16.8278 | 255.4540 | 43.2540 |
| compiled_fp32 | 8 | 0.8796 | 411.0318 | 19.4632 | 255.4540 | 3.7716 |
| compiled_fp32 | 16 | 0.8796 | 772.3930 | 20.7148 | 255.4540 | 2.5230 |
| compiled_fp32 | 32 | 0.8796 | 1659.9610 | 19.2776 | 255.4540 | 3.4040 |

## Notes

- Baseline FP32 is the original fine-tuned DistilBERT sentiment model.
- Dynamic quantization is CPU-focused and applied to Linear layers.
- Compiled execution reports the first compiled call separately from steady-state latency.
- All variants use the same validation split, max length, batch sizes, warmup steps, and measured steps.
