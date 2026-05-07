# Final Benchmark Summary

Benchmark device: CPU

## Results

| Variant | Batch Size | Accuracy | Latency ms | Throughput samples/sec | Model Size MB | Compile First Call sec |
|---|---:|---:|---:|---:|---:|---:|
| baseline_fp32 | 1 | 0.8784 | 56.3650 | 17.7415 | 255.4521 | 0.0000 |
| baseline_fp32 | 8 | 0.8784 | 440.3507 | 18.1673 | 255.4521 | 0.0000 |
| baseline_fp32 | 16 | 0.8784 | 779.3066 | 20.5311 | 255.4521 | 0.0000 |
| baseline_fp32 | 32 | 0.8784 | 1758.9950 | 18.1922 | 255.4521 | 0.0000 |
| dynamic_quantized_int8 | 1 | 0.8704 | 28.6115 | 34.9510 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 8 | 0.8784 | 207.4890 | 38.5563 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 16 | 0.8727 | 415.5956 | 38.4990 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 32 | 0.8704 | 908.5334 | 35.2216 | 132.2892 | 0.0000 |
| compiled_fp32 | 1 | 0.8784 | 58.6618 | 17.0469 | 255.4540 | 80.9181 |
| compiled_fp32 | 8 | 0.8784 | 357.3456 | 22.3873 | 255.4540 | 9.9219 |
| compiled_fp32 | 16 | 0.8784 | 735.1557 | 21.7641 | 255.4540 | 2.3471 |
| compiled_fp32 | 32 | 0.8784 | 1552.7003 | 20.6093 | 255.4540 | 3.3364 |

## Notes

- Baseline FP32 is the original fine-tuned DistilBERT sentiment model.
- Dynamic quantization is CPU-focused and applied to Linear layers.
- Compiled execution reports the first compiled call separately from steady-state latency.
- All variants use the same validation split, max length, batch sizes, warmup steps, and measured steps.
