# Final Benchmark Summary

Benchmark device: CPU

## Results

| Variant | Batch Size | Accuracy | Latency ms | Throughput samples/sec | Model Size MB | Compile First Call sec |
|---|---:|---:|---:|---:|---:|---:|
| baseline_fp32 | 1 | 0.8750 | 55.3627 | 18.0627 | 255.4521 | 0.0000 |
| baseline_fp32 | 8 | 0.8750 | 489.6071 | 16.3396 | 255.4521 | 0.0000 |
| baseline_fp32 | 16 | 0.8750 | 835.2836 | 19.1552 | 255.4521 | 0.0000 |
| baseline_fp32 | 32 | 0.8750 | 1721.3339 | 18.5902 | 255.4521 | 0.0000 |
| dynamic_quantized_int8 | 1 | 0.8761 | 28.0990 | 35.5885 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 8 | 0.8761 | 205.6288 | 38.9051 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 16 | 0.8739 | 449.3901 | 35.6038 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 32 | 0.8658 | 910.0038 | 35.1647 | 132.2892 | 0.0000 |
| compiled_fp32 | 1 | 0.8750 | 57.9191 | 17.2655 | 255.4540 | 66.1096 |
| compiled_fp32 | 8 | 0.8750 | 360.2087 | 22.2093 | 255.4540 | 3.5003 |
| compiled_fp32 | 16 | 0.8750 | 759.6697 | 21.0618 | 255.4540 | 2.6702 |
| compiled_fp32 | 32 | 0.8750 | 1528.3973 | 20.9370 | 255.4540 | 3.2385 |

## Notes

- Baseline FP32 is the original fine-tuned DistilBERT sentiment model.
- Dynamic quantization is CPU-focused and applied to Linear layers.
- Compiled execution reports the first compiled call separately from steady-state latency.
- All variants use the same validation split, max length, batch sizes, warmup steps, and measured steps.
