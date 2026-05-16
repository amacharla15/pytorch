# Final Benchmark Summary

Benchmark device: CPU

## Results

| Variant | Batch Size | Accuracy | Latency ms | Throughput samples/sec | Model Size MB | Compile First Call sec |
|---|---:|---:|---:|---:|---:|---:|
| baseline_fp32 | 8 | 0.9048 | 1142.6128 | 7.0015 | 255.4521 | 0.0000 |
| dynamic_quantized_int8 | 8 | 0.9002 | 467.5593 | 17.1101 | 132.2892 | 0.0000 |

## Notes

- Baseline FP32 is the original fine-tuned DistilBERT sentiment model.
- Dynamic quantization is CPU-focused and applied to Linear layers.
- Compiled execution reports the first compiled call separately from steady-state latency.
- All variants use the same validation split, max length, batch sizes, warmup steps, and measured steps.

## Compile Errors

- Batch size 8: CalledProcessError: Command '['/usr/bin/gcc', '/tmp/tmp89a5ace8/cuda_utils.c', '-O3', '-shared', '-fPIC', '-Wno-psabi', '-o', '/tmp/tmp89a5ace8/cuda_utils.cpython-312-x86_64-linux-gnu.so', '-l:libcuda.so.1', '-L/research/amacharla/venvs/distilbert/lib/python3.12/site-packages/triton/backends/nvidia/lib', '-L/lib/x86_64-linux-gnu', '-I/research/amacharla/venvs/distilbert/lib/python3.12/site-packages/triton/backends/nvidia/include', '-I/tmp/tmp89a5ace8', '-I/usr/include/python3.12']' returned non-zero exit status 1.

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

