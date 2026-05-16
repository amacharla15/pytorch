# Final Benchmark Summary

Benchmark device: CPU

## Results

| Variant | Batch Size | Accuracy | Latency ms | Throughput samples/sec | Model Size MB | Compile First Call sec |
|---|---:|---:|---:|---:|---:|---:|
| baseline_fp32 | 1 | 0.9048 | 31.5884 | 31.6572 | 255.4521 | 0.0000 |
| baseline_fp32 | 8 | 0.9048 | 1021.8287 | 7.8291 | 255.4521 | 0.0000 |
| baseline_fp32 | 16 | 0.9048 | 1023.0628 | 15.6393 | 255.4521 | 0.0000 |
| baseline_fp32 | 32 | 0.9048 | 1784.1051 | 17.9362 | 255.4521 | 0.0000 |
| dynamic_quantized_int8 | 1 | 0.9014 | 49.3588 | 20.2598 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 8 | 0.9002 | 310.6186 | 25.7551 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 16 | 0.9094 | 178.3275 | 89.7226 | 132.2892 | 0.0000 |
| dynamic_quantized_int8 | 32 | 0.9060 | 1130.7629 | 28.2995 | 132.2892 | 0.0000 |

## Notes

- Baseline FP32 is the original fine-tuned DistilBERT sentiment model.
- Dynamic quantization is CPU-focused and applied to Linear layers.
- Compiled execution reports the first compiled call separately from steady-state latency.
- All variants use the same validation split, max length, batch sizes, warmup steps, and measured steps.

## Compile Errors

- Batch size 1: CalledProcessError: Command '['/usr/bin/gcc', '/tmp/tmpio04zfyp/cuda_utils.c', '-O3', '-shared', '-fPIC', '-Wno-psabi', '-o', '/tmp/tmpio04zfyp/cuda_utils.cpython-312-x86_64-linux-gnu.so', '-l:libcuda.so.1', '-L/research/amacharla/venvs/distilbert/lib/python3.12/site-packages/triton/backends/nvidia/lib', '-L/lib/x86_64-linux-gnu', '-I/research/amacharla/venvs/distilbert/lib/python3.12/site-packages/triton/backends/nvidia/include', '-I/tmp/tmpio04zfyp', '-I/usr/include/python3.12']' returned non-zero exit status 1.

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- Batch size 8: CalledProcessError: Command '['/usr/bin/gcc', '/tmp/tmpo3ruiote/cuda_utils.c', '-O3', '-shared', '-fPIC', '-Wno-psabi', '-o', '/tmp/tmpo3ruiote/cuda_utils.cpython-312-x86_64-linux-gnu.so', '-l:libcuda.so.1', '-L/research/amacharla/venvs/distilbert/lib/python3.12/site-packages/triton/backends/nvidia/lib', '-L/lib/x86_64-linux-gnu', '-I/research/amacharla/venvs/distilbert/lib/python3.12/site-packages/triton/backends/nvidia/include', '-I/tmp/tmpo3ruiote', '-I/usr/include/python3.12']' returned non-zero exit status 1.

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- Batch size 16: CalledProcessError: Command '['/usr/bin/gcc', '/tmp/tmpl78axjhi/cuda_utils.c', '-O3', '-shared', '-fPIC', '-Wno-psabi', '-o', '/tmp/tmpl78axjhi/cuda_utils.cpython-312-x86_64-linux-gnu.so', '-l:libcuda.so.1', '-L/research/amacharla/venvs/distilbert/lib/python3.12/site-packages/triton/backends/nvidia/lib', '-L/lib/x86_64-linux-gnu', '-I/research/amacharla/venvs/distilbert/lib/python3.12/site-packages/triton/backends/nvidia/include', '-I/tmp/tmpl78axjhi', '-I/usr/include/python3.12']' returned non-zero exit status 1.

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- Batch size 32: CalledProcessError: Command '['/usr/bin/gcc', '/tmp/tmpag95hizy/cuda_utils.c', '-O3', '-shared', '-fPIC', '-Wno-psabi', '-o', '/tmp/tmpag95hizy/cuda_utils.cpython-312-x86_64-linux-gnu.so', '-l:libcuda.so.1', '-L/research/amacharla/venvs/distilbert/lib/python3.12/site-packages/triton/backends/nvidia/lib', '-L/lib/x86_64-linux-gnu', '-I/research/amacharla/venvs/distilbert/lib/python3.12/site-packages/triton/backends/nvidia/include', '-I/tmp/tmpag95hizy', '-I/usr/include/python3.12']' returned non-zero exit status 1.

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

