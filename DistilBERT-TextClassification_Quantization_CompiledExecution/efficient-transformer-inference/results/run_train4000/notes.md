# Run: Fine-tuned DistilBERT with 4,000 SST-2 Training Examples

## Training Setup

- Base model: distilbert-base-uncased
- Train examples: 4,000
- Validation examples: 872
- Epochs: 1
- Batch size: 8
- Max sequence length: 128
- Device: CPU

## Main Baseline Result

- Validation accuracy: 0.8784
- Validation loss: 0.2777

## Benchmark Summary

The fine-tuned FP32 baseline, dynamic quantized model, and compiled FP32 model were benchmarked on the SST-2 validation split.

Dynamic quantization reduced model size from about 255 MB to about 132 MB and improved latency across all tested batch sizes.

torch.compile preserved accuracy but introduced first-call compile overhead.
