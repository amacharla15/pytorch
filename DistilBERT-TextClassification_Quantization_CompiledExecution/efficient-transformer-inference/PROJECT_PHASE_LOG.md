
## Completed After Fine-Tuning

### Phase 2 - Baseline DistilBERT setup
Created:
- src/train_baseline.py

Completed:
- Loaded distilbert-base-uncased
- Added sequence classification head with num_labels=2
- Verified training works with a 64-example smoke test

### Phase 3 - Fine-tuned our baseline model
Completed:
- Fine-tuned DistilBERT on 4,000 SST-2 training examples
- Evaluated on 872 SST-2 validation examples
- Saved our checkpoint to checkpoints/our_finetuned_distilbert_sst2

Result:
- Validation accuracy: 0.8784

### Phase 4 - Baseline FP32 benchmark
Completed:
- Benchmarked our fine-tuned FP32 model on CPU
- Batch sizes: 1, 8, 16, 32
- Metrics: accuracy, latency, throughput, model size, memory

### Phase 5 - Dynamic quantization
Completed:
- Applied dynamic quantization to Linear layers
- Benchmarked the quantized model using the same settings

Main result:
- Model size reduced from about 255 MB to about 132 MB
- Latency improved significantly across tested batch sizes

### Phase 6 - torch.compile
Completed:
- Applied torch.compile to the fine-tuned FP32 model
- Measured compile first-call cost separately
- Benchmarked steady-state latency

Main result:
- Accuracy stayed the same
- Compile overhead was significant
- Steady-state speed improved for some larger batch sizes

### Phase 7 - Results saved
Created:
- results/final_results.csv
- results/final_summary.md
- figures/latency_comparison.png
- figures/throughput_comparison.png
- figures/model_size_comparison.png
- figures/accuracy_comparison.png

Backed up to:
- results/our_finetuned/
