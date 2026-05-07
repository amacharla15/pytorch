# Efficient Transformer Inference for Sentiment Classification

## Project Overview

This project studies inference optimization for transformer-based sentiment classification. We fine-tuned a DistilBERT model on SST-2, then compared three inference variants:

1. FP32 baseline
2. Dynamic quantized INT8 model
3. Compiled FP32 model using `torch.compile`

The goal was to evaluate the tradeoff between accuracy, latency, throughput, model size, and compile overhead.

## Dataset

We used SST-2 from the GLUE benchmark.

Task:

- Binary sentiment classification

Labels:

- 0: negative
- 1: positive

Validation set size:

- 872 examples

## Model

Base model:

- `distilbert-base-uncased`

Final selected checkpoint:

- `checkpoints/our_finetuned_distilbert_sst2_train4000_seed42`

Training setup:

- Training examples: 4,000
- Validation examples: 872
- Epochs: 1
- Batch size: 8
- Max sequence length: 128
- Seed: 42
- Device: CPU

## Project Pipeline

Raw sentence  
→ tokenizer  
→ input_ids / attention_mask  
→ DistilBERT classifier  
→ logits  
→ sentiment prediction  

## Optimization Methods

### 1. FP32 Baseline

The FP32 baseline is the fine-tuned DistilBERT model before inference optimization.

### 2. Dynamic Quantization

Dynamic quantization was applied to Linear layers. This reduced model size and improved CPU inference latency.

### 3. Compiled Execution

`torch.compile` was applied to the FP32 model. Compile first-call overhead was reported separately from steady-state latency.

## Final Results

Final model:

- `checkpoints/our_finetuned_distilbert_sst2_train4000_seed42`

Validation set:

- SST-2 validation split, 872 examples

Warmup steps:

- 5

Measured steps:

- 20

Device:

- CPU

| Variant | Batch Size | Accuracy | Correct / 872 | Latency ms | Throughput samples/sec | Model Size MB | Compile First Call |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 baseline | 1 | 0.8750 | 763 | 55.36 | 18.06 | 255.45 | 0.00 s |
| FP32 baseline | 8 | 0.8750 | 763 | 489.61 | 16.34 | 255.45 | 0.00 s |
| FP32 baseline | 16 | 0.8750 | 763 | 835.28 | 19.16 | 255.45 | 0.00 s |
| FP32 baseline | 32 | 0.8750 | 763 | 1721.33 | 18.59 | 255.45 | 0.00 s |
| Dynamic quantized INT8 | 1 | 0.8761 | 764 | 28.10 | 35.59 | 132.29 | 0.00 s |
| Dynamic quantized INT8 | 8 | 0.8761 | 764 | 205.63 | 38.91 | 132.29 | 0.00 s |
| Dynamic quantized INT8 | 16 | 0.8739 | 762 | 449.39 | 35.60 | 132.29 | 0.00 s |
| Dynamic quantized INT8 | 32 | 0.8658 | 755 | 910.00 | 35.16 | 132.29 | 0.00 s |
| Compiled FP32 | 1 | 0.8750 | 763 | 57.92 | 17.27 | 255.45 | 66.11 s |
| Compiled FP32 | 8 | 0.8750 | 763 | 360.21 | 22.21 | 255.45 | 3.50 s |
| Compiled FP32 | 16 | 0.8750 | 763 | 759.67 | 21.06 | 255.45 | 2.67 s |
| Compiled FP32 | 32 | 0.8750 | 763 | 1528.40 | 20.94 | 255.45 | 3.24 s |

## Main Batch-8 Comparison

| Variant | Accuracy | Correct / 872 | Latency ms | Speedup vs FP32 | Throughput | Model Size | Size Reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 baseline | 0.8750 | 763 | 489.61 | 1.00x | 16.34/s | 255.45 MB | - |
| Dynamic quantized INT8 | 0.8761 | 764 | 205.63 | 2.38x | 38.91/s | 132.29 MB | 48.2% smaller |
| Compiled FP32 | 0.8750 | 763 | 360.21 | 1.36x | 22.21/s | 255.45 MB | none |

## Quantization Diagnostic

The quantized model had one more correct validation example than the FP32 model at batch size 8. This should not be interpreted as a real accuracy improvement.

Prediction-level diagnostics showed:

- INT8 changed 29 out of 872 predictions
- 14 FP32-correct examples became wrong
- 15 FP32-wrong examples became correct

Therefore, dynamic quantization preserved aggregate accuracy approximately, but it did not preserve identical predictions.

## Key Findings

1. Dynamic quantization gave the strongest practical improvement.
   - Model size reduced from 255.45 MB to 132.29 MB.
   - Batch-8 latency reduced from 489.61 ms to 205.63 ms.
   - Aggregate accuracy stayed effectively unchanged.

2. `torch.compile` preserved accuracy and improved steady-state latency.
   - Batch-8 latency reduced from 489.61 ms to 360.21 ms.
   - However, compile first-call overhead must be considered.

3. Quantization affected mostly low-margin predictions.
   - The quantized model changed some predictions near the decision boundary.
   - The small accuracy difference was not a meaningful model-quality improvement.

## How to Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt

Inspect SST-2:

python3 src/phase1_2_inspect_sst2.py

Inspect tokenizer:

python3 src/phase1_3_tokenizer_inspection.py

Inspect DataLoader:

python3 src/phase1_data_pipeline.py

Train final baseline:

python3 src/train_baseline_seeded.py \
  --output-dir checkpoints/our_finetuned_distilbert_sst2_train4000_seed42 \
  --train-limit 4000 \
  --validation-limit 872 \
  --epochs 1 \
  --batch-size 8 \
  --log-every 25 \
  --seed 42

Run final benchmark:

python3 src/benchmark_all.py \
  --model-dir checkpoints/our_finetuned_distilbert_sst2_train4000_seed42 \
  --batch-sizes 1,8,16,32 \
  --warmup 5 \
  --steps 20 \
  --validation-limit 872

Run quantization diagnostics:

python3 src/compare_fp32_int8_predictions.py \
  --model-dir checkpoints/our_finetuned_distilbert_sst2_train4000_seed42 \
  --batch-size 8 \
  --validation-limit 872 \
  --output-csv results/diagnostics_4000_seed42_bs8.csv
Repository Structure
src/
  phase1_2_inspect_sst2.py
  phase1_3_tokenizer_inspection.py
  phase1_data_pipeline.py
  train_baseline.py
  train_baseline_seeded.py
  benchmark_all.py
  compare_fp32_int8_predictions.py

results/
  final_train4000_seed42/
  final_results.csv
  final_summary.md

figures/
  latency_comparison.png
  throughput_comparison.png
  model_size_comparison.png
  accuracy_comparison.png

checkpoints/
  our_finetuned_distilbert_sst2_train4000_seed42/
Limitations
Experiments were CPU-based.
Timing can vary by hardware and system load.
Dynamic quantization changed some predictions, even when aggregate accuracy stayed nearly the same.
torch.compile has first-call compile overhead.
The project used a subset of the SST-2 training set due to time and hardware constraints.
Final Conclusion

Dynamic quantization was the best optimization in this experiment. It reduced model size by about 48% and improved batch-8 latency by about 2.38x while keeping aggregate accuracy effectively unchanged. Compiled execution also improved steady-state latency but introduced compile overhead.
