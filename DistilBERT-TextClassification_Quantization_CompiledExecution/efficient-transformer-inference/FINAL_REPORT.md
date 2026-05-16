# Final Report: Efficient Transformer Inference for Sentiment Classification

## 1. Introduction

The goal of this project was to optimize inference for a transformer-based sentiment classifier. We fine-tuned DistilBERT on SST-2 and compared three inference variants: the FP32 baseline, a dynamic quantized INT8 model, and a compiled FP32 model using 
