import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("results/final_train4000_seed42/final_results.csv")

# 1. Batch-8 latency bar chart
batch8 = df[df["batch_size"] == 8].copy()

plt.figure(figsize=(8, 5))
plt.bar(batch8["variant"], batch8["avg_latency_ms"])
plt.ylabel("Latency per Batch (ms)")
plt.title("Batch-8 Latency Comparison")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("figures/presentation_batch8_latency.png", dpi=200)
plt.close()

# 2. Batch-8 throughput bar chart
plt.figure(figsize=(8, 5))
plt.bar(batch8["variant"], batch8["throughput_samples_per_sec"])
plt.ylabel("Throughput (samples/sec)")
plt.title("Batch-8 Throughput Comparison")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("figures/presentation_batch8_throughput.png", dpi=200)
plt.close()

# 3. Model size bar chart
size_df = batch8[["variant", "model_size_mb"]].copy()

plt.figure(figsize=(8, 5))
plt.bar(size_df["variant"], size_df["model_size_mb"])
plt.ylabel("Model Size (MB)")
plt.title("Model Size Comparison")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("figures/presentation_model_size.png", dpi=200)
plt.close()

# 4. Correct predictions bar chart
total = 872
batch8["correct_predictions"] = (batch8["accuracy"] * total).round().astype(int)

plt.figure(figsize=(8, 5))
plt.bar(batch8["variant"], batch8["correct_predictions"])
plt.ylabel("Correct Predictions out of 872")
plt.title("Batch-8 Accuracy as Correct Predictions")
plt.ylim(740, 872)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("figures/presentation_correct_predictions.png", dpi=200)
plt.close()

print("Saved presentation figures:")
print("figures/presentation_batch8_latency.png")
print("figures/presentation_batch8_throughput.png")
print("figures/presentation_model_size.png")
print("figures/presentation_correct_predictions.png")
