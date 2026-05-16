import os
import time
import json
import csv

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = "checkpoints/our_finetuned_distilbert_sst2_full_seed42"
OUTPUT_DIR = "results/compile_profiler_full"
BATCH_SIZE = 8
MAX_LENGTH = 128


def write_text(path, text):
    with open(path, "w") as f:
        f.write(str(text))


def export_top_ops(prof, path, limit=30):
    rows = []
    for item in prof.key_averages():
        rows.append({
            "name": item.key,
            "self_cpu_time_total_us": item.self_cpu_time_total,
            "cpu_time_total_us": item.cpu_time_total,
            "self_cpu_memory_usage": item.self_cpu_memory_usage,
            "cpu_memory_usage": item.cpu_memory_usage,
            "count": item.count,
        })

    rows.sort(key=lambda x: x["self_cpu_time_total_us"], reverse=True)
    rows = rows[:limit]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "self_cpu_time_total_us",
                "cpu_time_total_us",
                "self_cpu_memory_usage",
                "cpu_memory_usage",
                "count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_batch(tokenizer):
    texts = [
        "This movie was excellent and emotional.",
        "The acting was boring and weak.",
        "I really enjoyed the film.",
        "The story was slow and disappointing.",
        "The performances were powerful.",
        "This was a terrible movie.",
        "The film was smart, funny, and moving.",
        "I would not recommend this movie.",
    ]

    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return encoded["input_ids"], encoded["attention_mask"]


def run_forward(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits


def profile_model(model, input_ids, attention_mask, name):
    model.eval()

    with torch.no_grad():
        for _ in range(5):
            _ = run_forward(model, input_ids, attention_mask)

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                with record_function(name + "_forward"):
                    _ = run_forward(model, input_ids, attention_mask)

    table = prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=40,
    )

    write_text(os.path.join(OUTPUT_DIR, f"{name}_profile_table.txt"), table)
    export_top_ops(prof, os.path.join(OUTPUT_DIR, f"{name}_top_ops.csv"))
    prof.export_chrome_trace(os.path.join(OUTPUT_DIR, f"{name}_trace.json"))

    return table


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    input_ids, attention_mask = make_batch(tokenizer)

    batch_info = {
        "model_dir": MODEL_DIR,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "input_ids_shape": list(input_ids.shape),
        "attention_mask_shape": list(attention_mask.shape),
    }

    with open(os.path.join(OUTPUT_DIR, "batch_info.json"), "w") as f:
        json.dump(batch_info, f, indent=2)

    print("Profiling eager FP32...")
    profile_model(model, input_ids, attention_mask, "eager_fp32")

    print("Compiling model...")
    compiled_model = torch.compile(model)

    with torch.no_grad():
        start = time.perf_counter()
        _ = run_forward(compiled_model, input_ids, attention_mask)
        end = time.perf_counter()
        first_call_seconds = end - start

    print("First compiled call seconds:", first_call_seconds)

    print("Profiling compiled FP32 steady state...")
    profile_model(compiled_model, input_ids, attention_mask, "compiled_fp32")

    summary = {
        "first_compiled_call_seconds": first_call_seconds,
        "note": "First compiled call includes compilation overhead. Profile tables are steady-state after warmup.",
    }

    with open(os.path.join(OUTPUT_DIR, "compile_profile_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved profiler outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
