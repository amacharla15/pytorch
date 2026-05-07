import os
import gc
import time
import json
import argparse
import tempfile

import psutil
import torch
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

try:
    from torch.ao.quantization import quantize_dynamic
except Exception:
    from torch.quantization import quantize_dynamic


DEFAULT_MODEL_DIR = "checkpoints/baseline_distilbert_sst2"
DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--fallback-model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-sizes", type=str, default="1,8,16,32")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--validation-limit", type=int, default=872)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--num-threads", type=int, default=0)
    return parser.parse_args()


def get_model_source(args):
    if os.path.isdir(args.model_dir):
        return args.model_dir
    return args.fallback_model_name


def load_tokenizer(args):
    source = get_model_source(args)
    return AutoTokenizer.from_pretrained(source)


def load_fp32_model(args):
    source = get_model_source(args)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    model.eval()
    return model


def build_validation_loader(tokenizer, batch_size, max_length, validation_limit):
    raw_datasets = load_dataset("nyu-mll/glue", "sst2")
    dataset = raw_datasets["validation"]

    if validation_limit > 0:
        limit = min(validation_limit, len(dataset))
        dataset = dataset.select(range(limit))

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    dataset = dataset.map(tokenize_function, batched=True)

    if "label" in dataset.column_names:
        dataset = dataset.rename_column("label", "labels")

    remove_cols = []
    for col in ["sentence", "idx"]:
        if col in dataset.column_names:
            remove_cols.append(col)

    dataset = dataset.remove_columns(remove_cols)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    return loader


def move_batch_to_device(batch, device):
    moved = {}
    for key in batch:
        moved[key] = batch[key].to(device)
    return moved


def forward_logits(model, batch, device):
    batch = move_batch_to_device(batch, device)
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"]
    )
    return outputs.logits, batch["labels"]


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            logits, labels = forward_logits(model, batch, device)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

    if total == 0:
        return 0.0

    return correct / total


def get_next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_latency(model, loader, device, warmup_steps, measured_steps):
    model.eval()
    iterator = iter(loader)

    with torch.no_grad():
        for _ in range(warmup_steps):
            batch, iterator = get_next_batch(iterator, loader)
            _ = forward_logits(model, batch, device)

        synchronize_if_needed(device)

        latencies = []
        total_samples = 0

        for _ in range(measured_steps):
            batch, iterator = get_next_batch(iterator, loader)
            batch_size = batch["input_ids"].shape[0]

            start = time.perf_counter()
            _ = forward_logits(model, batch, device)
            synchronize_if_needed(device)
            end = time.perf_counter()

            latencies.append(end - start)
            total_samples += batch_size

    total_time = sum(latencies)

    if len(latencies) == 0:
        avg_latency_ms = 0.0
    else:
        avg_latency_ms = (total_time / len(latencies)) * 1000.0

    if total_time == 0:
        throughput = 0.0
    else:
        throughput = total_samples / total_time

    return avg_latency_ms, throughput


def model_size_mb(model):
    with tempfile.NamedTemporaryFile(delete=True) as f:
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name)
    return size / (1024 * 1024)


def rss_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_one_variant(variant_name, model, tokenizer, args, batch_sizes, device, compile_first_call_by_batch_size=None):
    rows = []

    model.to(device)
    model.eval()

    size_mb = model_size_mb(model)

    for batch_size in batch_sizes:
        loader = build_validation_loader(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=args.max_length,
            validation_limit=args.validation_limit
        )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        memory_before = rss_memory_mb()

        accuracy = evaluate_accuracy(model, loader, device)

        memory_after_accuracy = rss_memory_mb()

        avg_latency_ms, throughput = benchmark_latency(
            model=model,
            loader=loader,
            device=device,
            warmup_steps=args.warmup,
            measured_steps=args.steps
        )

        memory_after_benchmark = rss_memory_mb()

        compile_first_call_seconds = 0.0
        if compile_first_call_by_batch_size is not None:
            compile_first_call_seconds = compile_first_call_by_batch_size.get(batch_size, 0.0)

        row = {
            "variant": variant_name,
            "batch_size": batch_size,
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency_ms,
            "throughput_samples_per_sec": throughput,
            "model_size_mb": size_mb,
            "rss_memory_before_mb": memory_before,
            "rss_memory_after_accuracy_mb": memory_after_accuracy,
            "rss_memory_after_benchmark_mb": memory_after_benchmark,
            "compile_first_call_seconds": compile_first_call_seconds,
            "validation_examples": args.validation_limit,
            "max_length": args.max_length,
            "warmup_steps": args.warmup,
            "measured_steps": args.steps
        }

        rows.append(row)

        print()
        print("RESULT")
        print(json.dumps(row, indent=2))

    return rows


def make_quantized_model(args):
    model = load_fp32_model(args)
    model.cpu()
    model.eval()
    qmodel = quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
        inplace=False
    )
    qmodel.eval()
    return qmodel


def benchmark_compiled(args, tokenizer, batch_sizes):
    rows = []
    errors = []

    for batch_size in batch_sizes:
        device = torch.device("cpu")

        try:
            model = load_fp32_model(args)
            model.to(device)
            model.eval()

            compiled_model = torch.compile(model)
            loader = build_validation_loader(
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=args.max_length,
                validation_limit=args.validation_limit
            )

            batch = next(iter(loader))

            start = time.perf_counter()
            with torch.no_grad():
                _ = forward_logits(compiled_model, batch, device)
            end = time.perf_counter()

            compile_first_call_seconds = end - start

            compile_dict = {batch_size: compile_first_call_seconds}

            compiled_rows = run_one_variant(
                variant_name="compiled_fp32",
                model=compiled_model,
                tokenizer=tokenizer,
                args=args,
                batch_sizes=[batch_size],
                device=device,
                compile_first_call_by_batch_size=compile_dict
            )

            rows.extend(compiled_rows)

            del model
            del compiled_model
            gc.collect()

        except Exception as e:
            error = {
                "variant": "compiled_fp32",
                "batch_size": batch_size,
                "error": str(e)
            }
            errors.append(error)
            print()
            print("COMPILE ERROR")
            print(json.dumps(error, indent=2))

    return rows, errors


def make_graphs(df):
    os.makedirs("figures", exist_ok=True)

    metrics = [
        ("avg_latency_ms", "Average Latency (ms)", "figures/latency_comparison.png"),
        ("throughput_samples_per_sec", "Throughput (samples/sec)", "figures/throughput_comparison.png"),
        ("model_size_mb", "Model Size (MB)", "figures/model_size_comparison.png"),
        ("accuracy", "Accuracy", "figures/accuracy_comparison.png")
    ]

    for metric, title, path in metrics:
        plt.figure()

        for variant in df["variant"].unique():
            temp = df[df["variant"] == variant].sort_values("batch_size")
            plt.plot(temp["batch_size"], temp[metric], marker="o", label=variant)

        plt.xlabel("Batch Size")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


def write_markdown_summary(df, compile_errors):
    os.makedirs("results", exist_ok=True)

    lines = []
    lines.append("# Final Benchmark Summary")
    lines.append("")
    lines.append("Benchmark device: CPU")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Variant | Batch Size | Accuracy | Latency ms | Throughput samples/sec | Model Size MB | Compile First Call sec |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for _, row in df.iterrows():
        lines.append(
            f"| {row['variant']} | {int(row['batch_size'])} | "
            f"{row['accuracy']:.4f} | "
            f"{row['avg_latency_ms']:.4f} | "
            f"{row['throughput_samples_per_sec']:.4f} | "
            f"{row['model_size_mb']:.4f} | "
            f"{row['compile_first_call_seconds']:.4f} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Baseline FP32 is the original fine-tuned DistilBERT sentiment model.")
    lines.append("- Dynamic quantization is CPU-focused and applied to Linear layers.")
    lines.append("- Compiled execution reports the first compiled call separately from steady-state latency.")
    lines.append("- All variants use the same validation split, max length, batch sizes, warmup steps, and measured steps.")
    lines.append("")

    if len(compile_errors) > 0:
        lines.append("## Compile Errors")
        lines.append("")
        for item in compile_errors:
            lines.append(f"- Batch size {item['batch_size']}: {item['error']}")
        lines.append("")

    with open("results/final_summary.md", "w") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    batch_sizes = []
    for item in args.batch_sizes.split(","):
        item = item.strip()
        if item:
            batch_sizes.append(int(item))

    device = torch.device("cpu")

    print("Using device:", device)
    print("Batch sizes:", batch_sizes)
    print("Model source:", get_model_source(args))

    tokenizer = load_tokenizer(args)

    all_rows = []
    compile_errors = []

    print()
    print("RUNNING BASELINE FP32")
    baseline_model = load_fp32_model(args)
    baseline_rows = run_one_variant(
        variant_name="baseline_fp32",
        model=baseline_model,
        tokenizer=tokenizer,
        args=args,
        batch_sizes=batch_sizes,
        device=device
    )
    all_rows.extend(baseline_rows)
    del baseline_model
    gc.collect()

    print()
    print("RUNNING DYNAMIC QUANTIZED")
    quantized_model = make_quantized_model(args)
    torch.save(quantized_model.state_dict(), "checkpoints/dynamic_quantized_state_dict.pt")
    quantized_rows = run_one_variant(
        variant_name="dynamic_quantized_int8",
        model=quantized_model,
        tokenizer=tokenizer,
        args=args,
        batch_sizes=batch_sizes,
        device=device
    )
    all_rows.extend(quantized_rows)
    del quantized_model
    gc.collect()

    if not args.disable_compile:
        print()
        print("RUNNING COMPILED FP32")
        compiled_rows, compile_errors = benchmark_compiled(
            args=args,
            tokenizer=tokenizer,
            batch_sizes=batch_sizes
        )
        all_rows.extend(compiled_rows)

    df = pd.DataFrame(all_rows)
    df.to_csv("results/final_results.csv", index=False)

    make_graphs(df)
    write_markdown_summary(df, compile_errors)

    with open("results/compile_errors.json", "w") as f:
        json.dump(compile_errors, f, indent=2)

    print()
    print("DONE")
    print("Saved:")
    print("- results/final_results.csv")
    print("- results/final_summary.md")
    print("- figures/latency_comparison.png")
    print("- figures/throughput_comparison.png")
    print("- figures/model_size_comparison.png")
    print("- figures/accuracy_comparison.png")
    print("- checkpoints/dynamic_quantized_state_dict.pt")


if __name__ == "__main__":
    main()
