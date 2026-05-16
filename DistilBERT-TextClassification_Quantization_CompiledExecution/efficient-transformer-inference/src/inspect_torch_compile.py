import os
import time
import json
import contextlib

import torch
import torch._dynamo
from torch.profiler import profile
from torch.profiler import ProfilerActivity
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader


MODEL_DIR = "checkpoints/our_finetuned_distilbert_sst2_train4000_seed42"
OUTPUT_DIR = "results/compiler_inspection"
BATCH_SIZE = 8
MAX_LENGTH = 128
VALIDATION_LIMIT = 32


def build_loader(tokenizer):
    raw = load_dataset("nyu-mll/glue", "sst2")
    dataset = raw["validation"].select(range(VALIDATION_LIMIT))

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.remove_columns(["sentence", "idx"])

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt"
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator
    )

    return loader


def clean_batch(batch):
    inputs = {}
    inputs["input_ids"] = batch["input_ids"]
    inputs["attention_mask"] = batch["attention_mask"]
    return inputs


def run_forward(model, batch):
    inputs = clean_batch(batch)
    outputs = model(**inputs)
    return outputs.logits


def write_text(path, text):
    with open(path, "w") as f:
        f.write(str(text))


def try_dynamo_explain(model, batch):
    explain_path = os.path.join(OUTPUT_DIR, "dynamo_explain.txt")

    inputs = clean_batch(batch)

    try:
        explanation = torch._dynamo.explain(model)(**inputs)
        write_text(explain_path, explanation)
        print("Saved Dynamo explain:", explain_path)
    except Exception as e:
        write_text(explain_path, "Dynamo explain failed:\n" + repr(e))
        print("Dynamo explain failed. Saved error:", explain_path)


def capture_graph_with_custom_backend(model, batch):
    graph_dir = os.path.join(OUTPUT_DIR, "captured_graphs")
    os.makedirs(graph_dir, exist_ok=True)

    counter = {"n": 0}

    def debug_backend(gm, example_inputs):
        index = counter["n"]
        counter["n"] += 1

        graph_path = os.path.join(graph_dir, f"graph_{index}.txt")
        code_path = os.path.join(graph_dir, f"graph_{index}_code.py")

        with open(graph_path, "w") as f:
            f.write(str(gm.graph))

        with open(code_path, "w") as f:
            f.write(gm.code)

        print("Captured graph:", graph_path)
        print("Captured graph code:", code_path)

        return gm.forward

    try:
        torch._dynamo.reset()
        compiled_debug_model = torch.compile(
            model,
            backend=debug_backend,
            fullgraph=False
        )

        with torch.no_grad():
            _ = run_forward(compiled_debug_model, batch)

        summary = {
            "captured_graph_count": counter["n"],
            "note": "These are FX graphs captured by Dynamo using a custom backend. This backend captures graphs but does not apply Inductor optimization."
        }

        with open(os.path.join(OUTPUT_DIR, "graph_capture_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    except Exception as e:
        write_text(
            os.path.join(OUTPUT_DIR, "graph_capture_error.txt"),
            repr(e)
        )


def profile_model(model, batch, name):
    model.eval()

    table_path = os.path.join(OUTPUT_DIR, f"{name}_profile_table.txt")
    trace_path = os.path.join(OUTPUT_DIR, f"{name}_trace.json")

    with torch.no_grad():
        for _ in range(3):
            _ = run_forward(model, batch)

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        with torch.no_grad():
            for _ in range(5):
                _ = run_forward(model, batch)

    table = prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=30
    )

    write_text(table_path, table)
    prof.export_chrome_trace(trace_path)

    print("Saved profile table:", table_path)
    print("Saved trace:", trace_path)


def measure_first_and_second_call(model, batch):
    compiled = torch.compile(model)

    with torch.no_grad():
        start = time.perf_counter()
        _ = run_forward(compiled, batch)
        end = time.perf_counter()
        first_call = end - start

        start = time.perf_counter()
        _ = run_forward(compiled, batch)
        end = time.perf_counter()
        second_call = end - start

    result = {
        "first_compiled_call_seconds": first_call,
        "second_compiled_call_seconds": second_call,
        "note": "First call includes compilation overhead. Later calls reuse compiled artifacts when assumptions still hold."
    }

    path = os.path.join(OUTPUT_DIR, "compile_first_second_call.json")

    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print("Saved first/second compiled call timing:", path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    loader = build_loader(tokenizer)
    batch = next(iter(loader))

    batch_info = {
        "batch_size": int(batch["input_ids"].shape[0]),
        "sequence_length": int(batch["input_ids"].shape[1]),
        "input_ids_shape": list(batch["input_ids"].shape),
        "attention_mask_shape": list(batch["attention_mask"].shape),
        "model_dir": MODEL_DIR
    }

    with open(os.path.join(OUTPUT_DIR, "batch_info.json"), "w") as f:
        json.dump(batch_info, f, indent=2)

    print("Running Dynamo explain...")
    try_dynamo_explain(model, batch)

    print("Capturing Dynamo graphs with custom backend...")
    capture_graph_with_custom_backend(model, batch)

    print("Measuring first vs second compiled call...")
    measure_first_and_second_call(model, batch)

    print("Profiling FP32 eager model...")
    profile_model(model, batch, "fp32_eager")

    print("Profiling compiled model...")
    compiled_model = torch.compile(model)
    with torch.no_grad():
        _ = run_forward(compiled_model, batch)
    profile_model(compiled_model, batch, "compiled_fp32")

    print("Done. Check:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
