import argparse
import os
import json

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

try:
    from torch.ao.quantization import quantize_dynamic
except Exception:
    from torch.quantization import quantize_dynamic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-limit", type=int, default=872)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def build_loader(tokenizer, batch_size, validation_limit, max_length):
    raw = load_dataset("nyu-mll/glue", "sst2")
    dataset = raw["validation"]

    if validation_limit > 0:
        dataset = dataset.select(range(min(validation_limit, len(dataset))))

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    tokenized = dataset.map(tokenize_function, batched=True)

    sentences = tokenized["sentence"]
    labels = tokenized["label"]
    idxs = tokenized["idx"]

    tokenized = tokenized.rename_column("label", "labels")
    tokenized = tokenized.remove_columns(["sentence", "idx"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    loader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )

    return loader, sentences, labels, idxs


def get_outputs(model, loader):
    model.eval()

    all_logits = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            inputs = {}
            inputs["input_ids"] = batch["input_ids"]
            inputs["attention_mask"] = batch["attention_mask"]

            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_logits.extend(logits.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    return all_logits, all_preds


def margin(logits):
    return abs(float(logits[0]) - float(logits[1]))


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    fp32_model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    fp32_model.eval()

    int8_model = quantize_dynamic(
        fp32_model,
        {torch.nn.Linear},
        dtype=torch.qint8,
        inplace=False
    )
    int8_model.eval()

    loader, sentences, labels, idxs = build_loader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        validation_limit=args.validation_limit,
        max_length=args.max_length
    )

    fp32_logits, fp32_preds = get_outputs(fp32_model, loader)
    int8_logits, int8_preds = get_outputs(int8_model, loader)

    rows = []

    for i in range(len(labels)):
        label = int(labels[i])
        fp32_pred = int(fp32_preds[i])
        int8_pred = int(int8_preds[i])

        fp32_correct = fp32_pred == label
        int8_correct = int8_pred == label
        flipped = fp32_pred != int8_pred

        row = {
            "idx": int(idxs[i]),
            "sentence": sentences[i],
            "label": label,
            "fp32_pred": fp32_pred,
            "int8_pred": int8_pred,
            "fp32_correct": fp32_correct,
            "int8_correct": int8_correct,
            "flipped": flipped,
            "fp32_logit_0": fp32_logits[i][0],
            "fp32_logit_1": fp32_logits[i][1],
            "int8_logit_0": int8_logits[i][0],
            "int8_logit_1": int8_logits[i][1],
            "fp32_margin": margin(fp32_logits[i]),
            "int8_margin": margin(int8_logits[i]),
            "abs_logit0_diff": abs(fp32_logits[i][0] - int8_logits[i][0]),
            "abs_logit1_diff": abs(fp32_logits[i][1] - int8_logits[i][1])
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    total = len(df)
    fp32_acc = df["fp32_correct"].mean()
    int8_acc = df["int8_correct"].mean()
    flipped_count = int(df["flipped"].sum())

    fp32_correct_int8_wrong = int(((df["fp32_correct"] == True) & (df["int8_correct"] == False)).sum())
    fp32_wrong_int8_correct = int(((df["fp32_correct"] == False) & (df["int8_correct"] == True)).sum())
    both_correct = int(((df["fp32_correct"] == True) & (df["int8_correct"] == True)).sum())
    both_wrong = int(((df["fp32_correct"] == False) & (df["int8_correct"] == False)).sum())

    flipped_df = df[df["flipped"] == True]
    not_flipped_df = df[df["flipped"] == False]

    summary = {
        "model_dir": args.model_dir,
        "batch_size": args.batch_size,
        "validation_examples": total,
        "fp32_accuracy": float(fp32_acc),
        "int8_accuracy": float(int8_acc),
        "flipped_predictions": flipped_count,
        "fp32_correct_int8_wrong": fp32_correct_int8_wrong,
        "fp32_wrong_int8_correct": fp32_wrong_int8_correct,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "avg_fp32_margin_flipped": float(flipped_df["fp32_margin"].mean()) if len(flipped_df) > 0 else None,
        "avg_fp32_margin_not_flipped": float(not_flipped_df["fp32_margin"].mean()) if len(not_flipped_df) > 0 else None,
        "avg_abs_logit0_diff": float(df["abs_logit0_diff"].mean()),
        "avg_abs_logit1_diff": float(df["abs_logit1_diff"].mean())
    }

    summary_path = args.output_csv.replace(".csv", "_summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("Saved CSV:", args.output_csv)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
