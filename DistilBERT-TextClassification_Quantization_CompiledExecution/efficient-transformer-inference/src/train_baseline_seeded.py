import os
import json
import time
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from torch.optim import AdamW


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=str, default="checkpoints/our_finetuned_distilbert_sst2")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-limit", type=int, default=4000)
    parser.add_argument("--validation-limit", type=int, default=872)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def tokenize_dataset(tokenizer, max_length, train_limit, validation_limit):
    raw_datasets = load_dataset("nyu-mll/glue", "sst2")

    train_dataset = raw_datasets["train"]
    validation_dataset = raw_datasets["validation"]

    if train_limit > 0:
        train_limit = min(train_limit, len(train_dataset))
        train_dataset = train_dataset.select(range(train_limit))

    if validation_limit > 0:
        validation_limit = min(validation_limit, len(validation_dataset))
        validation_dataset = validation_dataset.select(range(validation_limit))

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=max_length
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    validation_dataset = validation_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    validation_dataset = validation_dataset.rename_column("label", "labels")

    remove_train_cols = []
    for col in ["sentence", "idx"]:
        if col in train_dataset.column_names:
            remove_train_cols.append(col)

    remove_val_cols = []
    for col in ["sentence", "idx"]:
        if col in validation_dataset.column_names:
            remove_val_cols.append(col)

    train_dataset = train_dataset.remove_columns(remove_train_cols)
    validation_dataset = validation_dataset.remove_columns(remove_val_cols)

    return train_dataset, validation_dataset


def build_loaders(train_dataset, validation_dataset, tokenizer, batch_size):
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    return train_loader, validation_loader


def move_batch(batch, device):
    moved = {}
    for key in batch:
        if key == "token_type_ids":
            continue
        moved[key] = batch[key].to(device)
    return moved


def evaluate(model, validation_loader, device):
    model.eval()

    total_loss = 0.0
    total_steps = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in validation_loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)

            loss = outputs.loss
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].numel()
            total_loss += loss.item()
            total_steps += 1

    if total_steps == 0:
        avg_loss = 0.0
    else:
        avg_loss = total_loss / total_steps

    if total == 0:
        accuracy = 0.0
    else:
        accuracy = correct / total

    return avg_loss, accuracy



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def train(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    print("Base model:", args.model_name)
    print("Output dir:", args.output_dir)
    print("Seed:", args.seed)
    print("Train limit:", args.train_limit)
    print("Validation limit:", args.validation_limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset, validation_dataset = tokenize_dataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit
    )

    train_loader, validation_loader = build_loaders(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )

    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate
    )

    history = []

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()

        total_train_loss = 0.0
        total_train_steps = 0

        print()
        print("Epoch", epoch + 1, "of", args.epochs)

        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_steps += 1

            if (step + 1) % args.log_every == 0:
                avg_train_loss = total_train_loss / total_train_steps
                print("step:", step + 1, "avg_train_loss:", avg_train_loss)

        if total_train_steps == 0:
            avg_train_loss = 0.0
        else:
            avg_train_loss = total_train_loss / total_train_steps

        validation_loss, validation_accuracy = evaluate(
            model=model,
            validation_loader=validation_loader,
            device=device
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy
        }

        history.append(row)

        print("epoch_train_loss:", avg_train_loss)
        print("validation_loss:", validation_loss)
        print("validation_accuracy:", validation_accuracy)

    end_time = time.time()

    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

    summary = {
        "base_model": args.model_name,
        "output_dir": args.output_dir,
        "train_limit": args.train_limit,
        "validation_limit": args.validation_limit,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "device": str(device),
        "seed": args.seed,
        "training_time_seconds": end_time - start_time,
        "history": history
    }

    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Saved our fine-tuned model to:", args.output_dir)
    print("Saved training summary to:", os.path.join(args.output_dir, "training_summary.json"))


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
