from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8

raw_datasets = load_dataset("nyu-mll/glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=MAX_LENGTH
    )

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"]
validation_dataset = tokenized_datasets["validation"]

train_dataset = train_dataset.rename_column("label", "labels")
validation_dataset = validation_dataset.rename_column("label", "labels")

train_dataset = train_dataset.remove_columns(["sentence", "idx"])
validation_dataset = validation_dataset.remove_columns(["sentence", "idx"])

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator
)

batch = next(iter(train_loader))

print("TRAIN DATASET:")
print(train_dataset)
print()
print("VALIDATION DATASET:")
print(validation_dataset)
print()
print("BATCH KEYS:")
print(batch.keys())
print()
print("input_ids shape:", batch["input_ids"].shape)
print("attention_mask shape:", batch["attention_mask"].shape)
print("labels shape:", batch["labels"].shape)
print()
print("input_ids:")
print(batch["input_ids"])
print()
print("attention_mask:")
print(batch["attention_mask"])
print()
print("labels:")
print(batch["labels"])
