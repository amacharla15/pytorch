from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

MODEL_NAME = "distilbert-base-uncased"
print(MODEL_NAME)

raw_dataset = load_dataset(
    "csv",
    data_files={
        "train": "toy_dataset/train.csv",
        "validation": "toy_dataset/validation.csv",
        "test": "toy_dataset/test.csv",
    }
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


print(raw_dataset["train"][0])

example = raw_dataset["train"][0]
sentence = example["sentence"]
label=example["label"]

print(sentence)
print(label)



tokenized_output=tokenizer(sentence,truncation=True, max_length=32)
print(tokenized_output)

print(tokenized_output["input_ids"])
print(tokenized_output["attention_mask"])


tokens = tokenizer.convert_ids_to_tokens(tokenized_output["input_ids"])
print(tokens)



def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=32
    )


tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)


print(tokenized_dataset["train"][0])


train_dataset = train_dataset.rename_column("label", "labels")
validation_dataset = validation_dataset.rename_column("label", "labels")

train_dataset = train_dataset.remove_columns(["sentence", "idx"])
validation_dataset = validation_dataset.remove_columns(["sentence", "idx"])

print(tokenized_dataset["train"][0])


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