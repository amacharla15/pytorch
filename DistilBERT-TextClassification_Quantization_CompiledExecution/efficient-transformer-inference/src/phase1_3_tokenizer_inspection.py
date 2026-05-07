from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128

raw_datasets = load_dataset("nyu-mll/glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

example = raw_datasets["train"][0]
sentence = example["sentence"]
label = example["label"]

encoded = tokenizer(
    sentence,
    truncation=True,
    max_length=MAX_LENGTH
)

input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]
tokens = tokenizer.convert_ids_to_tokens(input_ids)
decoded = tokenizer.decode(input_ids)

print("MODEL_NAME:", MODEL_NAME)
print("RAW SENTENCE:")
print(sentence)
print()
print("LABEL:", label)
print()
print("TOKENS:")
print(tokens)
print()
print("INPUT IDS:")
print(input_ids)
print()
print("ATTENTION MASK:")
print(attention_mask)
print()
print("DECODED TEXT:")
print(decoded)
print()
print("SEQUENCE LENGTH:", len(input_ids))
