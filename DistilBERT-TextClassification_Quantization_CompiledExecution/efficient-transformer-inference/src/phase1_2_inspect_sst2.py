from datasets import load_dataset

raw_datasets = load_dataset("nyu-mll/glue", "sst2")

print("DATASET OBJECT")
print(raw_datasets)
print()

print("AVAILABLE SPLITS")
for split_name in raw_datasets.keys():
    print(split_name)
print()

print("TRAIN FEATURES")
print(raw_datasets["train"].features)
print()

print("NUMBER OF EXAMPLES")
print("train:", len(raw_datasets["train"]))
print("validation:", len(raw_datasets["validation"]))
print("test:", len(raw_datasets["test"]))
print()

print("FIRST TRAINING EXAMPLE")
first_example = raw_datasets["train"][0]
print(first_example)
print()

print("FIRST FIVE TRAINING EXAMPLES")
for i in range(5):
    example = raw_datasets["train"][i]

    sentence = example["sentence"]
    label = example["label"]
    idx = example["idx"]

    if label == 0:
        label_name = "negative"
    else:
        label_name = "positive"

    print("Example index:", i)
    print("Dataset idx:", idx)
    print("Sentence:", sentence)
    print("Label:", label)
    print("Label meaning:", label_name)
    print()