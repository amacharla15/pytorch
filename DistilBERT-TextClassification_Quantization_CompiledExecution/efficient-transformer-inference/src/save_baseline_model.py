import os
import json
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
CHECKPOINT_DIR = "checkpoints/baseline_distilbert_sst2"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(CHECKPOINT_DIR)
model.save_pretrained(CHECKPOINT_DIR)

model.eval()

samples = [
    "This movie was excellent and emotional.",
    "The acting was boring and the story was weak.",
    "I really enjoyed the film.",
    "This was a complete waste of time."
]

predictions = []

with torch.no_grad():
    for text in samples:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        outputs = model(**encoded)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1).item()

        if predicted == 0:
            label = "negative"
        else:
            label = "positive"

        predictions.append({
            "text": text,
            "predicted_label_id": predicted,
            "predicted_label": label,
            "logits": logits.squeeze(0).tolist()
        })

with open(os.path.join(CHECKPOINT_DIR, "sample_predictions.json"), "w") as f:
    json.dump(predictions, f, indent=2)

print("Saved baseline model and tokenizer to:", CHECKPOINT_DIR)
print()
for item in predictions:
    print(item["text"])
    print("prediction:", item["predicted_label"])
    print("logits:", item["logits"])
    print()
