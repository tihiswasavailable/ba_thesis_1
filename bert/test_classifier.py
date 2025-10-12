# Quick test in Python terminal
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load your model (update the path!)
model_path = "cookie_classifier_20250731_094404"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Test a cookie button
text = "Einverstanden"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs).item()

labels = {0: 'PRIVACY_FRIENDLY', 1: 'NEUTRAL', 2: 'PRIVACY_RISKY'}
print(f"'{text}' -> {labels[pred]} ({probs[0][pred]:.3f})")