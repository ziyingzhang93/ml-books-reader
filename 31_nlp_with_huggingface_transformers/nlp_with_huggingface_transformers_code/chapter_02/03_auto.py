import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "ArthurZ/opt-350m-dummy-sc" # or "KernAI/stock-news-distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "Machine Learning Mastery is a nice website."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
