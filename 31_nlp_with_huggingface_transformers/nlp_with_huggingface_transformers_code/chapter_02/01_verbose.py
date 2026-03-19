import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model_name = "KernAI/stock-news-distilbert"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

text = "Machine Learning Mastery is a nice website."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
