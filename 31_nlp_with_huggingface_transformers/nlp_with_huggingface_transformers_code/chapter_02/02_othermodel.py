import torch
from transformers import GPT2Tokenizer, OPTForSequenceClassification

model_name = "ArthurZ/opt-350m-dummy-sc"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = OPTForSequenceClassification.from_pretrained(model_name)

text = "Machine Learning Mastery is a nice website."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
