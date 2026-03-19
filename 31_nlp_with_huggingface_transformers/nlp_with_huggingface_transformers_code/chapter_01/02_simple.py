import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
input_ids = tokenizer("Hello, world!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**input_ids)
output_tokens = outputs.logits.argmax(dim=-1)
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(output_text)
