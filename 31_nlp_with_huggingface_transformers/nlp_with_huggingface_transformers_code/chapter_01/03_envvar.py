import os
os.environ["HF_TOKEN"] = "hf_YourTokenHere"
os.environ["HF_HOME"] = "~/.cache/huggingface"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
input_ids = tokenizer("Hello, world!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**input_ids)
output_tokens = outputs.logits.argmax(dim=-1)
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(output_text)
