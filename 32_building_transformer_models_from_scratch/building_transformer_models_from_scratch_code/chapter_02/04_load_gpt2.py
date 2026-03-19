from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
print(model)
