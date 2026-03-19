from transformers import GPT2LMHeadModel, GPT2Config

config = GPT2Config()
model = GPT2LMHeadModel(config=config)
print(model)
