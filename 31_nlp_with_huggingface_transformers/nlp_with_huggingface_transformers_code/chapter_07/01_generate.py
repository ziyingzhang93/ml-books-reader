from transformers import GPT2LMHeadModel, GPT2Tokenizer

# create model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# tokenize input prompt to sequence of ids
prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")
# generate output as a sequence of token ids
output = model.generate(
    **inputs,
    max_length=50,
    num_return_sequences=1,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.0,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
# convert token ids into text strings
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Prompt: {prompt}")
print("Generated Text:")
print(generated_text)
