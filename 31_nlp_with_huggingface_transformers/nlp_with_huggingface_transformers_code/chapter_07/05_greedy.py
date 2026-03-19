from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The secret to happiness is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text with greedy decoding vs. sampling
print(f"Prompt: {prompt}\n")
print("Greedy Decoding (do_sample=False):")
output = model.generate(
    **inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.0,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
print()
print("Sampling (do_sample=True):")
output = model.generate(
    **inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.0,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
