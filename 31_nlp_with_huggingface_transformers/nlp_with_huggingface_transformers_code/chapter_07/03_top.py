from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The best way to learn programming is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text with different top_k values
top_k_values = [5, 20, 50]
print(f"Prompt: {prompt}")

for top_k in top_k_values:
    print()
    print(f"Top-K = {top_k}")
    output = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=1.0,
        top_k=top_k,
        top_p=1.0,
        repetition_penalty=1.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated Text:")
    print(generated_text)

# Generate text with different top_p values
top_p_values = [0.5, 0.7, 0.9]
for top_p in top_p_values:
    print()
    print(f"Top-P = {top_p}")
    output = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=1.0,
        top_k=0,
        top_p=top_p,
        repetition_penalty=1.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated Text:")
    print(generated_text)
