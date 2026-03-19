from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text with different temperature values
temperatures = [0.2, 0.5, 1.0, 1.5]
print(f"Prompt: {prompt}")
for temp in temperatures:
    print()
    print(f"Temperature: {temp}")
    output = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=temp,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated Text:")
    print(generated_text)
