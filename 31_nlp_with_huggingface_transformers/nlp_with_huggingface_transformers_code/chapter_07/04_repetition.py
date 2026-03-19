from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "Once upon a time, there was a"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text with different repetition penalties
penalties = [1.0, 1.2, 1.5, 2.0]
print(f"Prompt: {prompt}")
for penalty in penalties:
    print()
    print(f"Repetition penalty: {penalty}")
    output = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.3,
        top_k=50,
        top_p=1.0,
        repetition_penalty=penalty,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated Text:")
    print(generated_text)
