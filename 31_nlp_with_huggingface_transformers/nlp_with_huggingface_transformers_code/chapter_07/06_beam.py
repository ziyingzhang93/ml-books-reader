from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The key to successful machine learning is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text with greedy decoding vs. sampling
print(f"Prompt: {prompt}\n")
outputs = model.generate(
    **inputs,
    num_beams=5,             # Number of beams to use
    early_stopping=True,     # Stop when all beams have finished
    no_repeat_ngram_size=2,  # Avoid repeating n-grams
    num_return_sequences=3,  # Return multiple sequences
    max_length=100,
    temperature=1.5,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
for idx, output in enumerate(outputs):
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Generated Text ({idx+1}):")
    print(generated_text)
