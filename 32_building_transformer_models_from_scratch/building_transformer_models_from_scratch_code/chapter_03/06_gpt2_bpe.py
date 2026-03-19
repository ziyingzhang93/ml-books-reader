from transformers import GPT2Tokenizer

# Load the GPT-2 tokenizer (which uses BPE)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tokenize a text
text = "Pre-trained models are available."
tokens = tokenizer.encode(text)
print(f"Token IDs: {tokens}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
print(f"Decoded: {tokenizer.decode(tokens)}")
