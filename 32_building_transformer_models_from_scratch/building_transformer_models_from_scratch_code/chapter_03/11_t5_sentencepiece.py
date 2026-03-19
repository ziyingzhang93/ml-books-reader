from transformers import T5Tokenizer

# Load the T5 tokenizer (which uses SentencePiece+Unigram)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Tokenize a text
text = "SentencePiece is a subword tokenizer used in models such as XLNet and T5."
tokens = tokenizer.encode(text)
print(f"Token IDs: {tokens}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
print(f"Decoded: {tokenizer.decode(tokens)}")
