import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
text = "Pre-trained models are available."
tokens = encoding.encode(text)
print(f"Token IDs: {tokens}")
print(f"Tokens: {[encoding.decode_single_token_bytes(t) for t in tokens]}")
print(f"Decoded: {encoding.decode(tokens)}")
