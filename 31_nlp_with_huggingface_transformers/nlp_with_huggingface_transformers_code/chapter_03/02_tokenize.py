from transformers import BertTokenizer

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Text example
text = "I love machine learning!"

# Tokenize the text
tokens = tokenizer.tokenize(text)
print(f"Original text: {text}")
print(f"Tokenized text: {tokens}")

# Convert tokens to IDs
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Token IDs: {input_ids}")
