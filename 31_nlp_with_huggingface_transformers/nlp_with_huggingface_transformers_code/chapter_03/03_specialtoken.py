from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "I love machine learning!"

# Complete tokenization with special tokens
encoded = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    padding="max_length",
    max_length=10,
    return_tensors="pt"
)

print("Full encoded sequence:")
for token_id, token in zip(
    encoded["input_ids"][0],
    tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
):
    print(f"{token}: {token_id}")
