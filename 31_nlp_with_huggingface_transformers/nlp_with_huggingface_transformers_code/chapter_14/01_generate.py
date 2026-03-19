from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Define some example sentences
sentences = [
    "The cat sat on the mat.",
    "The dog slept on the floor.",
    "I love natural language processing."
]

def get_embeddings(sentences, model, tokenizer):
    "Function to get embeddings for a batch of sentences"

    # Tokenize input and get model output
    encoded_input = tokenizer(sentences, padding=True, truncation=True,
                              return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Use the CLS token embedding as the sentence embedding
    sentence_embeddings = model_output.last_hidden_state[:, 0, :]

    # Convert torch tensor to numpy array for easier handling
    return sentence_embeddings.numpy()

# Get embeddings for our example sentences
embeddings = get_embeddings(sentences, model, tokenizer)
print(f"Embedding shape: {embeddings.shape}")
print("First 5 dimensions of the sentences' embeddings:")
print(f"{np.round(embeddings[:, :5], 3)}")
