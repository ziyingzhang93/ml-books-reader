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
    "Function to get embeddings for a batch of sentences with mean pooling"

    # Tokenize input and get model output
    encoded_input = tokenizer(sentences, padding=True, truncation=True,
                              return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Extract the attention mask and output sequence
    attention_mask = encoded_input["attention_mask"]
    output_seq = model_output.last_hidden_state

    # Mean pooling: take the average of all token embeddings
    mask = attention_mask.unsqueeze(-1).expand(output_seq.size()).float()
    sum_embeddings = (output_seq * mask).sum(1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask

    # Convert torch tensor to numpy array for easier handling
    return mean_pooled.numpy()

# Get embeddings with mean pooling
embeddings = get_embeddings(sentences, model, tokenizer)
print(f"Embedding shape: {embeddings.shape}")
print("First 5 dimensions of the sentences' embeddings:")
print(f"{np.round(embeddings[:, :5], 3)}")
