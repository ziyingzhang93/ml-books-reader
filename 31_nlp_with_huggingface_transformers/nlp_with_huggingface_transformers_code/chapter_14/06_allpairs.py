from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define some example sentences
sentences = [
    "The cat sat on the mat.",
    "The dog slept on the floor.",
    "I love natural language processing."
]

# Load a pre-trained model and generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

# Get embeddings with mean pooling
print(f"Embedding shape: {embeddings.shape}")
print("First 5 dimensions of the sentences' embeddings:")
print(f"{np.round(embeddings[:, :5], 3)}")

# Calculate cosine similarity between the all pairs
print(cosine_similarity(embeddings, embeddings).round(3))
