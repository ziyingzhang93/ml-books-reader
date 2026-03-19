import torch
from sentence_transformers import SentenceTransformer, util

texts = [
  "The stock market reached a new high today, with technology stocks leading the gains.",
  "A new study shows that regular exercise can reduce the risk of heart disease.",
  "The latest smartphone from Apple features a better camera and longer battery life.",
  "Scientists have discovered a new species of frog in the Amazon rainforest."
]
categories = ["Business", "Health", "Technology", "Science"]

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
text_embeddings = model.encode(texts, convert_to_tensor=True)
category_embeddings = model.encode(categories, convert_to_tensor=True)

# Calculate cosine similarity between texts and categories
similarities = util.cos_sim(text_embeddings, category_embeddings)

# Get the most similar category for each text
best_categories = torch.argmax(similarities, dim=1)
for i, text in enumerate(texts):
    category = categories[best_categories[i]]
    similarity = similarities[i][best_categories[i]].item()
    print(f"Text: {text}")
    print(f"Category: {category} (Similarity: {similarity:.4f})\n")
