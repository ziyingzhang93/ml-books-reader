import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

texts_with_categories = [
    {"category": "Business", "text": "The stock market reached a new high today."},
    {"category": "Business", "text": "Investors are optimistic about the economy."},
    {"category": "Business", "text": "The company reported strong quarterly earnings."},
    {"category": "Business", "text":
       "The central bank has decided to keep interest rates unchanged."},
    {"category": "Health", "text":
      "A new study shows that regular exercise can reduce the risk of heart disease."},
    {"category": "Health", "text":
      "A balanced diet is essential for maintaining good health."},
    {"category": "Health", "text":
      "The new vaccine has been approved for use against the flu."},
    {"category": "Health", "text": "Sleep is important for physical and mental health."},
    {"category": "Technology", "text":
      "The latest smartphone features a better camera and longer battery life."},
    {"category": "Technology", "text":
      "The new laptop has a faster processor and more memory."},
    {"category": "Technology", "text":
      "The software update includes new security features."},
    {"category": "Technology", "text":
      "5G networks promise faster internet speeds for mobile devices."},
    {"category": "Science", "text":
      "Scientists have discovered a new species in the Amazon rainforest."},
    {"category": "Science", "text":
      "Astronomers have observed a supernova in a distant galaxy."},
    {"category": "Science", "text":
      "The Mars rover has sent back new images of the planet's surface."},
    {"category": "Science", "text":
      "Researchers have developed a new method for measuring ocean temperatures."}
]

# Extract texts and categories
texts = [item["text"] for item in texts_with_categories]
categories = [item["category"] for item in texts_with_categories]

# Generate embeddings, then reduce dimension with t-SNE
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

tsne = TSNE(n_components=2, perplexity=5, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings)

# Define colors for categories
unique_categories = list(set(categories))
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_categories)))
category_to_color = {cat: color for cat, color in zip(unique_categories, colors)}

# Create a scatter plot
plt.figure(figsize=(10, 8))
for i, (x, y) in enumerate(reduced_embeddings):
    category = categories[i]
    color = category_to_color[category]
    plt.scatter(x, y, color=color, alpha=0.7)
    plt.annotate(texts[i][:20] + "...", (x, y), fontsize=8)

# Add legend, mark the axes
for category, color in category_to_color.items():
    plt.scatter([], [], color=color, label=category)
plt.legend()
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("t-SNE Visualization of Text Embeddings")
plt.tight_layout()
plt.show()
