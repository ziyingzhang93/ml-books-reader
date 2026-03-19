import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

def get_context_vector(text, model, tokenizer):
    """Get context vector by mean pooling"""
    # Tokenize input, get model output
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling: take average across sequence length of the output
    pooled_vector = torch.mean(outputs.last_hidden_state, dim=1)
    return pooled_vector[0]

# Create a document corpus (more documents for clustering)
documents = [
    "Machine learning algorithms build models based on sample data to make predictions "
        "without being explicitly programmed.",
    "Deep learning uses neural networks with many layers to learn representations of "
        "data with multiple levels of abstraction.",
    "Neural networks are computing systems inspired by the biological neural networks "
        "that constitute animal brains.",
    "Convolutional neural networks are deep neural networks most commonly applied to "
        "analyzing visual imagery.",
    "Natural language processing is a subfield of linguistics, computer science, and "
        "artificial intelligence.",
    "Sentiment analysis uses NLP to identify and extract opinions within text to "
        "determine writer's attitude.",
    "Named entity recognition is a subtask of information extraction that seeks to "
        "locate and classify named entities in text.",
    "Computer vision is an interdisciplinary field that deals with how computers can "
        "gain high-level understanding from digital images.",
    "Image recognition is the ability of software to identify objects, places, people, "
        "writing and actions in images.",
    "Object detection is a computer technology related to computer vision and "
        "image processing."
]

# Generate context vectors for all documents
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
doc_vectors = np.array([get_context_vector(doc, model, tokenizer) for doc in documents])

# Perform K-means clustering on documents
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(doc_vectors)

# Print documents in each cluster
for i in range(num_clusters):
    print(f"\nCluster {i+1}:")
    cluster_docs = [doc for j, doc in enumerate(documents) if cluster_labels[j] == i]
    for doc in cluster_docs:
        print(f"- {doc}")

# Visualize the clusters in reduced dimensionality
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(doc_vectors)

plt.figure(figsize=(10, 6))
colors = ["red", "blue", "green"]
for i in range(num_clusters):
    # Plot points in each cluster
    cluster_points = reduced_vectors[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                c=colors[i], label=f"Cluster {i+1}")
plt.title("Document Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()
