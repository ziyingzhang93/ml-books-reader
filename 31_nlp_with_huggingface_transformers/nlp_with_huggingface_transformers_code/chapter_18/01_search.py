import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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

def semantic_search(query, documents, document_vectors, top_k=2):
    """Search the corpus"""
    # Calculate similarity between query and all documents
    query_vector = get_context_vector(query, model, tokenizer)
    similarities = cosine_similarity([query_vector], document_vectors)[0]

    # Get indices of top-k most similar documents
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Return top-k documents and their similarity scores
    results = []
    for idx in top_indices:
        results.append({
            "document": documents[idx],
            "similarity": similarities[idx]
        })
    return results

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Create a document corpus and convert them into context vectors
documents = [
    "Machine learning is a field of study that gives computers the ability to learn "
        "without being explicitly programmed.",
    "Deep learning is a subset of machine learning that uses neural networks with "
        "many layers.",
    "Natural language processing is a field of AI that focuses on the interaction "
        "between computers and human language.",
    "Computer vision is an interdisciplinary field that deals with how computers "
        "can gain high-level understanding from digital images or videos.",
    "Reinforcement learning is about taking suitable actions to maximize reward "
        "in a particular situation."
]
document_vectors = [get_context_vector(doc, model, tokenizer) for doc in documents]

# Example search
query = "How do computers learn from data?"
results = semantic_search(query, documents, document_vectors)

# Print results
print(f"Query: {query}\n")
for i, result in enumerate(results):
    print(f"Result {i+1} (Similarity: {result["similarity"]:.4f}):")
    print(result["document"])
    print()
