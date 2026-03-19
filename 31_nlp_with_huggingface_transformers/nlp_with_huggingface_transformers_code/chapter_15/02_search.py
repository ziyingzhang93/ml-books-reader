import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
  {
    "language": "English",
    "text": ("Machine learning is a field of study that gives computers the ability "
             "to learn without being explicitly programmed.")
  }, {
    "language": "Spanish",
    "text": ("El aprendizaje automático es un campo de estudio que da a las computadoras "
             "la capacidad de aprender sin ser programadas explícitamente.")
  }, {
    "language": "French",
    "text": ("L'apprentissage automatique est un domaine d'étude qui donne aux "
             "ordinateurs la capacité d'apprendre sans être explicitement programmés.")
  }, {
    "language": "German",
    "text": ("Maschinelles Lernen ist ein Studienbereich, der Computern die Fähigkeit "
             "gibt, zu lernen, ohne explizit programmiert zu werden.")
  }, {
    "language": "Italian",
    "text": ("Il machine learning è un campo di studio che conferisce ai computer la "
             "capacità di apprendere senza essere esplicitamente programmati.")
  }, {
    "language": "English",
    "text": ("Natural language processing is a subfield of linguistics, computer "
             "science, and artificial intelligence.")
  }, {
    "language": "English",
    "text": ("Computer vision is an interdisciplinary field that deals with how "
             "computers can gain high-level understanding from digital images or videos.")
  }
]

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Generate embeddings for the corpus
texts = [doc["text"] for doc in corpus]
embeddings = model.encode(texts)

# Define a query in English and generate an embedding
query = "What is machine learning?"
query_embedding = model.encode(query)

# Sort the embeddings of the corpus by descending similarity
similarities = cosine_similarity([query_embedding], embeddings)[0]
ranked_indices = np.argsort(similarities)[::-1]

# Print ranked results
print(f"Query: {query}\n")
for i, idx in enumerate(ranked_indices[:3]):  # Show top 3 results
    print(f"{i+1}. [{corpus[idx]["language"]}] {corpus[idx]["text"]} "
          f"(Similarity: {similarities[idx]:.4f})")
