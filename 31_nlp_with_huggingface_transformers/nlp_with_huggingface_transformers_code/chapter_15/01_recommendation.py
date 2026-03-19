import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Define a corpus of articles (title and content)
articles = [
  {
    "title": "Understanding Deep Learning",
    "content": ("Deep learning is a subset of machine learning where artificial neural "
                "networks, algorithms inspired by the human brain, learn from large "
                "amounts of data.")
  }, {
    "title": "Introduction to Natural Language Processing",
    "content": ("Natural Language Processing (NLP) is a field of AI that gives machines "
                "the ability to read, understand, and derive meaning from human "
                "languages.")
  }, {
    "title": "The Future of Computer Vision",
    "content": ("Computer vision is an interdisciplinary field that deals with how "
                "computers can gain high-level understanding from digital images or "
                "videos.")
  }, {
    "title": "Reinforcement Learning Explained",
    "content": ("Reinforcement learning is an area of machine learning concerned with "
                "how software agents ought to take actions in an environment so as to "
                "maximize some notion of cumulative reward.")
  }, {
    "title": "Neural Networks and Their Applications",
    "content": ("Neural networks are a set of algorithms, modeled loosely after the "
                "human brain, that are designed to recognize patterns in data.")
  }
]

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_article_embeddings(articles, model):
    """create embeddings for articles"""
    texts = [f"{article["title"]}. {article["content"]}" for article in articles]
    embeddings = model.encode(texts)
    return embeddings

def get_recommendations(article_id, articles, embeddings, top_n=2):
    """get recommendations for a given article ID based on cosine similarity"""
    similarities = cosine_similarity([embeddings[article_id]], embeddings)[0]
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    return [articles[idx] for idx in similar_indices]

# Create embeddings for all articles, and get recommendation for first article
embeddings = create_article_embeddings(articles, model)
recommendations = get_recommendations(0, articles, embeddings)

# Print the recommendations
print(f'Recommendations for "{articles[0]["title"]}":')
for i, rec in enumerate(recommendations):
    print(f"{i+1}. {rec["title"]}")
