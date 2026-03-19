import torch
from transformers import pipeline

# Create a sentiment analysis pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Test text
text = "I absolutely love this product! Would buy again."

# Get the sentiment
result = sentiment_analyzer(text)
print(f"Sentiment: {result[0]['label']}")
print(f"Confidence: {result[0]['score']:.4f}")
