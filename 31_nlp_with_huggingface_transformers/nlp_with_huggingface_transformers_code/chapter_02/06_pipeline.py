import torch
from transformers import pipeline

model_name = "KernAI/stock-news-distilbert"
classifier = pipeline(model=model_name)

text = "Machine Learning Mastery is a nice website."
prediction = classifier(text)
print(prediction)
