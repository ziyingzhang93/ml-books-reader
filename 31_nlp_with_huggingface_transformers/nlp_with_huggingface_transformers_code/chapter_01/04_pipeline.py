from transformers import pipeline

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_id)
result = classifier("Machine Learning Mastery is a great website for machine learning.")
print(result)
