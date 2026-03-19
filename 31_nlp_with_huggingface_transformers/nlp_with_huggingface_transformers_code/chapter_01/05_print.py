from transformers import pipeline

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_id)
print(classifier.model)
print(classifier.tokenizer)
