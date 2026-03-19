from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = DistilBertForSequenceClassification.from_pretrained(model_id)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_id)
