import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

model_name = "KernAI/stock-news-distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)

text = "Machine Learning Mastery is a nice website."
inputs = tokenizer(text, return_tensors="tf")
logits = model(**inputs).logits
predicted_class_id = tf.math.argmax(logits).numpy()
