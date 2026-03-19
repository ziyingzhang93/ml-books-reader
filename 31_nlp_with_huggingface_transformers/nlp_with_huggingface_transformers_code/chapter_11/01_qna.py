import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "distilbert-base-uncased-distilled-squad"

tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer,
                       device=device)
max_answer_length = 50
top_k = 3
question = "What is the capital of France?"
context = "France is a country in Western Europe. Its capital is Paris, which is " \
          "known for its art, fashion, gastronomy and culture."
result = qa_pipeline(question=question, context=context,
                     max_answer_len=max_answer_length, top_k=top_k)
print(f"Question: {question}")
print(f"Context: {context}")
print(result)
