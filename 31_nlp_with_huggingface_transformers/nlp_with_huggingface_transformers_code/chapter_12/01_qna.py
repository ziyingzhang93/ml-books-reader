from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# Define a context and a question
question = "What is machine learning?"
context = """Machine learning is a field of inquiry devoted to understanding and building
methods that 'learn', that is, methods that leverage data to improve performance on some
set of tasks. It is seen as a part of artificial intelligence.  Machine learning
algorithms build a model based on sample data, known as training data, in order to make
predictions or decisions without being explicitly programmed to do so. Machine learning
algorithms are used in a wide variety of applications, such as in medicine, email
filtering, speech recognition, and computer vision, where it is difficult or unfeasible to
develop conventional algorithms to perform the needed tasks."""

# Tokenize the input and run the model
inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Process the answer
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits)
answer_tokens = inputs.input_ids[0, answer_start: answer_end + 1]
answer = tokenizer.decode(answer_tokens)

print(f"Question: {question}")
print(f"Answer: {answer}")
