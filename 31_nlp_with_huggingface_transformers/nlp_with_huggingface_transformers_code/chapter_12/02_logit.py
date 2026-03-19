from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import numpy as np

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# Define multiple contexts
question = "What is deep learning?"
contexts = [
    """Machine learning is a field of inquiry devoted to understanding and building
    methods that 'learn', that is, methods that leverage data to improve performance
    on some set of tasks. It is seen as a part of artificial intelligence.""",

    """Deep learning is a subset of machine learning where artificial neural networks,
    algorithms inspired by the human brain, learn from large amounts of data. Deep
    learning is behind many recent advances in AI, including computer vision and
    speech recognition.""",

    """Natural Language Processing (NLP) is a field of AI that gives machines the
    ability to read, understand, and derive meaning from human languages. It's used
    in applications like chatbots, translation services, and sentiment analysis."""
]

# Function to get answer from a single context
def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)

    # Get the most likely answer span
    ans_start = torch.argmax(output.start_logits)
    ans_end = torch.argmax(output.end_logits)

    # Calculate the confidence score (simplified)
    confidence = float(output.start_logits[0, ans_start] + output.end_logits[0, ans_end])
    # Extract the answer
    answer_tokens = inputs.input_ids[0, ans_start : ans_end+1]
    answer = tokenizer.decode(answer_tokens)

    return answer, confidence

# Get answers from all contexts
answers_with_scores = [get_answer(question, context) for context in contexts]

# Find the answer with the highest confidence score
best_answer_idx = np.argmax([score for _, score in answers_with_scores])
best_answer, best_score = answers_with_scores[best_answer_idx]

print(f"Question: {question}")
print(f"Best Answer: {best_answer}")
print(f"From Context: {contexts[best_answer_idx][:100]}...")
print(f"Confidence Score: {best_score}")
