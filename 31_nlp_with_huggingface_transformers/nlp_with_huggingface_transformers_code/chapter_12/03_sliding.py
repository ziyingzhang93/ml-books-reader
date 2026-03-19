from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import numpy as np

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# Define a long context
question = "What is the capital of France?"
long_context = """Paris is the capital and most populous city of France, with an
estimated population of 2,175,601 residents as of 2018, in an area of more than 105
square kilometres. The City of Paris is the centre and seat of government of the
region and province of Île-de-France, or Paris Region, which has an estimated
population of 12,174,880, or about 18 percent of the population of France as of
2017."""

def get_answer_sliding_window(question, context, total_len=512, stride=128):
    """Function to get answer using sliding window"""
    # Tokenize the question and context
    question_tokens = tokenizer.tokenize(question)
    context_tokens = tokenizer.tokenize(context)

    # If the context is short enough, process it directly
    if len(question_tokens) + len(context_tokens) + 3 <= total_len:  # +3 for CLS & 2 SEP
        best_answer, best_score = get_answer(question, context)
        return best_answer, best_score, context

    # Otherwise, use sliding window
    max_question_len = 64  # Limit question length to leave room for content
    if len(question_tokens) > max_question_len:
        question_tokens = question_tokens[:max_question_len]

    # Calculate how many tokens we can allocate to the context
    max_len = total_len - len(question_tokens) - 3  # -3 for CLS & 2 SEP
    windows = []
    for i in range(0, len(context_tokens), stride):
        windows.append(tokenizer.convert_tokens_to_string(context_tokens[i:i+max_len]))
        if i + max_len >= len(context_tokens):
            break  # Last window

    # Get answers from all windows
    answers_with_scores = [get_answer(question, window) for window in windows]

    # Find the answer with the highest confidence score
    best_answer_idx = np.argmax([score for _, score in answers_with_scores])
    best_answer, best_score = answers_with_scores[best_answer_idx]
    return best_answer, best_score, windows[best_answer_idx]

def get_answer(question, context):
    """Function to get answer from a single context"""
    inputs = tokenizer(question, context, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)

    ans_start = torch.argmax(output.start_logits)
    ans_end = torch.argmax(output.end_logits)

    confidence = float(output.start_logits[0, ans_start] + output.end_logits[0, ans_end])
    answer_tokens = inputs.input_ids[0, ans_start: ans_end + 1]
    answer = tokenizer.decode(answer_tokens)
    return answer, confidence

# Get answer using sliding window
best_answer, best_score, best_window = get_answer_sliding_window(question, long_context)

print(f"Question: {question}")
print(f"Best Answer: {best_answer}")
print(f"From Window: {best_window[:100]}...")
print(f"Confidence Score: {best_score}")
