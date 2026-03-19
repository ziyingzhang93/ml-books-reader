from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, \
    BertTokenizer, BertForQuestionAnswering
import torch

# Load DistilBERT model and tokenizer
distilbert_model_name = "distilbert-base-uncased-distilled-squad"
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_name)
distilbert_model = DistilBertForQuestionAnswering.from_pretrained(distilbert_model_name)

# Load BERT model and tokenizer
bert_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForQuestionAnswering.from_pretrained(bert_model_name)

# Define a context and a question
question = "What is the capital of France?"
context = """Paris is the capital and most populous city of France, with an estimated
population of 2,175,601 residents as of 2018, in an area of more than 105 square
kilometres. The City of Paris is the centre and seat of government of the region
and province of Île-de-France, or Paris Region, which has an estimated population
of 12,174,880, or about 18 percent of the population of France as of 2017."""

# Function to get answer from DistilBERT
def get_distilbert_answer(question, context):
    inputs = distilbert_tokenizer(question, context, return_tensors="pt")

    with torch.no_grad():
        output = distilbert_model(**inputs)

    start = torch.argmax(output.start_logits)
    end = torch.argmax(output.end_logits)

    confidence = float(output.start_logits[0, start] + output.end_logits[0, end])
    answer_tokens = inputs.input_ids[0, start:end+1]
    answer = distilbert_tokenizer.decode(answer_tokens)

    return answer, confidence

# Function to get answer from BERT
def get_bert_answer(question, context):
    inputs = bert_tokenizer(question, context, return_tensors="pt")

    with torch.no_grad():
        output = bert_model(**inputs)

    start = torch.argmax(output.start_logits)
    end = torch.argmax(output.end_logits)

    confidence = float(output.start_logits[0, start] + output.end_logits[0, end])

    answer_tokens = inputs.input_ids[0, start:end+1]
    answer = bert_tokenizer.decode(answer_tokens)

    return answer, confidence

# Get answers from both models
distilbert_answer, distilbert_confidence = get_distilbert_answer(question, context)
bert_answer, bert_confidence = get_bert_answer(question, context)

# Simple ensemble: choose the answer with the highest confidence
if distilbert_confidence > bert_confidence:
    final_answer = distilbert_answer
    model_used = "DistilBERT"
    confidence = distilbert_confidence
else:
    final_answer = bert_answer
    model_used = "BERT"
    confidence = bert_confidence

print(f"Question: {question}")
print(f"Final Answer: {final_answer}")
print(f"Model Used: {model_used}")
print(f"Confidence Score: {confidence}")
