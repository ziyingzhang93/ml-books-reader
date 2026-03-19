import time
from dataclasses import dataclass

import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline

@dataclass
class QAConfig:
    """Configuration for QA settings"""
    max_sequence_length: int = 512
    max_answer_length: int = 50
    top_k: int = 3
    threshold: float = 0.5


class QASystem:
    """Q&A system with chunking"""
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)

        # Initialize pipeline for simple queries and answer cache
        self.qa_pipeline = pipeline("question-answering", model=self.model,
                                    tokenizer=self.tokenizer, device=self.device)
        self.answer_cache = {}

    def preprocess_context(self, context, max_length=512):
      """Split long contexts into chunks below max_length"""
      chunks = []
      current_chunk = []
      current_length = 0

      for word in context.split():
          if current_length + 1 + len(word) > max_length:
              chunks.append(" ".join(current_chunk))
              current_chunk = [word]
              current_length = len(word)
          else:
              current_chunk.append(word)
              current_length += 1 + len(word)  # length of space + word

      # Add the last chunk if it's not empty
      if current_chunk:
          chunks.append(" ".join(current_chunk))

      return chunks

    def get_answer(self, question, context, config):
        """Get answer with confidence score"""
        # Check cache
        cache_key = (question, context)
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]

        # Preprocess context into chunks
        context_chunks = self.preprocess_context(context, config.max_sequence_length)

        # Get answers from all chunks
        answers = []
        for chunk in context_chunks:
            result = self.qa_pipeline(question=question,
                                      context=chunk,
                                      max_answer_len=config.max_answer_length,
                                      top_k=config.top_k)
            assert isinstance(result, list)
            for answer in result:
                if answer["score"] >= config.threshold:
                    answers.append(answer)

        # Return the best answer or indicate no answer found
        if answers:
            best_answer = max(answers, key=lambda x: x["score"])
            result = {
                "answer": best_answer["answer"],
                "confidence": best_answer["score"],
            }
        else:
            result = {
                "answer": "No answer found",
                "confidence": 0.0,
            }

        # Cache the result
        self.answer_cache[cache_key] = result
        return result


config = QAConfig(max_sequence_length=512, max_answer_length=50, threshold=0.5)
qa_system = QASystem()
context = """
The Python programming language was created by Guido van Rossum and was released in 1991.
Python is known for its simple syntax and readability. It has become one of the most
popular programming languages, especially in fields like data science and machine
learning.  The language is maintained by the Python Steering Council and developed by a
large community of contributors.
"""
questions = [
    "Who created Python?",
    "When was Python released?",
    "Why is Python popular?",
    "What is Python known for?"
]
for question in questions:
    start_time = time.time()
    answer = qa_system.get_answer(question, context, config)
    duration = time.time() - start_time
    print(f"Question: {question}")
    print(f"Answer: {answer['answer']}")
    print(f"Confidence: {answer['confidence']:.2f}")
    print(f"Duration: {duration:.2f}s")
    print("-" * 50)
