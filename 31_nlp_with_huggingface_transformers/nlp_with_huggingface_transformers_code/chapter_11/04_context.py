import collections
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

class ContextManager:
    def __init__(self, max_contexts=10):
        self.contexts = collections.OrderedDict()
        self.max_contexts = max_contexts

    def add_context(self, context_id, context):
        """Add context with automatic cleanup"""
        if len(self.contexts) >= self.max_contexts:
            self.contexts.popitem(last=False)
        self.contexts[context_id] = context

    def get_context(self, context_id):
        """Get context by ID"""
        return self.contexts.get(context_id)

    def search_relevant_context(self, question, top_k=3):
        """Search for relevant contexts based on relevance score"""
        relevant_contexts = []
        for context_id, context in self.contexts.items():
            relevance_score = self._calculate_relevance(question, context)
            relevant_contexts.append((relevance_score, context_id))
        return sorted(relevant_contexts, reverse=True)[:top_k]

    def _calculate_relevance(self, question, context):
        """Calculate relevance score between question and context.
        This is a simple counting the number of overlap words
        """
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        return len(question_words.intersection(context_words)) / len(question_words)

context_manager = ContextManager(max_contexts=10)
context_manager.add_context("python", """
  Python is a high-level, interpreted programming language created by Guido van Rossum and
  released in 1991.  Python's design philosophy emphasizes code readability with its
  notable use of significant whitespace. Python features a dynamic type system and
  automatic memory management and supports multiple programming paradigms, including
  structured, object-oriented, and functional programming.
""")
context_manager.add_context("machine_learning", """
  Machine learning is a field of study that gives computers the ability to learn without
  being explicitly programmed. It is a branch of artificial intelligence based on the idea
  that systems can learn from data, identify patterns and make decisions with minimal
  human intervention.
""")

config = QAConfig(max_sequence_length=512, max_answer_length=50, threshold=0.5)
qa_system = QASystem()
question = "Who created Python?"
relevant_contexts = context_manager.search_relevant_context(question, top_k=1)
if relevant_contexts:
    relevance, context_id = relevant_contexts[0]
    context = context_manager.get_context(context_id)
    print(f"Question: {question}")
    print(f"Most relevant context: {context_id} (relevance: {relevance:.2f})")
    print(context)

    answer = qa_system.get_answer(question, context, config)
    print(f"Answer: {answer['answer']}")
    print(f"Confidence: {answer['confidence']:.2f}")
else:
    print("No relevant context found.")
