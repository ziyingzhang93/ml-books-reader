import functools
import pprint
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class AdvancedTextSummarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6", quantize=False):
        """Initialize the advanced summarizer with additional features.

        Args:
            model_name (str): Name of the pre-trained model to use
            quantize (bool): Whether to quantize the model for faster inference
        """
        self.device = "cuda" if torch.cuda.is_available() and not quantize else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if quantize:
            self.model = torch.quantization.quantize_dynamic(
                self.model, dtype=torch.qint8)
        self.model.to(self.device)

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize input text.

        Args:
            text (str): Raw input text

        Returns:
            str: Cleaned and normalized text
        """
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())
        # Remove URLs
        text = re.sub(r"^https?://[^\s/$.?#].[^\s]*$", "", text)
        # Remove special characters but keep punctuation
        text = re.sub(r"[^\w\s.,!?-]", "", text)

        return text

    def split_long_text(self, text: str, max_tokens: int = 1024) -> list[str]:
        """Split long text into chunks that fit within model's max token limit.

        Args:
            text (str): Input text
            max_tokens (int): Maximum tokens per chunk

        Returns:
            list[str]: List of text chunks
        """
        # Tokenize the full text
        tokens = self.tokenizer.tokenize(text)

        # Split into chunks, then convert back to strings
        chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
        return [self.tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

    @functools.lru_cache(maxsize=200)
    def cached_summarize(self, text: str, max_length: int = 130, min_length: int = 30,
                         length_penalty: float = 2.0, repetition_penalty: float = 2.0,
                         num_beams: int = 4, early_stopping: bool = True) -> str:
        """Cached version of the summarization function."""
        try:
            # Tokenize the input text
            inputs = self.tokenizer(text, max_length=1024, truncation=True,
                                    padding="max_length", return_tensors="pt"
                                    ).to(self.device)
            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
                early_stopping=early_stopping,
                no_repeat_ngram_size=3,   # Prevent repetition of phrases
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"Error during summarization: {str(e)}")
            return text

    def summarize_batch(self, texts: list[str], batch_size: int = 4, **kwargs
                        ) -> list[str]:
        """Summarize multiple texts efficiently in batches.

        Args:
            texts (list[str]): List of input texts
            batch_size (int): Number of texts to process at once
            **kwargs: Additional arguments for summarization

        Returns:
            list[str]: List of generated summaries
        """
        summaries = []

        for i in range(0, len(texts), batch_size):
            # Create batch and process each text in the batch
            batch = texts[i:i + batch_size]
            processed_batch = [self.preprocess_text(text) for text in batch]

            # Tokenize batch
            inputs = self.tokenizer(processed_batch, max_length=1024, truncation=True,
                                    padding=True, return_tensors="pt"
                                    ).to(self.device)
            # Generate summaries for batch
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **kwargs
            )
            # Decode summaries
            summaries.extend([self.tokenizer.decode(ids, skip_special_tokens=True)
                              for ids in summary_ids])
        return summaries

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30,
                  length_penalty: float = 2.0, repetition_penalty: float = 2.0,
                  num_beams: int = 4, early_stopping: bool = True) -> dict[str, str]:
        """Generate a summary with advanced features.

        Args:
            text (str): The text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            length_penalty (float): Penalty for longer summaries
            repetition_penalty (float): Penalty for repeated tokens
            num_beams (int): Number of beams for beam search
            early_stopping (bool): Whether to stop when all beams are finished

        Returns:
            dict[str, str]: Dictionary containing original and summarized text
        """
        # Preprocess the text
        cleaned_text = self.preprocess_text(text)

        # Handle long texts
        chunks = self.split_long_text(cleaned_text)
        chunk_summaries = []
        for chunk in chunks:
            summary = self.cached_summarize(
                chunk,
                max_length=max_length // len(chunks),  # Adjust length for chunks
                min_length=min_length // len(chunks),
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
                early_stopping=early_stopping
            )
            chunk_summaries.append(summary)

        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "summary": " ".join(chunk_summaries)
        }

# Initialize the advanced summarizer with caching enabled and quantization
adv_summarizer = AdvancedTextSummarizer(quantize=True)

# Sample text
long_text = """
The development of artificial intelligence (AI) has significantly impacted various
industries worldwide. From healthcare to finance, AI-powered applications have
streamlined operations, improved accuracy, and unlocked new possibilities. In healthcare,
AI assists in diagnostics, personalized treatment plans, and drug discovery. In finance,
it aids in fraud detection, algorithmic trading, and customer service. Despite its
benefits, AI raises concerns about data privacy, ethical implications, and job
displacement.
"""

# Generate a summary with default settings
adv_summary = adv_summarizer.summarize(long_text)
print("Advanced Summary:")
pprint.pprint(adv_summary)

# Batch summarization
texts = [
  "AI is revolutionizing healthcare with better diagnostics and personalized treatments.",
  "Self-driving cars are powered by machine learning algorithms that continuously " \
      "learn from traffic patterns.",
  "Natural language processing helps computers understand and communicate with humans " \
    "more effectively.",
  "Climate change is being studied using AI models to predict future environmental " \
    "patterns."
]

# Summarize multiple texts in a batch
batch_summaries = adv_summarizer.summarize_batch(texts, batch_size=2)
print("\nBatch Summaries:")
for i, s in enumerate(batch_summaries, 1):
    pprint.pprint(f"{i}. {s}")
