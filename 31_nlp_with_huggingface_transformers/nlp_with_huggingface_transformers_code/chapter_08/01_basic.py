import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TextSummarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """Initialize the summarizer with a pre-trained model.

        Args:
            model_name (str): Name of the pre-trained model to use.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def summarize(self, text, max_length=130, min_length=30, length_penalty=2.0,
                  repetition_penalty=2.0, num_beams=4, early_stopping=True):
        """Generate a summary for the given text.

        Args:
            text (str): The text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            length_penalty (float): Penalty for longer summaries
            repetition_penalty (float): Penalty for repeated tokens
            num_beams (int): Number of beams for beam search
            early_stopping (bool): Whether to stop when all beams are finished

        Returns:
            str: The generated summary
        """
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
                no_repeat_ngram_size=3,
                num_beams=num_beams,
                early_stopping=early_stopping
            )
            # Decode and return the summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"Error during summarization: {str(e)}")
            return text

# Initialize the basic summarizer
summarizer = TextSummarizer()

# Sample text to summarize
sample_text = """
Artificial intelligence (AI) is a rapidly advancing field that focuses on creating
intelligent systems capable of performing tasks that typically require human intelligence.
These tasks include natural language processing, computer vision, speech recognition,
and decision-making. With applications across healthcare, finance, education, and more,
AI is transforming the way we interact with technology and solve complex problems.
"""

# Generate a summary
summary = summarizer.summarize(sample_text)
print("Basic Summary:\n", summary)

# Test with shorter text
short_text = "AI is changing the world by automating tasks and providing insights " \
             "from large datasets."
short_summary = summarizer.summarize(short_text)
print("Short Text Summary:\n", short_summary)
