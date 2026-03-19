import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """Initialize the summarizer with model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def summarize(self, text, context_weight=0.5, max_length=150, min_length=50,
                  num_beams=4, length_penalty=2.0, repetition_penalty=1.0,
                  do_sample=False, temperature=1.0, early_stopping=True):
        """Generate a summary with context awareness."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024
                                ).to(self.device)
        # Generate summary using only the input tokens
        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            temperature=temperature,
            early_stopping=early_stopping,
        )
        # Decode and return the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

# Let's run an example to see how it works
summarizer = Summarizer()
text = """
The development of artificial intelligence has revolutionized numerous industries.
Machine learning algorithms now power everything from recommendation systems to
autonomous vehicles. Deep learning, in particular, has shown remarkable success
in tasks like image recognition and natural language processing. However, these
advances also raise important ethical considerations about AI's impact on society,
privacy, and employment.
"""

summary = summarizer.summarize(text)
print(f"Summary:\n{summary}")
