import torch
from rouge_score import rouge_scorer
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

class StyleControlledSummarizer(Summarizer):
    def summarize_with_style(self, text, style="concise"):
        """Generate summaries with different styles.

        Args:
            text (str): Input text to summarize
            style (str): Summary style ('concise', 'detailed', 'technical', 'simple')
        Returns:
            str: Generated summary with specified style
        """
        style_params = {
            "concise": {
                "max_length": 80,
                "min_length": 30,
                "length_penalty": 3.0,
                "num_beams": 4,
                "early_stopping": True
            },
            "detailed": {
                "max_length": 200,
                "min_length": 100,
                "length_penalty": 1.0,
                "num_beams": 6,
                "early_stopping": False
            },
            "technical": {
                "max_length": 150,
                "min_length": 50,
                "length_penalty": 2.0,
                "num_beams": 5,
                "repetition_penalty": 1.5
            },
            "simple": {
                "max_length": 100,
                "min_length": 30,
                "length_penalty": 2.0,
                "num_beams": 3,
                "do_sample": True,
                "temperature": 0.7
            }
        }
        params = style_params[style]
        return self.summarize(text, **params)

class SummaryEvaluator:
    def __init__(self):
        """Initialize with ROUGE metrics."""
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    def evaluate_summary(self, reference, candidate):
        """Calculate ROUGE scores for a summary.

        Args:
            reference (str): Reference summary
            candidate (str): Generated summary

        Returns:
            dict: ROUGE scores for different metrics
        """
        scores = self.scorer.score(reference, candidate)

        print("Summary Quality Metrics:")
        print(f"ROUGE-1: {scores['rouge1'].fmeasure:.3f}")
        print(f"ROUGE-2: {scores['rouge2'].fmeasure:.3f}")
        print(f"ROUGE-L: {scores['rougeL'].fmeasure:.3f}")

        return scores

# Checking the metrics implementation
summarizer = StyleControlledSummarizer()
evaluator = SummaryEvaluator()
text = """
Quantum computing leverages the principles of quantum mechanics to perform
computations. Unlike classical computers that use bits, quantum computers
use quantum bits or qubits. These qubits can exist in multiple states
simultaneously through superposition, potentially allowing quantum computers
to solve certain problems exponentially faster than classical computers.
However, maintaining quantum coherence and minimizing errors remains a
significant challenge in building practical quantum computers.
"""
reference = "Quantum computing uses qubits for faster computation but faces coherence challenges."
for style in ["concise", "detailed", "technical", "simple"]:
    summary = summarizer.summarize_with_style(text, style=style)
    print(f"\n{style.capitalize()} Summary:")
    print(summary)
    scores = evaluator.evaluate_summary(reference, summary)
