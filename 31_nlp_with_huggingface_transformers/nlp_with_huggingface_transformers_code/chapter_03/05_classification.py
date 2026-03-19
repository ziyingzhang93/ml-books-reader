import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BERTSentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.labels = ["NEGATIVE", "POSITIVE"]

    def preprocess_text(self, text):
        # Remove extra whitespace and normalize
        text = " ".join(text.split())

        # Tokenize with BERT-specific tokens
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Move to GPU if available
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict(self, text):
        # Prepare text for model
        inputs = self.preprocess_text(text)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Convert to human-readable format
        prediction_dict = {
            "text": text,
            "sentiment": self.labels[probabilities.argmax().item()],
            "confidence": probabilities.max().item(),
            "probabilities": {
                label: prob.item()
                for label, prob in zip(self.labels, probabilities[0])
            }
        }
        return prediction_dict
