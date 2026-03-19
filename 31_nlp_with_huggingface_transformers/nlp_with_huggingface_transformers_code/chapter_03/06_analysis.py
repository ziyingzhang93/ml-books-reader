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

def demonstrate_sentiment_analysis():
    # Initialize analyzer
    analyzer = BERTSentimentAnalyzer()

    # Test texts
    texts = [
        "This product completely transformed my workflow!",
        "Terrible experience, would not recommend.",
        "It's decent for the price, but nothing special."
    ]

    # Analyze each text
    for text in texts:
        result = analyzer.predict(text)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Detailed probabilities:")
        for label, prob in result["probabilities"].items():
            print(f"  {label}: {prob:.4f}")

# Running demonstration
demonstrate_sentiment_analysis()
