import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class BERTNamedEntityRecognizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model=AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def recognize_entities(self, text):
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(-1)

        # Convert predictions to entities
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.model.config.id2label[p.item()] for p in predictions[0]]

        # Extract entities
        entities = []
        current_entity = None
        for token, label in zip(tokens, labels):
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {"type": label[2:], "text": token}
            elif label.startswith("I-") and current_entity:
                if token.startswith("##"):
                    current_entity["text"] += token[2:]
                else:
                    current_entity["text"] += " " + token
            elif label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            if current_entity:
                entities.append(current_entity)

        return entities
