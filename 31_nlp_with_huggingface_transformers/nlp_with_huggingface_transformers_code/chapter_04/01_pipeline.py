from transformers import pipeline

# Initialize the NER pipeline
ner_pipeline = pipeline("ner",
                        model="dbmdz/bert-large-cased-finetuned-conll03-english",
                        aggregation_strategy="simple")

# Text example
text = "Apple CEO Tim Cook announced new iPhone models in California yesterday."

# Perform NER
entities = ner_pipeline(text)

# Print the results
for entity in entities:
    print(f"Entity: {entity['word']}")
    print(f"Type: {entity['entity_group']}")
    print(f"Confidence: {entity['score']:.4f}")
    print("-" * 30)
