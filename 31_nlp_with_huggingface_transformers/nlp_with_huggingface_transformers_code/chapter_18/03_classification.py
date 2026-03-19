from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def get_context_vector(text, model, tokenizer):
    """Get context vector by mean pooling"""
    # Tokenize input, get model output
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling: take average across sequence length of the output
    pooled_vector = torch.mean(outputs.last_hidden_state, dim=1)
    return pooled_vector[0]

# Create a dataset of texts with labels
texts = [
    "The stock market reached a new high today, with technology stocks leading the "
        "gains.",
    "The company reported strong quarterly earnings, exceeding analysts' expectations.",
    "Investors are optimistic about the economy despite recent inflation concerns.",
    "The new vaccine has shown high efficacy in clinical trials against all variants.",
    "Researchers have discovered a potential treatment for a previously incurable "
        "disease.",
    "The hospital announced expanded capacity to handle the increasing number of "
        "patients.",
    "The latest smartphone features a better camera and longer battery life.",
    "The software update includes new security features and performance improvements.",
    "The tech company unveiled its newest artificial intelligence system yesterday."
]
labels = [
    "Business",
    "Business",
    "Business",
    "Health",
    "Health",
    "Health",
    "Technology",
    "Technology",
    "Technology"
]

# Generate context vectors for all texts
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
text_vectors = np.array([get_context_vector(text, model, tokenizer) for text in texts])

# Split into training and testing sets, train a classifier, then evaluate
X_train, X_test, y_train, y_test = \
    train_test_split(text_vectors, labels, test_size=0.3, random_state=42)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# Classify new texts
new_texts = [
    "The central bank has decided to keep interest rates unchanged.",
    "A new study shows that regular exercise can reduce the risk of heart disease.",
    "The new laptop has a faster processor and more memory than previous models."
]
new_vectors = np.array([get_context_vector(text, model, tokenizer) for text in new_texts])
predictions = classifier.predict(new_vectors)

# Print predictions
for text, prediction in zip(new_texts, predictions):
    print(f"Text: {text}")
    print(f"Category: {prediction}\n")
