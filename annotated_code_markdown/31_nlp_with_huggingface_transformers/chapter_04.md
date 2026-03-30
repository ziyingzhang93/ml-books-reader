# HuggingFace Transformers NLP / NLP with HF Transformers
## Chapter 04

---

### Pipeline

# 01 — Pipeline / 管道

**Chapter 04 — File 1 of 3 / 第04章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Initialize the NER pipeline**.

本脚本演示 **Initialize the NER pipeline**。

---
## Step 1 — Step 1

```python
from transformers import pipeline
```

---
## Step 2 — Initialize the NER pipeline

```python
ner_pipeline = pipeline("ner",
                        model="dbmdz/bert-large-cased-finetuned-conll03-english",
                        aggregation_strategy="simple")
```

---
## Step 3 — Text example

```python
text = "Apple CEO Tim Cook announced new iPhone models in California yesterday."
```

---
## Step 4 — Perform NER

```python
entities = ner_pipeline(text)
```

---
## Step 5 — Print the results

```python
for entity in entities:
    print(f"Entity: {entity['word']}")
    print(f"Type: {entity['entity_group']}")
    print(f"Confidence: {entity['score']:.4f}")
    print("-" * 30)
```

---
## Learning Notes / 学习笔记

- **概念**: Initialize the NER pipeline 是机器学习中的常用技术。  
  *Initialize the NER pipeline is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pipeline / 管道
# Complete Code / 完整代码
# ===============================

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
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Automodel

# 03 — Automodel / 03 Automodel

**Chapter 04 — File 2 of 3 / 第04章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Load model and tokenizer**.

本脚本演示 **Load model and tokenizer**。

---
## Step 1 — Step 1

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
```

---
## Step 2 — Load model and tokenizer

```python
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
```

---
## Step 3 — Text example

```python
text = "Google and Microsoft are competing in the AI space while Elon " \
       "Musk founded SpaceX."
```

---
## Step 4 — Tokenize the text

```python
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
```

---
## Step 5 — Get predictions

```python
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
```

---
## Step 6 — Convert predictions to labels

```python
label_list = model.config.id2label
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predictions = predictions[0].tolist()
```

---
## Step 7 — Process results

```python
current_entity = []
current_entity_type = None

for token, prediction in zip(tokens, predictions):
    if token.startswith("##"):
        if current_entity:
            current_entity.append(token[2:])
    else:
        if current_entity:
            print(f"Entity: {''.join(current_entity)}")
            print(f"Type: {current_entity_type}")
            print("-" * 30)
            current_entity = []

        if label_list[prediction] != "O":
            current_entity = [token]
            current_entity_type = label_list[prediction]
```

---
## Step 8 — Print final entity if exists

```python
if current_entity:
    print(f"Entity: {''.join(current_entity)}")
    print(f"Type: {current_entity_type}")
```

---
## Learning Notes / 学习笔记

- **概念**: Load model and tokenizer 是机器学习中的常用技术。  
  *Load model and tokenizer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Automodel / 03 Automodel
# Complete Code / 完整代码
# ===============================

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load model and tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Text example
text = "Google and Microsoft are competing in the AI space while Elon " \
       "Musk founded SpaceX."

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

# Convert predictions to labels
label_list = model.config.id2label
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predictions = predictions[0].tolist()

# Process results
current_entity = []
current_entity_type = None

for token, prediction in zip(tokens, predictions):
    if token.startswith("##"):
        if current_entity:
            current_entity.append(token[2:])
    else:
        if current_entity:
            print(f"Entity: {''.join(current_entity)}")
            print(f"Type: {current_entity_type}")
            print("-" * 30)
            current_entity = []

        if label_list[prediction] != "O":
            current_entity = [token]
            current_entity_type = label_list[prediction]

# Print final entity if exists
if current_entity:
    print(f"Entity: {''.join(current_entity)}")
    print(f"Type: {current_entity_type}")
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Complete

# 08 — Complete / 08 Complete

**Chapter 04 — File 3 of 3 / 第04章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Get predictions**.

本脚本演示 **Get predictions**。

---
## Step 1 — Step 1

```python
from transformers import pipeline
import torch
import logging
from typing import List, Dict

class NERProcessor:
    def __init__(self,
                 model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
                 confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.ner_pipeline = pipeline("ner",
                                         model=model_name,
                                         aggregation_strategy="simple",
                                         device=self.device)
        except Exception as e:
            logging.error(f"Failed to initialize NER pipeline: {str(e)}")
            raise

    def process_text(self, text: str) -> List[Dict]:
        if not text or not isinstance(text, str):
            logging.warning("Invalid input text")
            return []

        try:
```

---
## Step 2 — Get predictions

```python
entities = self.ner_pipeline(text)
```

---
## Step 3 — Post-process results

```python
filtered_entities = [
                entity for entity in entities
                if entity["score"] >= self.confidence_threshold
            ]

            return filtered_entities
        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
            return []


if __name__ == "__main__":
```

---
## Step 4 — Initialize processor

```python
processor = NERProcessor()
```

---
## Step 5 — Text example

```python
text = """
    Apple Inc. CEO Tim Cook announced new partnerships with Microsoft
    and Google during a conference in New York City. The event was also
    attended by Sundar Pichai and Satya Nadella.
    """
```

---
## Step 6 — Process text

```python
results = processor.process_text(text)
```

---
## Step 7 — Print results

```python
for entity in results:
        print(f"Entity: {entity['word']}")
        print(f"Type: {entity['entity_group']}")
        print(f"Confidence: {entity['score']:.4f}")
        print("-" * 30)
```

---
## Learning Notes / 学习笔记

- **概念**: Get predictions 是机器学习中的常用技术。  
  *Get predictions is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Complete / 08 Complete
# Complete Code / 完整代码
# ===============================

from transformers import pipeline
import torch
import logging
from typing import List, Dict

class NERProcessor:
    def __init__(self,
                 model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
                 confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.ner_pipeline = pipeline("ner",
                                         model=model_name,
                                         aggregation_strategy="simple",
                                         device=self.device)
        except Exception as e:
            logging.error(f"Failed to initialize NER pipeline: {str(e)}")
            raise

    def process_text(self, text: str) -> List[Dict]:
        if not text or not isinstance(text, str):
            logging.warning("Invalid input text")
            return []

        try:
            # Get predictions
            entities = self.ner_pipeline(text)

            # Post-process results
            filtered_entities = [
                entity for entity in entities
                if entity["score"] >= self.confidence_threshold
            ]

            return filtered_entities
        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
            return []


if __name__ == "__main__":
    # Initialize processor
    processor = NERProcessor()

    # Text example
    text = """
    Apple Inc. CEO Tim Cook announced new partnerships with Microsoft
    and Google during a conference in New York City. The event was also
    attended by Sundar Pichai and Satya Nadella.
    """

    # Process text
    results = processor.process_text(text)

    # Print results
    for entity in results:
        print(f"Entity: {entity['word']}")
        print(f"Type: {entity['entity_group']}")
        print(f"Confidence: {entity['score']:.4f}")
        print("-" * 30)
```

---

### Chapter Summary / 章节总结

# Chapter 04 Summary / 第04章总结

## Theme / 主题: Chapter 04 / Chapter 04

This chapter contains **3 code files** demonstrating chapter 04.

本章包含 **3 个代码文件**，演示Chapter 04。

---
## Evolution / 演化路线

  1. `01_pipeline.ipynb` — Pipeline
  2. `03_automodel.ipynb` — Automodel
  3. `08_complete.ipynb` — Complete

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 04) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 04）是机器学习流水线中的基础构建块。

---
