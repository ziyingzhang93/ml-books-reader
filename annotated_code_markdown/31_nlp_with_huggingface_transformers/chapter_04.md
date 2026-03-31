# HuggingFace Transformers NLP / NLP with HF Transformers
## Chapter 04

---

### Pipeline



---

### Automodel



---

### Complete

# 08 — Complete / 08 Complete

**Chapter 04 — File 3 of 3 / 第04章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Get predictions**.

本脚本演示 **Get predictions**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import pipeline
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
import logging
from typing import List, Dict

class NERProcessor:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
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
        # 打印输出 / Print output
        print(f"Entity: {entity['word']}")
        # 打印输出 / Print output
        print(f"Type: {entity['entity_group']}")
        # 打印输出 / Print output
        print(f"Confidence: {entity['score']:.4f}")
        # 打印输出 / Print output
        print("-" * 30)
```

---
## Learning Notes / 学习笔记

- **概念**: Get predictions 是机器学习中的常用技术。  
  *Get predictions is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Complete / 08 Complete
# Complete Code / 完整代码
# ===============================

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import pipeline
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
import logging
from typing import List, Dict

class NERProcessor:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
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
        # 打印输出 / Print output
        print(f"Entity: {entity['word']}")
        # 打印输出 / Print output
        print(f"Type: {entity['entity_group']}")
        # 打印输出 / Print output
        print(f"Confidence: {entity['score']:.4f}")
        # 打印输出 / Print output
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
