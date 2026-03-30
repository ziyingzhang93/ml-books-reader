# HF Transformers
## Chapter 17

---

### Summarize

# 02 — Summarize / 02 Summarize

**Chapter 17 — File 2 of 2 / 第17章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Tokenize the input**.

本脚本演示 **Tokenize the input**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_sentence_embedding(sentence, model, tokenizer):
    """Sentence embedding extracted from the [CLS] prefix token"""
```

---
## Step 2 — Tokenize the input

```python
inputs = tokenizer(sentence, return_tensors="pt",
                       add_special_tokens=True, truncation=True, max_length=512)
```

---
## Step 3 — Forward pass, get hidden states

```python
with torch.no_grad():
        outputs = model(**inputs)
```

---
## Step 4 — Get the [CLS] token embedding at position 0 from the last layer

```python
cls_embedding = outputs.last_hidden_state[0, 0].numpy()
    return cls_embedding

def extractive_summarize(document, model, tokenizer, num_sentences=3):
```

---
## Step 5 — Split the document into sentences

```python
sentences = [s.strip() for s in document.split(".") if s.strip()]
    if len(sentences) <= num_sentences:
        return document
```

---
## Step 6 — Get embeddings for all sentences

```python
sentence_embeddings = []
    for sentence in sentences:
        embedding = get_sentence_embedding(sentence, model, tokenizer)
        sentence_embeddings.append(embedding)
```

---
## Step 7 — Calculate the document embedding (average of all sentence embeddings)
then find the most similar sentences

```python
document_embedding = np.mean(sentence_embeddings, axis=0)
    similarities = []
    for idx, embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(embedding, document_embedding)
        similarities.append((sim, idx))
    top_sentences = sorted(similarities, reverse=True)[:num_sentences]
```

---
## Step 8 — Extract the sentences, preserve the original order

```python
top_indices = sorted([x[1] for x in top_sentences])
    summary_sentences = [sentences[i] for i in top_indices]
```

---
## Step 9 — Join the sentences to form the summary

```python
summary = ". ".join(summary_sentences) + "."
    return summary
```

---
## Step 10 — Example document

```python
document = """
Transformer models have revolutionized natural language processing by
introducing mechanisms that can effectively capture contextual relationships in
text. One of the most powerful aspects of transformers is their ability to
generate context-aware vector representations, often referred to as context
vectors. Unlike traditional word embeddings that assign a fixed vector to each
word regardless of context, transformer models generate dynamic representations
that depend on the surrounding words. This allows them to capture the nuanced
meanings of words in different contexts. For example, in the sentences "I'm
going to the bank to deposit money" and "I'm going to sit by the river bank,"
the word "bank" has different meanings. A traditional word embedding would
assign the same vector to "bank" in both sentences, but a transformer model
generates different context vectors that capture the distinct meanings based on
the surrounding words. This contextual understanding enables transformers to
excel at a wide range of NLP tasks, from question answering and sentiment
analysis to machine translation and text summarization.
"""
```

---
## Step 11 — Generate a summary

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
summary = extractive_summarize(document, model, tokenizer, num_sentences=3)
```

---
## Step 12 — Print the original document and the summary

```python
print("Original Document:")
print(document)
print("Summary:")
print(summary)
```

---
## Learning Notes / 学习笔记

- **概念**: Tokenize the input 是机器学习中的常用技术。  
  *Tokenize the input is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `np.dot` | 矩阵点积/向量内积 | Matrix dot product / vector inner product |
| `np.linalg` | 线性代数运算 | Linear algebra operations |
| `np.mean` | 计算均值 | Calculate mean |
| `numpy` | 数值计算库 | Numerical computing library |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Summarize / 02 Summarize
# Complete Code / 完整代码
# ===============================

import numpy as np
import torch
from transformers import BertTokenizer, BertModel

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_sentence_embedding(sentence, model, tokenizer):
    """Sentence embedding extracted from the [CLS] prefix token"""
    # Tokenize the input
    inputs = tokenizer(sentence, return_tensors="pt",
                       add_special_tokens=True, truncation=True, max_length=512)

    # Forward pass, get hidden states
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the [CLS] token embedding at position 0 from the last layer
    cls_embedding = outputs.last_hidden_state[0, 0].numpy()
    return cls_embedding

def extractive_summarize(document, model, tokenizer, num_sentences=3):
    # Split the document into sentences
    sentences = [s.strip() for s in document.split(".") if s.strip()]
    if len(sentences) <= num_sentences:
        return document

    # Get embeddings for all sentences
    sentence_embeddings = []
    for sentence in sentences:
        embedding = get_sentence_embedding(sentence, model, tokenizer)
        sentence_embeddings.append(embedding)

    # Calculate the document embedding (average of all sentence embeddings)
    # then find the most similar sentences
    document_embedding = np.mean(sentence_embeddings, axis=0)
    similarities = []
    for idx, embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(embedding, document_embedding)
        similarities.append((sim, idx))
    top_sentences = sorted(similarities, reverse=True)[:num_sentences]

    # Extract the sentences, preserve the original order
    top_indices = sorted([x[1] for x in top_sentences])
    summary_sentences = [sentences[i] for i in top_indices]

    # Join the sentences to form the summary
    summary = ". ".join(summary_sentences) + "."
    return summary

# Example document
document = """
Transformer models have revolutionized natural language processing by
introducing mechanisms that can effectively capture contextual relationships in
text. One of the most powerful aspects of transformers is their ability to
generate context-aware vector representations, often referred to as context
vectors. Unlike traditional word embeddings that assign a fixed vector to each
word regardless of context, transformer models generate dynamic representations
that depend on the surrounding words. This allows them to capture the nuanced
meanings of words in different contexts. For example, in the sentences "I'm
going to the bank to deposit money" and "I'm going to sit by the river bank,"
the word "bank" has different meanings. A traditional word embedding would
assign the same vector to "bank" in both sentences, but a transformer model
generates different context vectors that capture the distinct meanings based on
the surrounding words. This contextual understanding enables transformers to
excel at a wide range of NLP tasks, from question answering and sentiment
analysis to machine translation and text summarization.
"""

# Generate a summary
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
summary = extractive_summarize(document, model, tokenizer, num_sentences=3)

# Print the original document and the summary
print("Original Document:")
print(document)
print("Summary:")
print(summary)
```

---

### Chapter Summary

# Chapter 17 Summary / 第17章总结

## Theme / 主题: Chapter 17 / Chapter 17

This chapter contains **2 code files** demonstrating chapter 17.

本章包含 **2 个代码文件**，演示Chapter 17。

---
## Evolution / 演化路线

  1. `01_keyword.ipynb` — Keyword
  2. `02_summarize.ipynb` — Summarize

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 17) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 17）是机器学习流水线中的基础构建块。

---
