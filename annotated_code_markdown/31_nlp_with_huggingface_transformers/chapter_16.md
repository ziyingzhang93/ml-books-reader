# HF Transformers
## Chapter 16

---

### Context

# 01 — Context / 01 Context

**Chapter 16 — File 1 of 4 / 第16章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load pre-trained model and tokenizer**.

本脚本演示 **Load pre-trained model and tokenizer**。

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
from transformers import BertModel, BertTokenizer
```

---
## Step 2 — Load pre-trained model and tokenizer

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # for safety: set to evaluation mode

def get_context_vectors(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
```

---
## Step 3 — Get the tokens (for reference)

```python
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
```

---
## Step 4 — Forward pass, get all hidden states from each layer

```python
with torch.no_grad():
        outputs = model(input_ids,
                        attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states
```

---
## Step 5 — Each element in hidden states has shape (batch_size, sequence_length, hidden_size)
Here takes the first element in the batch from the last layer

```python
last_layer_vectors = hidden_states[-1][0].numpy()  # Shape: (seq length, hidden size)

    return tokens, last_layer_vectors
```

---
## Step 6 — Get context vectors from example sentences with ambiguous words

```python
sentence1 = "I'm going to the bank to deposit money."
sentence2 = "I'm going to sit by the river bank."
tokens1, vectors1 = get_context_vectors(sentence1, model, tokenizer)
tokens2, vectors2 = get_context_vectors(sentence2, model, tokenizer)
```

---
## Step 7 — Print the tokens for reference

```python
print("Tokens in sentence 1:", tokens1)
print("Tokens in sentence 2:", tokens2)
```

---
## Step 8 — Find the index of "bank" in both sentences

```python
bank_idx1 = tokens1.index("bank")
bank_idx2 = tokens2.index("bank")
```

---
## Step 9 — Get the context vectors for "bank" in both sentences

```python
bank_vector1 = vectors1[bank_idx1]
bank_vector2 = vectors2[bank_idx2]
```

---
## Step 10 — Calculate cosine similarity between the two "bank" vectors
lower similarity means meaning is different

```python
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

similarity = cosine_similarity(bank_vector1, bank_vector2)
print(f"Cosine similarity between 'bank' vectors: {similarity:.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Load pre-trained model and tokenizer 是机器学习中的常用技术。  
  *Load pre-trained model and tokenizer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `np.dot` | 矩阵点积/向量内积 | Matrix dot product / vector inner product |
| `np.linalg` | 线性代数运算 | Linear algebra operations |
| `numpy` | 数值计算库 | Numerical computing library |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Context / 01 Context
# Complete Code / 完整代码
# ===============================

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # for safety: set to evaluation mode

def get_context_vectors(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get the tokens (for reference)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Forward pass, get all hidden states from each layer
    with torch.no_grad():
        outputs = model(input_ids,
                        attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    # Each element in hidden states has shape (batch_size, sequence_length, hidden_size)
    # Here takes the first element in the batch from the last layer
    last_layer_vectors = hidden_states[-1][0].numpy()  # Shape: (seq length, hidden size)

    return tokens, last_layer_vectors

# Get context vectors from example sentences with ambiguous words
sentence1 = "I'm going to the bank to deposit money."
sentence2 = "I'm going to sit by the river bank."
tokens1, vectors1 = get_context_vectors(sentence1, model, tokenizer)
tokens2, vectors2 = get_context_vectors(sentence2, model, tokenizer)

# Print the tokens for reference
print("Tokens in sentence 1:", tokens1)
print("Tokens in sentence 2:", tokens2)

# Find the index of "bank" in both sentences
bank_idx1 = tokens1.index("bank")
bank_idx2 = tokens2.index("bank")

# Get the context vectors for "bank" in both sentences
bank_vector1 = vectors1[bank_idx1]
bank_vector2 = vectors2[bank_idx2]

# Calculate cosine similarity between the two "bank" vectors
# lower similarity means meaning is different
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

similarity = cosine_similarity(bank_vector1, bank_vector2)
print(f"Cosine similarity between 'bank' vectors: {similarity:.4f}")
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Layers

# 02 — Layers / 02 Layers

**Chapter 16 — File 2 of 4 / 第16章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load pre-trained model and tokenizer**.

本脚本演示 **Load pre-trained model and tokenizer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
```

---
## Step 2 — Load pre-trained model and tokenizer

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # for safety: set to evaluation mode

def get_all_layer_vectors(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
```

---
## Step 3 — Get the tokens (for reference)

```python
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
```

---
## Step 4 — Forward pass, get all hidden states from each layer

```python
with torch.no_grad():
        outputs = model(input_ids,
                        attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states
```

---
## Step 5 — Convert from torch tensor to numpy arrays, take only the first element in the batch

```python
all_layers_vectors = [layer[0].numpy() for layer in hidden_states]

    return tokens, all_layers_vectors
```

---
## Step 6 — Get vectors from all layers for a sentence

```python
sentence = "The quick brown fox jumps over the lazy dog."
tokens, all_layers = get_all_layer_vectors(sentence, model, tokenizer)
print(f"Number of layers (including embedding layer): {len(all_layers)}")
```

---
## Step 7 — Let's analyze how the representation of a word changes across layers

```python
word = "fox"
word_idx = tokens.index(word)
```

---
## Step 8 — Extract the vector for this word from each layer

```python
word_vectors = [layer[word_idx] for layer in all_layers]
```

---
## Step 9 — Calculate the cosine similarity between consecutive layers

```python
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

similarities = []
for i in range(len(word_vectors) - 1):
    sim = cosine_similarity(word_vectors[i], word_vectors[i+1])
    similarities.append(sim)
```

---
## Step 10 — Plot the similarities

```python
plt.figure(figsize=(10, 6))
plt.plot(similarities, marker="o")
plt.title(f"Cosine Similarity Between Consecutive Layers for '{word}'")
plt.xlabel("Layer Transition")
plt.ylabel("Cosine Similarity")
plt.xticks(range(len(similarities)), [f"{i}->{i+1}" for i in range(len(similarities))])
plt.grid(True)
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load pre-trained model and tokenizer 是机器学习中的常用技术。  
  *Load pre-trained model and tokenizer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `matplotlib` | 绑图库 | Plotting library |
| `np.dot` | 矩阵点积/向量内积 | Matrix dot product / vector inner product |
| `np.linalg` | 线性代数运算 | Linear algebra operations |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Layers / 02 Layers
# Complete Code / 完整代码
# ===============================

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # for safety: set to evaluation mode

def get_all_layer_vectors(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get the tokens (for reference)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Forward pass, get all hidden states from each layer
    with torch.no_grad():
        outputs = model(input_ids,
                        attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    # Convert from torch tensor to numpy arrays, take only the first element in the batch
    all_layers_vectors = [layer[0].numpy() for layer in hidden_states]

    return tokens, all_layers_vectors

# Get vectors from all layers for a sentence
sentence = "The quick brown fox jumps over the lazy dog."
tokens, all_layers = get_all_layer_vectors(sentence, model, tokenizer)
print(f"Number of layers (including embedding layer): {len(all_layers)}")

# Let's analyze how the representation of a word changes across layers
word = "fox"
word_idx = tokens.index(word)

# Extract the vector for this word from each layer
word_vectors = [layer[word_idx] for layer in all_layers]

# Calculate the cosine similarity between consecutive layers
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

similarities = []
for i in range(len(word_vectors) - 1):
    sim = cosine_similarity(word_vectors[i], word_vectors[i+1])
    similarities.append(sim)

# Plot the similarities
plt.figure(figsize=(10, 6))
plt.plot(similarities, marker="o")
plt.title(f"Cosine Similarity Between Consecutive Layers for '{word}'")
plt.xlabel("Layer Transition")
plt.ylabel("Cosine Similarity")
plt.xticks(range(len(similarities)), [f"{i}->{i+1}" for i in range(len(similarities))])
plt.grid(True)
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Chapter Summary

# Chapter 16 Summary / 第16章总结

## Theme / 主题: Chapter 16 / Chapter 16

This chapter contains **4 code files** demonstrating chapter 16.

本章包含 **4 个代码文件**，演示Chapter 16。

---
## Evolution / 演化路线

  1. `01_context.ipynb` — Context
  2. `02_layers.ipynb` — Layers
  3. `03_wordsense.ipynb` — Wordsense
  4. `04_patterns.ipynb` — Patterns

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 16) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 16）是机器学习流水线中的基础构建块。

---
