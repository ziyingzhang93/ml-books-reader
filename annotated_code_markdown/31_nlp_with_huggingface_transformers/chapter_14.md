# HuggingFace Transformers NLP / NLP with HF Transformers
## Chapter 14

---

### Generate

# 01 — Generate / 01 Generate

**Chapter 14 — File 1 of 4 / 第14章 — 第1个文件（共4个）**

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
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — Step 1

```python
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModel
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
```

---
## Step 2 — Load pre-trained model and tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
```

---
## Step 3 — Define some example sentences

```python
sentences = [
    "The cat sat on the mat.",
    "The dog slept on the floor.",
    "I love natural language processing."
]

def get_embeddings(sentences, model, tokenizer):
    "Function to get embeddings for a batch of sentences"
```

---
## Step 4 — Tokenize input and get model output

```python
encoded_input = tokenizer(sentences, padding=True, truncation=True,
                              return_tensors="pt")
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        model_output = model(**encoded_input)
```

---
## Step 5 — Use the CLS token embedding as the sentence embedding

```python
sentence_embeddings = model_output.last_hidden_state[:, 0, :]
```

---
## Step 6 — Convert torch tensor to numpy array for easier handling

```python
return sentence_embeddings.numpy()
```

---
## Step 7 — Get embeddings for our example sentences

```python
embeddings = get_embeddings(sentences, model, tokenizer)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Embedding shape: {embeddings.shape}")
# 打印输出 / Print output
print("First 5 dimensions of the sentences' embeddings:")
# 打印输出 / Print output
print(f"{np.round(embeddings[:, :5], 3)}")
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
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `numpy` | 数值计算库 | Numerical computing library |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Generate / 01 Generate
# Complete Code / 完整代码
# ===============================

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModel
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Define some example sentences
sentences = [
    "The cat sat on the mat.",
    "The dog slept on the floor.",
    "I love natural language processing."
]

def get_embeddings(sentences, model, tokenizer):
    "Function to get embeddings for a batch of sentences"

    # Tokenize input and get model output
    encoded_input = tokenizer(sentences, padding=True, truncation=True,
                              return_tensors="pt")
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Use the CLS token embedding as the sentence embedding
    sentence_embeddings = model_output.last_hidden_state[:, 0, :]

    # Convert torch tensor to numpy array for easier handling
    return sentence_embeddings.numpy()

# Get embeddings for our example sentences
embeddings = get_embeddings(sentences, model, tokenizer)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Embedding shape: {embeddings.shape}")
# 打印输出 / Print output
print("First 5 dimensions of the sentences' embeddings:")
# 打印输出 / Print output
print(f"{np.round(embeddings[:, :5], 3)}")
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Meanpooling

# 03 — Meanpooling / 03 Meanpooling

**Chapter 14 — File 2 of 4 / 第14章 — 第2个文件（共4个）**

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
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — Step 1

```python
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModel
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
```

---
## Step 2 — Load pre-trained model and tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
```

---
## Step 3 — Define some example sentences

```python
sentences = [
    "The cat sat on the mat.",
    "The dog slept on the floor.",
    "I love natural language processing."
]

def get_embeddings(sentences, model, tokenizer):
    "Function to get embeddings for a batch of sentences with mean pooling"
```

---
## Step 4 — Tokenize input and get model output

```python
encoded_input = tokenizer(sentences, padding=True, truncation=True,
                              return_tensors="pt")
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        model_output = model(**encoded_input)
```

---
## Step 5 — Extract the attention mask and output sequence

```python
attention_mask = encoded_input["attention_mask"]
    output_seq = model_output.last_hidden_state
```

---
## Step 6 — Mean pooling: take the average of all token embeddings

```python
# 查看张量/数组形状 / Check tensor/array shape
mask = attention_mask.unsqueeze(-1).expand(output_seq.size()).float()
    sum_embeddings = (output_seq * mask).sum(1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask
```

---
## Step 7 — Convert torch tensor to numpy array for easier handling

```python
return mean_pooled.numpy()
```

---
## Step 8 — Get embeddings with mean pooling

```python
embeddings = get_embeddings(sentences, model, tokenizer)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Embedding shape: {embeddings.shape}")
# 打印输出 / Print output
print("First 5 dimensions of the sentences' embeddings:")
# 打印输出 / Print output
print(f"{np.round(embeddings[:, :5], 3)}")
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
| `numpy` | 数值计算库 | Numerical computing library |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Meanpooling / 03 Meanpooling
# Complete Code / 完整代码
# ===============================

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModel
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Define some example sentences
sentences = [
    "The cat sat on the mat.",
    "The dog slept on the floor.",
    "I love natural language processing."
]

def get_embeddings(sentences, model, tokenizer):
    "Function to get embeddings for a batch of sentences with mean pooling"

    # Tokenize input and get model output
    encoded_input = tokenizer(sentences, padding=True, truncation=True,
                              return_tensors="pt")
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Extract the attention mask and output sequence
    attention_mask = encoded_input["attention_mask"]
    output_seq = model_output.last_hidden_state

    # Mean pooling: take the average of all token embeddings
    # 查看张量/数组形状 / Check tensor/array shape
    mask = attention_mask.unsqueeze(-1).expand(output_seq.size()).float()
    sum_embeddings = (output_seq * mask).sum(1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask

    # Convert torch tensor to numpy array for easier handling
    return mean_pooled.numpy()

# Get embeddings with mean pooling
embeddings = get_embeddings(sentences, model, tokenizer)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Embedding shape: {embeddings.shape}")
# 打印输出 / Print output
print("First 5 dimensions of the sentences' embeddings:")
# 打印输出 / Print output
print(f"{np.round(embeddings[:, :5], 3)}")
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Sentence

# 05 — Sentence / 05 Sentence

**Chapter 14 — File 3 of 4 / 第14章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Define some example sentences**.

本脚本演示 **Define some example sentences**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
from sentence_transformers import SentenceTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics.pairwise import cosine_similarity
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
```

---
## Step 2 — Define some example sentences

```python
sentences = [
    "The cat sat on the mat.",
    "The dog slept on the floor.",
    "I love natural language processing."
]
```

---
## Step 3 — Load a pre-trained model and generate embeddings

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)
```

---
## Step 4 — Get embeddings with mean pooling

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Embedding shape: {embeddings.shape}")
# 打印输出 / Print output
print("First 5 dimensions of the sentences' embeddings:")
# 打印输出 / Print output
print(f"{np.round(embeddings[:, :5], 3)}")
```

---
## Step 5 — Calculate cosine similarity between the first two sentences

```python
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
# 打印输出 / Print output
print(f"Cosine similarity between '{sentences[0]}' and '{sentences[1]}':",
      np.round(similarity[0][0], 3))
```

---
## Learning Notes / 学习笔记

- **概念**: Define some example sentences 是机器学习中的常用技术。  
  *Define some example sentences is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `numpy` | 数值计算库 | Numerical computing library |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sentence / 05 Sentence
# Complete Code / 完整代码
# ===============================

from sentence_transformers import SentenceTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics.pairwise import cosine_similarity
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# Define some example sentences
sentences = [
    "The cat sat on the mat.",
    "The dog slept on the floor.",
    "I love natural language processing."
]

# Load a pre-trained model and generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

# Get embeddings with mean pooling
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Embedding shape: {embeddings.shape}")
# 打印输出 / Print output
print("First 5 dimensions of the sentences' embeddings:")
# 打印输出 / Print output
print(f"{np.round(embeddings[:, :5], 3)}")

# Calculate cosine similarity between the first two sentences
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
# 打印输出 / Print output
print(f"Cosine similarity between '{sentences[0]}' and '{sentences[1]}':",
      np.round(similarity[0][0], 3))
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Allpairs

# 06 — Allpairs / 06 Allpairs

**Chapter 14 — File 4 of 4 / 第14章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Define some example sentences**.

本脚本演示 **Define some example sentences**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
from sentence_transformers import SentenceTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics.pairwise import cosine_similarity
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
```

---
## Step 2 — Define some example sentences

```python
sentences = [
    "The cat sat on the mat.",
    "The dog slept on the floor.",
    "I love natural language processing."
]
```

---
## Step 3 — Load a pre-trained model and generate embeddings

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)
```

---
## Step 4 — Get embeddings with mean pooling

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Embedding shape: {embeddings.shape}")
# 打印输出 / Print output
print("First 5 dimensions of the sentences' embeddings:")
# 打印输出 / Print output
print(f"{np.round(embeddings[:, :5], 3)}")
```

---
## Step 5 — Calculate cosine similarity between the all pairs

```python
# 打印输出 / Print output
print(cosine_similarity(embeddings, embeddings).round(3))
```

---
## Learning Notes / 学习笔记

- **概念**: Define some example sentences 是机器学习中的常用技术。  
  *Define some example sentences is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `numpy` | 数值计算库 | Numerical computing library |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Allpairs / 06 Allpairs
# Complete Code / 完整代码
# ===============================

from sentence_transformers import SentenceTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics.pairwise import cosine_similarity
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# Define some example sentences
sentences = [
    "The cat sat on the mat.",
    "The dog slept on the floor.",
    "I love natural language processing."
]

# Load a pre-trained model and generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

# Get embeddings with mean pooling
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Embedding shape: {embeddings.shape}")
# 打印输出 / Print output
print("First 5 dimensions of the sentences' embeddings:")
# 打印输出 / Print output
print(f"{np.round(embeddings[:, :5], 3)}")

# Calculate cosine similarity between the all pairs
# 打印输出 / Print output
print(cosine_similarity(embeddings, embeddings).round(3))
```

---

### Chapter Summary / 章节总结

# Chapter 14 Summary / 第14章总结

## Theme / 主题: Chapter 14 / Chapter 14

This chapter contains **4 code files** demonstrating chapter 14.

本章包含 **4 个代码文件**，演示Chapter 14。

---
## Evolution / 演化路线

  1. `01_generate.ipynb` — Generate
  2. `03_meanpooling.ipynb` — Meanpooling
  3. `05_sentence.ipynb` — Sentence
  4. `06_allpairs.ipynb` — Allpairs

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 14) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 14）是机器学习流水线中的基础构建块。

---
