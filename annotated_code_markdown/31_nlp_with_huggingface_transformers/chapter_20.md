# HuggingFace Transformers NLP / NLP with HF Transformers
## Chapter 20

---

### Query

# 01 — Query / 01 Query

**Chapter 20 — File 1 of 3 / 第20章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Load BART model and tokenizer**.

本脚本演示 **Load BART model and tokenizer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import BartForConditionalGeneration, BartTokenizer
```

---
## Step 2 — Load BART model and tokenizer

```python
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

def reformulate_query(query, n=2):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=64,
        num_beams=10,
        num_return_sequences=n,
        temperature=1.5,  # High temperature for diversity
        top_k=50,
        do_sample=True
    )
```

---
## Step 3 — Decode the outputs one by one

```python
reformulations = [tokenizer.decode(output, skip_special_tokens=True)
                      for output in outputs]
    all_queries = [query] + reformulations
    return all_queries
```

---
## Step 4 — Generate reformulations from an example query

```python
query = "How do transformer-based systems process natural language?"
reformulated_queries = reformulate_query(query)
# 打印输出 / Print output
print(f"Original Query: {query}")
# 打印输出 / Print output
print("Reformulated Queries:")
# 同时获取索引和值 / Get both index and value
for i, q in enumerate(reformulated_queries[1:], 1):
    # 打印输出 / Print output
    print(f"{i}. {q}")
```

---
## Learning Notes / 学习笔记

- **概念**: Load BART model and tokenizer 是机器学习中的常用技术。  
  *Load BART model and tokenizer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Query / 01 Query
# Complete Code / 完整代码
# ===============================

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import BartForConditionalGeneration, BartTokenizer

# Load BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

def reformulate_query(query, n=2):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=64,
        num_beams=10,
        num_return_sequences=n,
        temperature=1.5,  # High temperature for diversity
        top_k=50,
        do_sample=True
    )
    # Decode the outputs one by one
    reformulations = [tokenizer.decode(output, skip_special_tokens=True)
                      for output in outputs]
    all_queries = [query] + reformulations
    return all_queries

# Generate reformulations from an example query
query = "How do transformer-based systems process natural language?"
reformulated_queries = reformulate_query(query)
# 打印输出 / Print output
print(f"Original Query: {query}")
# 打印输出 / Print output
print("Reformulated Queries:")
# 同时获取索引和值 / Get both index and value
for i, q in enumerate(reformulated_queries[1:], 1):
    # 打印输出 / Print output
    print(f"{i}. {q}")
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Hybrid



---

### Multistage

# 04 — Multistage / 04 Multistage

**Chapter 20 — File 3 of 3 / 第20章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Load pre-trained model and tokenizer for re-ranking**.

本脚本演示 **Load pre-trained model and tokenizer for re-ranking**。

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
from rank_bm25 import BM25Okapi
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import faiss
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

dense_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
dense_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
```

---
## Step 2 — Load pre-trained model and tokenizer for re-ranking

```python
reranker_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker_model = AutoModelForSequenceClassification.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L-6-v2")

def generate_embedding(text):
    """Generate dense vector using mean pooling"""
    inputs = dense_tokenizer(text, padding=True, truncation=True, return_tensors="pt",
                             max_length=512)
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        outputs = dense_model(**inputs)

    attention_mask = inputs["attention_mask"]
    embeddings = outputs.last_hidden_state

    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
    sum_embeddings = torch.sum(embeddings * expanded_mask, axis=1)
    sum_mask = torch.clamp(expanded_mask.sum(axis=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings.cpu().numpy()

def hybrid_retrieval(query, k=3, alpha=0.5):
    """Hybrid retrieval: Use both the BM25 and L2 index on FAISS"""
```

---
## Step 3 — Sparse score of each document with BM25

```python
tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
```

---
## Step 4 — Normalize BM25 scores to [0,1] unless all elements are zero

```python
if max(bm25_scores) > 0:
        bm25_scores = bm25_scores / max(bm25_scores)
```

---
## Step 5 — Sort all documents according to L2 distance to query

```python
query_embedding = generate_embedding(query)
    # 获取长度 / Get length
    distances, indices = index.search(query_embedding, len(documents))
```

---
## Step 6 — Dense score: 1/distance as similarity metric, then normalize to [0,1]

```python
eps = 1e-5  # a small value to prevent division by zero
    # 创建NumPy数组 / Create NumPy array
    dense_scores = 1 / (eps + np.array(distances[0]))
    dense_scores = dense_scores / max(dense_scores)
```

---
## Step 7 — Combine scores = affine combination of sparse and dense scores

```python
combined_scores = alpha * dense_scores + (1 - alpha) * bm25_scores
```

---
## Step 8 — Get top-k documents

```python
top_indices = np.argsort(combined_scores)[::-1][:k]
    results = [(documents[idx], combined_scores[idx]) for idx in top_indices]
    return results

def rerank(query, documents, top_k=3):
    """Sort documents by the reranker model and select top-k"""
```

---
## Step 9 — Prepare inputs for the re-ranker

```python
pairs = [[query, doc] for doc in documents]
    features = reranker_tokenizer(pairs, padding=True, truncation=True,
                                  return_tensors="pt")
```

---
## Step 10 — Get re-ranking scores

```python
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
        scores = reranker_model(**features).logits.squeeze(-1).cpu().numpy()
```

---
## Step 11 — Sort documents by score, then pick top-k

```python
ranked_indices = np.argsort(scores)[::-1][:top_k]
    reranked_docs = [(documents[idx], float(scores[idx])) for idx in ranked_indices]
    return reranked_docs

def multi_stage_retrieval(query, documents, initial_k=5, final_k=3):
    """Multi-stage retrieval: Hybrid retrievel to shortlist documents, then pick
    with a reranker
    """
```

---
## Step 12 — Stage 1: Initial retrieval using hybrid method

```python
initial_results = hybrid_retrieval(query, k=initial_k)
    initial_docs = [doc for doc, _ in initial_results]
```

---
## Step 13 — Stage 2: Re-ranking

```python
reranked_results = rerank(query, initial_docs, top_k=final_k)
    return reranked_results
```

---
## Step 14 — Sample document collection

```python
documents = [
    "Transformers use self-attention mechanisms to process input sequences in "
        "parallel, making them efficient for long sequences.",
    "The attention mechanism in transformers allows the model to focus on different "
        "parts of the input sequence when generating each output element.",
    "Transformer models have a fixed context length determined by the positional "
        "encoding and self-attention mechanisms.",
    "To handle sequences longer than the context length, transformers can use "
        "techniques like sliding windows or hierarchical processing.",
    "Recurrent Neural Networks (RNNs) process sequences sequentially, which can be "
        "inefficient for long sequences.",
    "Long Short-Term Memory (LSTM) networks are a type of RNN designed to handle "
        "long-term dependencies in sequences.",
    "The Transformer architecture was introduced in the paper 'Attention Is All "
        "You Need' by Vaswani et al.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a "
        "transformer-based model designed for understanding the context of words.",
    "GPT (Generative Pre-trained Transformer) is a transformer-based model designed "
        "for natural language generation.",
    "Transformer-XL extends the context length of transformers by using a "
        "segment-level recurrence mechanism."
]
```

---
## Step 15 — Prepare for sparse retrieval (BM25)

```python
tokenized_corpus = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)
```

---
## Step 16 — Prepare for dense retrieval (FAISS)

```python
document_embeddings = generate_embedding(documents)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)
```

---
## Step 17 — Example query

```python
query = "How do transformers handle long sequences?"
results = multi_stage_retrieval(query, documents)
# 打印输出 / Print output
print(f"Query: {query}")
# 打印输出 / Print output
print("Re-ranked Results:")
# 同时获取索引和值 / Get both index and value
for i, (doc, score) in enumerate(results):
    # 打印输出 / Print output
    print(f"Document {i+1} (Score: {score:.4f}):")
    # 打印输出 / Print output
    print(doc)
    # 打印输出 / Print output
    print()
```

---
## Learning Notes / 学习笔记

- **概念**: Load pre-trained model and tokenizer for re-ranking 是机器学习中的常用技术。  
  *Load pre-trained model and tokenizer for re-ranking is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Multistage / 04 Multistage
# Complete Code / 完整代码
# ===============================

from rank_bm25 import BM25Okapi
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import faiss
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

dense_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
dense_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Load pre-trained model and tokenizer for re-ranking
reranker_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker_model = AutoModelForSequenceClassification.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L-6-v2")

def generate_embedding(text):
    """Generate dense vector using mean pooling"""
    inputs = dense_tokenizer(text, padding=True, truncation=True, return_tensors="pt",
                             max_length=512)
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        outputs = dense_model(**inputs)

    attention_mask = inputs["attention_mask"]
    embeddings = outputs.last_hidden_state

    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
    sum_embeddings = torch.sum(embeddings * expanded_mask, axis=1)
    sum_mask = torch.clamp(expanded_mask.sum(axis=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings.cpu().numpy()

def hybrid_retrieval(query, k=3, alpha=0.5):
    """Hybrid retrieval: Use both the BM25 and L2 index on FAISS"""
    # Sparse score of each document with BM25
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize BM25 scores to [0,1] unless all elements are zero
    if max(bm25_scores) > 0:
        bm25_scores = bm25_scores / max(bm25_scores)

    # Sort all documents according to L2 distance to query
    query_embedding = generate_embedding(query)
    # 获取长度 / Get length
    distances, indices = index.search(query_embedding, len(documents))

    # Dense score: 1/distance as similarity metric, then normalize to [0,1]
    eps = 1e-5  # a small value to prevent division by zero
    # 创建NumPy数组 / Create NumPy array
    dense_scores = 1 / (eps + np.array(distances[0]))
    dense_scores = dense_scores / max(dense_scores)

    # Combine scores = affine combination of sparse and dense scores
    combined_scores = alpha * dense_scores + (1 - alpha) * bm25_scores

    # Get top-k documents
    top_indices = np.argsort(combined_scores)[::-1][:k]
    results = [(documents[idx], combined_scores[idx]) for idx in top_indices]
    return results

def rerank(query, documents, top_k=3):
    """Sort documents by the reranker model and select top-k"""
    # Prepare inputs for the re-ranker
    pairs = [[query, doc] for doc in documents]
    features = reranker_tokenizer(pairs, padding=True, truncation=True,
                                  return_tensors="pt")
    # Get re-ranking scores
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        scores = reranker_model(**features).logits.squeeze(-1).cpu().numpy()
    # Sort documents by score, then pick top-k
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    reranked_docs = [(documents[idx], float(scores[idx])) for idx in ranked_indices]
    return reranked_docs

def multi_stage_retrieval(query, documents, initial_k=5, final_k=3):
    """Multi-stage retrieval: Hybrid retrievel to shortlist documents, then pick
    with a reranker
    """
    # Stage 1: Initial retrieval using hybrid method
    initial_results = hybrid_retrieval(query, k=initial_k)
    initial_docs = [doc for doc, _ in initial_results]
    # Stage 2: Re-ranking
    reranked_results = rerank(query, initial_docs, top_k=final_k)
    return reranked_results

# Sample document collection
documents = [
    "Transformers use self-attention mechanisms to process input sequences in "
        "parallel, making them efficient for long sequences.",
    "The attention mechanism in transformers allows the model to focus on different "
        "parts of the input sequence when generating each output element.",
    "Transformer models have a fixed context length determined by the positional "
        "encoding and self-attention mechanisms.",
    "To handle sequences longer than the context length, transformers can use "
        "techniques like sliding windows or hierarchical processing.",
    "Recurrent Neural Networks (RNNs) process sequences sequentially, which can be "
        "inefficient for long sequences.",
    "Long Short-Term Memory (LSTM) networks are a type of RNN designed to handle "
        "long-term dependencies in sequences.",
    "The Transformer architecture was introduced in the paper 'Attention Is All "
        "You Need' by Vaswani et al.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a "
        "transformer-based model designed for understanding the context of words.",
    "GPT (Generative Pre-trained Transformer) is a transformer-based model designed "
        "for natural language generation.",
    "Transformer-XL extends the context length of transformers by using a "
        "segment-level recurrence mechanism."
]

# Prepare for sparse retrieval (BM25)
tokenized_corpus = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# Prepare for dense retrieval (FAISS)
document_embeddings = generate_embedding(documents)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Example query
query = "How do transformers handle long sequences?"
results = multi_stage_retrieval(query, documents)
# 打印输出 / Print output
print(f"Query: {query}")
# 打印输出 / Print output
print("Re-ranked Results:")
# 同时获取索引和值 / Get both index and value
for i, (doc, score) in enumerate(results):
    # 打印输出 / Print output
    print(f"Document {i+1} (Score: {score:.4f}):")
    # 打印输出 / Print output
    print(doc)
    # 打印输出 / Print output
    print()
```

---

### Chapter Summary / 章节总结

# Chapter 20 Summary / 第20章总结

## Theme / 主题: Chapter 20 / Chapter 20

This chapter contains **3 code files** demonstrating chapter 20.

本章包含 **3 个代码文件**，演示Chapter 20。

---
## Evolution / 演化路线

  1. `01_query.ipynb` — Query
  2. `03_hybrid.ipynb` — Hybrid
  3. `04_multistage.ipynb` — Multistage

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 20) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 20）是机器学习流水线中的基础构建块。

---
