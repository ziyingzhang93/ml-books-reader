# HF Transformers
## Chapter 19

---

### Retrieval

# 04 — Retrieval / 模型评估

**Chapter 19 — File 3 of 5 / 第19章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Tokenize each text and convert to PyTorch tensors**.

本脚本演示 **Tokenize each text and convert to PyTorch tensors**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def generate_embedding(docs, model, tokenizer):
```

---
## Step 2 — Tokenize each text and convert to PyTorch tensors

```python
inputs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt",
                       max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
```

---
## Step 3 — Embedding defined as mean pooling of all tokens

```python
attention_mask = inputs["attention_mask"]
    embeddings = outputs.last_hidden_state

    expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
    sum_embeddings = torch.sum(embeddings * expanded_mask, axis=1)
    sum_mask = torch.clamp(expanded_mask.sum(axis=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
```

---
## Step 4 — Convert to numpy array

```python
return mean_embeddings.cpu().numpy()

def retrieve_documents(query, index, documents, k=3):
```

---
## Step 5 — Generate embedding for the query

```python
query_embedding = generate_embedding(query, model, tokenizer)   # 1xD matrix
```

---
## Step 6 — Search the index for similar documents

```python
distances, indices = index.search(query_embedding, k)  # 1xk matrices
```

---
## Step 7 — Return the retrieved documents and their distances

```python
retrieved_docs = [(documents[idx], float(distances[0][i]))
                      for i, idx in enumerate(indices[0])]
    return retrieved_docs
```

---
## Step 8 — Sample document collection

```python
documents = [
    "Transformers are a type of deep learning model introduced in the paper 'Attention "
        "Is All You Need'.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a "
        "transformer-based model designed to understand the context of a word based on "
        "its surroundings.",
    "GPT (Generative Pre-trained Transformer) is a transformer-based model designed for "
        "natural language generation tasks.",
    "T5 (Text-to-Text Transfer Transformer) treats every NLP problem as a text-to-text "
        "problem, where both the input and output are text strings.",
    "RoBERTa is an optimized version of BERT with improved training methodology and more "
        "training data.",
    "DistilBERT is a smaller, faster version of BERT that retains 97% of its language "
        "understanding capabilities.",
    "ALBERT reduces the parameters of BERT by sharing parameters across layers and using "
        "embedding factorization.",
    "XLNet is a generalized autoregressive pretraining method that overcomes the "
        "limitations of BERT by using permutation language modeling.",
    "ELECTRA uses a generator-discriminator architecture for more efficient pretraining.",
    "DeBERTa enhances BERT with disentangled attention and an enhanced mask decoder."
]
```

---
## Step 9 — Generate embeddings for all documents,
then create FAISS index for efficient similarity search

```python
document_embeddings = generate_embedding(documents, model, tokenizer)
dimension = document_embeddings.shape[1]   # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)       # Using L2 (Euclidean) distance
index.add(document_embeddings)             # Add embeddings to the index
print(f"Created index with {index.ntotal} documents")
```

---
## Step 10 — Example query

```python
query = "What is BERT?"
retrieved_docs = retrieve_documents(query, index, documents)
```

---
## Step 11 — Print the retrieved documents

```python
print(f"Query: {query}\n")
for i, (doc, distance) in enumerate(retrieved_docs):
    print(f"Document {i+1} (Distance: {distance:.4f}):")
    print(doc)
    print()
```

---
## Learning Notes / 学习笔记

- **概念**: Tokenize each text and convert to PyTorch tensors 是机器学习中的常用技术。  
  *Tokenize each text and convert to PyTorch tensors is a common technique in machine learning.*

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
# Retrieval / 模型评估
# Complete Code / 完整代码
# ===============================

import faiss
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def generate_embedding(docs, model, tokenizer):
    # Tokenize each text and convert to PyTorch tensors
    inputs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt",
                       max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    # Embedding defined as mean pooling of all tokens
    attention_mask = inputs["attention_mask"]
    embeddings = outputs.last_hidden_state

    expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
    sum_embeddings = torch.sum(embeddings * expanded_mask, axis=1)
    sum_mask = torch.clamp(expanded_mask.sum(axis=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask

    # Convert to numpy array
    return mean_embeddings.cpu().numpy()

def retrieve_documents(query, index, documents, k=3):
    # Generate embedding for the query
    query_embedding = generate_embedding(query, model, tokenizer)   # 1xD matrix
    # Search the index for similar documents
    distances, indices = index.search(query_embedding, k)  # 1xk matrices
    # Return the retrieved documents and their distances
    retrieved_docs = [(documents[idx], float(distances[0][i]))
                      for i, idx in enumerate(indices[0])]
    return retrieved_docs

# Sample document collection
documents = [
    "Transformers are a type of deep learning model introduced in the paper 'Attention "
        "Is All You Need'.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a "
        "transformer-based model designed to understand the context of a word based on "
        "its surroundings.",
    "GPT (Generative Pre-trained Transformer) is a transformer-based model designed for "
        "natural language generation tasks.",
    "T5 (Text-to-Text Transfer Transformer) treats every NLP problem as a text-to-text "
        "problem, where both the input and output are text strings.",
    "RoBERTa is an optimized version of BERT with improved training methodology and more "
        "training data.",
    "DistilBERT is a smaller, faster version of BERT that retains 97% of its language "
        "understanding capabilities.",
    "ALBERT reduces the parameters of BERT by sharing parameters across layers and using "
        "embedding factorization.",
    "XLNet is a generalized autoregressive pretraining method that overcomes the "
        "limitations of BERT by using permutation language modeling.",
    "ELECTRA uses a generator-discriminator architecture for more efficient pretraining.",
    "DeBERTa enhances BERT with disentangled attention and an enhanced mask decoder."
]

# Generate embeddings for all documents,
# then create FAISS index for efficient similarity search
document_embeddings = generate_embedding(documents, model, tokenizer)
dimension = document_embeddings.shape[1]   # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)       # Using L2 (Euclidean) distance
index.add(document_embeddings)             # Add embeddings to the index
print(f"Created index with {index.ntotal} documents")

# Example query
query = "What is BERT?"
retrieved_docs = retrieve_documents(query, index, documents)

# Print the retrieved documents
print(f"Query: {query}\n")
for i, (doc, distance) in enumerate(retrieved_docs):
    print(f"Document {i+1} (Distance: {distance:.4f}):")
    print(doc)
    print()
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Generator

# 05 — Generator / 05 Generator

**Chapter 19 — File 4 of 5 / 第19章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Tokenize each text and convert to PyTorch tensors**.

本脚本演示 **Tokenize each text and convert to PyTorch tensors**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — Step 1

```python
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
gen_tokenizer = AutoTokenizer.from_pretrained("t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def generate_embedding(docs, model, tokenizer):
```

---
## Step 2 — Tokenize each text and convert to PyTorch tensors

```python
inputs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt",
                       max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
```

---
## Step 3 — Embedding defined as mean pooling of all tokens

```python
attention_mask = inputs["attention_mask"]
    embeddings = outputs.last_hidden_state

    expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
    sum_embeddings = torch.sum(embeddings * expanded_mask, axis=1)
    sum_mask = torch.clamp(expanded_mask.sum(axis=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
```

---
## Step 4 — Convert to numpy array

```python
return mean_embeddings.cpu().numpy()

def retrieve_documents(query, index, documents, k=3):
```

---
## Step 5 — Generate embedding for the query

```python
query_embedding = generate_embedding(query, model, tokenizer)   # 1xD matrix
```

---
## Step 6 — Search the index for similar documents

```python
distances, indices = index.search(query_embedding, k)  # 1xk matrices
```

---
## Step 7 — Return the retrieved documents and their distances

```python
retrieved_docs = [(documents[idx], float(distances[0][i]))
                      for i, idx in enumerate(indices[0])]
    return retrieved_docs

def generate_response(query, retrieved_docs, max_length=150):
```

---
## Step 8 — Combine the query and retrieved documents into a single prompt

```python
context = "\n".join(retrieved_docs)
    prompt = f"question: {query} context: {context}"
```

---
## Step 9 — Generate a response

```python
inputs = gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = gen_model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    response = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
```

---
## Step 10 — Sample document collection

```python
documents = [
    "Transformers are a type of deep learning model introduced in the paper 'Attention "
        "Is All You Need'.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a "
        "transformer-based model designed to understand the context of a word based on "
        "its surroundings.",
    "GPT (Generative Pre-trained Transformer) is a transformer-based model designed for "
        "natural language generation tasks.",
    "T5 (Text-to-Text Transfer Transformer) treats every NLP problem as a text-to-text "
        "problem, where both the input and output are text strings.",
    "RoBERTa is an optimized version of BERT with improved training methodology and more "
        "training data.",
    "DistilBERT is a smaller, faster version of BERT that retains 97% of its language "
        "understanding capabilities.",
    "ALBERT reduces the parameters of BERT by sharing parameters across layers and using "
        "embedding factorization.",
    "XLNet is a generalized autoregressive pretraining method that overcomes the "
        "limitations of BERT by using permutation language modeling.",
    "ELECTRA uses a generator-discriminator architecture for more efficient pretraining.",
    "DeBERTa enhances BERT with disentangled attention and an enhanced mask decoder."
]
document_embeddings = generate_embedding(documents, model, tokenizer)
dimension = document_embeddings.shape[1]   # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)       # Using L2 (Euclidean) distance
index.add(document_embeddings)             # Add embeddings to the index

query = "What is BERT?"
retrieved_docs = retrieve_documents(query, index, documents)
```

---
## Step 11 — Generate a response for the example query

```python
response = generate_response(query, [doc for doc, score in retrieved_docs])
print("Generated Response:")
print(response)
```

---
## Learning Notes / 学习笔记

- **概念**: Tokenize each text and convert to PyTorch tensors 是机器学习中的常用技术。  
  *Tokenize each text and convert to PyTorch tensors is a common technique in machine learning.*

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
# Generator / 05 Generator
# Complete Code / 完整代码
# ===============================

import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
gen_tokenizer = AutoTokenizer.from_pretrained("t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def generate_embedding(docs, model, tokenizer):
    # Tokenize each text and convert to PyTorch tensors
    inputs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt",
                       max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    # Embedding defined as mean pooling of all tokens
    attention_mask = inputs["attention_mask"]
    embeddings = outputs.last_hidden_state

    expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
    sum_embeddings = torch.sum(embeddings * expanded_mask, axis=1)
    sum_mask = torch.clamp(expanded_mask.sum(axis=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask

    # Convert to numpy array
    return mean_embeddings.cpu().numpy()

def retrieve_documents(query, index, documents, k=3):
    # Generate embedding for the query
    query_embedding = generate_embedding(query, model, tokenizer)   # 1xD matrix
    # Search the index for similar documents
    distances, indices = index.search(query_embedding, k)  # 1xk matrices
    # Return the retrieved documents and their distances
    retrieved_docs = [(documents[idx], float(distances[0][i]))
                      for i, idx in enumerate(indices[0])]
    return retrieved_docs

def generate_response(query, retrieved_docs, max_length=150):
    # Combine the query and retrieved documents into a single prompt
    context = "\n".join(retrieved_docs)
    prompt = f"question: {query} context: {context}"

    # Generate a response
    inputs = gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = gen_model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    response = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Sample document collection
documents = [
    "Transformers are a type of deep learning model introduced in the paper 'Attention "
        "Is All You Need'.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a "
        "transformer-based model designed to understand the context of a word based on "
        "its surroundings.",
    "GPT (Generative Pre-trained Transformer) is a transformer-based model designed for "
        "natural language generation tasks.",
    "T5 (Text-to-Text Transfer Transformer) treats every NLP problem as a text-to-text "
        "problem, where both the input and output are text strings.",
    "RoBERTa is an optimized version of BERT with improved training methodology and more "
        "training data.",
    "DistilBERT is a smaller, faster version of BERT that retains 97% of its language "
        "understanding capabilities.",
    "ALBERT reduces the parameters of BERT by sharing parameters across layers and using "
        "embedding factorization.",
    "XLNet is a generalized autoregressive pretraining method that overcomes the "
        "limitations of BERT by using permutation language modeling.",
    "ELECTRA uses a generator-discriminator architecture for more efficient pretraining.",
    "DeBERTa enhances BERT with disentangled attention and an enhanced mask decoder."
]
document_embeddings = generate_embedding(documents, model, tokenizer)
dimension = document_embeddings.shape[1]   # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)       # Using L2 (Euclidean) distance
index.add(document_embeddings)             # Add embeddings to the index

query = "What is BERT?"
retrieved_docs = retrieve_documents(query, index, documents)

# Generate a response for the example query
response = generate_response(query, [doc for doc, score in retrieved_docs])
print("Generated Response:")
print(response)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Complete

# 08 — Complete / 08 Complete

**Chapter 19 — File 5 of 5 / 第19章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Model to use in retriever**.

本脚本演示 **Model to use in retriever**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSeq2SeqLM
```

---
## Step 2 — Model to use in retriever

```python
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
```

---
## Step 3 — Model to use in generator

```python
gen_tokenizer = AutoTokenizer.from_pretrained("t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def generate_embedding(docs, model, tokenizer):
```

---
## Step 4 — Tokenize each text and convert to PyTorch tensors

```python
inputs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt",
                       max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
```

---
## Step 5 — Embedding defined as mean pooling of all tokens

```python
attention_mask = inputs["attention_mask"]
    embeddings = outputs.last_hidden_state

    expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
    sum_embeddings = torch.sum(embeddings * expanded_mask, axis=1)
    sum_mask = torch.clamp(expanded_mask.sum(axis=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
```

---
## Step 6 — Convert to numpy array

```python
return mean_embeddings.cpu().numpy()

def retrieve_documents(query, index, documents, k=3):
```

---
## Step 7 — Generate embedding for the query

```python
query_embedding = generate_embedding(query, model, tokenizer)   # 1xD matrix
```

---
## Step 8 — Search the index for similar documents

```python
distances, indices = index.search(query_embedding, k)  # 1xk matrices
```

---
## Step 9 — Return the retrieved documents and their distances

```python
retrieved_docs = [(documents[idx], float(distances[0][i]))
                      for i, idx in enumerate(indices[0])]
    return retrieved_docs

def generate_response(query, retrieved_docs, max_length=150):
```

---
## Step 10 — Combine the query and retrieved documents into a single prompt

```python
if retrieved_docs:
        context = "\n".join(retrieved_docs)
        prompt = f"question: {query} context: {context}"
    else:
        prompt = f"question: {query}"
```

---
## Step 11 — Generate a response

```python
inputs = gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = gen_model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    response = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def rag_pipeline(query, documents, retriever_k=3, max_length=150):
    retrieved_docs = retrieve_documents(query, index, documents, k=retriever_k)
    docs = [doc for doc, distance in retrieved_docs]
    response = generate_response(query, docs, max_length=max_length)
    return response, retrieved_docs
```

---
## Step 12 — Sample document collection

```python
documents = [
    "Transformers are a type of deep learning model introduced in the paper 'Attention "
        "Is All You Need'.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a "
        "transformer-based model designed to understand the context of a word based on "
        "its surroundings.",
    "GPT (Generative Pre-trained Transformer) is a transformer-based model designed for "
        "natural language generation tasks.",
    "T5 (Text-to-Text Transfer Transformer) treats every NLP problem as a text-to-text "
        "problem, where both the input and output are text strings.",
    "RoBERTa is an optimized version of BERT with improved training methodology and more "
        "training data.",
    "DistilBERT is a smaller, faster version of BERT that retains 97% of its language "
        "understanding capabilities.",
    "ALBERT reduces the parameters of BERT by sharing parameters across layers and using "
        "embedding factorization.",
    "XLNet is a generalized autoregressive pretraining method that overcomes the "
        "limitations of BERT by using permutation language modeling.",
    "ELECTRA uses a generator-discriminator architecture for more efficient pretraining.",
    "DeBERTa enhances BERT with disentangled attention and an enhanced mask decoder."
]
```

---
## Step 13 — Generate embeddings for all documents,
then create FAISS index for efficient similarity search

```python
document_embeddings = generate_embedding(documents, model, tokenizer)
dimension = document_embeddings.shape[1]   # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)       # Using L2 (Euclidean) distance
index.add(document_embeddings)             # Add embeddings to the index
print(f"Created index with {index.ntotal} documents")
```

---
## Step 14 — Example queries

```python
queries = [
    "What is BERT?",
    "How does GPT work?",
    "What is the difference between BERT and GPT?",
    "What is a smaller version of BERT?"
]
```

---
## Step 15 — Run the RAG pipeline for each query

```python
for query in queries:
    response, retrieved_docs = rag_pipeline(query, documents)
    print(f"Query: {query}")
    print()
    print("Retrieved Documents:")
    for i, (doc, distance) in enumerate(retrieved_docs):
        print(f"Document {i+1} (Distance: {distance:.4f}):")
        print(doc)
    print()
    print("Generated Response:")
    print(response)
    print("-" * 20)
```

---
## Learning Notes / 学习笔记

- **概念**: Model to use in retriever 是机器学习中的常用技术。  
  *Model to use in retriever is a common technique in machine learning.*

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
# Complete / 08 Complete
# Complete Code / 完整代码
# ===============================

import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSeq2SeqLM

# Model to use in retriever
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# Model to use in generator
gen_tokenizer = AutoTokenizer.from_pretrained("t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def generate_embedding(docs, model, tokenizer):
    # Tokenize each text and convert to PyTorch tensors
    inputs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt",
                       max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    # Embedding defined as mean pooling of all tokens
    attention_mask = inputs["attention_mask"]
    embeddings = outputs.last_hidden_state

    expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
    sum_embeddings = torch.sum(embeddings * expanded_mask, axis=1)
    sum_mask = torch.clamp(expanded_mask.sum(axis=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask

    # Convert to numpy array
    return mean_embeddings.cpu().numpy()

def retrieve_documents(query, index, documents, k=3):
    # Generate embedding for the query
    query_embedding = generate_embedding(query, model, tokenizer)   # 1xD matrix
    # Search the index for similar documents
    distances, indices = index.search(query_embedding, k)  # 1xk matrices
    # Return the retrieved documents and their distances
    retrieved_docs = [(documents[idx], float(distances[0][i]))
                      for i, idx in enumerate(indices[0])]
    return retrieved_docs

def generate_response(query, retrieved_docs, max_length=150):
    # Combine the query and retrieved documents into a single prompt
    if retrieved_docs:
        context = "\n".join(retrieved_docs)
        prompt = f"question: {query} context: {context}"
    else:
        prompt = f"question: {query}"

    # Generate a response
    inputs = gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = gen_model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    response = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def rag_pipeline(query, documents, retriever_k=3, max_length=150):
    retrieved_docs = retrieve_documents(query, index, documents, k=retriever_k)
    docs = [doc for doc, distance in retrieved_docs]
    response = generate_response(query, docs, max_length=max_length)
    return response, retrieved_docs

# Sample document collection
documents = [
    "Transformers are a type of deep learning model introduced in the paper 'Attention "
        "Is All You Need'.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a "
        "transformer-based model designed to understand the context of a word based on "
        "its surroundings.",
    "GPT (Generative Pre-trained Transformer) is a transformer-based model designed for "
        "natural language generation tasks.",
    "T5 (Text-to-Text Transfer Transformer) treats every NLP problem as a text-to-text "
        "problem, where both the input and output are text strings.",
    "RoBERTa is an optimized version of BERT with improved training methodology and more "
        "training data.",
    "DistilBERT is a smaller, faster version of BERT that retains 97% of its language "
        "understanding capabilities.",
    "ALBERT reduces the parameters of BERT by sharing parameters across layers and using "
        "embedding factorization.",
    "XLNet is a generalized autoregressive pretraining method that overcomes the "
        "limitations of BERT by using permutation language modeling.",
    "ELECTRA uses a generator-discriminator architecture for more efficient pretraining.",
    "DeBERTa enhances BERT with disentangled attention and an enhanced mask decoder."
]

# Generate embeddings for all documents,
# then create FAISS index for efficient similarity search
document_embeddings = generate_embedding(documents, model, tokenizer)
dimension = document_embeddings.shape[1]   # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)       # Using L2 (Euclidean) distance
index.add(document_embeddings)             # Add embeddings to the index
print(f"Created index with {index.ntotal} documents")

# Example queries
queries = [
    "What is BERT?",
    "How does GPT work?",
    "What is the difference between BERT and GPT?",
    "What is a smaller version of BERT?"
]
# Run the RAG pipeline for each query
for query in queries:
    response, retrieved_docs = rag_pipeline(query, documents)
    print(f"Query: {query}")
    print()
    print("Retrieved Documents:")
    for i, (doc, distance) in enumerate(retrieved_docs):
        print(f"Document {i+1} (Distance: {distance:.4f}):")
        print(doc)
    print()
    print("Generated Response:")
    print(response)
    print("-" * 20)
```

---

### Chapter Summary

# Chapter 19 Summary / 第19章总结

## Theme / 主题: Chapter 19 / Chapter 19

This chapter contains **5 code files** demonstrating chapter 19.

本章包含 **5 个代码文件**，演示Chapter 19。

---
## Evolution / 演化路线

  1. `01_index.ipynb` — Index
  2. `03_cosine.ipynb` — Cosine
  3. `04_retrieval.ipynb` — Retrieval
  4. `05_generator.ipynb` — Generator
  5. `08_complete.ipynb` — Complete

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 19) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 19）是机器学习流水线中的基础构建块。

---
