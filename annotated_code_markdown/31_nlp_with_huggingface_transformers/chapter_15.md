# HF Transformers
## Chapter 15

---

### Recommendation

# 01 — Recommendation / 01 Recommendation

**Chapter 15 — File 1 of 5 / 第15章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Define a corpus of articles (title and content)**.

本脚本演示 **Define a corpus of articles (title and content)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
```

---
## Step 2 — Define a corpus of articles (title and content)

```python
articles = [
  {
    "title": "Understanding Deep Learning",
    "content": ("Deep learning is a subset of machine learning where artificial neural "
                "networks, algorithms inspired by the human brain, learn from large "
                "amounts of data.")
  }, {
    "title": "Introduction to Natural Language Processing",
    "content": ("Natural Language Processing (NLP) is a field of AI that gives machines "
                "the ability to read, understand, and derive meaning from human "
                "languages.")
  }, {
    "title": "The Future of Computer Vision",
    "content": ("Computer vision is an interdisciplinary field that deals with how "
                "computers can gain high-level understanding from digital images or "
                "videos.")
  }, {
    "title": "Reinforcement Learning Explained",
    "content": ("Reinforcement learning is an area of machine learning concerned with "
                "how software agents ought to take actions in an environment so as to "
                "maximize some notion of cumulative reward.")
  }, {
    "title": "Neural Networks and Their Applications",
    "content": ("Neural networks are a set of algorithms, modeled loosely after the "
                "human brain, that are designed to recognize patterns in data.")
  }
]

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_article_embeddings(articles, model):
    """create embeddings for articles"""
    texts = [f"{article["title"]}. {article["content"]}" for article in articles]
    embeddings = model.encode(texts)
    return embeddings

def get_recommendations(article_id, articles, embeddings, top_n=2):
    """get recommendations for a given article ID based on cosine similarity"""
    similarities = cosine_similarity([embeddings[article_id]], embeddings)[0]
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    return [articles[idx] for idx in similar_indices]
```

---
## Step 3 — Create embeddings for all articles, and get recommendation for first article

```python
embeddings = create_article_embeddings(articles, model)
recommendations = get_recommendations(0, articles, embeddings)
```

---
## Step 4 — Print the recommendations

```python
print(f'Recommendations for "{articles[0]["title"]}":')
for i, rec in enumerate(recommendations):
    print(f"{i+1}. {rec["title"]}")
```

---
## Learning Notes / 学习笔记

- **概念**: Define a corpus of articles (title and content) 是机器学习中的常用技术。  
  *Define a corpus of articles (title and content) is a common technique in machine learning.*

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
# Recommendation / 01 Recommendation
# Complete Code / 完整代码
# ===============================

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Define a corpus of articles (title and content)
articles = [
  {
    "title": "Understanding Deep Learning",
    "content": ("Deep learning is a subset of machine learning where artificial neural "
                "networks, algorithms inspired by the human brain, learn from large "
                "amounts of data.")
  }, {
    "title": "Introduction to Natural Language Processing",
    "content": ("Natural Language Processing (NLP) is a field of AI that gives machines "
                "the ability to read, understand, and derive meaning from human "
                "languages.")
  }, {
    "title": "The Future of Computer Vision",
    "content": ("Computer vision is an interdisciplinary field that deals with how "
                "computers can gain high-level understanding from digital images or "
                "videos.")
  }, {
    "title": "Reinforcement Learning Explained",
    "content": ("Reinforcement learning is an area of machine learning concerned with "
                "how software agents ought to take actions in an environment so as to "
                "maximize some notion of cumulative reward.")
  }, {
    "title": "Neural Networks and Their Applications",
    "content": ("Neural networks are a set of algorithms, modeled loosely after the "
                "human brain, that are designed to recognize patterns in data.")
  }
]

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_article_embeddings(articles, model):
    """create embeddings for articles"""
    texts = [f"{article["title"]}. {article["content"]}" for article in articles]
    embeddings = model.encode(texts)
    return embeddings

def get_recommendations(article_id, articles, embeddings, top_n=2):
    """get recommendations for a given article ID based on cosine similarity"""
    similarities = cosine_similarity([embeddings[article_id]], embeddings)[0]
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    return [articles[idx] for idx in similar_indices]

# Create embeddings for all articles, and get recommendation for first article
embeddings = create_article_embeddings(articles, model)
recommendations = get_recommendations(0, articles, embeddings)

# Print the recommendations
print(f'Recommendations for "{articles[0]["title"]}":')
for i, rec in enumerate(recommendations):
    print(f"{i+1}. {rec["title"]}")
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Search

# 02 — Search / 02 Search

**Chapter 15 — File 2 of 5 / 第15章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Generate embeddings for the corpus**.

本脚本演示 **Generate embeddings for the corpus**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
  {
    "language": "English",
    "text": ("Machine learning is a field of study that gives computers the ability "
             "to learn without being explicitly programmed.")
  }, {
    "language": "Spanish",
    "text": ("El aprendizaje automático es un campo de estudio que da a las computadoras "
             "la capacidad de aprender sin ser programadas explícitamente.")
  }, {
    "language": "French",
    "text": ("L'apprentissage automatique est un domaine d'étude qui donne aux "
             "ordinateurs la capacité d'apprendre sans être explicitement programmés.")
  }, {
    "language": "German",
    "text": ("Maschinelles Lernen ist ein Studienbereich, der Computern die Fähigkeit "
             "gibt, zu lernen, ohne explizit programmiert zu werden.")
  }, {
    "language": "Italian",
    "text": ("Il machine learning è un campo di studio che conferisce ai computer la "
             "capacità di apprendere senza essere esplicitamente programmati.")
  }, {
    "language": "English",
    "text": ("Natural language processing is a subfield of linguistics, computer "
             "science, and artificial intelligence.")
  }, {
    "language": "English",
    "text": ("Computer vision is an interdisciplinary field that deals with how "
             "computers can gain high-level understanding from digital images or videos.")
  }
]

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
```

---
## Step 2 — Generate embeddings for the corpus

```python
texts = [doc["text"] for doc in corpus]
embeddings = model.encode(texts)
```

---
## Step 3 — Define a query in English and generate an embedding

```python
query = "What is machine learning?"
query_embedding = model.encode(query)
```

---
## Step 4 — Sort the embeddings of the corpus by descending similarity

```python
similarities = cosine_similarity([query_embedding], embeddings)[0]
ranked_indices = np.argsort(similarities)[::-1]
```

---
## Step 5 — Print ranked results

```python
print(f"Query: {query}\n")
for i, idx in enumerate(ranked_indices[:3]):  # Show top 3 results
    print(f"{i+1}. [{corpus[idx]["language"]}] {corpus[idx]["text"]} "
          f"(Similarity: {similarities[idx]:.4f})")
```

---
## Learning Notes / 学习笔记

- **概念**: Generate embeddings for the corpus 是机器学习中的常用技术。  
  *Generate embeddings for the corpus is a common technique in machine learning.*

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
# Search / 02 Search
# Complete Code / 完整代码
# ===============================

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
  {
    "language": "English",
    "text": ("Machine learning is a field of study that gives computers the ability "
             "to learn without being explicitly programmed.")
  }, {
    "language": "Spanish",
    "text": ("El aprendizaje automático es un campo de estudio que da a las computadoras "
             "la capacidad de aprender sin ser programadas explícitamente.")
  }, {
    "language": "French",
    "text": ("L'apprentissage automatique est un domaine d'étude qui donne aux "
             "ordinateurs la capacité d'apprendre sans être explicitement programmés.")
  }, {
    "language": "German",
    "text": ("Maschinelles Lernen ist ein Studienbereich, der Computern die Fähigkeit "
             "gibt, zu lernen, ohne explizit programmiert zu werden.")
  }, {
    "language": "Italian",
    "text": ("Il machine learning è un campo di studio che conferisce ai computer la "
             "capacità di apprendere senza essere esplicitamente programmati.")
  }, {
    "language": "English",
    "text": ("Natural language processing is a subfield of linguistics, computer "
             "science, and artificial intelligence.")
  }, {
    "language": "English",
    "text": ("Computer vision is an interdisciplinary field that deals with how "
             "computers can gain high-level understanding from digital images or videos.")
  }
]

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Generate embeddings for the corpus
texts = [doc["text"] for doc in corpus]
embeddings = model.encode(texts)

# Define a query in English and generate an embedding
query = "What is machine learning?"
query_embedding = model.encode(query)

# Sort the embeddings of the corpus by descending similarity
similarities = cosine_similarity([query_embedding], embeddings)[0]
ranked_indices = np.argsort(similarities)[::-1]

# Print ranked results
print(f"Query: {query}\n")
for i, idx in enumerate(ranked_indices[:3]):  # Show top 3 results
    print(f"{i+1}. [{corpus[idx]["language"]}] {corpus[idx]["text"]} "
          f"(Similarity: {similarities[idx]:.4f})")
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Classification

# 03 — Classification / 分类

**Chapter 15 — File 3 of 5 / 第15章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Business articles**.

本脚本演示 **Business articles**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — Step 1

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

articles = [
```

---
## Step 2 — Business articles

```python
{"category": "Business", "text":
    "The stock market reached a new high today, with technology stocks leading the "
     "gains."},
    {"category": "Business", "text":
    "The government announced a new tax policy that will affect small businesses."},
    {"category": "Business", "text":
    "The central bank has decided to keep interest rates unchanged."},
    {"category": "Business", "text":
    "Quarterly earnings reports exceeded expectations for most Fortune 500 companies."},
    {"category": "Business", "text":
    "Inflation rates have decreased for the third consecutive month."},
    {"category": "Business", "text":
    "The merger between two major corporations has been approved by regulators."},
    {"category": "Business", "text":
    "Unemployment rates have fallen to a five-year low according to new data."},
    {"category": "Business", "text":
    "The cryptocurrency market experienced significant volatility this week."},
```

---
## Step 3 — Health articles

```python
{"category": "Health", "text":
    "A new study shows that regular exercise can reduce the risk of heart disease."},
    {"category": "Health", "text":
    "A clinical trial for a new cancer treatment has shown promising results."},
    {"category": "Health", "text":
    "A balanced diet and regular sleep are essential for maintaining good health."},
    {"category": "Health", "text":
    "Medical researchers have identified a new gene linked to Alzheimer's disease."},
    {"category": "Health", "text":
    "The WHO has issued new guidelines for managing diabetes in elderly patients."},
    {"category": "Health", "text":
    "A new technique for early detection of breast cancer has been developed."},
    {"category": "Health", "text":
    "Studies show that mindfulness meditation can help reduce stress and anxiety."},
    {"category": "Health", "text":
    "Public health officials warn of a potential flu outbreak this winter season."},
```

---
## Step 4 — Technology articles

```python
{"category": "Technology", "text":
    "The latest smartphone from Apple features a better camera and longer battery life."},
    {"category": "Technology", "text":
    "The new electric car from Tesla has a range of over 400 miles."},
    {"category": "Technology", "text":
    "The latest update to the operating system includes new security features."},
    {"category": "Technology", "text":
    "The tech company unveiled its new virtual reality headset at the annual "
    "conference."},
    {"category": "Technology", "text":
    "Researchers have developed a quantum computer that can solve complex problems."},
    {"category": "Technology", "text":
    "The new social media platform has gained millions of users in just a few months."},
    {"category": "Technology", "text":
    "Cybersecurity experts warn of a new type of malware targeting smart home devices."},
```

---
## Step 5 — Science articles

```python
{"category": "Science", "text":
    "Scientists have discovered a new species of frog in the Amazon rainforest."},
    {"category": "Science", "text":
    "Astronomers have observed a supernova in a distant galaxy."},
    {"category": "Science", "text":
    "Researchers have developed a new method for measuring ocean temperatures."},
    {"category": "Science", "text":
    "A fossil discovery suggests that dinosaurs may have been warm-blooded."},
    {"category": "Science", "text":
    "Climate scientists report that Arctic ice is melting at an unprecedented rate."},
    {"category": "Science", "text":
    "Physicists have confirmed the existence of a new subatomic particle."},
    {"category": "Science", "text":
    "A study of coral reefs shows signs of recovery in protected marine areas."},
    {"category": "Science", "text":
    "Biologists have sequenced the genome of an endangered species of tiger."}
]
```

---
## Step 6 — Prepare data for classification training

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [article["text"] for article in articles]
X = model.encode(texts)
y = [article["category"] for article in articles]
```

---
## Step 7 — Normalize features

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---
## Step 8 — Split data into training and testing sets with stratification

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```

---
## Step 9 — Train a logistic regression classifier with regularization

```python
classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
classifier.fit(X_train, y_train)
```

---
## Step 10 — Evaluate the classifier

```python
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
```

---
## Step 11 — Classify new articles

```python
new_articles = [
    "The company reported a 20% increase in quarterly profits.",
    "A new vaccine has been approved for use against the flu.",
    "The new laptop features a faster processor and more memory.",
    "The Mars rover has sent back new images of the planet\"s surface."
]
new_embeddings = model.encode(new_articles)
new_embeddings_scaled = scaler.transform(new_embeddings)
new_predictions = classifier.predict(new_embeddings_scaled)
for article, prediction in zip(new_articles, new_predictions):
    print(f"Article: {article}\nPredicted Category: {prediction}\n")
```

---
## Learning Notes / 学习笔记

- **概念**: Business articles 是机器学习中的常用技术。  
  *Business articles is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `classification_report` | 分类报告：精确率/召回率/F1 | Classification report: precision/recall/F1 |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `regularization` | 正则化：防止过拟合 | Regularization: prevent overfitting |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Classification / 分类
# Complete Code / 完整代码
# ===============================

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

articles = [
    # Business articles
    {"category": "Business", "text":
    "The stock market reached a new high today, with technology stocks leading the "
     "gains."},
    {"category": "Business", "text":
    "The government announced a new tax policy that will affect small businesses."},
    {"category": "Business", "text":
    "The central bank has decided to keep interest rates unchanged."},
    {"category": "Business", "text":
    "Quarterly earnings reports exceeded expectations for most Fortune 500 companies."},
    {"category": "Business", "text":
    "Inflation rates have decreased for the third consecutive month."},
    {"category": "Business", "text":
    "The merger between two major corporations has been approved by regulators."},
    {"category": "Business", "text":
    "Unemployment rates have fallen to a five-year low according to new data."},
    {"category": "Business", "text":
    "The cryptocurrency market experienced significant volatility this week."},

    # Health articles
    {"category": "Health", "text":
    "A new study shows that regular exercise can reduce the risk of heart disease."},
    {"category": "Health", "text":
    "A clinical trial for a new cancer treatment has shown promising results."},
    {"category": "Health", "text":
    "A balanced diet and regular sleep are essential for maintaining good health."},
    {"category": "Health", "text":
    "Medical researchers have identified a new gene linked to Alzheimer's disease."},
    {"category": "Health", "text":
    "The WHO has issued new guidelines for managing diabetes in elderly patients."},
    {"category": "Health", "text":
    "A new technique for early detection of breast cancer has been developed."},
    {"category": "Health", "text":
    "Studies show that mindfulness meditation can help reduce stress and anxiety."},
    {"category": "Health", "text":
    "Public health officials warn of a potential flu outbreak this winter season."},

    # Technology articles
    {"category": "Technology", "text":
    "The latest smartphone from Apple features a better camera and longer battery life."},
    {"category": "Technology", "text":
    "The new electric car from Tesla has a range of over 400 miles."},
    {"category": "Technology", "text":
    "The latest update to the operating system includes new security features."},
    {"category": "Technology", "text":
    "The tech company unveiled its new virtual reality headset at the annual "
    "conference."},
    {"category": "Technology", "text":
    "Researchers have developed a quantum computer that can solve complex problems."},
    {"category": "Technology", "text":
    "The new social media platform has gained millions of users in just a few months."},
    {"category": "Technology", "text":
    "Cybersecurity experts warn of a new type of malware targeting smart home devices."},

    # Science articles
    {"category": "Science", "text":
    "Scientists have discovered a new species of frog in the Amazon rainforest."},
    {"category": "Science", "text":
    "Astronomers have observed a supernova in a distant galaxy."},
    {"category": "Science", "text":
    "Researchers have developed a new method for measuring ocean temperatures."},
    {"category": "Science", "text":
    "A fossil discovery suggests that dinosaurs may have been warm-blooded."},
    {"category": "Science", "text":
    "Climate scientists report that Arctic ice is melting at an unprecedented rate."},
    {"category": "Science", "text":
    "Physicists have confirmed the existence of a new subatomic particle."},
    {"category": "Science", "text":
    "A study of coral reefs shows signs of recovery in protected marine areas."},
    {"category": "Science", "text":
    "Biologists have sequenced the genome of an endangered species of tiger."}
]

# Prepare data for classification training
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [article["text"] for article in articles]
X = model.encode(texts)
y = [article["category"] for article in articles]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train a logistic regression classifier with regularization
classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# Classify new articles
new_articles = [
    "The company reported a 20% increase in quarterly profits.",
    "A new vaccine has been approved for use against the flu.",
    "The new laptop features a faster processor and more memory.",
    "The Mars rover has sent back new images of the planet\"s surface."
]
new_embeddings = model.encode(new_articles)
new_embeddings_scaled = scaler.transform(new_embeddings)
new_predictions = classifier.predict(new_embeddings_scaled)
for article, prediction in zip(new_articles, new_predictions):
    print(f"Article: {article}\nPredicted Category: {prediction}\n")
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Zeroshot

# 04 — Zeroshot / 04 Zeroshot

**Chapter 15 — File 4 of 5 / 第15章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Load a pre-trained Sentence Transformer model**.

本脚本演示 **Load a pre-trained Sentence Transformer model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
import torch
from sentence_transformers import SentenceTransformer, util

texts = [
  "The stock market reached a new high today, with technology stocks leading the gains.",
  "A new study shows that regular exercise can reduce the risk of heart disease.",
  "The latest smartphone from Apple features a better camera and longer battery life.",
  "Scientists have discovered a new species of frog in the Amazon rainforest."
]
categories = ["Business", "Health", "Technology", "Science"]
```

---
## Step 2 — Load a pre-trained Sentence Transformer model

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
text_embeddings = model.encode(texts, convert_to_tensor=True)
category_embeddings = model.encode(categories, convert_to_tensor=True)
```

---
## Step 3 — Calculate cosine similarity between texts and categories

```python
similarities = util.cos_sim(text_embeddings, category_embeddings)
```

---
## Step 4 — Get the most similar category for each text

```python
best_categories = torch.argmax(similarities, dim=1)
for i, text in enumerate(texts):
    category = categories[best_categories[i]]
    similarity = similarities[i][best_categories[i]].item()
    print(f"Text: {text}")
    print(f"Category: {category} (Similarity: {similarity:.4f})\n")
```

---
## Learning Notes / 学习笔记

- **概念**: Load a pre-trained Sentence Transformer model 是机器学习中的常用技术。  
  *Load a pre-trained Sentence Transformer model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Zeroshot / 04 Zeroshot
# Complete Code / 完整代码
# ===============================

import torch
from sentence_transformers import SentenceTransformer, util

texts = [
  "The stock market reached a new high today, with technology stocks leading the gains.",
  "A new study shows that regular exercise can reduce the risk of heart disease.",
  "The latest smartphone from Apple features a better camera and longer battery life.",
  "Scientists have discovered a new species of frog in the Amazon rainforest."
]
categories = ["Business", "Health", "Technology", "Science"]

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
text_embeddings = model.encode(texts, convert_to_tensor=True)
category_embeddings = model.encode(categories, convert_to_tensor=True)

# Calculate cosine similarity between texts and categories
similarities = util.cos_sim(text_embeddings, category_embeddings)

# Get the most similar category for each text
best_categories = torch.argmax(similarities, dim=1)
for i, text in enumerate(texts):
    category = categories[best_categories[i]]
    similarity = similarities[i][best_categories[i]].item()
    print(f"Text: {text}")
    print(f"Category: {category} (Similarity: {similarity:.4f})\n")
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Visualize

# 05 — Visualize / 05 Visualize

**Chapter 15 — File 5 of 5 / 第15章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Extract texts and categories**.

本脚本演示 **Extract texts and categories**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

texts_with_categories = [
    {"category": "Business", "text": "The stock market reached a new high today."},
    {"category": "Business", "text": "Investors are optimistic about the economy."},
    {"category": "Business", "text": "The company reported strong quarterly earnings."},
    {"category": "Business", "text":
       "The central bank has decided to keep interest rates unchanged."},
    {"category": "Health", "text":
      "A new study shows that regular exercise can reduce the risk of heart disease."},
    {"category": "Health", "text":
      "A balanced diet is essential for maintaining good health."},
    {"category": "Health", "text":
      "The new vaccine has been approved for use against the flu."},
    {"category": "Health", "text": "Sleep is important for physical and mental health."},
    {"category": "Technology", "text":
      "The latest smartphone features a better camera and longer battery life."},
    {"category": "Technology", "text":
      "The new laptop has a faster processor and more memory."},
    {"category": "Technology", "text":
      "The software update includes new security features."},
    {"category": "Technology", "text":
      "5G networks promise faster internet speeds for mobile devices."},
    {"category": "Science", "text":
      "Scientists have discovered a new species in the Amazon rainforest."},
    {"category": "Science", "text":
      "Astronomers have observed a supernova in a distant galaxy."},
    {"category": "Science", "text":
      "The Mars rover has sent back new images of the planet's surface."},
    {"category": "Science", "text":
      "Researchers have developed a new method for measuring ocean temperatures."}
]
```

---
## Step 2 — Extract texts and categories

```python
texts = [item["text"] for item in texts_with_categories]
categories = [item["category"] for item in texts_with_categories]
```

---
## Step 3 — Generate embeddings, then reduce dimension with t-SNE

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

tsne = TSNE(n_components=2, perplexity=5, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings)
```

---
## Step 4 — Define colors for categories

```python
unique_categories = list(set(categories))
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_categories)))
category_to_color = {cat: color for cat, color in zip(unique_categories, colors)}
```

---
## Step 5 — Create a scatter plot

```python
plt.figure(figsize=(10, 8))
for i, (x, y) in enumerate(reduced_embeddings):
    category = categories[i]
    color = category_to_color[category]
    plt.scatter(x, y, color=color, alpha=0.7)
    plt.annotate(texts[i][:20] + "...", (x, y), fontsize=8)
```

---
## Step 6 — Add legend, mark the axes

```python
for category, color in category_to_color.items():
    plt.scatter([], [], color=color, label=category)
plt.legend()
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("t-SNE Visualization of Text Embeddings")
plt.tight_layout()
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Extract texts and categories 是机器学习中的常用技术。  
  *Extract texts and categories is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Visualize / 05 Visualize
# Complete Code / 完整代码
# ===============================

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

texts_with_categories = [
    {"category": "Business", "text": "The stock market reached a new high today."},
    {"category": "Business", "text": "Investors are optimistic about the economy."},
    {"category": "Business", "text": "The company reported strong quarterly earnings."},
    {"category": "Business", "text":
       "The central bank has decided to keep interest rates unchanged."},
    {"category": "Health", "text":
      "A new study shows that regular exercise can reduce the risk of heart disease."},
    {"category": "Health", "text":
      "A balanced diet is essential for maintaining good health."},
    {"category": "Health", "text":
      "The new vaccine has been approved for use against the flu."},
    {"category": "Health", "text": "Sleep is important for physical and mental health."},
    {"category": "Technology", "text":
      "The latest smartphone features a better camera and longer battery life."},
    {"category": "Technology", "text":
      "The new laptop has a faster processor and more memory."},
    {"category": "Technology", "text":
      "The software update includes new security features."},
    {"category": "Technology", "text":
      "5G networks promise faster internet speeds for mobile devices."},
    {"category": "Science", "text":
      "Scientists have discovered a new species in the Amazon rainforest."},
    {"category": "Science", "text":
      "Astronomers have observed a supernova in a distant galaxy."},
    {"category": "Science", "text":
      "The Mars rover has sent back new images of the planet's surface."},
    {"category": "Science", "text":
      "Researchers have developed a new method for measuring ocean temperatures."}
]

# Extract texts and categories
texts = [item["text"] for item in texts_with_categories]
categories = [item["category"] for item in texts_with_categories]

# Generate embeddings, then reduce dimension with t-SNE
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

tsne = TSNE(n_components=2, perplexity=5, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings)

# Define colors for categories
unique_categories = list(set(categories))
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_categories)))
category_to_color = {cat: color for cat, color in zip(unique_categories, colors)}

# Create a scatter plot
plt.figure(figsize=(10, 8))
for i, (x, y) in enumerate(reduced_embeddings):
    category = categories[i]
    color = category_to_color[category]
    plt.scatter(x, y, color=color, alpha=0.7)
    plt.annotate(texts[i][:20] + "...", (x, y), fontsize=8)

# Add legend, mark the axes
for category, color in category_to_color.items():
    plt.scatter([], [], color=color, label=category)
plt.legend()
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("t-SNE Visualization of Text Embeddings")
plt.tight_layout()
plt.show()
```

---

### Chapter Summary

# Chapter 15 Summary / 第15章总结

## Theme / 主题: Chapter 15 / Chapter 15

This chapter contains **5 code files** demonstrating chapter 15.

本章包含 **5 个代码文件**，演示Chapter 15。

---
## Evolution / 演化路线

  1. `01_recommendation.ipynb` — Recommendation
  2. `02_search.ipynb` — Search
  3. `03_classification.ipynb` — Classification
  4. `04_zeroshot.ipynb` — Zeroshot
  5. `05_visualize.ipynb` — Visualize

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 15) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 15）是机器学习流水线中的基础构建块。

---
