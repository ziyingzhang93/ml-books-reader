# HuggingFace Transformers NLP / NLP with HF Transformers
## Chapter 18

---

### Search

# 01 — Search / 01 Search

**Chapter 18 — File 1 of 3 / 第18章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Tokenize input, get model output**.

本脚本演示 **Tokenize input, get model output**。

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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics.pairwise import cosine_similarity
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModel

def get_context_vector(text, model, tokenizer):
    """Get context vector by mean pooling"""
```

---
## Step 2 — Tokenize input, get model output

```python
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        outputs = model(**inputs)
```

---
## Step 3 — Mean pooling: take average across sequence length of the output

```python
pooled_vector = torch.mean(outputs.last_hidden_state, dim=1)
    return pooled_vector[0]

def semantic_search(query, documents, document_vectors, top_k=2):
    """Search the corpus"""
```

---
## Step 4 — Calculate similarity between query and all documents

```python
query_vector = get_context_vector(query, model, tokenizer)
    similarities = cosine_similarity([query_vector], document_vectors)[0]
```

---
## Step 5 — Get indices of top-k most similar documents

```python
top_indices = np.argsort(similarities)[::-1][:top_k]
```

---
## Step 6 — Return top-k documents and their similarity scores

```python
results = []
    for idx in top_indices:
        # 添加元素到列表末尾 / Append element to list end
        results.append({
            "document": documents[idx],
            "similarity": similarities[idx]
        })
    return results
```

---
## Step 7 — Load pre-trained model and tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
```

---
## Step 8 — Create a document corpus and convert them into context vectors

```python
documents = [
    "Machine learning is a field of study that gives computers the ability to learn "
        "without being explicitly programmed.",
    "Deep learning is a subset of machine learning that uses neural networks with "
        "many layers.",
    "Natural language processing is a field of AI that focuses on the interaction "
        "between computers and human language.",
    "Computer vision is an interdisciplinary field that deals with how computers "
        "can gain high-level understanding from digital images or videos.",
    "Reinforcement learning is about taking suitable actions to maximize reward "
        "in a particular situation."
]
document_vectors = [get_context_vector(doc, model, tokenizer) for doc in documents]
```

---
## Step 9 — Example search

```python
query = "How do computers learn from data?"
results = semantic_search(query, documents, document_vectors)
```

---
## Step 10 — Print results

```python
# 打印输出 / Print output
print(f"Query: {query}\n")
# 同时获取索引和值 / Get both index and value
for i, result in enumerate(results):
    # 打印输出 / Print output
    print(f"Result {i+1} (Similarity: {result["similarity"]:.4f}):")
    # 打印输出 / Print output
    print(result["document"])
    # 打印输出 / Print output
    print()
```

---
## Learning Notes / 学习笔记

- **概念**: Tokenize input, get model output 是机器学习中的常用技术。  
  *Tokenize input, get model output is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Search / 01 Search
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics.pairwise import cosine_similarity
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModel

def get_context_vector(text, model, tokenizer):
    """Get context vector by mean pooling"""
    # Tokenize input, get model output
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling: take average across sequence length of the output
    pooled_vector = torch.mean(outputs.last_hidden_state, dim=1)
    return pooled_vector[0]

def semantic_search(query, documents, document_vectors, top_k=2):
    """Search the corpus"""
    # Calculate similarity between query and all documents
    query_vector = get_context_vector(query, model, tokenizer)
    similarities = cosine_similarity([query_vector], document_vectors)[0]

    # Get indices of top-k most similar documents
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Return top-k documents and their similarity scores
    results = []
    for idx in top_indices:
        # 添加元素到列表末尾 / Append element to list end
        results.append({
            "document": documents[idx],
            "similarity": similarities[idx]
        })
    return results

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Create a document corpus and convert them into context vectors
documents = [
    "Machine learning is a field of study that gives computers the ability to learn "
        "without being explicitly programmed.",
    "Deep learning is a subset of machine learning that uses neural networks with "
        "many layers.",
    "Natural language processing is a field of AI that focuses on the interaction "
        "between computers and human language.",
    "Computer vision is an interdisciplinary field that deals with how computers "
        "can gain high-level understanding from digital images or videos.",
    "Reinforcement learning is about taking suitable actions to maximize reward "
        "in a particular situation."
]
document_vectors = [get_context_vector(doc, model, tokenizer) for doc in documents]

# Example search
query = "How do computers learn from data?"
results = semantic_search(query, documents, document_vectors)

# Print results
# 打印输出 / Print output
print(f"Query: {query}\n")
# 同时获取索引和值 / Get both index and value
for i, result in enumerate(results):
    # 打印输出 / Print output
    print(f"Result {i+1} (Similarity: {result["similarity"]:.4f}):")
    # 打印输出 / Print output
    print(result["document"])
    # 打印输出 / Print output
    print()
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Clustering

# 02 — Clustering / 聚类

**Chapter 18 — File 2 of 3 / 第18章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Tokenize input, get model output**.

本脚本演示 **Tokenize input, get model output**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — Step 1

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.cluster import KMeans
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModel

def get_context_vector(text, model, tokenizer):
    """Get context vector by mean pooling"""
```

---
## Step 2 — Tokenize input, get model output

```python
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        outputs = model(**inputs)
```

---
## Step 3 — Mean pooling: take average across sequence length of the output

```python
pooled_vector = torch.mean(outputs.last_hidden_state, dim=1)
    return pooled_vector[0]
```

---
## Step 4 — Create a document corpus (more documents for clustering)

```python
documents = [
    "Machine learning algorithms build models based on sample data to make predictions "
        "without being explicitly programmed.",
    "Deep learning uses neural networks with many layers to learn representations of "
        "data with multiple levels of abstraction.",
    "Neural networks are computing systems inspired by the biological neural networks "
        "that constitute animal brains.",
    "Convolutional neural networks are deep neural networks most commonly applied to "
        "analyzing visual imagery.",
    "Natural language processing is a subfield of linguistics, computer science, and "
        "artificial intelligence.",
    "Sentiment analysis uses NLP to identify and extract opinions within text to "
        "determine writer's attitude.",
    "Named entity recognition is a subtask of information extraction that seeks to "
        "locate and classify named entities in text.",
    "Computer vision is an interdisciplinary field that deals with how computers can "
        "gain high-level understanding from digital images.",
    "Image recognition is the ability of software to identify objects, places, people, "
        "writing and actions in images.",
    "Object detection is a computer technology related to computer vision and "
        "image processing."
]
```

---
## Step 5 — Generate context vectors for all documents

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
# 创建NumPy数组 / Create NumPy array
doc_vectors = np.array([get_context_vector(doc, model, tokenizer) for doc in documents])
```

---
## Step 6 — Perform K-means clustering on documents

```python
num_clusters = 3
# K均值聚类：将数据分成K组 / KMeans: group data into K clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(doc_vectors)
```

---
## Step 7 — Print documents in each cluster

```python
# 生成整数序列 / Generate integer sequence
for i in range(num_clusters):
    # 打印输出 / Print output
    print(f"\nCluster {i+1}:")
    # 同时获取索引和值 / Get both index and value
    cluster_docs = [doc for j, doc in enumerate(documents) if cluster_labels[j] == i]
    for doc in cluster_docs:
        # 打印输出 / Print output
        print(f"- {doc}")
```

---
## Step 8 — Visualize the clusters in reduced dimensionality

```python
# 主成分分析：降维，保留最重要的特征 / PCA: reduce dimensions, keep key features
pca = PCA(n_components=2)
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
reduced_vectors = pca.fit_transform(doc_vectors)

# 创建画布 / Create figure canvas
plt.figure(figsize=(10, 6))
colors = ["red", "blue", "green"]
# 生成整数序列 / Generate integer sequence
for i in range(num_clusters):
```

---
## Step 9 — Plot points in each cluster

```python
cluster_points = reduced_vectors[cluster_labels == i]
    # 绘制散点图 / Draw scatter plot
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                c=colors[i], label=f"Cluster {i+1}")
# 设置图表标题 / Set chart title
plt.title("Document Clusters")
# 设置X轴标签 / Set X-axis label
plt.xlabel("PCA Component 1")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("PCA Component 2")
# 显示图例 / Show legend
plt.legend()
plt.grid(True)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Tokenize input, get model output 是机器学习中的常用技术。  
  *Tokenize input, get model output is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `KMeans` | K均值聚类 | K-Means clustering |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `matplotlib` | 绑图库 | Plotting library |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Clustering / 聚类
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.cluster import KMeans
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModel

def get_context_vector(text, model, tokenizer):
    """Get context vector by mean pooling"""
    # Tokenize input, get model output
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling: take average across sequence length of the output
    pooled_vector = torch.mean(outputs.last_hidden_state, dim=1)
    return pooled_vector[0]

# Create a document corpus (more documents for clustering)
documents = [
    "Machine learning algorithms build models based on sample data to make predictions "
        "without being explicitly programmed.",
    "Deep learning uses neural networks with many layers to learn representations of "
        "data with multiple levels of abstraction.",
    "Neural networks are computing systems inspired by the biological neural networks "
        "that constitute animal brains.",
    "Convolutional neural networks are deep neural networks most commonly applied to "
        "analyzing visual imagery.",
    "Natural language processing is a subfield of linguistics, computer science, and "
        "artificial intelligence.",
    "Sentiment analysis uses NLP to identify and extract opinions within text to "
        "determine writer's attitude.",
    "Named entity recognition is a subtask of information extraction that seeks to "
        "locate and classify named entities in text.",
    "Computer vision is an interdisciplinary field that deals with how computers can "
        "gain high-level understanding from digital images.",
    "Image recognition is the ability of software to identify objects, places, people, "
        "writing and actions in images.",
    "Object detection is a computer technology related to computer vision and "
        "image processing."
]

# Generate context vectors for all documents
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
# 创建NumPy数组 / Create NumPy array
doc_vectors = np.array([get_context_vector(doc, model, tokenizer) for doc in documents])

# Perform K-means clustering on documents
num_clusters = 3
# K均值聚类：将数据分成K组 / KMeans: group data into K clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(doc_vectors)

# Print documents in each cluster
# 生成整数序列 / Generate integer sequence
for i in range(num_clusters):
    # 打印输出 / Print output
    print(f"\nCluster {i+1}:")
    # 同时获取索引和值 / Get both index and value
    cluster_docs = [doc for j, doc in enumerate(documents) if cluster_labels[j] == i]
    for doc in cluster_docs:
        # 打印输出 / Print output
        print(f"- {doc}")

# Visualize the clusters in reduced dimensionality
# 主成分分析：降维，保留最重要的特征 / PCA: reduce dimensions, keep key features
pca = PCA(n_components=2)
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
reduced_vectors = pca.fit_transform(doc_vectors)

# 创建画布 / Create figure canvas
plt.figure(figsize=(10, 6))
colors = ["red", "blue", "green"]
# 生成整数序列 / Generate integer sequence
for i in range(num_clusters):
    # Plot points in each cluster
    cluster_points = reduced_vectors[cluster_labels == i]
    # 绘制散点图 / Draw scatter plot
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                c=colors[i], label=f"Cluster {i+1}")
# 设置图表标题 / Set chart title
plt.title("Document Clusters")
# 设置X轴标签 / Set X-axis label
plt.xlabel("PCA Component 1")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("PCA Component 2")
# 显示图例 / Show legend
plt.legend()
plt.grid(True)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Classification

# 03 — Classification / 分类

**Chapter 18 — File 3 of 3 / 第18章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Tokenize input, get model output**.

本脚本演示 **Tokenize input, get model output**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
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
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModel
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import classification_report

def get_context_vector(text, model, tokenizer):
    """Get context vector by mean pooling"""
```

---
## Step 2 — Tokenize input, get model output

```python
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        outputs = model(**inputs)
```

---
## Step 3 — Mean pooling: take average across sequence length of the output

```python
pooled_vector = torch.mean(outputs.last_hidden_state, dim=1)
    return pooled_vector[0]
```

---
## Step 4 — Create a dataset of texts with labels

```python
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
```

---
## Step 5 — Generate context vectors for all texts

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
# 创建NumPy数组 / Create NumPy array
text_vectors = np.array([get_context_vector(text, model, tokenizer) for text in texts])
```

---
## Step 6 — Split into training and testing sets, train a classifier, then evaluate

```python
X_train, X_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    train_test_split(text_vectors, labels, test_size=0.3, random_state=42)
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# 生成分类报告：精确率/召回率/F1 / Classification report: precision/recall/F1
print(classification_report(y_test, y_pred))
```

---
## Step 7 — Classify new texts

```python
new_texts = [
    "The central bank has decided to keep interest rates unchanged.",
    "A new study shows that regular exercise can reduce the risk of heart disease.",
    "The new laptop has a faster processor and more memory than previous models."
]
# 创建NumPy数组 / Create NumPy array
new_vectors = np.array([get_context_vector(text, model, tokenizer) for text in new_texts])
predictions = classifier.predict(new_vectors)
```

---
## Step 8 — Print predictions

```python
# 将多个序列配对 / Pair multiple sequences
for text, prediction in zip(new_texts, predictions):
    # 打印输出 / Print output
    print(f"Text: {text}")
    # 打印输出 / Print output
    print(f"Category: {prediction}\n")
```

---
## Learning Notes / 学习笔记

- **概念**: Tokenize input, get model output 是机器学习中的常用技术。  
  *Tokenize input, get model output is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `classification_report` | 分类报告：精确率/召回率/F1 | Classification report: precision/recall/F1 |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
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

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModel
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import classification_report

def get_context_vector(text, model, tokenizer):
    """Get context vector by mean pooling"""
    # Tokenize input, get model output
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
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
# 创建NumPy数组 / Create NumPy array
text_vectors = np.array([get_context_vector(text, model, tokenizer) for text in texts])

# Split into training and testing sets, train a classifier, then evaluate
X_train, X_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    train_test_split(text_vectors, labels, test_size=0.3, random_state=42)
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# 生成分类报告：精确率/召回率/F1 / Classification report: precision/recall/F1
print(classification_report(y_test, y_pred))

# Classify new texts
new_texts = [
    "The central bank has decided to keep interest rates unchanged.",
    "A new study shows that regular exercise can reduce the risk of heart disease.",
    "The new laptop has a faster processor and more memory than previous models."
]
# 创建NumPy数组 / Create NumPy array
new_vectors = np.array([get_context_vector(text, model, tokenizer) for text in new_texts])
predictions = classifier.predict(new_vectors)

# Print predictions
# 将多个序列配对 / Pair multiple sequences
for text, prediction in zip(new_texts, predictions):
    # 打印输出 / Print output
    print(f"Text: {text}")
    # 打印输出 / Print output
    print(f"Category: {prediction}\n")
```

---

### Chapter Summary / 章节总结

# Chapter 18 Summary / 第18章总结

## Theme / 主题: Chapter 18 / Chapter 18

This chapter contains **3 code files** demonstrating chapter 18.

本章包含 **3 个代码文件**，演示Chapter 18。

---
## Evolution / 演化路线

  1. `01_search.ipynb` — Search
  2. `02_clustering.ipynb` — Clustering
  3. `03_classification.ipynb` — Classification

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 18) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 18）是机器学习流水线中的基础构建块。

---
