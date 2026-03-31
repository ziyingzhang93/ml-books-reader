# 从零构建Transformer / Building Transformers from Scratch
## Chapter 04

---

### Glove

# 01 — Glove / 01 Glove

**Chapter 04 — File 1 of 4 / 第04章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load pretrained GloVe embeddings**.

本脚本演示 **Load pretrained GloVe embeddings**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from gensim.models import KeyedVectors
```

---
## Step 2 — Load pretrained GloVe embeddings

```python
model = KeyedVectors.load_word2vec_format('glove.6B.50d.txt', binary=False,
                                          no_header=True)
```

---
## Step 3 — Find similar words

```python
similar_words = model.most_similar('king')
# 打印输出 / Print output
print(similar_words)
# 打印输出 / Print output
print()
```

---
## Step 4 — Word analogies

```python
result = model.most_similar(positive=['king', 'woman'], negative=['man'])
# 打印输出 / Print output
print(result)
```

---
## Learning Notes / 学习笔记

- **概念**: Load pretrained GloVe embeddings 是机器学习中的常用技术。  
  *Load pretrained GloVe embeddings is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Glove / 01 Glove
# Complete Code / 完整代码
# ===============================

from gensim.models import KeyedVectors

# Load pretrained GloVe embeddings
model = KeyedVectors.load_word2vec_format('glove.6B.50d.txt', binary=False,
                                          no_header=True)
# Find similar words
similar_words = model.most_similar('king')
# 打印输出 / Print output
print(similar_words)
# 打印输出 / Print output
print()

# Word analogies
result = model.most_similar(positive=['king', 'woman'], negative=['man'])
# 打印输出 / Print output
print(result)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Gensim

# 02 — Gensim / 02 Gensim

**Chapter 04 — File 2 of 4 / 第04章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Prepare your text data**.

本脚本演示 **Prepare your text data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
```

---
## Step 2 — Prepare your text data

```python
sentences = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog jumps over the lazy fox",
```

---
## Step 3 — ... more sentences

```python
]
```

---
## Step 4 — Preprocess the sentences

```python
tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]
```

---
## Step 5 — Train the model

```python
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,  # dimension of the word vectors
    window=5,         # context window size
    min_count=1,      # ignore words with frequency < min_count
    workers=4,        # number of CPU cores to use
    sg=0              # 0 for CBOW, 1 for Skip-gram
)
```

---
## Step 6 — Save the model

```python
# 保存模型到文件 / Save model to file
model.save("word2vec.model")
```

---
## Step 7 — Use the model

```python
model = Word2Vec.load("word2vec.model")
vector = model.wv['quick']  # get the vector for a word
similar_words = model.wv.most_similar('quick')
# 打印输出 / Print output
print(similar_words)
```

---
## Learning Notes / 学习笔记

- **概念**: Prepare your text data 是机器学习中的常用技术。  
  *Prepare your text data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gensim / 02 Gensim
# Complete Code / 完整代码
# ===============================

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Prepare your text data
sentences = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog jumps over the lazy fox",
    # ... more sentences
]

# Preprocess the sentences
tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# Train the model
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,  # dimension of the word vectors
    window=5,         # context window size
    min_count=1,      # ignore words with frequency < min_count
    workers=4,        # number of CPU cores to use
    sg=0              # 0 for CBOW, 1 for Skip-gram
)

# Save the model
# 保存模型到文件 / Save model to file
model.save("word2vec.model")

# Use the model
model = Word2Vec.load("word2vec.model")
vector = model.wv['quick']  # get the vector for a word
similar_words = model.wv.most_similar('quick')
# 打印输出 / Print output
print(similar_words)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Word2Vec

# 03 — Word2Vec / 03 Word2Vec

**Chapter 04 — File 3 of 4 / 第04章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Prepare your text data**.

本脚本演示 **Prepare your text data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model

## Code Flow / 代码流程

```
   
┌───────────────────────┐
│  定义模型 Define Model  │
└───────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Word2VecModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, vocab_size, embedding_dim):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(embedding_dim, vocab_size)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return out
```

---
## Step 2 — Prepare your text data

```python
sentences = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog jumps over the lazy fox",
```

---
## Step 3 — ... more sentences

```python
]
```

---
## Step 4 — Create a dataset for training

```python
skipgram_size = 2
dataset = []
vocab = set()
for sentence in sentences:
    tokens = sentence.split()
    vocab.update(tokens)
    # 获取长度 / Get length
    for i in range(len(tokens)):
        context = tokens[i-skipgram_size:i] + tokens[i+1:i+skipgram_size+1]
        target = tokens[i]
        # 添加元素到列表末尾 / Append element to list end
        dataset.append((context, target))

# 同时获取索引和值 / Get both index and value
vocab_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
# 获取长度 / Get length
vocab_size = len(vocab)
```

---
## Step 5 — Training setup

```python
embedding_dim = 50
model = Word2VecModel(vocab_size, embedding_dim)
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
criterion = nn.CrossEntropyLoss()
# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.001)
num_epochs = 10
```

---
## Step 6 — Training loop

```python
# 生成整数序列 / Generate integer sequence
for epoch in range(num_epochs):
    for context, target in dataset:
        context_idx = [vocab_to_idx[word] for word in context]
        # 获取长度 / Get length
        target_idx = [vocab_to_idx[target]] * len(context)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        output = model(torch.tensor(target_idx))
        loss = criterion(output, torch.tensor(context_idx))
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
```

---
## Step 7 — Save the model

```python
# 获取模型参数字典 / Get model parameter dictionary
torch.save(model.state_dict(), "word2vec.pt")
```

---
## Learning Notes / 学习笔记

- **概念**: Prepare your text data 是机器学习中的常用技术。  
  *Prepare your text data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `nn.Embedding` | 将整数ID映射为稠密向量 | Map integer IDs to dense vectors |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Word2Vec / 03 Word2Vec
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Word2VecModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, vocab_size, embedding_dim):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(embedding_dim, vocab_size)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return out

# Prepare your text data
sentences = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog jumps over the lazy fox",
    # ... more sentences
]

# Create a dataset for training
skipgram_size = 2
dataset = []
vocab = set()
for sentence in sentences:
    tokens = sentence.split()
    vocab.update(tokens)
    # 获取长度 / Get length
    for i in range(len(tokens)):
        context = tokens[i-skipgram_size:i] + tokens[i+1:i+skipgram_size+1]
        target = tokens[i]
        # 添加元素到列表末尾 / Append element to list end
        dataset.append((context, target))

# 同时获取索引和值 / Get both index and value
vocab_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
# 获取长度 / Get length
vocab_size = len(vocab)

# Training setup
embedding_dim = 50
model = Word2VecModel(vocab_size, embedding_dim)
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
criterion = nn.CrossEntropyLoss()
# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.001)
num_epochs = 10

# Training loop
# 生成整数序列 / Generate integer sequence
for epoch in range(num_epochs):
    for context, target in dataset:
        context_idx = [vocab_to_idx[word] for word in context]
        # 获取长度 / Get length
        target_idx = [vocab_to_idx[target]] * len(context)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        output = model(torch.tensor(target_idx))
        loss = criterion(output, torch.tensor(context_idx))
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()

# Save the model
# 获取模型参数字典 / Get model parameter dictionary
torch.save(model.state_dict(), "word2vec.pt")
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Bert Embed

# 04 — Bert Embed / 04 Bert Embed

**Chapter 04 — File 4 of 4 / 第04章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Bert Embed**.

本脚本演示 **04 Bert Embed**。

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
from transformers import BertModel, BertConfig

config = BertConfig()
model = BertModel(config=config)
# 打印输出 / Print output
print(model)
# 打印输出 / Print output
print(model.embeddings.word_embeddings.state_dict())
```

---
## Learning Notes / 学习笔记

- **概念**: Bert Embed 是机器学习中的常用技术。  
  *Bert Embed is a common technique in machine learning.*

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
# Bert Embed / 04 Bert Embed
# Complete Code / 完整代码
# ===============================

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import BertModel, BertConfig

config = BertConfig()
model = BertModel(config=config)
# 打印输出 / Print output
print(model)
# 打印输出 / Print output
print(model.embeddings.word_embeddings.state_dict())
```

---

### Chapter Summary / 章节总结

# Chapter 04 Summary / 第04章总结

## Theme / 主题: Chapter 04 / Chapter 04

This chapter contains **4 code files** demonstrating chapter 04.

本章包含 **4 个代码文件**，演示Chapter 04。

---
## Evolution / 演化路线

  1. `01_glove.ipynb` — Glove
  2. `02_gensim.ipynb` — Gensim
  3. `03_word2vec.ipynb` — Word2Vec
  4. `04_bert_embed.ipynb` — Bert Embed

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 04) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 04）是机器学习流水线中的基础构建块。

---
