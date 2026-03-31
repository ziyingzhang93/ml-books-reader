# Python 机器学习 / Python for Machine Learning
## Chapter 10

---

### Set Trace

# 01 — Set Trace / 01 Set Trace

**Chapter 10 — File 1 of 5 / 第10章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **importing the Python debugger module**.

本脚本演示 **importing the Python debugger module**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import random
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import dot
from scipy.special import softmax
```

---
## Step 2 — importing the Python debugger module

```python
import pdb
```

---
## Step 3 — encoder representations of four different words

```python
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])
```

---
## Step 4 — stacking the word embeddings into a single array

```python
words = array([word_1, word_2, word_3, word_4])
```

---
## Step 5 — generating the weight matrices

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
random.seed(42)
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))
```

---
## Step 6 — generating the queries, keys and values

```python
Q = dot(words, W_Q)
K = dot(words, W_K)
V = dot(words, W_V)
```

---
## Step 7 — inserting a breakpoint

```python
pdb.set_trace()
```

---
## Step 8 — scoring the query vectors against all key vectors

```python
scores = dot(Q, K.transpose())
```

---
## Step 9 — computing the weights by a softmax operation

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
weights = softmax(scores / K.shape[1] ** 0.5, axis=1)
```

---
## Step 10 — computing the attention by a weighted sum of the value vectors

```python
attention = dot(weights, V)

# 打印输出 / Print output
print(attention)
```

---
## Learning Notes / 学习笔记

- **概念**: importing the Python debugger module 是机器学习中的常用技术。  
  *importing the Python debugger module is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Set Trace / 01 Set Trace
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import random
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import dot
from scipy.special import softmax

# importing the Python debugger module
import pdb

# encoder representations of four different words
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])

# stacking the word embeddings into a single array
words = array([word_1, word_2, word_3, word_4])

# generating the weight matrices
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
random.seed(42)
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))

# generating the queries, keys and values
Q = dot(words, W_Q)
K = dot(words, W_K)
V = dot(words, W_V)

# inserting a breakpoint
pdb.set_trace()

# scoring the query vectors against all key vectors
scores = dot(Q, K.transpose())

# computing the weights by a softmax operation
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
weights = softmax(scores / K.shape[1] ** 0.5, axis=1)

# computing the attention by a weighted sum of the value vectors
attention = dot(weights, V)

# 打印输出 / Print output
print(attention)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Breakpoint

# 02 — Breakpoint / 02 Breakpoint

**Chapter 10 — File 2 of 5 / 第10章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **encoder representations of four different words**.

本脚本演示 **encoder representations of four different words**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import random
from scipy.special import softmax
```

---
## Step 2 — encoder representations of four different words

```python
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])
```

---
## Step 3 — stacking the word embeddings into a single array

```python
words = array([word_1, word_2, word_3, word_4])
```

---
## Step 4 — generating the weight matrices

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
random.seed(42)
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))
```

---
## Step 5 — generating the queries, keys and values

```python
Q = words @ W_Q
K = words @ W_K
V = words @ W_V
```

---
## Step 6 — inserting a breakpoint

```python
breakpoint()
```

---
## Step 7 — scoring the query vectors against all key vectors

```python
scores = Q @ K.transpose()
```

---
## Step 8 — computing the weights by a softmax operation

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
weights = softmax(scores / K.shape[1] ** 0.5, axis=1)
```

---
## Step 9 — computing the attention by a weighted sum of the value vectors

```python
attention = weights @ V

# 打印输出 / Print output
print(attention)
```

---
## Learning Notes / 学习笔记

- **概念**: encoder representations of four different words 是机器学习中的常用技术。  
  *encoder representations of four different words is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Breakpoint / 02 Breakpoint
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import random
from scipy.special import softmax

# encoder representations of four different words
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])

# stacking the word embeddings into a single array
words = array([word_1, word_2, word_3, word_4])

# generating the weight matrices
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
random.seed(42)
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))

# generating the queries, keys and values
Q = words @ W_Q
K = words @ W_K
V = words @ W_V

# inserting a breakpoint
breakpoint()

# scoring the query vectors against all key vectors
scores = Q @ K.transpose()

# computing the weights by a softmax operation
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
weights = softmax(scores / K.shape[1] ** 0.5, axis=1)

# computing the attention by a weighted sum of the value vectors
attention = weights @ V

# 打印输出 / Print output
print(attention)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Env Var

# 04 — Env Var / 04 Env Var

**Chapter 10 — File 3 of 5 / 第10章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **setting the value of the PYTHONBREAKPOINT environment variable**.

本脚本演示 **setting the value of the PYTHONBREAKPOINT environment variable**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import random
from scipy.special import softmax
```

---
## Step 2 — setting the value of the PYTHONBREAKPOINT environment variable

```python
# 导入操作系统接口 / Import OS interface
import os
os.environ['PYTHONBREAKPOINT'] = '0'
```

---
## Step 3 — encoder representations of four different words

```python
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])
```

---
## Step 4 — stacking the word embeddings into a single array

```python
words = array([word_1, word_2, word_3, word_4])
```

---
## Step 5 — generating the weight matrices

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
random.seed(42)
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))
```

---
## Step 6 — generating the queries, keys and values

```python
Q = words @ W_Q
K = words @ W_K
V = words @ W_V
```

---
## Step 7 — inserting a breakpoint

```python
breakpoint()
```

---
## Step 8 — scoring the query vectors against all key vectors

```python
scores = Q @ K.transpose()
```

---
## Step 9 — computing the weights by a softmax operation

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
weights = softmax(scores / K.shape[1] ** 0.5, axis=1)
```

---
## Step 10 — computing the attention by a weighted sum of the value vectors

```python
attention = weights @ V

# 打印输出 / Print output
print(attention)
```

---
## Learning Notes / 学习笔记

- **概念**: setting the value of the PYTHONBREAKPOINT environment variable 是机器学习中的常用技术。  
  *setting the value of the PYTHONBREAKPOINT environment variable is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Env Var / 04 Env Var
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import random
from scipy.special import softmax

# setting the value of the PYTHONBREAKPOINT environment variable
# 导入操作系统接口 / Import OS interface
import os
os.environ['PYTHONBREAKPOINT'] = '0'

# encoder representations of four different words
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])

# stacking the word embeddings into a single array
words = array([word_1, word_2, word_3, word_4])

# generating the weight matrices
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
random.seed(42)
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))

# generating the queries, keys and values
Q = words @ W_Q
K = words @ W_K
V = words @ W_V

# inserting a breakpoint
breakpoint()

# scoring the query vectors against all key vectors
scores = Q @ K.transpose()

# computing the weights by a softmax operation
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
weights = softmax(scores / K.shape[1] ** 0.5, axis=1)

# computing the attention by a weighted sum of the value vectors
attention = weights @ V

# 打印输出 / Print output
print(attention)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Breakpoint



---

### Postmortem

# 10 — Postmortem / 10 Postmortem

**Chapter 10 — File 5 of 5 / 第10章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Experimentally find the average of 1/x where x is a random integer in 0 to 9999**.

本脚本演示 **Experimentally find the average of 1/x where x is a random integer in 0 to 9999**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入系统相关功能 / Import system utilities
import sys
import pdb
# 导入随机数生成模块 / Import random number module
import random

def debughook(etype, value, tb):
    pdb.pm() # post-mortem debugger
sys.excepthook = debughook
```

---
## Step 2 — Experimentally find the average of 1/x where x is a random integer in 0 to 9999

```python
N = 1000
randomsum = 0
# 生成整数序列 / Generate integer sequence
for i in range(N):
    x = random.randint(0,10000)
    randomsum += 1/x

# 打印输出 / Print output
print("Average is", randomsum/N)
```

---
## Learning Notes / 学习笔记

- **概念**: Experimentally find the average of 1/x where x is a random integer in 0 to 9999 是机器学习中的常用技术。  
  *Experimentally find the average of 1/x where x is a random integer in 0 to 9999 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Postmortem / 10 Postmortem
# Complete Code / 完整代码
# ===============================

# 导入系统相关功能 / Import system utilities
import sys
import pdb
# 导入随机数生成模块 / Import random number module
import random

def debughook(etype, value, tb):
    pdb.pm() # post-mortem debugger
sys.excepthook = debughook

# Experimentally find the average of 1/x where x is a random integer in 0 to 9999
N = 1000
randomsum = 0
# 生成整数序列 / Generate integer sequence
for i in range(N):
    x = random.randint(0,10000)
    randomsum += 1/x

# 打印输出 / Print output
print("Average is", randomsum/N)
```

---

### Chapter Summary / 章节总结

# Chapter 10 Summary / 第10章总结

## Theme / 主题: Chapter 10 / Chapter 10

This chapter contains **5 code files** demonstrating chapter 10.

本章包含 **5 个代码文件**，演示Chapter 10。

---
## Evolution / 演化路线

  1. `01_set_trace.ipynb` — Set Trace
  2. `02_breakpoint.ipynb` — Breakpoint
  3. `04_env_var.ipynb` — Env Var
  4. `07_breakpoint.ipynb` — Breakpoint
  5. `10_postmortem.ipynb` — Postmortem

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 10) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 10）是机器学习流水线中的基础构建块。

---
