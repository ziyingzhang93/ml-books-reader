# Transformer
## Chapter 15

---

### Testattention

# 11 — Testattention / 注意力机制

**Chapter 15 — File 1 of 1 / 第15章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Implementing the Scaled-Dot Product Attention**.

本脚本演示 **Implementing the Scaled-Dot Product Attention**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — Step 1

```python
from numpy import random
from tensorflow import matmul, math, cast, float32
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import softmax

input_seq_length = 5  # Maximum length of the input sequence
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
batch_size = 64  # Batch size from the training process
```

---
## Step 2 — Implementing the Scaled-Dot Product Attention

```python
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
```

---
## Step 3 — Scoring the queries against the keys after transposing the latter, and scaling

```python
scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))
```

---
## Step 4 — Apply mask to the attention scores

```python
if mask is not None:
            scores += -1e9 * mask
```

---
## Step 5 — Computing the weights by a softmax operation

```python
weights = softmax(scores)
```

---
## Step 6 — Computing the attention by a weighted sum of the value vectors

```python
return matmul(weights, values)

queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))

attention = DotProductAttention()
print(attention(queries, keys, values, d_k))
```

---
## Learning Notes / 学习笔记

- **概念**: Implementing the Scaled-Dot Product Attention 是机器学习中的常用技术。  
  *Implementing the Scaled-Dot Product Attention is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Testattention / 注意力机制
# Complete Code / 完整代码
# ===============================

from numpy import random
from tensorflow import matmul, math, cast, float32
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import softmax

input_seq_length = 5  # Maximum length of the input sequence
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
batch_size = 64  # Batch size from the training process

# Implementing the Scaled-Dot Product Attention
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation
        weights = softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)

queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))

attention = DotProductAttention()
print(attention(queries, keys, values, d_k))
```

---

### Chapter Summary

# Chapter 15 Summary / 第15章总结

## Theme / 主题: Chapter 15 / Chapter 15

This chapter contains **1 code files** demonstrating chapter 15.

本章包含 **1 个代码文件**，演示Chapter 15。

---
## Evolution / 演化路线

  1. `11_testattention.ipynb` — Testattention

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 15) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 15）是机器学习流水线中的基础构建块。

---
