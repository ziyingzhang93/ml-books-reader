# 从零构建Transformer / Building Transformers from Scratch
## Chapter 09

---

### Attention



---

### Mask Functions

# 05 — Mask Functions / 05 Mask Functions

**Chapter 09 — File 2 of 4 / 第09章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Mask Functions**.

本脚本演示 **05 Mask Functions**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

def create_causal_mask(seq_len):
    """
    Create a causal mask for autoregressive attention.

    Args:
        seq_len: Length of the sequence

    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask

def create_padding_mask(batch, padding_token_id):
    """
    Create a padding mask for a batch of sequences.

    Args:
        batch: Batch of sequences, shape (batch_size, seq_len)
        padding_token_id: ID of the padding token

    Returns:
        Padding mask of shape (batch_size, seq_len, seq_len)
    """
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    batch_size, seq_len = batch.shape
    padded = torch.zeros_like(batch).float() \
                  .masked_fill(batch == padding_token_id, float('-inf'))
    # 创建全零张量 / Create tensor of zeros
    mask = torch.zeros(batch_size, seq_len, seq_len) + padded[:,:,None] + padded[:,None,:]
    return mask[:, None, :, :]

# 打印输出 / Print output
print(create_causal_mask(5))
batch = torch.tensor([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 0, 0, 0],
    [1, 2, 3, 4, 0, 0]
])
# 打印输出 / Print output
print(create_padding_mask(batch, 0))
```

---
## Learning Notes / 学习笔记

- **概念**: Mask Functions 是机器学习中的常用技术。  
  *Mask Functions is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mask Functions / 05 Mask Functions
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

def create_causal_mask(seq_len):
    """
    Create a causal mask for autoregressive attention.

    Args:
        seq_len: Length of the sequence

    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask

def create_padding_mask(batch, padding_token_id):
    """
    Create a padding mask for a batch of sequences.

    Args:
        batch: Batch of sequences, shape (batch_size, seq_len)
        padding_token_id: ID of the padding token

    Returns:
        Padding mask of shape (batch_size, seq_len, seq_len)
    """
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    batch_size, seq_len = batch.shape
    padded = torch.zeros_like(batch).float() \
                  .masked_fill(batch == padding_token_id, float('-inf'))
    # 创建全零张量 / Create tensor of zeros
    mask = torch.zeros(batch_size, seq_len, seq_len) + padded[:,:,None] + padded[:,None,:]
    return mask[:, None, :, :]

# 打印输出 / Print output
print(create_causal_mask(5))
batch = torch.tensor([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 0, 0, 0],
    [1, 2, 3, 4, 0, 0]
])
# 打印输出 / Print output
print(create_padding_mask(batch, 0))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Torch Mha

# 07 — Torch Mha / 07 Torch Mha

**Chapter 09 — File 3 of 4 / 第09章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Input tensor: 0 = padding**.

本脚本演示 **Input tensor: 0 = padding**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

dim = 16
num_heads = 4
# 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
attn_layer = torch.nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
```

---
## Step 2 — Input tensor: 0 = padding

```python
batch = torch.tensor([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 0, 0, 0],
    [1, 2, 3, 4, 0, 0]
])
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
batch_size, seq_len = batch.shape
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, dim)

padding_mask = (batch == 0)
y = attn_layer(x, x, x, key_padding_mask=padding_mask, attn_mask=None)
```

---
## Learning Notes / 学习笔记

- **概念**: Input tensor: 0 = padding 是机器学习中的常用技术。  
  *Input tensor: 0 = padding is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `nn.MultiheadAttention` | 多头注意力机制 | Multi-head attention mechanism |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Torch Mha / 07 Torch Mha
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

dim = 16
num_heads = 4
# 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
attn_layer = torch.nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)

# Input tensor: 0 = padding
batch = torch.tensor([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 0, 0, 0],
    [1, 2, 3, 4, 0, 0]
])
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
batch_size, seq_len = batch.shape
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, dim)

padding_mask = (batch == 0)
y = attn_layer(x, x, x, key_padding_mask=padding_mask, attn_mask=None)
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Padding

# 08 — Padding / 08 Padding

**Chapter 09 — File 4 of 4 / 第09章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Input tensor: 0 = padding**.

本脚本演示 **Input tensor: 0 = padding**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

def create_mask(query, key, padding_token_id):
    """
    Create a padding mask for a batch of sequences.

    Args:
        query: Batch of sequences for query, shape (batch_size, query_len)
        key: Batch of sequences for key, shape (batch_size, key_len)
        padding_token_id: ID of the padding token

    Returns:
        Padding mask of shape (batch_size, query_len, key_len)
    """
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    batch_size, query_len = query.shape
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    _, key_len = key.shape
    q_padded = torch.zeros_like(query).float() \
                    .masked_fill(query == padding_token_id, float('-inf'))
    k_padded = torch.zeros_like(key).float() \
                    .masked_fill(key == padding_token_id, float('-inf'))
    # 创建全零张量 / Create tensor of zeros
    mask = torch.zeros(batch_size, query_len, key_len) + \
           q_padded[:,:,None] + \
           k_padded[:,None,:]
    return mask

dim = 16
num_heads = 4
# 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
attn_layer = torch.nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
```

---
## Step 2 — Input tensor: 0 = padding

```python
batch = torch.tensor([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 0, 0, 0],
    [1, 2, 3, 4, 0, 0]
])
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
batch_size, seq_len = batch.shape
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, dim)

attn_mask = create_mask(batch, batch, 0)
attn_mask = attn_mask.repeat(1, num_heads, 1, 1).view(-1, seq_len, seq_len)

y = attn_layer(x, x, x, key_padding_mask=None, attn_mask=attn_mask)
```

---
## Learning Notes / 学习笔记

- **概念**: Input tensor: 0 = padding 是机器学习中的常用技术。  
  *Input tensor: 0 = padding is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `nn.MultiheadAttention` | 多头注意力机制 | Multi-head attention mechanism |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Padding / 08 Padding
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

def create_mask(query, key, padding_token_id):
    """
    Create a padding mask for a batch of sequences.

    Args:
        query: Batch of sequences for query, shape (batch_size, query_len)
        key: Batch of sequences for key, shape (batch_size, key_len)
        padding_token_id: ID of the padding token

    Returns:
        Padding mask of shape (batch_size, query_len, key_len)
    """
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    batch_size, query_len = query.shape
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    _, key_len = key.shape
    q_padded = torch.zeros_like(query).float() \
                    .masked_fill(query == padding_token_id, float('-inf'))
    k_padded = torch.zeros_like(key).float() \
                    .masked_fill(key == padding_token_id, float('-inf'))
    # 创建全零张量 / Create tensor of zeros
    mask = torch.zeros(batch_size, query_len, key_len) + \
           q_padded[:,:,None] + \
           k_padded[:,None,:]
    return mask

dim = 16
num_heads = 4
# 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
attn_layer = torch.nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)

# Input tensor: 0 = padding
batch = torch.tensor([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 0, 0, 0],
    [1, 2, 3, 4, 0, 0]
])
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
batch_size, seq_len = batch.shape
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, dim)

attn_mask = create_mask(batch, batch, 0)
attn_mask = attn_mask.repeat(1, num_heads, 1, 1).view(-1, seq_len, seq_len)

y = attn_layer(x, x, x, key_padding_mask=None, attn_mask=attn_mask)
```

---

### Chapter Summary / 章节总结

# Chapter 09 Summary / 第09章总结

## Theme / 主题: Chapter 09 / Chapter 09

This chapter contains **4 code files** demonstrating chapter 09.

本章包含 **4 个代码文件**，演示Chapter 09。

---
## Evolution / 演化路线

  1. `03_attention.ipynb` — Attention
  2. `05_mask_functions.ipynb` — Mask Functions
  3. `07_torch_mha.ipynb` — Torch Mha
  4. `08_padding.ipynb` — Padding

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 09) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 09）是机器学习流水线中的基础构建块。

---
