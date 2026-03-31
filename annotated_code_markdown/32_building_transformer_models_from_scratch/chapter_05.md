# 从零构建Transformer / Building Transformers from Scratch
## Chapter 05

---

### Sinusoidal

# 01 — Sinusoidal / 01 Sinusoidal

**Chapter 05 — File 1 of 4 / 第05章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Example usage**.

本脚本演示 **Example usage**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

def create_sinusoidal_encodings(seq_len, dim):
    N = 10000
    # 生成整数序列 / Generate integer sequence
    i = torch.arange(0, dim//2)
    div_term = torch.exp(-np.log(N) * (2*i / dim))
    # 生成整数序列 / Generate integer sequence
    position = torch.arange(seq_len).unsqueeze(1)

    # 创建全零张量 / Create tensor of zeros
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
```

---
## Step 2 — Example usage

```python
seq_len = 512
dim = 768
positional_encodings = create_sinusoidal_encodings(seq_len, dim)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
sequence = torch.randn(seq_len, dim)
sequence = sequence + positional_encodings
```

---
## Learning Notes / 学习笔记

- **概念**: Example usage 是机器学习中的常用技术。  
  *Example usage is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sinusoidal / 01 Sinusoidal
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

def create_sinusoidal_encodings(seq_len, dim):
    N = 10000
    # 生成整数序列 / Generate integer sequence
    i = torch.arange(0, dim//2)
    div_term = torch.exp(-np.log(N) * (2*i / dim))
    # 生成整数序列 / Generate integer sequence
    position = torch.arange(seq_len).unsqueeze(1)

    # 创建全零张量 / Create tensor of zeros
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Example usage
seq_len = 512
dim = 768
positional_encodings = create_sinusoidal_encodings(seq_len, dim)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
sequence = torch.randn(seq_len, dim)
sequence = sequence + positional_encodings
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Learned

# 02 — Learned / 02 Learned

**Chapter 05 — File 2 of 4 / 第05章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Example usage**.

本脚本演示 **Example usage**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class LearnedPositionalEncoding(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, max_seq_len, dim):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.position_embeddings = nn.Embedding(max_seq_len, dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # 生成整数序列 / Generate integer sequence
        positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
        position_embeddings = self.position_embeddings(positions)
        return x + position_embeddings
```

---
## Step 2 — Example usage

```python
model = LearnedPositionalEncoding(max_seq_len=512, dim=768)
```

---
## Learning Notes / 学习笔记

- **概念**: Example usage 是机器学习中的常用技术。  
  *Example usage is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `nn.Embedding` | 将整数ID映射为稠密向量 | Map integer IDs to dense vectors |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Learned / 02 Learned
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class LearnedPositionalEncoding(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, max_seq_len, dim):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.position_embeddings = nn.Embedding(max_seq_len, dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # 生成整数序列 / Generate integer sequence
        positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
        position_embeddings = self.position_embeddings(positions)
        return x + position_embeddings

# Example usage
model = LearnedPositionalEncoding(max_seq_len=512, dim=768)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Rope

# 03 — Rope / 03 Rope

**Chapter 05 — File 3 of 4 / 第05章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Rope**.

本脚本演示 **03 Rope**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class RotaryPositionalEncoding(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, max_seq_len=512):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        N = 10000
        # 生成整数序列 / Generate integer sequence
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        # 生成整数序列 / Generate integer sequence
        position = torch.arange(max_seq_len).float()
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)
```

---
## Learning Notes / 学习笔记

- **概念**: Rope 是机器学习中的常用技术。  
  *Rope is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Rope / 03 Rope
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class RotaryPositionalEncoding(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, max_seq_len=512):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        N = 10000
        # 生成整数序列 / Generate integer sequence
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        # 生成整数序列 / Generate integer sequence
        position = torch.arange(max_seq_len).float()
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Relative

# 04 — Relative / 04 Relative

**Chapter 05 — File 4 of 4 / 第05章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Relative**.

本脚本演示 **04 Relative**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class RelativePositionalEncoding(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, max_relative_position, d_model):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.max_relative_position = max_relative_position
        self.relative_attention_bias = nn.Parameter(
            # 生成正态分布随机张量 / Generate random tensor from normal distribution
            torch.randn(2 * max_relative_position + 1, d_model)
        )

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, length):
        # 生成整数序列 / Generate integer sequence
        context_position = torch.arange(length, dtype=torch.long)[:, None]
        # 生成整数序列 / Generate integer sequence
        memory_position = torch.arange(length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = relative_position + self.max_relative_position
        return self.relative_attention_bias[relative_position_bucket]
```

---
## Learning Notes / 学习笔记

- **概念**: Relative 是机器学习中的常用技术。  
  *Relative is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Relative / 04 Relative
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class RelativePositionalEncoding(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, max_relative_position, d_model):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.max_relative_position = max_relative_position
        self.relative_attention_bias = nn.Parameter(
            # 生成正态分布随机张量 / Generate random tensor from normal distribution
            torch.randn(2 * max_relative_position + 1, d_model)
        )

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, length):
        # 生成整数序列 / Generate integer sequence
        context_position = torch.arange(length, dtype=torch.long)[:, None]
        # 生成整数序列 / Generate integer sequence
        memory_position = torch.arange(length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = relative_position + self.max_relative_position
        return self.relative_attention_bias[relative_position_bucket]
```

---

### Chapter Summary / 章节总结

# Chapter 05 Summary / 第05章总结

## Theme / 主题: Chapter 05 / Chapter 05

This chapter contains **4 code files** demonstrating chapter 05.

本章包含 **4 个代码文件**，演示Chapter 05。

---
## Evolution / 演化路线

  1. `01_sinusoidal.ipynb` — Sinusoidal
  2. `02_learned.ipynb` — Learned
  3. `03_rope.ipynb` — Rope
  4. `04_relative.ipynb` — Relative

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 05) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 05）是机器学习流水线中的基础构建块。

---
