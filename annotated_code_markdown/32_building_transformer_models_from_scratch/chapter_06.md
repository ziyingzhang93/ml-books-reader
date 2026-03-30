# 从零构建Transformer
## Chapter 06

---

### Extrapolate

# 01 — Extrapolate / 01 Extrapolate

**Chapter 06 — File 1 of 2 / 第06章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Normal case: use learned embeddings**.

本脚本演示 **Normal case: use learned embeddings**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
import torch
import torch.nn as nn

class ExtrapolatingLearnedEncoding(nn.Module):
    def __init__(self, max_trained_len, d):
        super().__init__()
        self.max_trained_len = max_trained_len
        self.position_embeddings = nn.Embedding(max_trained_len, d)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len <= self.max_trained_len:
```

---
## Step 2 — Normal case: use learned embeddings

```python
positions = torch.arange(seq_len, device=x.device)
            return x + self.position_embeddings(positions)
        else:
```

---
## Step 3 — Extrapolation case: use interpolation

```python
positions = torch.arange(seq_len, device=x.device)
```

---
## Step 4 — Interpolate between existing positions

```python
scale = (self.max_trained_len - 1) / (seq_len - 1)
            scaled_positions = positions * scale
```

---
## Step 5 — Get floor and ceiling positions

```python
pos_floor = torch.floor(scaled_positions).long()
            pos_ceil = torch.ceil(scaled_positions).long()
```

---
## Step 6 — Get weights for interpolation

```python
weights = (scaled_positions - pos_floor.float()).unsqueeze(-1)
```

---
## Step 7 — Interpolate

```python
emb_floor = self.position_embeddings(pos_floor)
            emb_ceil = self.position_embeddings(pos_ceil)
            return x + (1 - weights) * emb_floor + weights * emb_ceil
```

---
## Learning Notes / 学习笔记

- **概念**: Normal case: use learned embeddings 是机器学习中的常用技术。  
  *Normal case: use learned embeddings is a common technique in machine learning.*

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
# Extrapolate / 01 Extrapolate
# Complete Code / 完整代码
# ===============================

import torch
import torch.nn as nn

class ExtrapolatingLearnedEncoding(nn.Module):
    def __init__(self, max_trained_len, d):
        super().__init__()
        self.max_trained_len = max_trained_len
        self.position_embeddings = nn.Embedding(max_trained_len, d)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len <= self.max_trained_len:
            # Normal case: use learned embeddings
            positions = torch.arange(seq_len, device=x.device)
            return x + self.position_embeddings(positions)
        else:
            # Extrapolation case: use interpolation
            positions = torch.arange(seq_len, device=x.device)
            # Interpolate between existing positions
            scale = (self.max_trained_len - 1) / (seq_len - 1)
            scaled_positions = positions * scale
            # Get floor and ceiling positions
            pos_floor = torch.floor(scaled_positions).long()
            pos_ceil = torch.ceil(scaled_positions).long()
            # Get weights for interpolation
            weights = (scaled_positions - pos_floor.float()).unsqueeze(-1)
            # Interpolate
            emb_floor = self.position_embeddings(pos_floor)
            emb_ceil = self.position_embeddings(pos_ceil)
            return x + (1 - weights) * emb_floor + weights * emb_ceil
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Yarn

# 02 — Yarn / 02 Yarn

**Chapter 06 — File 2 of 2 / 第06章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Original RoPE multiplied with a scaling factor**.

本脚本演示 **Original RoPE multiplied with a scaling factor**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
import torch
import torch.nn as nn
import numpy as np

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class YaRN(nn.Module):
    def __init__(self, dim, orig_seq_len=512, scale=4, alpha=1, beta=32):
        super().__init__()
        N = 10000
        pos_freq = N ** (torch.arange(0, dim, 2).float() / dim)
        inv_freq_extrapolation = 1. / pos_freq
        inv_freq_interpolation = 1. / (scale * pos_freq)

        low = dim * np.log(orig_seq_len / (2*np.pi*beta)) / (2*np.log(N))
        high = dim * np.log(orig_seq_len / (2*np.pi*alpha)) / (2*np.log(N))
        low = max(np.floor(low), 0)
        high = min(np.ceil(high), dim-1)

        linear_func = (torch.arange(dim // 2).float() - low) / (high - low)
        ramp_func = torch.clamp(linear_func, 0, 1)
        inv_freq_factor = 1 - ramp_func
        inv_freq = inv_freq_interpolation * (1-inv_freq_factor) + \
                   inv_freq_extrapolation * inv_freq_factor
```

---
## Step 2 — Original RoPE multiplied with a scaling factor

```python
scaling_factor = 0.1 * np.log(scale) + 1.0
        position = torch.arange(orig_seq_len * scale).float()
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos() * scaling_factor)
        self.register_buffer("sin", sinusoid_inp.sin() * scaling_factor)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)
```

---
## Learning Notes / 学习笔记

- **概念**: Original RoPE multiplied with a scaling factor 是机器学习中的常用技术。  
  *Original RoPE multiplied with a scaling factor is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `numpy` | 数值计算库 | Numerical computing library |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Yarn / 02 Yarn
# Complete Code / 完整代码
# ===============================

import torch
import torch.nn as nn
import numpy as np

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class YaRN(nn.Module):
    def __init__(self, dim, orig_seq_len=512, scale=4, alpha=1, beta=32):
        super().__init__()
        N = 10000
        pos_freq = N ** (torch.arange(0, dim, 2).float() / dim)
        inv_freq_extrapolation = 1. / pos_freq
        inv_freq_interpolation = 1. / (scale * pos_freq)

        low = dim * np.log(orig_seq_len / (2*np.pi*beta)) / (2*np.log(N))
        high = dim * np.log(orig_seq_len / (2*np.pi*alpha)) / (2*np.log(N))
        low = max(np.floor(low), 0)
        high = min(np.ceil(high), dim-1)

        linear_func = (torch.arange(dim // 2).float() - low) / (high - low)
        ramp_func = torch.clamp(linear_func, 0, 1)
        inv_freq_factor = 1 - ramp_func
        inv_freq = inv_freq_interpolation * (1-inv_freq_factor) + \
                   inv_freq_extrapolation * inv_freq_factor

        # Original RoPE multiplied with a scaling factor
        scaling_factor = 0.1 * np.log(scale) + 1.0
        position = torch.arange(orig_seq_len * scale).float()
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos() * scaling_factor)
        self.register_buffer("sin", sinusoid_inp.sin() * scaling_factor)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)
```

---

### Chapter Summary

# Chapter 06 Summary / 第06章总结

## Theme / 主题: Chapter 06 / Chapter 06

This chapter contains **2 code files** demonstrating chapter 06.

本章包含 **2 个代码文件**，演示Chapter 06。

---
## Evolution / 演化路线

  1. `01_extrapolate.ipynb` — Extrapolate
  2. `02_yarn.ipynb` — Yarn

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 06) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 06）是机器学习流水线中的基础构建块。

---
