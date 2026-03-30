# 从零构建Transformer / Building Transformers from Scratch
## Chapter 07

---

### Mha

# 01 — Mha / 01 Mha

**Chapter 07 — File 1 of 3 / 第07章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Project queries, keys, and values**.

本脚本演示 **Project queries, keys, and values**。

---
## Step 1 — Step 1

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
```

---
## Step 2 — Project queries, keys, and values

```python
q = self.q_proj(x) \
                .view(batch_size, seq_length, self.num_heads, self.head_dim) \
                .transpose(1, 2)
        k = self.k_proj(x) \
                .view(batch_size, seq_length, self.num_heads, self.head_dim) \
                .transpose(1, 2)
        v = self.v_proj(x) \
                .view(batch_size, seq_length, self.num_heads, self.head_dim) \
                .transpose(1, 2)
```

---
## Step 3 — Compute attention scores, optionally add attention mask to the score

```python
scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
```

---
## Step 4 — optional: attn_weights = F.dropout(attn_weights, p=0.2)
Apply attention weights to values

```python
context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.d_model)

        return self.out_proj(context)
```

---
## Learning Notes / 学习笔记

- **概念**: Project queries, keys, and values 是机器学习中的常用技术。  
  *Project queries, keys, and values is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mha / 01 Mha
# Complete Code / 完整代码
# ===============================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Project queries, keys, and values
        q = self.q_proj(x) \
                .view(batch_size, seq_length, self.num_heads, self.head_dim) \
                .transpose(1, 2)
        k = self.k_proj(x) \
                .view(batch_size, seq_length, self.num_heads, self.head_dim) \
                .transpose(1, 2)
        v = self.v_proj(x) \
                .view(batch_size, seq_length, self.num_heads, self.head_dim) \
                .transpose(1, 2)

        # Compute attention scores, optionally add attention mask to the score
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        # optional: attn_weights = F.dropout(attn_weights, p=0.2)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.d_model)

        return self.out_proj(context)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Gqa

# 02 — Gqa / 02 Gqa

**Chapter 07 — File 2 of 3 / 第07章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Project queries, keys, and values**.

本脚本演示 **Project queries, keys, and values**。

---
## Step 1 — Step 1

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads     # num of query heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
```

---
## Step 2 — Project queries, keys, and values

```python
q = self.q_proj(x) \
                .view(batch_size, seq_length, self.num_heads, self.head_dim) \
                .transpose(1, 2)
        k = self.k_proj(x) \
                .view(batch_size, seq_length, self.num_groups, self.head_dim) \
                .transpose(1, 2)
        v = self.v_proj(x) \
                .view(batch_size, seq_length, self.num_groups, self.head_dim) \
                .transpose(1, 2)
```

---
## Step 3 — Expand k and v to match the number of query heads

```python
k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
```

---
## Step 4 — Compute attention scores

```python
scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
```

---
## Step 5 — optional: attn_weights = F.dropout(attn_weights, p=0.2)
Apply attention weights to values

```python
context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.d_model)

        return self.out_proj(context)
```

---
## Learning Notes / 学习笔记

- **概念**: Project queries, keys, and values 是机器学习中的常用技术。  
  *Project queries, keys, and values is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gqa / 02 Gqa
# Complete Code / 完整代码
# ===============================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads     # num of query heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Project queries, keys, and values
        q = self.q_proj(x) \
                .view(batch_size, seq_length, self.num_heads, self.head_dim) \
                .transpose(1, 2)
        k = self.k_proj(x) \
                .view(batch_size, seq_length, self.num_groups, self.head_dim) \
                .transpose(1, 2)
        v = self.v_proj(x) \
                .view(batch_size, seq_length, self.num_groups, self.head_dim) \
                .transpose(1, 2)

        # Expand k and v to match the number of query heads
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        # optional: attn_weights = F.dropout(attn_weights, p=0.2)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.d_model)

        return self.out_proj(context)
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Gqa Torch

# 03 — Gqa Torch / 03 Gqa Torch

**Chapter 07 — File 3 of 3 / 第07章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Project queries, keys, and values**.

本脚本演示 **Project queries, keys, and values**。

---
## Step 1 — Step 1

```python
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads     # num of query heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
```

---
## Step 2 — Project queries, keys, and values

```python
q = self.q_proj(x) \
                .view(batch_size, seq_length, self.num_heads, self.head_dim) \
                .transpose(1, 2)
        k = self.k_proj(x) \
                .view(batch_size, seq_length, self.num_groups, self.head_dim) \
                .transpose(1, 2)
        v = self.v_proj(x) \
                .view(batch_size, seq_length, self.num_groups, self.head_dim) \
                .transpose(1, 2)
```

---
## Step 3 — Compute attention scores using PyTorch's built-in function

```python
attn_output = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
```

---
## Step 4 — Output projection

```python
context = attn_output.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.d_model)
        return self.out_proj(context)
```

---
## Learning Notes / 学习笔记

- **概念**: Project queries, keys, and values 是机器学习中的常用技术。  
  *Project queries, keys, and values is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gqa Torch / 03 Gqa Torch
# Complete Code / 完整代码
# ===============================

import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads     # num of query heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Project queries, keys, and values
        q = self.q_proj(x) \
                .view(batch_size, seq_length, self.num_heads, self.head_dim) \
                .transpose(1, 2)
        k = self.k_proj(x) \
                .view(batch_size, seq_length, self.num_groups, self.head_dim) \
                .transpose(1, 2)
        v = self.v_proj(x) \
                .view(batch_size, seq_length, self.num_groups, self.head_dim) \
                .transpose(1, 2)

        # Compute attention scores using PyTorch's built-in function
        attn_output = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        # Output projection
        context = attn_output.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.d_model)
        return self.out_proj(context)
```

---

### Chapter Summary / 章节总结

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **3 code files** demonstrating chapter 07.

本章包含 **3 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `01_mha.ipynb` — Mha
  2. `02_gqa.ipynb` — Gqa
  3. `03_gqa_torch.ipynb` — Gqa Torch

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---
