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
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F
# 导入数学函数库 / Import math functions library
import math

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class MultiHeadAttention(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, d_model, num_heads):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.q_proj = nn.Linear(d_model, d_model)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.k_proj = nn.Linear(d_model, d_model)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.v_proj = nn.Linear(d_model, d_model)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out_proj = nn.Linear(d_model, d_model)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
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

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mha / 01 Mha
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F
# 导入数学函数库 / Import math functions library
import math

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class MultiHeadAttention(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, d_model, num_heads):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.q_proj = nn.Linear(d_model, d_model)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.k_proj = nn.Linear(d_model, d_model)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.v_proj = nn.Linear(d_model, d_model)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out_proj = nn.Linear(d_model, d_model)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
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
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F
# 导入数学函数库 / Import math functions library
import math

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class GroupedQueryAttention(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, d_model, num_heads, num_groups):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads     # num of query heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.head_dim = d_model // num_heads

        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.k_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.v_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_model)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
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

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gqa / 02 Gqa
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F
# 导入数学函数库 / Import math functions library
import math

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class GroupedQueryAttention(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, d_model, num_heads, num_groups):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads     # num of query heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.head_dim = d_model // num_heads

        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.k_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.v_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_model)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
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
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class GroupedQueryAttention(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, d_model, num_heads, num_groups):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads     # num of query heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.head_dim = d_model // num_heads

        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.k_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.v_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_model)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
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

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gqa Torch / 03 Gqa Torch
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class GroupedQueryAttention(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, d_model, num_heads, num_groups):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads     # num of query heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.head_dim = d_model // num_heads

        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.k_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.v_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_model)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
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
