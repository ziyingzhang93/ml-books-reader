# 从零构建Transformer / Building Transformers from Scratch
## Chapter 13

---

### Bert Skip

# 01 — Bert Skip / 01 Bert Skip

**Chapter 13 — File 1 of 2 / 第13章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Skip connection around attention sub-layer**.

本脚本演示 **Skip connection around attention sub-layer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class BertLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim, num_heads):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.attention = nn.MultiheadAttention(dim, num_heads)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear1 = nn.Linear(dim, intermediate_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear2 = nn.Linear(intermediate_dim, dim)
        self.act = nn.GELU()
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm1 = nn.LayerNorm(dim)
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm2 = nn.LayerNorm(dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
```

---
## Step 2 — Skip connection around attention sub-layer

```python
attn_output = self.attention(x, x, x)[0]  # extract first element of the tuple
        x = x + attn_output  # Residual connection
        x = self.norm1(x)    # Layer normalization
```

---
## Step 3 — Skip connection around MLP sub-layer

```python
mlp_output = self.linear1(x)
        mlp_output = self.act(mlp_output)
        mlp_output = self.linear2(mlp_output)
        x = x + mlp_output   # Residual connection
        x = self.norm2(x)    # Layer normalization
        return x
```

---
## Learning Notes / 学习笔记

- **概念**: Skip connection around attention sub-layer 是机器学习中的常用技术。  
  *Skip connection around attention sub-layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `nn.LayerNorm` | 层归一化，Transformer常用 | Layer normalization, common in Transformers |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.MultiheadAttention` | 多头注意力机制 | Multi-head attention mechanism |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Bert Skip / 01 Bert Skip
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class BertLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim, num_heads):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.attention = nn.MultiheadAttention(dim, num_heads)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear1 = nn.Linear(dim, intermediate_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear2 = nn.Linear(intermediate_dim, dim)
        self.act = nn.GELU()
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm1 = nn.LayerNorm(dim)
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm2 = nn.LayerNorm(dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # Skip connection around attention sub-layer
        attn_output = self.attention(x, x, x)[0]  # extract first element of the tuple
        x = x + attn_output  # Residual connection
        x = self.norm1(x)    # Layer normalization

        # Skip connection around MLP sub-layer
        mlp_output = self.linear1(x)
        mlp_output = self.act(mlp_output)
        mlp_output = self.linear2(mlp_output)
        x = x + mlp_output   # Residual connection
        x = self.norm2(x)    # Layer normalization
        return x
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Prenorm



---

### Chapter Summary / 章节总结

# Chapter 13 Summary / 第13章总结

## Theme / 主题: Chapter 13 / Chapter 13

This chapter contains **2 code files** demonstrating chapter 13.

本章包含 **2 个代码文件**，演示Chapter 13。

---
## Evolution / 演化路线

  1. `01_bert_skip.ipynb` — Bert Skip
  2. `02_prenorm.ipynb` — Prenorm

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 13) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 13）是机器学习流水线中的基础构建块。

---
