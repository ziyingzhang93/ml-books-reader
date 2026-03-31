# 从零构建Transformer / Building Transformers from Scratch
## Chapter 11

---

### Bert Mlp

# 01 — Bert Mlp / 01 Bert Mlp

**Chapter 11 — File 1 of 2 / 第11章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Bert Mlp**.

本脚本演示 **01 Bert Mlp**。

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
class BertMLP(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc1 = nn.Linear(dim, intermediate_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc2 = nn.Linear(intermediate_dim, dim)
        self.gelu = nn.GELU()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
```

---
## Learning Notes / 学习笔记

- **概念**: Bert Mlp 是机器学习中的常用技术。  
  *Bert Mlp is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Bert Mlp / 01 Bert Mlp
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class BertMLP(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc1 = nn.Linear(dim, intermediate_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc2 = nn.Linear(intermediate_dim, dim)
        self.gelu = nn.GELU()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Llama Mlp

# 02 — Llama Mlp / 02 Llama Mlp

**Chapter 11 — File 2 of 2 / 第11章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Llama Mlp**.

本脚本演示 **02 Llama Mlp**。

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
class LlamaMLP(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.gate_proj = nn.Linear(dim, intermediate_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.up_proj = nn.Linear(dim, intermediate_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.down_proj = nn.Linear(intermediate_dim, dim)
        self.act = nn.SiLU()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        swish = self.act(up)
        output = self.down_proj(swish * gate)
        return output
```

---
## Learning Notes / 学习笔记

- **概念**: Llama Mlp 是机器学习中的常用技术。  
  *Llama Mlp is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Llama Mlp / 02 Llama Mlp
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class LlamaMLP(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.gate_proj = nn.Linear(dim, intermediate_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.up_proj = nn.Linear(dim, intermediate_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.down_proj = nn.Linear(intermediate_dim, dim)
        self.act = nn.SiLU()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        swish = self.act(up)
        output = self.down_proj(swish * gate)
        return output
```

---

### Chapter Summary / 章节总结

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **2 code files** demonstrating chapter 11.

本章包含 **2 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `01_bert_mlp.ipynb` — Bert Mlp
  2. `02_llama_mlp.ipynb` — Llama Mlp

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---
