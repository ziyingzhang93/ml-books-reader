# 从零构建Transformer / Building Transformers from Scratch
## Chapter 12

---

### Moe

# 01 — Moe / 01 Moe

**Chapter 12 — File 1 of 3 / 第12章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Create expert networks**.

本脚本演示 **Create expert networks**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Expert(nn.Module):
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
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swish = self.act(gate)
        output = self.down_proj(swish * up)
        return output

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class MoELayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim, num_experts, top_k=2):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
```

---
## Step 2 — Create expert networks

```python
self.experts = nn.ModuleList([
            # 生成整数序列 / Generate integer sequence
            Expert(dim, intermediate_dim) for _ in range(num_experts)
        ])
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.router = nn.Linear(dim, num_experts)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, hidden_states):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        batch_size, seq_len, hidden_dim = hidden_states.shape
```

---
## Step 3 — Reshape for expert processing, then compute routing probabilities

```python
hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
```

---
## Step 4 — shape of router_logits: (batch_size * seq_len, num_experts)

```python
router_logits = self.router(hidden_states_reshaped)
        routing_probs = F.softmax(router_logits, dim=-1)
```

---
## Step 5 — Select top-k experts, and scale the probabilities to sum to 1
output shape: (batch_size * seq_len, k)

```python
top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
```

---
## Step 6 — Process through selected experts

```python
output = []
        # 生成整数序列 / Generate integer sequence
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i]
```

---
## Step 7 — Process each vector in the batch and sequence with the selected expert

```python
expert_output = torch.stack([
                self.experts[exp_idx](hidden_states_reshaped[j])
                # 同时获取索引和值 / Get both index and value
                for j, exp_idx in enumerate(expert_idx)
            ], dim=0)
```

---
## Step 8 — Weighted sum by routing probability

```python
# 添加元素到列表末尾 / Append element to list end
output.append(expert_probs.unsqueeze(-1) * expert_output)
```

---
## Step 9 — Reshape back to original shape

```python
output = sum(output).view(batch_size, seq_len, hidden_dim)
        return output

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class MoETransformerLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim, num_experts, top_k=2, num_heads=8):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.moe = MoELayer(dim, intermediate_dim, num_experts, top_k)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
```

---
## Step 10 — Attention sublayer

```python
input_x = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        input_x = input_x + attn_output
```

---
## Step 11 — MoE sublayer

```python
x = self.norm2(input_x)
        moe_output = self.moe(x)
        return input_x + moe_output
```

---
## Learning Notes / 学习笔记

- **概念**: Create expert networks 是机器学习中的常用技术。  
  *Create expert networks is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.MultiheadAttention` | 多头注意力机制 | Multi-head attention mechanism |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Moe / 01 Moe
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Expert(nn.Module):
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
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swish = self.act(gate)
        output = self.down_proj(swish * up)
        return output

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class MoELayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim, num_experts, top_k=2):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
        # Create expert networks
        self.experts = nn.ModuleList([
            # 生成整数序列 / Generate integer sequence
            Expert(dim, intermediate_dim) for _ in range(num_experts)
        ])
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.router = nn.Linear(dim, num_experts)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, hidden_states):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Reshape for expert processing, then compute routing probabilities
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        # shape of router_logits: (batch_size * seq_len, num_experts)
        router_logits = self.router(hidden_states_reshaped)
        routing_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts, and scale the probabilities to sum to 1
        # output shape: (batch_size * seq_len, k)
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Process through selected experts
        output = []
        # 生成整数序列 / Generate integer sequence
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i]
            # Process each vector in the batch and sequence with the selected expert
            expert_output = torch.stack([
                self.experts[exp_idx](hidden_states_reshaped[j])
                # 同时获取索引和值 / Get both index and value
                for j, exp_idx in enumerate(expert_idx)
            ], dim=0)
            # Weighted sum by routing probability
            # 添加元素到列表末尾 / Append element to list end
            output.append(expert_probs.unsqueeze(-1) * expert_output)

        # Reshape back to original shape
        output = sum(output).view(batch_size, seq_len, hidden_dim)
        return output

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class MoETransformerLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim, num_experts, top_k=2, num_heads=8):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.moe = MoELayer(dim, intermediate_dim, num_experts, top_k)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # Attention sublayer
        input_x = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        input_x = input_x + attn_output

        # MoE sublayer
        x = self.norm2(input_x)
        moe_output = self.moe(x)
        return input_x + moe_output
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Test Moe

# 02 — Test Moe / 02 Test Moe

**Chapter 12 — File 2 of 3 / 第12章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Create expert networks**.

本脚本演示 **Create expert networks**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Expert(nn.Module):
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
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swish = self.act(gate)
        output = self.down_proj(swish * up)
        return output

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class MoELayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim, num_experts, top_k=2):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
```

---
## Step 2 — Create expert networks

```python
self.experts = nn.ModuleList([
            # 生成整数序列 / Generate integer sequence
            Expert(dim, intermediate_dim) for _ in range(num_experts)
        ])
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.router = nn.Linear(dim, num_experts)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, hidden_states):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        batch_size, seq_len, hidden_dim = hidden_states.shape
```

---
## Step 3 — Reshape for expert processing, the compute routing probabilities

```python
hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
```

---
## Step 4 — shape of router_logits: (batch_size * seq_len, num_experts)

```python
router_logits = self.router(hidden_states_reshaped)
        routing_probs = F.softmax(router_logits, dim=-1)
```

---
## Step 5 — Select top-k experts, and scale the probabilities to sum to 1
output shape: (batch_size * seq_len, k)

```python
top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
```

---
## Step 6 — Process through selected experts

```python
output = []
        # 生成整数序列 / Generate integer sequence
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i]
```

---
## Step 7 — Process each vector in the batch and sequence with the selected expert

```python
expert_output = torch.stack([
                self.experts[exp_idx](hidden_states_reshaped[j])
                # 同时获取索引和值 / Get both index and value
                for j, exp_idx in enumerate(expert_idx)
            ], dim=0)
```

---
## Step 8 — Weighted sum by routing probability

```python
# 添加元素到列表末尾 / Append element to list end
output.append(expert_probs.unsqueeze(-1) * expert_output)
```

---
## Step 9 — Reshape back to original shape

```python
output = sum(output).view(batch_size, seq_len, hidden_dim)
        return output

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class MoETransformerLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim, num_experts, top_k=2, num_heads=8):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.moe = MoELayer(dim, intermediate_dim, num_experts, top_k)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
```

---
## Step 10 — Attention sublayer

```python
input_x = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        input_x = input_x + attn_output
```

---
## Step 11 — MoE sublayer

```python
x = self.norm2(input_x)
        moe_output = self.moe(x)
        return input_x + moe_output

batch_size = 4
seq_len = 10
dim = 16
intermediate_dim = 72
num_experts = 8

# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, dim)
model = MoETransformerLayer(dim, intermediate_dim, num_experts)
y = model(x)
```

---
## Learning Notes / 学习笔记

- **概念**: Create expert networks 是机器学习中的常用技术。  
  *Create expert networks is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.MultiheadAttention` | 多头注意力机制 | Multi-head attention mechanism |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Test Moe / 02 Test Moe
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Expert(nn.Module):
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
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swish = self.act(gate)
        output = self.down_proj(swish * up)
        return output

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class MoELayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim, num_experts, top_k=2):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
        # Create expert networks
        self.experts = nn.ModuleList([
            # 生成整数序列 / Generate integer sequence
            Expert(dim, intermediate_dim) for _ in range(num_experts)
        ])
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.router = nn.Linear(dim, num_experts)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, hidden_states):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Reshape for expert processing, the compute routing probabilities
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        # shape of router_logits: (batch_size * seq_len, num_experts)
        router_logits = self.router(hidden_states_reshaped)
        routing_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts, and scale the probabilities to sum to 1
        # output shape: (batch_size * seq_len, k)
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Process through selected experts
        output = []
        # 生成整数序列 / Generate integer sequence
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i]
            # Process each vector in the batch and sequence with the selected expert
            expert_output = torch.stack([
                self.experts[exp_idx](hidden_states_reshaped[j])
                # 同时获取索引和值 / Get both index and value
                for j, exp_idx in enumerate(expert_idx)
            ], dim=0)
            # Weighted sum by routing probability
            # 添加元素到列表末尾 / Append element to list end
            output.append(expert_probs.unsqueeze(-1) * expert_output)

        # Reshape back to original shape
        output = sum(output).view(batch_size, seq_len, hidden_dim)
        return output

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class MoETransformerLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim, num_experts, top_k=2, num_heads=8):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.moe = MoELayer(dim, intermediate_dim, num_experts, top_k)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # Attention sublayer
        input_x = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        input_x = input_x + attn_output

        # MoE sublayer
        x = self.norm2(input_x)
        moe_output = self.moe(x)
        return input_x + moe_output

batch_size = 4
seq_len = 10
dim = 16
intermediate_dim = 72
num_experts = 8

# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, dim)
model = MoETransformerLayer(dim, intermediate_dim, num_experts)
y = model(x)
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Moe Shared

# 03 — Moe Shared / 03 Moe Shared

**Chapter 12 — File 3 of 3 / 第12章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Create expert networks**.

本脚本演示 **Create expert networks**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Expert(nn.Module):
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
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swish = self.act(gate)
        output = self.down_proj(swish * up)
        return output

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class MoELayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim, num_experts, top_k=2):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
```

---
## Step 2 — Create expert networks

```python
self.experts = nn.ModuleList([
            # 生成整数序列 / Generate integer sequence
            Expert(dim, intermediate_dim) for _ in range(num_experts)
        ])
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.router = nn.Linear(dim, num_experts)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, hidden_states):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        batch_size, seq_len, hidden_dim = hidden_states.shape
```

---
## Step 3 — Reshape for expert processing, the compute routing probabilities

```python
hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
```

---
## Step 4 — shape of router_logits: (batch_size * seq_len, num_experts)

```python
router_logits = self.router(hidden_states_reshaped)
        routing_probs = F.softmax(router_logits, dim=-1)
```

---
## Step 5 — Select top-k experts, and scale the probabilities to sum to 1
output shape: (batch_size * seq_len, k)

```python
top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
```

---
## Step 6 — Process through selected experts

```python
output = []
        # 生成整数序列 / Generate integer sequence
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i]
```

---
## Step 7 — Process each vector in the batch and sequence with the selected expert

```python
expert_output = torch.stack([
                self.experts[exp_idx](hidden_states_reshaped[j])
                # 同时获取索引和值 / Get both index and value
                for j, exp_idx in enumerate(expert_idx)
            ], dim=0)
```

---
## Step 8 — Weighted sum by routing probability

```python
# 添加元素到列表末尾 / Append element to list end
output.append(expert_probs.unsqueeze(-1) * expert_output)
```

---
## Step 9 — Reshape back to original shape

```python
output = sum(output).view(batch_size, seq_len, hidden_dim)
        return output

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class MoETransformerLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim, num_experts,
                 top_k=2, num_heads=8, num_shared_experts=1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.moe = MoELayer(dim, intermediate_dim, num_experts, top_k)
```

---
## Step 10 — shared experts

```python
self.shared_experts = nn.ModuleList([
            # 生成整数序列 / Generate integer sequence
            Expert(dim, intermediate_dim) for _ in range(num_shared_experts)
        ])
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
```

---
## Step 11 — Attention sublayer

```python
input_x = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        input_x = input_x + attn_output
```

---
## Step 12 — MoE sublayer

```python
x = self.norm2(input_x)
        moe_output = self.moe(x)
        for expert in self.shared_experts:
            moe_output += expert(x)
        return input_x + moe_output
```

---
## Learning Notes / 学习笔记

- **概念**: Create expert networks 是机器学习中的常用技术。  
  *Create expert networks is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.MultiheadAttention` | 多头注意力机制 | Multi-head attention mechanism |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Moe Shared / 03 Moe Shared
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Expert(nn.Module):
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
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swish = self.act(gate)
        output = self.down_proj(swish * up)
        return output

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class MoELayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim, num_experts, top_k=2):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
        # Create expert networks
        self.experts = nn.ModuleList([
            # 生成整数序列 / Generate integer sequence
            Expert(dim, intermediate_dim) for _ in range(num_experts)
        ])
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.router = nn.Linear(dim, num_experts)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, hidden_states):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Reshape for expert processing, the compute routing probabilities
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        # shape of router_logits: (batch_size * seq_len, num_experts)
        router_logits = self.router(hidden_states_reshaped)
        routing_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts, and scale the probabilities to sum to 1
        # output shape: (batch_size * seq_len, k)
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Process through selected experts
        output = []
        # 生成整数序列 / Generate integer sequence
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i]
            # Process each vector in the batch and sequence with the selected expert
            expert_output = torch.stack([
                self.experts[exp_idx](hidden_states_reshaped[j])
                # 同时获取索引和值 / Get both index and value
                for j, exp_idx in enumerate(expert_idx)
            ], dim=0)
            # Weighted sum by routing probability
            # 添加元素到列表末尾 / Append element to list end
            output.append(expert_probs.unsqueeze(-1) * expert_output)

        # Reshape back to original shape
        output = sum(output).view(batch_size, seq_len, hidden_dim)
        return output

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class MoETransformerLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, intermediate_dim, num_experts,
                 top_k=2, num_heads=8, num_shared_experts=1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.moe = MoELayer(dim, intermediate_dim, num_experts, top_k)
        # shared experts
        self.shared_experts = nn.ModuleList([
            # 生成整数序列 / Generate integer sequence
            Expert(dim, intermediate_dim) for _ in range(num_shared_experts)
        ])
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # Attention sublayer
        input_x = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        input_x = input_x + attn_output

        # MoE sublayer
        x = self.norm2(input_x)
        moe_output = self.moe(x)
        for expert in self.shared_experts:
            moe_output += expert(x)
        return input_x + moe_output
```

---

### Chapter Summary / 章节总结

# Chapter 12 Summary / 第12章总结

## Theme / 主题: Chapter 12 / Chapter 12

This chapter contains **3 code files** demonstrating chapter 12.

本章包含 **3 个代码文件**，演示Chapter 12。

---
## Evolution / 演化路线

  1. `01_moe.ipynb` — Moe
  2. `02_test_moe.ipynb` — Test Moe
  3. `03_moe_shared.ipynb` — Moe Shared

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 12) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 12）是机器学习流水线中的基础构建块。

---
