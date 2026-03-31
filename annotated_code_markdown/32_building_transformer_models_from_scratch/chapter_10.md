# 从零构建Transformer / Building Transformers from Scratch
## Chapter 10

---

### Layernorm

# 01 — Layernorm / 01 Layernorm

**Chapter 10 — File 1 of 6 / 第10章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Calculate mean and variance across the last dimension(s)**.

本脚本演示 **Calculate mean and variance across the last dimension(s)**。

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
class LayerNorm(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, eps=1e-5):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.eps = eps

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
```

---
## Step 2 — Calculate mean and variance across the last dimension(s)

```python
mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
```

---
## Step 3 — Normalize

```python
x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm
```

---
## Step 4 — Example usage

```python
batch_size, seq_len, hidden_dim = 2, 5, 128
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, hidden_dim)
layer_norm = LayerNorm()
output = layer_norm(x)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Input shape: {x.shape}")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Output shape: {output.shape}")
# 打印输出 / Print output
print(f"Output mean:\n{output.mean(axis=2)}")
# 打印输出 / Print output
print(f"Output std:\n{output.std(axis=2, correction=0)}")
```

---
## Learning Notes / 学习笔记

- **概念**: Calculate mean and variance across the last dimension(s) 是机器学习中的常用技术。  
  *Calculate mean and variance across the last dimension(s) is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Layernorm / 01 Layernorm
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class LayerNorm(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, eps=1e-5):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.eps = eps

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # Calculate mean and variance across the last dimension(s)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm

# Example usage
batch_size, seq_len, hidden_dim = 2, 5, 128
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, hidden_dim)
layer_norm = LayerNorm()
output = layer_norm(x)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Input shape: {x.shape}")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Output shape: {output.shape}")
# 打印输出 / Print output
print(f"Output mean:\n{output.mean(axis=2)}")
# 打印输出 / Print output
print(f"Output std:\n{output.std(axis=2, correction=0)}")
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Layernorm

# 02 — Layernorm / 02 Layernorm

**Chapter 10 — File 2 of 6 / 第10章 — 第2个文件（共6个）**

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
class LayerNorm(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, eps=1e-5):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.eps = eps
        # 创建全一张量 / Create tensor of ones
        self.weight = nn.Parameter(torch.ones(dim))
        # 创建全零张量 / Create tensor of zeros
        self.bias = nn.Parameter(torch.zeros(dim))

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias
```

---
## Step 2 — Example usage

```python
batch_size, seq_len, hidden_dim = 2, 5, 128
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, hidden_dim)
layer_norm = LayerNorm(hidden_dim)
output = layer_norm(x)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Input shape: {x.shape}")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Output shape: {output.shape}")
# 打印输出 / Print output
print(f"Output mean:\n{output.mean(axis=2)}")
# 打印输出 / Print output
print(f"Output std:\n{output.std(axis=2, correction=0)}")
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
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Layernorm / 02 Layernorm
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class LayerNorm(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, eps=1e-5):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.eps = eps
        # 创建全一张量 / Create tensor of ones
        self.weight = nn.Parameter(torch.ones(dim))
        # 创建全零张量 / Create tensor of zeros
        self.bias = nn.Parameter(torch.zeros(dim))

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias

# Example usage
batch_size, seq_len, hidden_dim = 2, 5, 128
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, hidden_dim)
layer_norm = LayerNorm(hidden_dim)
output = layer_norm(x)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Input shape: {x.shape}")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Output shape: {output.shape}")
# 打印输出 / Print output
print(f"Output mean:\n{output.mean(axis=2)}")
# 打印输出 / Print output
print(f"Output std:\n{output.std(axis=2, correction=0)}")
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Adanorm

# 03 — Adanorm / 03 Adanorm

**Chapter 10 — File 3 of 6 / 第10章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Adaptive parameters**.

本脚本演示 **Adaptive parameters**。

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
class AdaptiveLayerNorm(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, eps=1e-5):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.dim = dim
        self.eps = eps
```

---
## Step 2 — Adaptive parameters

```python
# 全连接层：y = xW + b / Fully connected layer: y = xW + b
self.ada_weight = nn.Linear(dim, dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.ada_bias = nn.Linear(dim, dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
```

---
## Step 3 — Standard LayerNorm

```python
mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
```

---
## Step 4 — Adaptive scaling and shifting

```python
ada_w = self.ada_weight(x)
        ada_b = self.ada_bias(x)

        return x_norm * ada_w + ada_b
```

---
## Step 5 — Example usage

```python
batch_size, seq_len, hidden_dim = 2, 5, 8
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, hidden_dim)

ada_ln = AdaptiveLayerNorm(hidden_dim)
output = ada_ln(x)
```

---
## Learning Notes / 学习笔记

- **概念**: Adaptive parameters 是机器学习中的常用技术。  
  *Adaptive parameters is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Adanorm / 03 Adanorm
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class AdaptiveLayerNorm(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, eps=1e-5):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.dim = dim
        self.eps = eps

        # Adaptive parameters
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.ada_weight = nn.Linear(dim, dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.ada_bias = nn.Linear(dim, dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # Standard LayerNorm
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Adaptive scaling and shifting
        ada_w = self.ada_weight(x)
        ada_b = self.ada_bias(x)

        return x_norm * ada_w + ada_b


# Example usage
batch_size, seq_len, hidden_dim = 2, 5, 8
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, hidden_dim)

ada_ln = AdaptiveLayerNorm(hidden_dim)
output = ada_ln(x)
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Rmsnorm

# 04 — Rmsnorm / 04 Rmsnorm

**Chapter 10 — File 4 of 6 / 第10章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Calculate RMS across the last dimension(s)**.

本脚本演示 **Calculate RMS across the last dimension(s)**。

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
class RMSNorm(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, eps=1e-6):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.dim = dim
        self.eps = eps
        # 创建全一张量 / Create tensor of ones
        self.weight = nn.Parameter(torch.ones(dim))

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
```

---
## Step 2 — Calculate RMS across the last dimension(s)

```python
rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
```

---
## Step 3 — Normalize

```python
x_norm = x * rms * self.weight
        return x_norm
```

---
## Step 4 — Example usage

```python
batch_size, seq_len, hidden_dim = 2, 5, 8
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, hidden_dim)
rms_norm = RMSNorm(hidden_dim)
output = rms_norm(x)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Input shape: {x.shape}")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Output shape: {output.shape}")
# 打印输出 / Print output
print(f"Output RMS: {torch.sqrt((output**2).mean(axis=2))}")
```

---
## Learning Notes / 学习笔记

- **概念**: Calculate RMS across the last dimension(s) 是机器学习中的常用技术。  
  *Calculate RMS across the last dimension(s) is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Rmsnorm / 04 Rmsnorm
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class RMSNorm(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, eps=1e-6):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.dim = dim
        self.eps = eps
        # 创建全一张量 / Create tensor of ones
        self.weight = nn.Parameter(torch.ones(dim))

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # Calculate RMS across the last dimension(s)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Normalize
        x_norm = x * rms * self.weight
        return x_norm

# Example usage
batch_size, seq_len, hidden_dim = 2, 5, 8
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, hidden_dim)
rms_norm = RMSNorm(hidden_dim)
output = rms_norm(x)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Input shape: {x.shape}")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Output shape: {output.shape}")
# 打印输出 / Print output
print(f"Output RMS: {torch.sqrt((output**2).mean(axis=2))}")
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Torch Norms

# 05 — Torch Norms / 05 Torch Norms

**Chapter 10 — File 5 of 6 / 第10章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **PyTorch's LayerNorm**.

本脚本演示 **PyTorch's LayerNorm**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


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
```

---
## Step 2 — PyTorch's LayerNorm

```python
batch_size, seq_len, hidden_dim = 2, 5, 8
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, hidden_dim)
```

---
## Step 3 — LayerNorm normalizes over the last dimension

```python
# 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
layer_norm = nn.LayerNorm(hidden_dim)
output_ln = layer_norm(x)
```

---
## Step 4 — RMSNorm normalizes over the last dimension

```python
rms_norm = nn.RMSNorm(hidden_dim)
output_rms = rms_norm(x)
```

---
## Learning Notes / 学习笔记

- **概念**: PyTorch's LayerNorm 是机器学习中的常用技术。  
  *PyTorch's LayerNorm is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `nn.LayerNorm` | 层归一化，Transformer常用 | Layer normalization, common in Transformers |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Torch Norms / 05 Torch Norms
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# PyTorch's LayerNorm
batch_size, seq_len, hidden_dim = 2, 5, 8
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, hidden_dim)

# LayerNorm normalizes over the last dimension
# 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
layer_norm = nn.LayerNorm(hidden_dim)
output_ln = layer_norm(x)

# RMSNorm normalizes over the last dimension
rms_norm = nn.RMSNorm(hidden_dim)
output_rms = rms_norm(x)
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Norm Weights

# 06 — Norm Weights / 06 Norm Weights

**Chapter 10 — File 6 of 6 / 第10章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **PyTorch's LayerNorm**.

本脚本演示 **PyTorch's LayerNorm**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


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
```

---
## Step 2 — PyTorch's LayerNorm

```python
batch_size, seq_len, hidden_dim = 2, 5, 8
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, hidden_dim)
```

---
## Step 3 — LayerNorm normalizes over the last dimension

```python
# 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
layer_norm = nn.LayerNorm(hidden_dim)
output_ln = layer_norm(x)
```

---
## Step 4 — RMSNorm normalizes over the last dimension

```python
rms_norm = nn.RMSNorm(hidden_dim)
output_rms = rms_norm(x)

# 打印输出 / Print output
print(layer_norm.weight) # nn.Parameter
# 打印输出 / Print output
print(layer_norm.bias)   # nn.Parameter
# 打印输出 / Print output
print(rms_norm.weight)   # nn.Parameter
```

---
## Learning Notes / 学习笔记

- **概念**: PyTorch's LayerNorm 是机器学习中的常用技术。  
  *PyTorch's LayerNorm is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `nn.LayerNorm` | 层归一化，Transformer常用 | Layer normalization, common in Transformers |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Norm Weights / 06 Norm Weights
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# PyTorch's LayerNorm
batch_size, seq_len, hidden_dim = 2, 5, 8
# 生成正态分布随机张量 / Generate random tensor from normal distribution
x = torch.randn(batch_size, seq_len, hidden_dim)

# LayerNorm normalizes over the last dimension
# 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
layer_norm = nn.LayerNorm(hidden_dim)
output_ln = layer_norm(x)

# RMSNorm normalizes over the last dimension
rms_norm = nn.RMSNorm(hidden_dim)
output_rms = rms_norm(x)

# 打印输出 / Print output
print(layer_norm.weight) # nn.Parameter
# 打印输出 / Print output
print(layer_norm.bias)   # nn.Parameter
# 打印输出 / Print output
print(rms_norm.weight)   # nn.Parameter
```

---

### Chapter Summary / 章节总结

# Chapter 10 Summary / 第10章总结

## Theme / 主题: Chapter 10 / Chapter 10

This chapter contains **6 code files** demonstrating chapter 10.

本章包含 **6 个代码文件**，演示Chapter 10。

---
## Evolution / 演化路线

  1. `01_layernorm.ipynb` — Layernorm
  2. `02_layernorm.ipynb` — Layernorm
  3. `03_adanorm.ipynb` — Adanorm
  4. `04_rmsnorm.ipynb` — Rmsnorm
  5. `05_torch_norms.ipynb` — Torch Norms
  6. `06_norm_weights.ipynb` — Norm Weights

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 10) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 10）是机器学习流水线中的基础构建块。

---
