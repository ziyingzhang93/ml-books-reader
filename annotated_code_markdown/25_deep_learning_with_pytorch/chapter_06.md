# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 06

---

### Sequential

# 02 — Sequential / 02 Sequential

**Chapter 06 — File 1 of 7 / 第06章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Sequential**.

本脚本演示 **02 Sequential**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(764, 100),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(100, 50),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(50, 10),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
# 打印输出 / Print output
print(model)
```

---
## Learning Notes / 学习笔记

- **概念**: Sequential 是机器学习中的常用技术。  
  *Sequential is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sequential / 02 Sequential
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(764, 100),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(100, 50),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(50, 10),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
# 打印输出 / Print output
print(model)
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Sequential



---

### Sequential

# 04 — Sequential / 04 Sequential

**Chapter 06 — File 3 of 7 / 第06章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Sequential**.

本脚本演示 **04 Sequential**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential()
# 全连接层：y = xW + b / Fully connected layer: y = xW + b
model.add_module("dense1", nn.Linear(8, 12))
# ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
model.add_module("act1", nn.ReLU())
# 全连接层：y = xW + b / Fully connected layer: y = xW + b
model.add_module("dense2", nn.Linear(12, 8))
# ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
model.add_module("act2", nn.ReLU())
# 全连接层：y = xW + b / Fully connected layer: y = xW + b
model.add_module("output", nn.Linear(8, 1))
# Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
model.add_module("outact", nn.Sigmoid())
# 打印输出 / Print output
print(model)
```

---
## Learning Notes / 学习笔记

- **概念**: Sequential 是机器学习中的常用技术。  
  *Sequential is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sequential / 04 Sequential
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential()
# 全连接层：y = xW + b / Fully connected layer: y = xW + b
model.add_module("dense1", nn.Linear(8, 12))
# ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
model.add_module("act1", nn.ReLU())
# 全连接层：y = xW + b / Fully connected layer: y = xW + b
model.add_module("dense2", nn.Linear(12, 8))
# ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
model.add_module("act2", nn.ReLU())
# 全连接层：y = xW + b / Fully connected layer: y = xW + b
model.add_module("output", nn.Linear(8, 1))
# Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
model.add_module("outact", nn.Sigmoid())
# 打印输出 / Print output
print(model)
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Save

# 12 — Save / 保存/加载模型

**Chapter 06 — File 4 of 7 / 第06章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Save**.

本脚本演示 **保存/加载模型**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(764, 100),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(100, 50),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(50, 10),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
torch.save(model, "my_model.pth")
```

---
## Learning Notes / 学习笔记

- **概念**: Save 是机器学习中的常用技术。  
  *Save is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Save / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(764, 100),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(100, 50),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(50, 10),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
torch.save(model, "my_model.pth")
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Load

# 13 — Load / 13 Load

**Chapter 06 — File 5 of 7 / 第06章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Load**.

本脚本演示 **13 Load**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

model = torch.load("my_model.pth")
```

---
## Learning Notes / 学习笔记

- **概念**: Load 是机器学习中的常用技术。  
  *Load is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load / 13 Load
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

model = torch.load("my_model.pth")
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Save

# 14 — Save / 保存/加载模型

**Chapter 06 — File 6 of 7 / 第06章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Save**.

本脚本演示 **保存/加载模型**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(764, 100),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(100, 50),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(50, 10),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
# 获取模型参数字典 / Get model parameter dictionary
torch.save(model.state_dict(), "my_model.pth")
```

---
## Learning Notes / 学习笔记

- **概念**: Save 是机器学习中的常用技术。  
  *Save is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Save / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(764, 100),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(100, 50),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(50, 10),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
# 获取模型参数字典 / Get model parameter dictionary
torch.save(model.state_dict(), "my_model.pth")
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Load

# 15 — Load / 15 Load

**Chapter 06 — File 7 of 7 / 第06章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Load**.

本脚本演示 **15 Load**。

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

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(764, 100),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(100, 50),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(50, 10),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
# 加载模型参数 / Load model parameters
model.load_state_dict(torch.load("my_model.pth"))
```

---
## Learning Notes / 学习笔记

- **概念**: Load 是机器学习中的常用技术。  
  *Load is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load / 15 Load
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(764, 100),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(100, 50),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(50, 10),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
# 加载模型参数 / Load model parameters
model.load_state_dict(torch.load("my_model.pth"))
```

---

### Chapter Summary / 章节总结

# Chapter 06 Summary / 第06章总结

## Theme / 主题: Chapter 06 / Chapter 06

This chapter contains **7 code files** demonstrating chapter 06.

本章包含 **7 个代码文件**，演示Chapter 06。

---
## Evolution / 演化路线

  1. `02_sequential.ipynb` — Sequential
  2. `03_sequential.ipynb` — Sequential
  3. `04_sequential.ipynb` — Sequential
  4. `12_save.ipynb` — Save
  5. `13_load.ipynb` — Load
  6. `14_save.ipynb` — Save
  7. `15_load.ipynb` — Load

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 06) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 06）是机器学习流水线中的基础构建块。

---
