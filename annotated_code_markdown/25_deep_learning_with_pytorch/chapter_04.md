# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 04

---

### Matrix

# 01 — Matrix / 01 Matrix

**Chapter 04 — File 1 of 9 / 第04章 — 第1个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Matrix**.

本脚本演示 **01 Matrix**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

x = torch.tensor([1, 2, 3])
# 打印输出 / Print output
print(x)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(x.shape)
# 打印输出 / Print output
print(x.dtype)
```

---
## Learning Notes / 学习笔记

- **概念**: Matrix 是机器学习中的常用技术。  
  *Matrix is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Matrix / 01 Matrix
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

x = torch.tensor([1, 2, 3])
# 打印输出 / Print output
print(x)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(x.shape)
# 打印输出 / Print output
print(x.dtype)
```

---

➡️ **Next / 下一步**: File 2 of 9

---

### Grad

# 02 — Grad / 02 Grad

**Chapter 04 — File 2 of 9 / 第04章 — 第2个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Grad**.

本脚本演示 **02 Grad**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
# 打印输出 / Print output
print(x)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(x.shape)
# 打印输出 / Print output
print(x.dtype)
```

---
## Learning Notes / 学习笔记

- **概念**: Grad 是机器学习中的常用技术。  
  *Grad is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grad / 02 Grad
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
# 打印输出 / Print output
print(x)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(x.shape)
# 打印输出 / Print output
print(x.dtype)
```

---

➡️ **Next / 下一步**: File 3 of 9

---

### Ops

# 03 — Ops / 03 Ops

**Chapter 04 — File 3 of 9 / 第04章 — 第3个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Ops**.

本脚本演示 **03 Ops**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

x = torch.tensor(3.6, requires_grad=True)
y = x * x
# 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
y.backward()
# 打印输出 / Print output
print("x =", x)
# 打印输出 / Print output
print("y =", y)
# 打印输出 / Print output
print("x.grad =", x.grad)
```

---
## Learning Notes / 学习笔记

- **概念**: Ops 是机器学习中的常用技术。  
  *Ops is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ops / 03 Ops
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

x = torch.tensor(3.6, requires_grad=True)
y = x * x
# 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
y.backward()
# 打印输出 / Print output
print("x =", x)
# 打印输出 / Print output
print("y =", y)
# 打印输出 / Print output
print("x.grad =", x.grad)
```

---

➡️ **Next / 下一步**: File 4 of 9

---

### Polynomial

# 04 — Polynomial / 04 Polynomial

**Chapter 04 — File 4 of 9 / 第04章 — 第4个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Polynomial**.

本脚本演示 **04 Polynomial**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

polynomial = np.poly1d([1, 2, 3])
# 打印输出 / Print output
print(polynomial)
```

---
## Learning Notes / 学习笔记

- **概念**: Polynomial 是机器学习中的常用技术。  
  *Polynomial is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Polynomial / 04 Polynomial
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

polynomial = np.poly1d([1, 2, 3])
# 打印输出 / Print output
print(polynomial)
```

---

➡️ **Next / 下一步**: File 5 of 9

---

### Eval

# 05 — Eval / 模型评估

**Chapter 04 — File 5 of 9 / 第04章 — 第5个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Eval**.

本脚本演示 **模型评估**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

polynomial = np.poly1d([1, 2, 3])
# 打印输出 / Print output
print(polynomial(1.5))
```

---
## Learning Notes / 学习笔记

- **概念**: Eval 是机器学习中的常用技术。  
  *Eval is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Eval / 模型评估
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

polynomial = np.poly1d([1, 2, 3])
# 打印输出 / Print output
print(polynomial(1.5))
```

---

➡️ **Next / 下一步**: File 6 of 9

---

### Samples

# 06 — Samples / 06 Samples

**Chapter 04 — File 6 of 9 / 第04章 — 第6个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Generate random samples roughly between -10 to +10**.

本脚本演示 **Generate random samples roughly between -10 to +10**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

polynomial = np.poly1d([1, 2, 3])
N = 20   # number of samples
```

---
## Step 2 — Generate random samples roughly between -10 to +10

```python
# 生成随机数 / Generate random numbers
X = np.random.randn(N,1) * 5
Y = polynomial(X)
# 打印输出 / Print output
print(X)
# 打印输出 / Print output
print(Y)
```

---
## Learning Notes / 学习笔记

- **概念**: Generate random samples roughly between -10 to +10 是机器学习中的常用技术。  
  *Generate random samples roughly between -10 to +10 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Samples / 06 Samples
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

polynomial = np.poly1d([1, 2, 3])
N = 20   # number of samples

# Generate random samples roughly between -10 to +10
# 生成随机数 / Generate random numbers
X = np.random.randn(N,1) * 5
Y = polynomial(X)
# 打印输出 / Print output
print(X)
# 打印输出 / Print output
print(Y)
```

---

➡️ **Next / 下一步**: File 7 of 9

---

### Grad



---

### Fitpoly



---

### Puzzle

# 09 — Puzzle / 09 Puzzle

**Chapter 04 — File 9 of 9 / 第04章 — 第9个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Gradient descent loop**.

本脚本演示 **Gradient descent loop**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model


---
## Code Flow / 代码流程

```
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  🏋️ 训练模型 / Train Model
```

---
## Step 1 — Step 1

```python
# 导入随机数生成模块 / Import random number module
import random
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

A = torch.tensor(random.random(), requires_grad=True)
B = torch.tensor(random.random(), requires_grad=True)
C = torch.tensor(random.random(), requires_grad=True)
D = torch.tensor(random.random(), requires_grad=True)
```

---
## Step 2 — Gradient descent loop

```python
EPOCHS = 2000
optimizer = torch.optim.NAdam([A, B, C, D], lr=0.01)
# 生成整数序列 / Generate integer sequence
for _ in range(EPOCHS):
    y1 = A + B - 9
    y2 = C - D - 1
    y3 = A + C - 8
    y4 = B - D - 2
    sqerr = y1*y1 + y2*y2 + y3*y3 + y4*y4
    # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
    optimizer.zero_grad()
    # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
    sqerr.backward()
    # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
    optimizer.step()

# 打印输出 / Print output
print(A)
# 打印输出 / Print output
print(B)
# 打印输出 / Print output
print(C)
# 打印输出 / Print output
print(D)
```

---
## Learning Notes / 学习笔记

- **概念**: Gradient descent loop 是机器学习中的常用技术。  
  *Gradient descent loop is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Puzzle / 09 Puzzle
# Complete Code / 完整代码
# ===============================

# 导入随机数生成模块 / Import random number module
import random
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

A = torch.tensor(random.random(), requires_grad=True)
B = torch.tensor(random.random(), requires_grad=True)
C = torch.tensor(random.random(), requires_grad=True)
D = torch.tensor(random.random(), requires_grad=True)

# Gradient descent loop
EPOCHS = 2000
optimizer = torch.optim.NAdam([A, B, C, D], lr=0.01)
# 生成整数序列 / Generate integer sequence
for _ in range(EPOCHS):
    y1 = A + B - 9
    y2 = C - D - 1
    y3 = A + C - 8
    y4 = B - D - 2
    sqerr = y1*y1 + y2*y2 + y3*y3 + y4*y4
    # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
    optimizer.zero_grad()
    # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
    sqerr.backward()
    # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
    optimizer.step()

# 打印输出 / Print output
print(A)
# 打印输出 / Print output
print(B)
# 打印输出 / Print output
print(C)
# 打印输出 / Print output
print(D)
```

---

### Chapter Summary / 章节总结

# Chapter 04 Summary / 第04章总结

## Theme / 主题: Chapter 04 / Chapter 04

This chapter contains **9 code files** demonstrating chapter 04.

本章包含 **9 个代码文件**，演示Chapter 04。

---
## Evolution / 演化路线

  1. `01_matrix.ipynb` — Matrix
  2. `02_grad.ipynb` — Grad
  3. `03_ops.ipynb` — Ops
  4. `04_polynomial.ipynb` — Polynomial
  5. `05_eval.ipynb` — Eval
  6. `06_samples.ipynb` — Samples
  7. `07_grad.ipynb` — Grad
  8. `08_fitpoly.ipynb` — Fitpoly
  9. `09_puzzle.ipynb` — Puzzle

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 04) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 04）是机器学习流水线中的基础构建块。

---
