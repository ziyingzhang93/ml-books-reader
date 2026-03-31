# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 03

---

### Tensor

# 01 — Tensor / 01 Tensor

**Chapter 03 — File 1 of 40 / 第03章 — 第1个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Tensor**.

本脚本演示 **01 Tensor**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
a = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.int32)
# 打印输出 / Print output
print(a)
```

---
## Learning Notes / 学习笔记

- **概念**: Tensor 是机器学习中的常用技术。  
  *Tensor is a common technique in machine learning.*

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
# Tensor / 01 Tensor
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
a = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.int32)
# 打印输出 / Print output
print(a)
```

---

➡️ **Next / 下一步**: File 2 of 40

---

### Linspace

# 02 — Linspace / 02 Linspace

**Chapter 03 — File 2 of 40 / 第03章 — 第2个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Linspace**.

本脚本演示 **02 Linspace**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
a = torch.linspace(-1, 1, 10)
# 打印输出 / Print output
print(a)
```

---
## Learning Notes / 学习笔记

- **概念**: Linspace 是机器学习中的常用技术。  
  *Linspace is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Linspace / 02 Linspace
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
a = torch.linspace(-1, 1, 10)
# 打印输出 / Print output
print(a)
```

---

➡️ **Next / 下一步**: File 3 of 40

---

### Random

# 03 — Random / 03 Random

**Chapter 03 — File 3 of 40 / 第03章 — 第3个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Random**.

本脚本演示 **03 Random**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
a = torch.rand(3,4)
# 打印输出 / Print output
print(a)
```

---
## Learning Notes / 学习笔记

- **概念**: Random 是机器学习中的常用技术。  
  *Random is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Random / 03 Random
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
a = torch.rand(3,4)
# 打印输出 / Print output
print(a)
```

---

➡️ **Next / 下一步**: File 4 of 40

---

### Randn

# 04 — Randn / 04 Randn

**Chapter 03 — File 4 of 40 / 第03章 — 第4个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Randn**.

本脚本演示 **04 Randn**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
```

---
## Learning Notes / 学习笔记

- **概念**: Randn 是机器学习中的常用技术。  
  *Randn is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Randn / 04 Randn
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
```

---

➡️ **Next / 下一步**: File 5 of 40

---

### Randint

# 05 — Randint / 05 Randint

**Chapter 03 — File 5 of 40 / 第03章 — 第5个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Randint**.

本脚本演示 **05 Randint**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
a = torch.randint(3, 10, size=(3,4))
# 打印输出 / Print output
print(a)
```

---
## Learning Notes / 学习笔记

- **概念**: Randint 是机器学习中的常用技术。  
  *Randint is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Randint / 05 Randint
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
a = torch.randint(3, 10, size=(3,4))
# 打印输出 / Print output
print(a)
```

---

➡️ **Next / 下一步**: File 6 of 40

---

### Randint

# 06 — Randint / 06 Randint

**Chapter 03 — File 6 of 40 / 第03章 — 第6个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Randint**.

本脚本演示 **06 Randint**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
a = torch.randint(10, size=(3,4))
# 打印输出 / Print output
print(a)
```

---
## Learning Notes / 学习笔记

- **概念**: Randint 是机器学习中的常用技术。  
  *Randint is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Randint / 06 Randint
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
a = torch.randint(10, size=(3,4))
# 打印输出 / Print output
print(a)
```

---

➡️ **Next / 下一步**: File 7 of 40

---

### Zeros

# 07 — Zeros / 07 Zeros

**Chapter 03 — File 7 of 40 / 第03章 — 第7个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Zeros**.

本脚本演示 **07 Zeros**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 创建全零张量 / Create tensor of zeros
a = torch.zeros(2, 3, 4)
# 打印输出 / Print output
print(a)
```

---
## Learning Notes / 学习笔记

- **概念**: Zeros 是机器学习中的常用技术。  
  *Zeros is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Zeros / 07 Zeros
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 创建全零张量 / Create tensor of zeros
a = torch.zeros(2, 3, 4)
# 打印输出 / Print output
print(a)
```

---

➡️ **Next / 下一步**: File 8 of 40

---

### Full

# 08 — Full / 08 Full

**Chapter 03 — File 8 of 40 / 第03章 — 第8个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Full**.

本脚本演示 **08 Full**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
a = torch.full((2,3,4), 5)
# 打印输出 / Print output
print(a)
```

---
## Learning Notes / 学习笔记

- **概念**: Full 是机器学习中的常用技术。  
  *Full is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Full / 08 Full
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
a = torch.full((2,3,4), 5)
# 打印输出 / Print output
print(a)
```

---

➡️ **Next / 下一步**: File 9 of 40

---

### Ones

# 09 — Ones / 09 Ones

**Chapter 03 — File 9 of 40 / 第03章 — 第9个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Ones**.

本脚本演示 **09 Ones**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 创建全一张量 / Create tensor of ones
a = torch.ones(2,3,4)
# 打印输出 / Print output
print(a)
```

---
## Learning Notes / 学习笔记

- **概念**: Ones 是机器学习中的常用技术。  
  *Ones is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ones / 09 Ones
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 创建全一张量 / Create tensor of ones
a = torch.ones(2,3,4)
# 打印输出 / Print output
print(a)
```

---

➡️ **Next / 下一步**: File 10 of 40

---

### Identity

# 10 — Identity / 10 Identity

**Chapter 03 — File 10 of 40 / 第03章 — 第10个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Identity**.

本脚本演示 **10 Identity**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
a = torch.eye(4)
# 打印输出 / Print output
print(a)
```

---
## Learning Notes / 学习笔记

- **概念**: Identity 是机器学习中的常用技术。  
  *Identity is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Identity / 10 Identity
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
a = torch.eye(4)
# 打印输出 / Print output
print(a)
```

---

➡️ **Next / 下一步**: File 11 of 40

---

### Check

# 11 — Check / 11 Check

**Chapter 03 — File 11 of 40 / 第03章 — 第11个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Check**.

本脚本演示 **11 Check**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 创建全零张量 / Create tensor of zeros
a = torch.zeros(2, 3, 4)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(a.shape)
# 打印输出 / Print output
print(a.size())
```

---
## Learning Notes / 学习笔记

- **概念**: Check 是机器学习中的常用技术。  
  *Check is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Check / 11 Check
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 创建全零张量 / Create tensor of zeros
a = torch.zeros(2, 3, 4)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(a.shape)
# 打印输出 / Print output
print(a.size())
```

---

➡️ **Next / 下一步**: File 12 of 40

---

### Dim

# 12 — Dim / 12 Dim

**Chapter 03 — File 12 of 40 / 第03章 — 第12个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Dim**.

本脚本演示 **12 Dim**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 创建全零张量 / Create tensor of zeros
a = torch.zeros(2, 3, 4)
# 打印输出 / Print output
print(a.ndim)
```

---
## Learning Notes / 学习笔记

- **概念**: Dim 是机器学习中的常用技术。  
  *Dim is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dim / 12 Dim
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 创建全零张量 / Create tensor of zeros
a = torch.zeros(2, 3, 4)
# 打印输出 / Print output
print(a.ndim)
```

---

➡️ **Next / 下一步**: File 13 of 40

---

### Firstdim

# 13 — Firstdim / 13 Firstdim

**Chapter 03 — File 13 of 40 / 第03章 — 第13个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Firstdim**.

本脚本演示 **13 Firstdim**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 创建全零张量 / Create tensor of zeros
a = torch.zeros(2, 3, 4)
# 打印输出 / Print output
print(len(a))
```

---
## Learning Notes / 学习笔记

- **概念**: Firstdim 是机器学习中的常用技术。  
  *Firstdim is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Firstdim / 13 Firstdim
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 创建全零张量 / Create tensor of zeros
a = torch.zeros(2, 3, 4)
# 打印输出 / Print output
print(len(a))
```

---

➡️ **Next / 下一步**: File 14 of 40

---

### Dtype

# 14 — Dtype / 14 Dtype

**Chapter 03 — File 14 of 40 / 第03章 — 第14个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Dtype**.

本脚本演示 **14 Dtype**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 创建全零张量 / Create tensor of zeros
a = torch.zeros(2, 3, 4)
# 打印输出 / Print output
print(a.dtype)
```

---
## Learning Notes / 学习笔记

- **概念**: Dtype 是机器学习中的常用技术。  
  *Dtype is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dtype / 14 Dtype
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 创建全零张量 / Create tensor of zeros
a = torch.zeros(2, 3, 4)
# 打印输出 / Print output
print(a.dtype)
```

---

➡️ **Next / 下一步**: File 15 of 40

---

### Dtype

# 15 — Dtype / 15 Dtype

**Chapter 03 — File 15 of 40 / 第03章 — 第15个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Dtype**.

本脚本演示 **15 Dtype**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 创建全零张量 / Create tensor of zeros
a = torch.zeros(2, 3, 4)
b = a.type(torch.int32)
# 打印输出 / Print output
print(a.dtype)
# 打印输出 / Print output
print(b.dtype)
```

---
## Learning Notes / 学习笔记

- **概念**: Dtype 是机器学习中的常用技术。  
  *Dtype is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dtype / 15 Dtype
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 创建全零张量 / Create tensor of zeros
a = torch.zeros(2, 3, 4)
b = a.type(torch.int32)
# 打印输出 / Print output
print(a.dtype)
# 打印输出 / Print output
print(b.dtype)
```

---

➡️ **Next / 下一步**: File 16 of 40

---

### Randn

# 16 — Randn / 16 Randn

**Chapter 03 — File 16 of 40 / 第03章 — 第16个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Randn**.

本脚本演示 **16 Randn**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4,5)
# 打印输出 / Print output
print(a)
```

---
## Learning Notes / 学习笔记

- **概念**: Randn 是机器学习中的常用技术。  
  *Randn is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Randn / 16 Randn
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4,5)
# 打印输出 / Print output
print(a)
```

---

➡️ **Next / 下一步**: File 17 of 40

---

### Slicing

# 17 — Slicing / 17 Slicing

**Chapter 03 — File 17 of 40 / 第03章 — 第17个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Slicing**.

本脚本演示 **17 Slicing**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4,5)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(a[1])
```

---
## Learning Notes / 学习笔记

- **概念**: Slicing 是机器学习中的常用技术。  
  *Slicing is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Slicing / 17 Slicing
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4,5)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(a[1])
```

---

➡️ **Next / 下一步**: File 18 of 40

---

### Slicing

# 18 — Slicing / 18 Slicing

**Chapter 03 — File 18 of 40 / 第03章 — 第18个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Slicing**.

本脚本演示 **18 Slicing**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4,5)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(a[1:, 2:4])
```

---
## Learning Notes / 学习笔记

- **概念**: Slicing 是机器学习中的常用技术。  
  *Slicing is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Slicing / 18 Slicing
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4,5)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(a[1:, 2:4])
```

---

➡️ **Next / 下一步**: File 19 of 40

---

### Newdim

# 19 — Newdim / 19 Newdim

**Chapter 03 — File 19 of 40 / 第03章 — 第19个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Newdim**.

本脚本演示 **19 Newdim**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4,5)
# 打印输出 / Print output
print(a)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(a[:, None, :, None].shape)
```

---
## Learning Notes / 学习笔记

- **概念**: Newdim 是机器学习中的常用技术。  
  *Newdim is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Newdim / 19 Newdim
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4,5)
# 打印输出 / Print output
print(a)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(a[:, None, :, None].shape)
```

---

➡️ **Next / 下一步**: File 20 of 40

---

### Unsqueeze

# 20 — Unsqueeze / 20 Unsqueeze

**Chapter 03 — File 20 of 40 / 第03章 — 第20个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Unsqueeze**.

本脚本演示 **20 Unsqueeze**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4,5)
b = torch.unsqueeze(a, dim=2)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(a.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(b.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: Unsqueeze 是机器学习中的常用技术。  
  *Unsqueeze is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Unsqueeze / 20 Unsqueeze
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4,5)
b = torch.unsqueeze(a, dim=2)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(a.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(b.shape)
```

---

➡️ **Next / 下一步**: File 21 of 40

---

### Boolean

# 21 — Boolean / 21 Boolean

**Chapter 03 — File 21 of 40 / 第03章 — 第21个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Boolean**.

本脚本演示 **21 Boolean**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(a[:, (a > -1).all(axis=0)])
```

---
## Learning Notes / 学习笔记

- **概念**: Boolean 是机器学习中的常用技术。  
  *Boolean is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Boolean / 21 Boolean
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(a[:, (a > -1).all(axis=0)])
```

---

➡️ **Next / 下一步**: File 22 of 40

---

### Indexing

# 22 — Indexing / 22 Indexing

**Chapter 03 — File 22 of 40 / 第03章 — 第22个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Indexing**.

本脚本演示 **22 Indexing**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(a[:, [1,0,0,1]])
```

---
## Learning Notes / 学习笔记

- **概念**: Indexing 是机器学习中的常用技术。  
  *Indexing is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Indexing / 22 Indexing
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(a[:, [1,0,0,1]])
```

---

➡️ **Next / 下一步**: File 23 of 40

---

### 1D

# 23 — 1D / 23 1D

**Chapter 03 — File 23 of 40 / 第03章 — 第23个文件（共40个）**

---

## Summary / 总结

This script demonstrates **1D**.

本脚本演示 **23 1D**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(a.ravel())
```

---
## Learning Notes / 学习笔记

- **概念**: 1D 是机器学习中的常用技术。  
  *1D is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# 1D / 23 1D
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(a.ravel())
```

---

➡️ **Next / 下一步**: File 24 of 40

---

### Reshape

# 24 — Reshape / 24 Reshape

**Chapter 03 — File 24 of 40 / 第03章 — 第24个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Reshape**.

本脚本演示 **24 Reshape**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
print(a.reshape(-1))
```

---
## Learning Notes / 学习笔记

- **概念**: Reshape 是机器学习中的常用技术。  
  *Reshape is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Reshape / 24 Reshape
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
print(a.reshape(-1))
```

---

➡️ **Next / 下一步**: File 25 of 40

---

### Reshape

# 25 — Reshape / 25 Reshape

**Chapter 03 — File 25 of 40 / 第03章 — 第25个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Reshape**.

本脚本演示 **25 Reshape**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
print(a.reshape(3,2,2))
```

---
## Learning Notes / 学习笔记

- **概念**: Reshape 是机器学习中的常用技术。  
  *Reshape is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Reshape / 25 Reshape
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
print(a.reshape(3,2,2))
```

---

➡️ **Next / 下一步**: File 26 of 40

---

### Transpose

# 26 — Transpose / 26 Transpose

**Chapter 03 — File 26 of 40 / 第03章 — 第26个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Transpose**.

本脚本演示 **26 Transpose**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(a.T)
```

---
## Learning Notes / 学习笔记

- **概念**: Transpose 是机器学习中的常用技术。  
  *Transpose is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Transpose / 26 Transpose
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(a.T)
```

---

➡️ **Next / 下一步**: File 27 of 40

---

### Transpose

# 27 — Transpose / 27 Transpose

**Chapter 03 — File 27 of 40 / 第03章 — 第27个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Transpose**.

本脚本演示 **27 Transpose**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(a.transpose(0, 1))
```

---
## Learning Notes / 学习笔记

- **概念**: Transpose 是机器学习中的常用技术。  
  *Transpose is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Transpose / 27 Transpose
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(a.transpose(0, 1))
```

---

➡️ **Next / 下一步**: File 28 of 40

---

### Vstack

# 28 — Vstack / 堆叠方法

**Chapter 03 — File 28 of 40 / 第03章 — 第28个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Vstack**.

本脚本演示 **堆叠方法**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,3)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
b = torch.randn(3,3)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(b)
# 打印输出 / Print output
print(torch.vstack([a,b]))
```

---
## Learning Notes / 学习笔记

- **概念**: Vstack 是机器学习中的常用技术。  
  *Vstack is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Vstack / 堆叠方法
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,3)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
b = torch.randn(3,3)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(b)
# 打印输出 / Print output
print(torch.vstack([a,b]))
```

---

➡️ **Next / 下一步**: File 29 of 40

---

### Concat

# 29 — Concat / 29 Concat

**Chapter 03 — File 29 of 40 / 第03章 — 第29个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Concat**.

本脚本演示 **29 Concat**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,3)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
b = torch.randn(3,3)
c = torch.concatenate([a, b])
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(b)
# 打印输出 / Print output
print(c)
```

---
## Learning Notes / 学习笔记

- **概念**: Concat 是机器学习中的常用技术。  
  *Concat is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Concat / 29 Concat
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,3)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
b = torch.randn(3,3)
c = torch.concatenate([a, b])
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(b)
# 打印输出 / Print output
print(c)
```

---

➡️ **Next / 下一步**: File 30 of 40

---

### Vsplit

# 30 — Vsplit / 30 Vsplit

**Chapter 03 — File 30 of 40 / 第03章 — 第30个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Vsplit**.

本脚本演示 **30 Vsplit**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,3)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
b = torch.randn(3,3)
c = torch.concatenate([a, b])
# 打印输出 / Print output
print(c)
# 打印输出 / Print output
print(torch.vsplit(c, 2))
```

---
## Learning Notes / 学习笔记

- **概念**: Vsplit 是机器学习中的常用技术。  
  *Vsplit is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Vsplit / 30 Vsplit
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,3)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
b = torch.randn(3,3)
c = torch.concatenate([a, b])
# 打印输出 / Print output
print(c)
# 打印输出 / Print output
print(torch.vsplit(c, 2))
```

---

➡️ **Next / 下一步**: File 31 of 40

---

### Split

# 31 — Split / 31 Split

**Chapter 03 — File 31 of 40 / 第03章 — 第31个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Split**.

本脚本演示 **31 Split**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,3)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
b = torch.randn(3,3)
c = torch.concatenate([a, b])
# 打印输出 / Print output
print(c)
# 打印输出 / Print output
print(torch.split(c, 3, dim=0))
```

---
## Learning Notes / 学习笔记

- **概念**: Split 是机器学习中的常用技术。  
  *Split is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Split / 31 Split
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,3)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
b = torch.randn(3,3)
c = torch.concatenate([a, b])
# 打印输出 / Print output
print(c)
# 打印输出 / Print output
print(torch.split(c, 3, dim=0))
```

---

➡️ **Next / 下一步**: File 32 of 40

---

### Functions

# 32 — Functions / 32 Functions

**Chapter 03 — File 32 of 40 / 第03章 — 第32个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Functions**.

本脚本演示 **32 Functions**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(2,3)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print("exp =", torch.exp(a))
# 打印输出 / Print output
print("log =", torch.log(a))
# 打印输出 / Print output
print("sin =", torch.sin(a))
# 打印输出 / Print output
print("arctan =", torch.arctan(a))
# 打印输出 / Print output
print("abs =", torch.abs(a))
# 打印输出 / Print output
print("square =", torch.square(a))
# 打印输出 / Print output
print("sqrt =", torch.sqrt(a))
# 打印输出 / Print output
print("ceil =", torch.ceil(a))
# 打印输出 / Print output
print("round =", torch.round(a))
# 打印输出 / Print output
print("clip =", torch.clip(a, 0.1, 0.9))
```

---
## Learning Notes / 学习笔记

- **概念**: Functions 是机器学习中的常用技术。  
  *Functions is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Functions / 32 Functions
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(2,3)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print("exp =", torch.exp(a))
# 打印输出 / Print output
print("log =", torch.log(a))
# 打印输出 / Print output
print("sin =", torch.sin(a))
# 打印输出 / Print output
print("arctan =", torch.arctan(a))
# 打印输出 / Print output
print("abs =", torch.abs(a))
# 打印输出 / Print output
print("square =", torch.square(a))
# 打印输出 / Print output
print("sqrt =", torch.sqrt(a))
# 打印输出 / Print output
print("ceil =", torch.ceil(a))
# 打印输出 / Print output
print("round =", torch.round(a))
# 打印输出 / Print output
print("clip =", torch.clip(a, 0.1, 0.9))
```

---

➡️ **Next / 下一步**: File 33 of 40

---

### Nan

# 33 — Nan / 33 Nan

**Chapter 03 — File 33 of 40 / 第03章 — 第33个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Nan**.

本脚本演示 **33 Nan**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
b = torch.sqrt(a)
# 打印输出 / Print output
print(b)
# 打印输出 / Print output
print(torch.isnan(b))
```

---
## Learning Notes / 学习笔记

- **概念**: Nan 是机器学习中的常用技术。  
  *Nan is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nan / 33 Nan
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
b = torch.sqrt(a)
# 打印输出 / Print output
print(b)
# 打印输出 / Print output
print(torch.isnan(b))
```

---

➡️ **Next / 下一步**: File 34 of 40

---

### Ops

# 34 — Ops / 34 Ops

**Chapter 03 — File 34 of 40 / 第03章 — 第34个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Ops**.

本脚本演示 **34 Ops**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(2, 3)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
b = torch.randn(2, 3)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(b)
# 打印输出 / Print output
print(a+b)
# 打印输出 / Print output
print(a/b)
# 打印输出 / Print output
print(a ** 2)
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
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ops / 34 Ops
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(2, 3)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
b = torch.randn(2, 3)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(b)
# 打印输出 / Print output
print(a+b)
# 打印输出 / Print output
print(a/b)
# 打印输出 / Print output
print(a ** 2)
```

---

➡️ **Next / 下一步**: File 35 of 40

---

### Matmul

# 35 — Matmul / 35 Matmul

**Chapter 03 — File 35 of 40 / 第03章 — 第35个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Matmul**.

本脚本演示 **35 Matmul**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(2, 3)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
b = torch.randn(2, 3)
# 打印输出 / Print output
print(torch.matmul(a, b.T))
# 打印输出 / Print output
print(a @ b.T)
```

---
## Learning Notes / 学习笔记

- **概念**: Matmul 是机器学习中的常用技术。  
  *Matmul is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Matmul / 35 Matmul
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(2, 3)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
b = torch.randn(2, 3)
# 打印输出 / Print output
print(torch.matmul(a, b.T))
# 打印输出 / Print output
print(a @ b.T)
```

---

➡️ **Next / 下一步**: File 36 of 40

---

### Dot

# 36 — Dot / 36 Dot

**Chapter 03 — File 36 of 40 / 第03章 — 第36个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Dot**.

本脚本演示 **36 Dot**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
b = torch.randn(3)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(b)
# 打印输出 / Print output
print(torch.dot(a, b))
# 打印输出 / Print output
print(a @ b)
```

---
## Learning Notes / 学习笔记

- **概念**: Dot 是机器学习中的常用技术。  
  *Dot is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dot / 36 Dot
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
b = torch.randn(3)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(b)
# 打印输出 / Print output
print(torch.dot(a, b))
# 打印输出 / Print output
print(a @ b)
```

---

➡️ **Next / 下一步**: File 37 of 40

---

### Statistics

# 37 — Statistics / 37 Statistics

**Chapter 03 — File 37 of 40 / 第03章 — 第37个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Statistics**.

本脚本演示 **37 Statistics**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(torch.mean(a, dim=0))
# 打印输出 / Print output
print(torch.std(a, dim=0))
# 打印输出 / Print output
print(torch.cumsum(a, dim=0))
# 打印输出 / Print output
print(torch.cumprod(a, dim=0))
```

---
## Learning Notes / 学习笔记

- **概念**: Statistics 是机器学习中的常用技术。  
  *Statistics is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Statistics / 37 Statistics
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(torch.mean(a, dim=0))
# 打印输出 / Print output
print(torch.std(a, dim=0))
# 打印输出 / Print output
print(torch.cumsum(a, dim=0))
# 打印输出 / Print output
print(torch.cumprod(a, dim=0))
```

---

➡️ **Next / 下一步**: File 38 of 40

---

### Svd

# 38 — Svd / 38 Svd

**Chapter 03 — File 38 of 40 / 第03章 — 第38个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Svd**.

本脚本演示 **38 Svd**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(torch.linalg.svd(a))
```

---
## Learning Notes / 学习笔记

- **概念**: Svd 是机器学习中的常用技术。  
  *Svd is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Svd / 38 Svd
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(torch.linalg.svd(a))
```

---

➡️ **Next / 下一步**: File 39 of 40

---

### Pad

# 39 — Pad / 39 Pad

**Chapter 03 — File 39 of 40 / 第03章 — 第39个文件（共40个）**

---

## Summary / 总结

This script demonstrates **Pad**.

本脚本演示 **39 Pad**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
b = torch.nn.functional.pad(a, (1,1,0,2), value=0)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(b)
```

---
## Learning Notes / 学习笔记

- **概念**: Pad 是机器学习中的常用技术。  
  *Pad is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pad / 39 Pad
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 生成正态分布随机张量 / Generate random tensor from normal distribution
a = torch.randn(3,4)
b = torch.nn.functional.pad(a, (1,1,0,2), value=0)
# 打印输出 / Print output
print(a)
# 打印输出 / Print output
print(b)
```

---

➡️ **Next / 下一步**: File 40 of 40

---

### Plot

# 40 — Plot / 40 Plot

**Chapter 03 — File 40 of 40 / 第03章 — 第40个文件（共40个）**

---

## Summary / 总结

This script demonstrates **create tensors**.

本脚本演示 **create tensors**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
```

---
## Step 2 — create tensors

```python
x = torch.linspace(-1, 1, 100)
y = torch.linspace(-2, 2, 100)
```

---
## Step 3 — create the surface

```python
xx, yy = torch.meshgrid(x, y, indexing="xy")  # xy-indexing is matching numpy
z = torch.sqrt(1 - xx**2 - (yy/2)**2)
# 打印输出 / Print output
print(xx)

# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection="3d")
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 2])
ax.plot_surface(xx, yy, z, cmap="cividis")
ax.view_init(45, 35)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create tensors 是机器学习中的常用技术。  
  *create tensors is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot / 40 Plot
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

# create tensors
x = torch.linspace(-1, 1, 100)
y = torch.linspace(-2, 2, 100)
# create the surface
xx, yy = torch.meshgrid(x, y, indexing="xy")  # xy-indexing is matching numpy
z = torch.sqrt(1 - xx**2 - (yy/2)**2)
# 打印输出 / Print output
print(xx)

# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection="3d")
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 2])
ax.plot_surface(xx, yy, z, cmap="cividis")
ax.view_init(45, 35)
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 03 Summary / 第03章总结

## Theme / 主题: Chapter 03 / Chapter 03

This chapter contains **40 code files** demonstrating chapter 03.

本章包含 **40 个代码文件**，演示Chapter 03。

---
## Evolution / 演化路线

  1. `01_tensor.ipynb` — Tensor
  2. `02_linspace.ipynb` — Linspace
  3. `03_random.ipynb` — Random
  4. `04_randn.ipynb` — Randn
  5. `05_randint.ipynb` — Randint
  6. `06_randint.ipynb` — Randint
  7. `07_zeros.ipynb` — Zeros
  8. `08_full.ipynb` — Full
  9. `09_ones.ipynb` — Ones
  10. `10_identity.ipynb` — Identity
  11. `11_check.ipynb` — Check
  12. `12_dim.ipynb` — Dim
  13. `13_firstdim.ipynb` — Firstdim
  14. `14_dtype.ipynb` — Dtype
  15. `15_dtype.ipynb` — Dtype
  16. `16_randn.ipynb` — Randn
  17. `17_slicing.ipynb` — Slicing
  18. `18_slicing.ipynb` — Slicing
  19. `19_newdim.ipynb` — Newdim
  20. `20_unsqueeze.ipynb` — Unsqueeze
  21. `21_boolean.ipynb` — Boolean
  22. `22_indexing.ipynb` — Indexing
  23. `23_1D.ipynb` — 1D
  24. `24_reshape.ipynb` — Reshape
  25. `25_reshape.ipynb` — Reshape
  26. `26_transpose.ipynb` — Transpose
  27. `27_transpose.ipynb` — Transpose
  28. `28_vstack.ipynb` — Vstack
  29. `29_concat.ipynb` — Concat
  30. `30_vsplit.ipynb` — Vsplit
  31. `31_split.ipynb` — Split
  32. `32_functions.ipynb` — Functions
  33. `33_nan.ipynb` — Nan
  34. `34_ops.ipynb` — Ops
  35. `35_matmul.ipynb` — Matmul
  36. `36_dot.ipynb` — Dot
  37. `37_statistics.ipynb` — Statistics
  38. `38_svd.ipynb` — Svd
  39. `39_pad.ipynb` — Pad
  40. `40_plot.ipynb` — Plot

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 03) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 03）是机器学习流水线中的基础构建块。

---
