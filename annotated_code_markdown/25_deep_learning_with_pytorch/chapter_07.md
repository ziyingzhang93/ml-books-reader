# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 07

---

### Convert

# 03 — Convert / 03 Convert

**Chapter 07 — File 1 of 3 / 第07章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **load the dataset, split into input (X) and output (y) variables**.

本脚本演示 **load the dataset, split into input (X) and output (y) variables**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
```

---
## Step 2 — load the dataset, split into input (X) and output (y) variables

```python
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Learning Notes / 学习笔记

- **概念**: load the dataset, split into input (X) and output (y) variables 是机器学习中的常用技术。  
  *load the dataset, split into input (X) and output (y) variables is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Convert / 03 Convert
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Mlp



---

### Predict



---

### Chapter Summary / 章节总结

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **3 code files** demonstrating chapter 07.

本章包含 **3 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `03_convert.ipynb` — Convert
  2. `10_mlp.ipynb` — Mlp
  3. `13_predict.ipynb` — Predict

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---
