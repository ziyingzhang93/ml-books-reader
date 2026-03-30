# PyTorch DL
## Chapter 02

---

### Test

# 01 — Test / 01 Test

**Chapter 02 — File 1 of 1 / 第02章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **create a tensor of size (2,3)**.

本脚本演示 **create a tensor of size (2,3)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import torch
```

---
## Step 2 — create a tensor of size (2,3)

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
```

---
## Step 3 — create a tensor of size (3,2)

```python
y = torch.tensor([[1, 2], [3, 4], [5, 6]])
```

---
## Step 4 — matrix multiplication of x and y

```python
z = torch.matmul(x, y)

print(z)
```

---
## Learning Notes / 学习笔记

- **概念**: create a tensor of size (2,3) 是机器学习中的常用技术。  
  *create a tensor of size (2,3) is a common technique in machine learning.*

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
# Test / 01 Test
# Complete Code / 完整代码
# ===============================

import torch

# create a tensor of size (2,3)
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# create a tensor of size (3,2)
y = torch.tensor([[1, 2], [3, 4], [5, 6]])

# matrix multiplication of x and y
z = torch.matmul(x, y)

print(z)
```

---

### Chapter Summary

# Chapter 02 Summary / 第02章总结

## Theme / 主题: Chapter 02 / Chapter 02

This chapter contains **1 code files** demonstrating chapter 02.

本章包含 **1 个代码文件**，演示Chapter 02。

---
## Evolution / 演化路线

  1. `01_test.ipynb` — Test

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 02) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 02）是机器学习流水线中的基础构建块。

---
