# 机器学习微积分 / Calculus for Machine Learning
## Chapter 21

---

### Jacobian

# 01 — Jacobian / 01 Jacobian

**Chapter 21 — File 1 of 1 / 第21章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Vector-valued function**.

本脚本演示 **Vector-valued function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy.abc import x, y, p, q, r, s, t, u
from sympy import exp, Matrix, simplify, pprint

def sigmoid(x):
    return 1/(1+exp(-x))
```

---
## Step 2 — Vector-valued function

```python
f = Matrix([sigmoid(p*x+q*y), sigmoid(r*x+s*y), sigmoid(t*x+u*y)])
variables = Matrix([x,y])
```

---
## Step 3 — Find and print the Jacobian

```python
# 打印输出 / Print output
pprint(f.jacobian(variables))
```

---
## Learning Notes / 学习笔记

- **概念**: Vector-valued function 是机器学习中的常用技术。  
  *Vector-valued function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Jacobian / 01 Jacobian
# Complete Code / 完整代码
# ===============================

from sympy.abc import x, y, p, q, r, s, t, u
from sympy import exp, Matrix, simplify, pprint

def sigmoid(x):
    return 1/(1+exp(-x))

# Vector-valued function
f = Matrix([sigmoid(p*x+q*y), sigmoid(r*x+s*y), sigmoid(t*x+u*y)])
variables = Matrix([x,y])
# Find and print the Jacobian
# 打印输出 / Print output
pprint(f.jacobian(variables))
```

---

### Chapter Summary / 章节总结

# Chapter 21 Summary / 第21章总结

## Theme / 主题: Chapter 21 / Chapter 21

This chapter contains **1 code files** demonstrating chapter 21.

本章包含 **1 个代码文件**，演示Chapter 21。

---
## Evolution / 演化路线

  1. `01_jacobian.ipynb` — Jacobian

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 21) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 21）是机器学习流水线中的基础构建块。

---
