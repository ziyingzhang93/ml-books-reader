# 机器学习微积分 / Calculus for Machine Learning
## Chapter 10

---

### Derivatives

# 06 — Derivatives / 06 Derivatives

**Chapter 10 — File 1 of 1 / 第10章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **finding the derivative of the sine and cosine functions**.

本脚本演示 **finding the derivative of the sine and cosine functions**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — finding the derivative of the sine and cosine functions

```python
from sympy import diff
from sympy import sin
from sympy import cos
from sympy import symbols
```

---
## Step 2 — define variable as symbol

```python
x = symbols('x')
```

---
## Step 3 — find the first derivative of sine and cosine with respect to x

```python
# 打印输出 / Print output
print('The first derivative of sine is:', diff(sin(x), x))
# 打印输出 / Print output
print('The first derivative of cosine is:', diff(cos(x), x))
```

---
## Step 4 — find the second derivative of sine and cosine with respect to x

```python
# 打印输出 / Print output
print('\nThe second derivative of sine is:', diff(sin(x), x, x))
# 打印输出 / Print output
print('The second derivative of cosine is:', diff(cos(x), x, x))
```

---
## Step 5 — find the second derivative of sine and cosine with respect to x

```python
# 打印输出 / Print output
print('\nThe second derivative of sine is:', diff(sin(x), x, 2))
# 打印输出 / Print output
print('The second derivative of cosine is:', diff(cos(x), x, 2))
```

---
## Learning Notes / 学习笔记

- **概念**: finding the derivative of the sine and cosine functions 是机器学习中的常用技术。  
  *finding the derivative of the sine and cosine functions is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Derivatives / 06 Derivatives
# Complete Code / 完整代码
# ===============================

# finding the derivative of the sine and cosine functions
from sympy import diff
from sympy import sin
from sympy import cos
from sympy import symbols

# define variable as symbol
x = symbols('x')

# find the first derivative of sine and cosine with respect to x
# 打印输出 / Print output
print('The first derivative of sine is:', diff(sin(x), x))
# 打印输出 / Print output
print('The first derivative of cosine is:', diff(cos(x), x))

# find the second derivative of sine and cosine with respect to x
# 打印输出 / Print output
print('\nThe second derivative of sine is:', diff(sin(x), x, x))
# 打印输出 / Print output
print('The second derivative of cosine is:', diff(cos(x), x, x))

# find the second derivative of sine and cosine with respect to x
# 打印输出 / Print output
print('\nThe second derivative of sine is:', diff(sin(x), x, 2))
# 打印输出 / Print output
print('The second derivative of cosine is:', diff(cos(x), x, 2))
```

---

### Chapter Summary / 章节总结

# Chapter 10 Summary / 第10章总结

## Theme / 主题: Chapter 10 / Chapter 10

This chapter contains **1 code files** demonstrating chapter 10.

本章包含 **1 个代码文件**，演示Chapter 10。

---
## Evolution / 演化路线

  1. `06_derivatives.ipynb` — Derivatives

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 10) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 10）是机器学习流水线中的基础构建块。

---
