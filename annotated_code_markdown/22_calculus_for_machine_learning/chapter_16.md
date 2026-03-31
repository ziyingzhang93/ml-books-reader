# 机器学习微积分 / Calculus for Machine Learning
## Chapter 16

---

### Deriv

# 01 — Deriv / 01 Deriv

**Chapter 16 — File 1 of 1 / 第16章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Deriv**.

本脚本演示 **01 Deriv**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy.abc import x, y
from sympy import diff, pprint

f = x**2 + 2 * y**2
dx = diff(f, x)
dy = diff(f, y)
# 打印输出 / Print output
print("Derivative of")
# 打印输出 / Print output
pprint(f)
# 打印输出 / Print output
print("with respect to x is")
# 打印输出 / Print output
pprint(dx)
# 打印输出 / Print output
print("and with respect to y is")
# 打印输出 / Print output
pprint(dy)
```

---
## Learning Notes / 学习笔记

- **概念**: Deriv 是机器学习中的常用技术。  
  *Deriv is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Deriv / 01 Deriv
# Complete Code / 完整代码
# ===============================

from sympy.abc import x, y
from sympy import diff, pprint

f = x**2 + 2 * y**2
dx = diff(f, x)
dy = diff(f, y)
# 打印输出 / Print output
print("Derivative of")
# 打印输出 / Print output
pprint(f)
# 打印输出 / Print output
print("with respect to x is")
# 打印输出 / Print output
pprint(dx)
# 打印输出 / Print output
print("and with respect to y is")
# 打印输出 / Print output
pprint(dy)
```

---

### Chapter Summary / 章节总结

# Chapter 16 Summary / 第16章总结

## Theme / 主题: Chapter 16 / Chapter 16

This chapter contains **1 code files** demonstrating chapter 16.

本章包含 **1 个代码文件**，演示Chapter 16。

---
## Evolution / 演化路线

  1. `01_deriv.ipynb` — Deriv

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 16) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 16）是机器学习流水线中的基础构建块。

---
