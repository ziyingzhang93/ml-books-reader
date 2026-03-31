# 机器学习微积分 / Calculus for Machine Learning
## Chapter 13

---

### Two Derivs

# 01 — Two Derivs / 01 Two Derivs

**Chapter 13 — File 1 of 1 / 第13章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Two Derivs**.

本脚本演示 **01 Two Derivs**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy import diff, sin, pprint
from sympy.abc import x

f = -x * sin(x)
d1 = diff(f, x)
d2 = diff(f, x, x)
# 打印输出 / Print output
print("Function")
# 打印输出 / Print output
pprint(f)
# 打印输出 / Print output
print("has first derivative")
# 打印输出 / Print output
pprint(d1)
# 打印输出 / Print output
print("and second derivative")
# 打印输出 / Print output
pprint(d2)
```

---
## Learning Notes / 学习笔记

- **概念**: Two Derivs 是机器学习中的常用技术。  
  *Two Derivs is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Two Derivs / 01 Two Derivs
# Complete Code / 完整代码
# ===============================

from sympy import diff, sin, pprint
from sympy.abc import x

f = -x * sin(x)
d1 = diff(f, x)
d2 = diff(f, x, x)
# 打印输出 / Print output
print("Function")
# 打印输出 / Print output
pprint(f)
# 打印输出 / Print output
print("has first derivative")
# 打印输出 / Print output
pprint(d1)
# 打印输出 / Print output
print("and second derivative")
# 打印输出 / Print output
pprint(d2)
```

---

### Chapter Summary / 章节总结

# Chapter 13 Summary / 第13章总结

## Theme / 主题: Chapter 13 / Chapter 13

This chapter contains **1 code files** demonstrating chapter 13.

本章包含 **1 个代码文件**，演示Chapter 13。

---
## Evolution / 演化路线

  1. `01_two_derivs.ipynb` — Two Derivs

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 13) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 13）是机器学习流水线中的基础构建块。

---
