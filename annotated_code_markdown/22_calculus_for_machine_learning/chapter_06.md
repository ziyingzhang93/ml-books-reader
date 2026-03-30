# ML微积分
## Chapter 06

---

### Example4

# 01 — Example4 / 01 Example4

**Chapter 06 — File 1 of 2 / 第06章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Example4**.

本脚本演示 **01 Example4**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy import limit, sqrt, pprint
from sympy.abc import x

expression = sqrt(x+1)
result = limit(expression, x, -1)
print("Limit of")
pprint(expression)
print("at x=-1 is", result)
```

---
## Learning Notes / 学习笔记

- **概念**: Example4 是机器学习中的常用技术。  
  *Example4 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example4 / 01 Example4
# Complete Code / 完整代码
# ===============================

from sympy import limit, sqrt, pprint
from sympy.abc import x

expression = sqrt(x+1)
result = limit(expression, x, -1)
print("Limit of")
pprint(expression)
print("at x=-1 is", result)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Sandwich

# 02 — Sandwich / 02 Sandwich

**Chapter 06 — File 2 of 2 / 第06章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Sandwich**.

本脚本演示 **02 Sandwich**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy import limit, sin, pprint
from sympy.abc import x

expression = x**2 * sin(1/x)
result = limit(expression, x, 0)
print("Limit of")
pprint(expression)
print("at x=0 is", result)
```

---
## Learning Notes / 学习笔记

- **概念**: Sandwich 是机器学习中的常用技术。  
  *Sandwich is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sandwich / 02 Sandwich
# Complete Code / 完整代码
# ===============================

from sympy import limit, sin, pprint
from sympy.abc import x

expression = x**2 * sin(1/x)
result = limit(expression, x, 0)
print("Limit of")
pprint(expression)
print("at x=0 is", result)
```

---

### Chapter Summary

# Chapter 06 Summary / 第06章总结

## Theme / 主题: Chapter 06 / Chapter 06

This chapter contains **2 code files** demonstrating chapter 06.

本章包含 **2 个代码文件**，演示Chapter 06。

---
## Evolution / 演化路线

  1. `01_example4.ipynb` — Example4
  2. `02_sandwich.ipynb` — Sandwich

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 06) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 06）是机器学习流水线中的基础构建块。

---
