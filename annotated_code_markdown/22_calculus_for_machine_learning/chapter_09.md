# ML微积分
## Chapter 09

---

### Verify

# 01 — Verify / 01 Verify

**Chapter 09 — File 1 of 2 / 第09章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Verify**.

本脚本演示 **01 Verify**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy import diff, pprint
from sympy.abc import x

expressions = [x**2, 3*x**5, 4*x**9]
for expression in expressions:
    result = diff(expression, x)
    print("Derivative of")
    pprint(expression)
    print("with respect to x is")
    pprint(result)
    print()
```

---
## Learning Notes / 学习笔记

- **概念**: Verify 是机器学习中的常用技术。  
  *Verify is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Verify / 01 Verify
# Complete Code / 完整代码
# ===============================

from sympy import diff, pprint
from sympy.abc import x

expressions = [x**2, 3*x**5, 4*x**9]
for expression in expressions:
    result = diff(expression, x)
    print("Derivative of")
    pprint(expression)
    print("with respect to x is")
    pprint(result)
    print()
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Real Power

# 02 — Real Power / 02 Real Power

**Chapter 09 — File 2 of 2 / 第09章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Real Power**.

本脚本演示 **02 Real Power**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy import diff, pprint, powsimp, simplify
from sympy.abc import x

expressions = ["k*x**a", "x**0.2", "x**pi", "x**(-3/4)"]
for expression in expressions:
    expression = simplify(expression)
    result = diff(expression, x)
    print("Derivative of")
    pprint(expression)
    print("with respect to x is")
    pprint(powsimp(result))
    print()
```

---
## Learning Notes / 学习笔记

- **概念**: Real Power 是机器学习中的常用技术。  
  *Real Power is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Real Power / 02 Real Power
# Complete Code / 完整代码
# ===============================

from sympy import diff, pprint, powsimp, simplify
from sympy.abc import x

expressions = ["k*x**a", "x**0.2", "x**pi", "x**(-3/4)"]
for expression in expressions:
    expression = simplify(expression)
    result = diff(expression, x)
    print("Derivative of")
    pprint(expression)
    print("with respect to x is")
    pprint(powsimp(result))
    print()
```

---

### Chapter Summary

# Chapter 09 Summary / 第09章总结

## Theme / 主题: Chapter 09 / Chapter 09

This chapter contains **2 code files** demonstrating chapter 09.

本章包含 **2 个代码文件**，演示Chapter 09。

---
## Evolution / 演化路线

  1. `01_verify.ipynb` — Verify
  2. `02_real_power.ipynb` — Real Power

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 09) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 09）是机器学习流水线中的基础构建块。

---
