# ML微积分
## Chapter 15

---

### Integration

# 01 — Integration / 01 Integration

**Chapter 15 — File 1 of 1 / 第15章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Integration**.

本脚本演示 **01 Integration**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy import integrate, pprint
from sympy.abc import x
import numpy as np

f = 3 * x**2
result = integrate(f, x)
print("Antiderivative of")
pprint(f)
print("is")
pprint(result)
print()

result = integrate(f, (x, 2, 3))
print("Integration of")
pprint(f)
print("for x=2 to x=3 is")
pprint(result)
print()

dx = 0.001
x = np.arange(2, 3, dx)
y = 3 * x**2
result = (y * dx).sum()
print("Numerically using left sum:", result)
x = np.arange(2, 3, dx) + dx
y = 3 * x**2
result = (y * dx).sum()
print("Numerically using right sum:", result)
x = np.arange(2, 3, dx) + dx/2
y = 3 * x**2
result = (y * dx).sum()
print("Numerically using midpoint sum:", result)
```

---
## Learning Notes / 学习笔记

- **概念**: Integration 是机器学习中的常用技术。  
  *Integration is a common technique in machine learning.*

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
# Integration / 01 Integration
# Complete Code / 完整代码
# ===============================

from sympy import integrate, pprint
from sympy.abc import x
import numpy as np

f = 3 * x**2
result = integrate(f, x)
print("Antiderivative of")
pprint(f)
print("is")
pprint(result)
print()

result = integrate(f, (x, 2, 3))
print("Integration of")
pprint(f)
print("for x=2 to x=3 is")
pprint(result)
print()

dx = 0.001
x = np.arange(2, 3, dx)
y = 3 * x**2
result = (y * dx).sum()
print("Numerically using left sum:", result)
x = np.arange(2, 3, dx) + dx
y = 3 * x**2
result = (y * dx).sum()
print("Numerically using right sum:", result)
x = np.arange(2, 3, dx) + dx/2
y = 3 * x**2
result = (y * dx).sum()
print("Numerically using midpoint sum:", result)
```

---

### Chapter Summary

# Chapter 15 Summary / 第15章总结

## Theme / 主题: Chapter 15 / Chapter 15

This chapter contains **1 code files** demonstrating chapter 15.

本章包含 **1 个代码文件**，演示Chapter 15。

---
## Evolution / 演化路线

  1. `01_integration.ipynb` — Integration

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 15) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 15）是机器学习流水线中的基础构建块。

---
