# 机器学习微积分 / Calculus for Machine Learning
## Chapter 22

---

### Hessian

# 01 — Hessian / 01 Hessian

**Chapter 22 — File 1 of 1 / 第22章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Hessian**.

本脚本演示 **01 Hessian**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy.abc import x, y
from sympy import pprint, hessian

g = x**3 + 2*y**2 + 3*x*y**2
variables = [x, y]
h = hessian(g, variables)
d = h.det()
# 打印输出 / Print output
print("Function")
# 打印输出 / Print output
pprint(g)
# 打印输出 / Print output
print("Hessian")
# 打印输出 / Print output
pprint(h)
# 打印输出 / Print output
print("Discriminant")
# 打印输出 / Print output
pprint(d)
for xval,yval in [(0,0), (1,0), (0,1), (-1,0)]:
    val = d.subs([(x,xval),(y,yval)])
    # 打印输出 / Print output
    print(f"Discriminant at ({xval},{yval}) = {val}")
```

---
## Learning Notes / 学习笔记

- **概念**: Hessian 是机器学习中的常用技术。  
  *Hessian is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hessian / 01 Hessian
# Complete Code / 完整代码
# ===============================

from sympy.abc import x, y
from sympy import pprint, hessian

g = x**3 + 2*y**2 + 3*x*y**2
variables = [x, y]
h = hessian(g, variables)
d = h.det()
# 打印输出 / Print output
print("Function")
# 打印输出 / Print output
pprint(g)
# 打印输出 / Print output
print("Hessian")
# 打印输出 / Print output
pprint(h)
# 打印输出 / Print output
print("Discriminant")
# 打印输出 / Print output
pprint(d)
for xval,yval in [(0,0), (1,0), (0,1), (-1,0)]:
    val = d.subs([(x,xval),(y,yval)])
    # 打印输出 / Print output
    print(f"Discriminant at ({xval},{yval}) = {val}")
```

---

### Chapter Summary / 章节总结

# Chapter 22 Summary / 第22章总结

## Theme / 主题: Chapter 22 / Chapter 22

This chapter contains **1 code files** demonstrating chapter 22.

本章包含 **1 个代码文件**，演示Chapter 22。

---
## Evolution / 演化路线

  1. `01_hessian.ipynb` — Hessian

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 22) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 22）是机器学习流水线中的基础构建块。

---
