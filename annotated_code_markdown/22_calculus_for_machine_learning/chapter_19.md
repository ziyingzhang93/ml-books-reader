# ML微积分
## Chapter 19

---

### High Order

# 01 — High Order / 01 High Order

**Chapter 19 — File 1 of 2 / 第19章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **High Order**.

本脚本演示 **01 High Order**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy.abc import x
from sympy import diff, pprint

f = x**3 + 2*x**2 - 4*x + 1
df1 = diff(f, x)
df2 = diff(f, x, x)
df3 = diff(f, x, x, x)
df4 = diff(f, x, x, x, x)
df5 = diff(f, x, x, x, x, x)
print("Function")
pprint(f)
print("First derivative")
pprint(df1)
print("Second derivative")
pprint(df2)
print("Third derivative")
pprint(df3)
print("Fourth derivative")
pprint(df4)
print("Fifth derivative")
pprint(df5)
```

---
## Learning Notes / 学习笔记

- **概念**: High Order 是机器学习中的常用技术。  
  *High Order is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# High Order / 01 High Order
# Complete Code / 完整代码
# ===============================

from sympy.abc import x
from sympy import diff, pprint

f = x**3 + 2*x**2 - 4*x + 1
df1 = diff(f, x)
df2 = diff(f, x, x)
df3 = diff(f, x, x, x)
df4 = diff(f, x, x, x, x)
df5 = diff(f, x, x, x, x, x)
print("Function")
pprint(f)
print("First derivative")
pprint(df1)
print("Second derivative")
pprint(df2)
print("Third derivative")
pprint(df3)
print("Fourth derivative")
pprint(df4)
print("Fifth derivative")
pprint(df5)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Deriv F

# 02 — Deriv F / 02 Deriv F

**Chapter 19 — File 2 of 2 / 第19章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Deriv F**.

本脚本演示 **02 Deriv F**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy.abc import x, y
from sympy import diff, pprint

f = x**2 + 3*x*y + 4*y**2
fx = diff(f, x)
fy = diff(f, y)
fxx = diff(fx, x)
fyy = diff(fy, y)
fxy = diff(fx, y)
fyx = diff(fy, x)
print("Function")
pprint(f)
print("f_x =")
pprint(fx)
print("f_y =")
pprint(fy)
print("f_xx =")
pprint(fxx)
print("f_yy =")
pprint(fyy)
print("f_xy =")
pprint(fxy)
print("f_yx =")
pprint(fyx)
```

---
## Learning Notes / 学习笔记

- **概念**: Deriv F 是机器学习中的常用技术。  
  *Deriv F is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Deriv F / 02 Deriv F
# Complete Code / 完整代码
# ===============================

from sympy.abc import x, y
from sympy import diff, pprint

f = x**2 + 3*x*y + 4*y**2
fx = diff(f, x)
fy = diff(f, y)
fxx = diff(fx, x)
fyy = diff(fy, y)
fxy = diff(fx, y)
fyx = diff(fy, x)
print("Function")
pprint(f)
print("f_x =")
pprint(fx)
print("f_y =")
pprint(fy)
print("f_xx =")
pprint(fxx)
print("f_yy =")
pprint(fyy)
print("f_xy =")
pprint(fxy)
print("f_yx =")
pprint(fyx)
```

---

### Chapter Summary

# Chapter 19 Summary / 第19章总结

## Theme / 主题: Chapter 19 / Chapter 19

This chapter contains **2 code files** demonstrating chapter 19.

本章包含 **2 个代码文件**，演示Chapter 19。

---
## Evolution / 演化路线

  1. `01_high_order.ipynb` — High Order
  2. `02_deriv_f.ipynb` — Deriv F

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 19) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 19）是机器学习流水线中的基础构建块。

---
