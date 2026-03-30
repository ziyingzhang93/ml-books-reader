# ML微积分
## Chapter 18

---

### F2

# 01 — F2 / 01 F2

**Chapter 18 — File 1 of 2 / 第18章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **F2**.

本脚本演示 **01 F2**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy.abc import x, y
from sympy import diff, pprint

f2 = x**2 + y**2
df2dx = diff(f2, x)
df2dy = diff(f2, y)
print("Partial derivative of")
pprint(f2)
print("with respect to x is")
pprint(df2dx)
print("and with respect to y is")
pprint(df2dy)
print("gradient at (1,1) is ({},{})".format(df2dx.subs([(x,1),(y,1)]), df2dy.subs([(x,1),(y,1)])))
print("gradient at (2,1) is ({},{})".format(df2dx.subs([(x,2),(y,1)]), df2dy.subs([(x,2),(y,1)])))
```

---
## Learning Notes / 学习笔记

- **概念**: F2 是机器学习中的常用技术。  
  *F2 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# F2 / 01 F2
# Complete Code / 完整代码
# ===============================

from sympy.abc import x, y
from sympy import diff, pprint

f2 = x**2 + y**2
df2dx = diff(f2, x)
df2dy = diff(f2, y)
print("Partial derivative of")
pprint(f2)
print("with respect to x is")
pprint(df2dx)
print("and with respect to y is")
pprint(df2dy)
print("gradient at (1,1) is ({},{})".format(df2dx.subs([(x,1),(y,1)]), df2dy.subs([(x,1),(y,1)])))
print("gradient at (2,1) is ({},{})".format(df2dx.subs([(x,2),(y,1)]), df2dy.subs([(x,2),(y,1)])))
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Maxrate

# 03 — Maxrate / 03 Maxrate

**Chapter 18 — File 2 of 2 / 第18章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Maxrate**.

本脚本演示 **03 Maxrate**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import numpy as np

def f(x, y):
    return x**2 + y**2

x, y = 1, 1
step = 0.001
angles = np.arange(0, 360, 5) # 0 to 360 degrees at 5-degree steps
maxdf, maxangle = -np.inf, 0
for angle in angles:
    rad = angle * np.pi / 180 # convert degree to radian
    dx, dy = np.sin(rad)*step, np.cos(rad)*step
    df = (f(x+dx, y+dy) - f(x,y))/step
    if df > maxdf:
        maxdf, maxangle = df, angle
    print(f"Rate of change at {angle} degrees = {df}")

dx, dy = np.sin(maxangle*np.pi/180), np.cos(maxangle*np.pi/180)
gradx, grady = dx*maxdf, dy*maxdf
print(f"Max rate of change at {maxangle} degrees")
print(f"Gradient vector at ({x},{y}) is ({dx*maxdf},{dy*maxdf})")
```

---
## Learning Notes / 学习笔记

- **概念**: Maxrate 是机器学习中的常用技术。  
  *Maxrate is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Maxrate / 03 Maxrate
# Complete Code / 完整代码
# ===============================

import numpy as np

def f(x, y):
    return x**2 + y**2

x, y = 1, 1
step = 0.001
angles = np.arange(0, 360, 5) # 0 to 360 degrees at 5-degree steps
maxdf, maxangle = -np.inf, 0
for angle in angles:
    rad = angle * np.pi / 180 # convert degree to radian
    dx, dy = np.sin(rad)*step, np.cos(rad)*step
    df = (f(x+dx, y+dy) - f(x,y))/step
    if df > maxdf:
        maxdf, maxangle = df, angle
    print(f"Rate of change at {angle} degrees = {df}")

dx, dy = np.sin(maxangle*np.pi/180), np.cos(maxangle*np.pi/180)
gradx, grady = dx*maxdf, dy*maxdf
print(f"Max rate of change at {maxangle} degrees")
print(f"Gradient vector at ({x},{y}) is ({dx*maxdf},{dy*maxdf})")
```

---

### Chapter Summary

# Chapter 18 Summary / 第18章总结

## Theme / 主题: Chapter 18 / Chapter 18

This chapter contains **2 code files** demonstrating chapter 18.

本章包含 **2 个代码文件**，演示Chapter 18。

---
## Evolution / 演化路线

  1. `01_f2.ipynb` — F2
  2. `03_maxrate.ipynb` — Maxrate

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 18) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 18）是机器学习流水线中的基础构建块。

---
