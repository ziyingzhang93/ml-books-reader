# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 18

---

### Ackley

# 01 — Ackley / 01 Ackley

**Chapter 18 — File 1 of 3 / 第18章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **ackley multimodal function**.

本脚本演示 **ackley multimodal function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — ackley multimodal function

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import exp
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import cos
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import e
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import pi
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
```

---
## Step 2 — objective function

```python
def objective(x, y):
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
```

---
## Step 3 — define range for input

```python
r_min, r_max = -5.0, 5.0
```

---
## Step 4 — sample input range uniformly at 0.1 increments

```python
# 生成整数序列 / Generate integer sequence
xaxis = arange(r_min, r_max, 0.1)
# 生成整数序列 / Generate integer sequence
yaxis = arange(r_min, r_max, 0.1)
```

---
## Step 5 — create a mesh from the axis

```python
x, y = meshgrid(xaxis, yaxis)
```

---
## Step 6 — compute targets

```python
results = objective(x, y)
```

---
## Step 7 — create a surface plot with the jet color scheme

```python
fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x, y, results, cmap='jet')
```

---
## Step 8 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: ackley multimodal function 是机器学习中的常用技术。  
  *ackley multimodal function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ackley / 01 Ackley
# Complete Code / 完整代码
# ===============================

# ackley multimodal function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import exp
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import cos
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import e
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import pi
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# objective function
def objective(x, y):
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
xaxis = arange(r_min, r_max, 0.1)
# 生成整数序列 / Generate integer sequence
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a surface plot with the jet color scheme
fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x, y, results, cmap='jet')
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Comma



---

### Plus



---

### Chapter Summary / 章节总结

# Chapter 18 Summary / 第18章总结

## Theme / 主题: Chapter 18 / Chapter 18

This chapter contains **3 code files** demonstrating chapter 18.

本章包含 **3 个代码文件**，演示Chapter 18。

---
## Evolution / 演化路线

  1. `01_ackley.ipynb` — Ackley
  2. `12_comma.ipynb` — Comma
  3. `15_plus.ipynb` — Plus

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 18) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 18）是机器学习流水线中的基础构建块。

---
