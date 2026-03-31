# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 21

---

### Plot Function

# 10 — Plot Function / 10 Plot Function

**Chapter 21 — File 1 of 3 / 第21章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **plot of simple function**.

本脚本演示 **plot of simple function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — plot of simple function

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — objective function

```python
def objective(x):
	return x**2.0
```

---
## Step 3 — define range for input

```python
r_min, r_max = -1.0, 1.0
```

---
## Step 4 — sample input range uniformly at 0.1 increments

```python
# 生成整数序列 / Generate integer sequence
inputs = arange(r_min, r_max+0.1, 0.1)
```

---
## Step 5 — compute targets

```python
results = objective(inputs)
```

---
## Step 6 — create a line plot of input vs result

```python
pyplot.plot(inputs, results)
```

---
## Step 7 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot of simple function 是机器学习中的常用技术。  
  *plot of simple function is a common technique in machine learning.*

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
# Plot Function / 10 Plot Function
# Complete Code / 完整代码
# ===============================

# plot of simple function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# objective function
def objective(x):
	return x**2.0

# define range for input
r_min, r_max = -1.0, 1.0
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
inputs = arange(r_min, r_max+0.1, 0.1)
# compute targets
results = objective(inputs)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Gradient Descent



---

### Plot Gradient Descent



---

### Chapter Summary / 章节总结

# Chapter 21 Summary / 第21章总结

## Theme / 主题: Chapter 21 / Chapter 21

This chapter contains **3 code files** demonstrating chapter 21.

本章包含 **3 个代码文件**，演示Chapter 21。

---
## Evolution / 演化路线

  1. `10_plot_function.ipynb` — Plot Function
  2. `13_gradient_descent.ipynb` — Gradient Descent
  3. `20_plot_gradient_descent.ipynb` — Plot Gradient Descent

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 21) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 21）是机器学习流水线中的基础构建块。

---
