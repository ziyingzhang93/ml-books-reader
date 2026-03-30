# 优化深度学习
## Chapter 14

---

### Problem

# 01 — Problem / 01 Problem

**Chapter 14 — File 1 of 4 / 第14章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **scatter plot of circles dataset**.

本脚本演示 **scatter plot of circles dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Step 1 — scatter plot of circles dataset

```python
from sklearn.datasets import make_circles
from matplotlib import pyplot
from numpy import where
```

---
## Step 2 — generate 2d classification dataset

```python
X, y = make_circles(n_samples=100, noise=0.1, random_state=1)
```

---
## Step 3 — scatter plot for each class value

```python
for class_value in range(2):
```

---
## Step 4 — select indices of points with the class label

```python
row_ix = where(y == class_value)
```

---
## Step 5 — scatter plot for points with a different color

```python
pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
```

---
## Step 6 — show plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: scatter plot of circles dataset 是机器学习中的常用技术。  
  *scatter plot of circles dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Problem / 01 Problem
# Complete Code / 完整代码
# ===============================

# scatter plot of circles dataset
from sklearn.datasets import make_circles
from matplotlib import pyplot
from numpy import where
# generate 2d classification dataset
X, y = make_circles(n_samples=100, noise=0.1, random_state=1)
# scatter plot for each class value
for class_value in range(2):
	# select indices of points with the class label
	row_ix = where(y == class_value)
	# scatter plot for points with a different color
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Chapter Summary

# Chapter 14 Summary / 第14章总结

## Theme / 主题: Chapter 14 / Chapter 14

This chapter contains **4 code files** demonstrating chapter 14.

本章包含 **4 个代码文件**，演示Chapter 14。

---
## Evolution / 演化路线

  1. `01_problem.ipynb` — Problem
  2. `02_mlp.ipynb` — Mlp
  3. `03_mlp_activity_reg_before.ipynb` — Mlp Activity Reg Before
  4. `04_mlp_activity_reg_after.ipynb` — Mlp Activity Reg After

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 14) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 14）是机器学习流水线中的基础构建块。

---
