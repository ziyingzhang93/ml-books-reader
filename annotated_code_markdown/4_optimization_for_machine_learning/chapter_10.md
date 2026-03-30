# 机器学习优化方法
## Chapter 10

---

### Plot Function

# 01 — Plot Function / 01 Plot Function

**Chapter 10 — File 1 of 2 / 第10章 — 第1个文件（共2个）**

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
from numpy import arange
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
# Plot Function / 01 Plot Function
# Complete Code / 完整代码
# ===============================

# plot of simple function
from numpy import arange
from matplotlib import pyplot

# objective function
def objective(x):
	return x**2.0

# define range for input
r_min, r_max = -1.0, 1.0
# sample input range uniformly at 0.1 increments
inputs = arange(r_min, r_max+0.1, 0.1)
# compute targets
results = objective(inputs)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 2

---
