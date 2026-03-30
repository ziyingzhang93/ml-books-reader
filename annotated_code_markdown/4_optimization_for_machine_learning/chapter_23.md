# 机器学习优化方法
## Chapter 23

---

### 3D Plot

# 02 — 3D Plot / 02 3D Plot

**Chapter 23 — File 1 of 4 / 第23章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **3d plot of the test function**.

本脚本演示 **3d plot of the test function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — 3d plot of the test function

```python
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot
```

---
## Step 2 — objective function

```python
def objective(x, y):
	return x**2.0 + y**2.0
```

---
## Step 3 — define range for input

```python
r_min, r_max = -1.0, 1.0
```

---
## Step 4 — sample input range uniformly at 0.1 increments

```python
xaxis = arange(r_min, r_max, 0.1)
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

- **概念**: 3d plot of the test function 是机器学习中的常用技术。  
  *3d plot of the test function is a common technique in machine learning.*

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
# 3D Plot / 02 3D Plot
# Complete Code / 完整代码
# ===============================

# 3d plot of the test function
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# define range for input
r_min, r_max = -1.0, 1.0
# sample input range uniformly at 0.1 increments
xaxis = arange(r_min, r_max, 0.1)
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

➡️ **Next / 下一步**: File 2 of 4

---

### Contour Plot

# 03 — Contour Plot / 03 Contour Plot

**Chapter 23 — File 2 of 4 / 第23章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **contour plot of the test function**.

本脚本演示 **contour plot of the test function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — contour plot of the test function

```python
from numpy import asarray
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot
```

---
## Step 2 — objective function

```python
def objective(x, y):
	return x**2.0 + y**2.0
```

---
## Step 3 — define range for input

```python
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
```

---
## Step 4 — sample input range uniformly at 0.1 increments

```python
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
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
## Step 7 — create a filled contour plot with 50 levels and jet color scheme

```python
pyplot.contourf(x, y, results, levels=50, cmap='jet')
```

---
## Step 8 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: contour plot of the test function 是机器学习中的常用技术。  
  *contour plot of the test function is a common technique in machine learning.*

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
# Contour Plot / 03 Contour Plot
# Complete Code / 完整代码
# ===============================

# contour plot of the test function
from numpy import asarray
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# sample input range uniformly at 0.1 increments
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Adagrad

# 14 — Adagrad / 14 Adagrad

**Chapter 23 — File 3 of 4 / 第23章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **gradient descent optimization with adagrad for a two-dimensional test function**.

本脚本演示 **gradient descent optimization with adagrad for a two-dimensional test function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — gradient descent optimization with adagrad for a two-dimensional test function

```python
from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed
```

---
## Step 2 — objective function

```python
def objective(x, y):
	return x**2.0 + y**2.0
```

---
## Step 3 — derivative of objective function

```python
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])
```

---
## Step 4 — gradient descent algorithm with adagrad

```python
def adagrad(objective, derivative, bounds, n_iter, step_size):
```

---
## Step 5 — generate an initial point

```python
solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
```

---
## Step 6 — list of the sum square gradients for each variable

```python
sq_grad_sums = [0.0 for _ in range(bounds.shape[0])]
```

---
## Step 7 — run the gradient descent

```python
for it in range(n_iter):
```

---
## Step 8 — calculate gradient

```python
gradient = derivative(solution[0], solution[1])
```

---
## Step 9 — update the sum of the squared partial derivatives

```python
for i in range(gradient.shape[0]):
			sq_grad_sums[i] += gradient[i]**2.0
```

---
## Step 10 — build a solution one variable at a time

```python
new_solution = list()
		for i in range(solution.shape[0]):
```

---
## Step 11 — calculate the step size for this variable

```python
alpha = step_size / (1e-8 + sqrt(sq_grad_sums[i]))
```

---
## Step 12 — calculate the new position in this variable

```python
value = solution[i] - alpha * gradient[i]
```

---
## Step 13 — store this variable

```python
new_solution.append(value)
```

---
## Step 14 — evaluate candidate point

```python
solution = asarray(new_solution)
		solution_eval = objective(solution[0], solution[1])
```

---
## Step 15 — report progress

```python
print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
	return [solution, solution_eval]
```

---
## Step 16 — seed the pseudo random number generator

```python
seed(1)
```

---
## Step 17 — define range for input

```python
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
```

---
## Step 18 — define the total iterations

```python
n_iter = 50
```

---
## Step 19 — define the step size

```python
step_size = 0.1
```

---
## Step 20 — perform the gradient descent search with adagrad

```python
best, score = adagrad(objective, derivative, bounds, n_iter, step_size)
print('Done!')
print('f(%s) = %f' % (best, score))
```

---
## Learning Notes / 学习笔记

- **概念**: gradient descent optimization with adagrad for a two-dimensional test function 是机器学习中的常用技术。  
  *gradient descent optimization with adagrad for a two-dimensional test function is a common technique in machine learning.*

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
# Adagrad / 14 Adagrad
# Complete Code / 完整代码
# ===============================

# gradient descent optimization with adagrad for a two-dimensional test function
from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# derivative of objective function
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])

# gradient descent algorithm with adagrad
def adagrad(objective, derivative, bounds, n_iter, step_size):
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# list of the sum square gradients for each variable
	sq_grad_sums = [0.0 for _ in range(bounds.shape[0])]
	# run the gradient descent
	for it in range(n_iter):
		# calculate gradient
		gradient = derivative(solution[0], solution[1])
		# update the sum of the squared partial derivatives
		for i in range(gradient.shape[0]):
			sq_grad_sums[i] += gradient[i]**2.0
		# build a solution one variable at a time
		new_solution = list()
		for i in range(solution.shape[0]):
			# calculate the step size for this variable
			alpha = step_size / (1e-8 + sqrt(sq_grad_sums[i]))
			# calculate the new position in this variable
			value = solution[i] - alpha * gradient[i]
			# store this variable
			new_solution.append(value)
		# evaluate candidate point
		solution = asarray(new_solution)
		solution_eval = objective(solution[0], solution[1])
		# report progress
		print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
	return [solution, solution_eval]

# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# define the step size
step_size = 0.1
# perform the gradient descent search with adagrad
best, score = adagrad(objective, derivative, bounds, n_iter, step_size)
print('Done!')
print('f(%s) = %f' % (best, score))
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Chapter Summary

# Chapter 23 Summary / 第23章总结

## Theme / 主题: Chapter 23 / Chapter 23

This chapter contains **4 code files** demonstrating chapter 23.

本章包含 **4 个代码文件**，演示Chapter 23。

---
## Evolution / 演化路线

  1. `02_3d_plot.ipynb` — 3D Plot
  2. `03_contour_plot.ipynb` — Contour Plot
  3. `14_adagrad.ipynb` — Adagrad
  4. `19_plot_adagrad.ipynb` — Plot Adagrad

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 23) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 23）是机器学习流水线中的基础构建块。

---
