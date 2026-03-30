# 机器学习优化方法
## Chapter 24

---

### 3D Plot

# 02 — 3D Plot / 02 3D Plot

**Chapter 24 — File 1 of 4 / 第24章 — 第1个文件（共4个）**

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

**Chapter 24 — File 2 of 4 / 第24章 — 第2个文件（共4个）**

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

### Plot Rmsprop

# 18 — Plot Rmsprop / 18 Plot Rmsprop

**Chapter 24 — File 4 of 4 / 第24章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of plotting the rmsprop search on a contour plot of the test function**.

本脚本演示 **example of plotting the rmsprop search on a contour plot of the test function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — example of plotting the rmsprop search on a contour plot of the test function

```python
from math import sqrt
from numpy import asarray
from numpy import arange
from numpy.random import rand
from numpy.random import seed
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
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
## Step 4 — gradient descent algorithm with rmsprop

```python
def rmsprop(objective, derivative, bounds, n_iter, step_size, rho):
```

---
## Step 5 — track all solutions

```python
solutions = list()
```

---
## Step 6 — generate an initial point

```python
solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
```

---
## Step 7 — list of the average square gradients for each variable

```python
sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
```

---
## Step 8 — run the gradient descent

```python
for it in range(n_iter):
```

---
## Step 9 — calculate gradient

```python
gradient = derivative(solution[0], solution[1])
```

---
## Step 10 — update the average of the squared partial derivatives

```python
for i in range(gradient.shape[0]):
```

---
## Step 11 — calculate the squared gradient

```python
sg = gradient[i]**2.0
```

---
## Step 12 — update the moving average of the squared gradient

```python
sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (sg * (1.0-rho))
```

---
## Step 13 — build solution

```python
new_solution = list()
		for i in range(solution.shape[0]):
```

---
## Step 14 — calculate the learning rate for this variable

```python
alpha = step_size / (1e-8 + sqrt(sq_grad_avg[i]))
```

---
## Step 15 — calculate the new position in this variable

```python
value = solution[i] - alpha * gradient[i]
			new_solution.append(value)
```

---
## Step 16 — store the new solution

```python
solution = asarray(new_solution)
		solutions.append(solution)
```

---
## Step 17 — evaluate candidate point

```python
solution_eval = objective(solution[0], solution[1])
```

---
## Step 18 — report progress

```python
print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
	return solutions
```

---
## Step 19 — seed the pseudo random number generator

```python
seed(1)
```

---
## Step 20 — define range for input

```python
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
```

---
## Step 21 — define the total iterations

```python
n_iter = 50
```

---
## Step 22 — define the step size

```python
step_size = 0.01
```

---
## Step 23 — momentum for rmsprop

```python
rho = 0.99
```

---
## Step 24 — perform the gradient descent search with rmsprop

```python
solutions = rmsprop(objective, derivative, bounds, n_iter, step_size, rho)
```

---
## Step 25 — sample input range uniformly at 0.1 increments

```python
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
```

---
## Step 26 — create a mesh from the axis

```python
x, y = meshgrid(xaxis, yaxis)
```

---
## Step 27 — compute targets

```python
results = objective(x, y)
```

---
## Step 28 — create a filled contour plot with 50 levels and jet color scheme

```python
pyplot.contourf(x, y, results, levels=50, cmap='jet')
```

---
## Step 29 — plot the sample as black circles

```python
solutions = asarray(solutions)
pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
```

---
## Step 30 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of plotting the rmsprop search on a contour plot of the test function 是机器学习中的常用技术。  
  *example of plotting the rmsprop search on a contour plot of the test function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Rmsprop / 18 Plot Rmsprop
# Complete Code / 完整代码
# ===============================

# example of plotting the rmsprop search on a contour plot of the test function
from math import sqrt
from numpy import asarray
from numpy import arange
from numpy.random import rand
from numpy.random import seed
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# derivative of objective function
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])

# gradient descent algorithm with rmsprop
def rmsprop(objective, derivative, bounds, n_iter, step_size, rho):
	# track all solutions
	solutions = list()
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# list of the average square gradients for each variable
	sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
	# run the gradient descent
	for it in range(n_iter):
		# calculate gradient
		gradient = derivative(solution[0], solution[1])
		# update the average of the squared partial derivatives
		for i in range(gradient.shape[0]):
			# calculate the squared gradient
			sg = gradient[i]**2.0
			# update the moving average of the squared gradient
			sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (sg * (1.0-rho))
		# build solution
		new_solution = list()
		for i in range(solution.shape[0]):
			# calculate the learning rate for this variable
			alpha = step_size / (1e-8 + sqrt(sq_grad_avg[i]))
			# calculate the new position in this variable
			value = solution[i] - alpha * gradient[i]
			new_solution.append(value)
		# store the new solution
		solution = asarray(new_solution)
		solutions.append(solution)
		# evaluate candidate point
		solution_eval = objective(solution[0], solution[1])
		# report progress
		print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
	return solutions

# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# define the step size
step_size = 0.01
# momentum for rmsprop
rho = 0.99
# perform the gradient descent search with rmsprop
solutions = rmsprop(objective, derivative, bounds, n_iter, step_size, rho)
# sample input range uniformly at 0.1 increments
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# plot the sample as black circles
solutions = asarray(solutions)
pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
# show the plot
pyplot.show()
```

---

### Chapter Summary

# Chapter 24 Summary / 第24章总结

## Theme / 主题: Chapter 24 / Chapter 24

This chapter contains **4 code files** demonstrating chapter 24.

本章包含 **4 个代码文件**，演示Chapter 24。

---
## Evolution / 演化路线

  1. `02_3d_plot.ipynb` — 3D Plot
  2. `03_contour_plot.ipynb` — Contour Plot
  3. `13_rmsprop.ipynb` — Rmsprop
  4. `18_plot_rmsprop.ipynb` — Plot Rmsprop

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 24) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 24）是机器学习流水线中的基础构建块。

---
