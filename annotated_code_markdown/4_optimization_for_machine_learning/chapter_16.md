# 机器学习优化方法
## Chapter 16

---

### Ackley Function

# 01 — Ackley Function / 01 Ackley Function

**Chapter 16 — File 1 of 4 / 第16章 — 第1个文件（共4个）**

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
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
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
# Ackley Function / 01 Ackley Function
# Complete Code / 完整代码
# ===============================

# ackley multimodal function
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# objective function
def objective(x, y):
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# define range for input
r_min, r_max = -5.0, 5.0
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

### Random Restart Ackley

# 12 — Random Restart Ackley / 12 Random Restart Ackley

**Chapter 16 — File 3 of 4 / 第16章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **hill climbing search with random restarts of the ackley objective function**.

本脚本演示 **hill climbing search with random restarts of the ackley objective function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — hill climbing search with random restarts of the ackley objective function

```python
from numpy import asarray
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
```

---
## Step 2 — objective function

```python
def objective(v):
	x, y = v
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
```

---
## Step 3 — check if a point is within the bounds of the search

```python
def in_bounds(point, bounds):
```

---
## Step 4 — enumerate all dimensions of the point

```python
for d in range(len(bounds)):
```

---
## Step 5 — check if out of bounds for this dimension

```python
if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
			return False
	return True
```

---
## Step 6 — hill climbing local search algorithm

```python
def hillclimbing(objective, bounds, n_iterations, step_size, start_pt):
```

---
## Step 7 — store the initial point

```python
solution = start_pt
```

---
## Step 8 — evaluate the initial point

```python
solution_eval = objective(solution)
```

---
## Step 9 — run the hill climb

```python
for i in range(n_iterations):
```

---
## Step 10 — take a step

```python
candidate = None
		while candidate is None or not in_bounds(candidate, bounds):
			candidate = solution + randn(len(bounds)) * step_size
```

---
## Step 11 — evaluate candidate point

```python
candidte_eval = objective(candidate)
```

---
## Step 12 — check if we should keep the new point

```python
if candidte_eval <= solution_eval:
```

---
## Step 13 — store the new point

```python
solution, solution_eval = candidate, candidte_eval
	return [solution, solution_eval]
```

---
## Step 14 — hill climbing with random restarts algorithm

```python
def random_restarts(objective, bounds, n_iter, step_size, n_restarts):
	best, best_eval = None, 1e+10
```

---
## Step 15 — enumerate restarts

```python
for n in range(n_restarts):
```

---
## Step 16 — generate a random initial point for the search

```python
start_pt = None
		while start_pt is None or not in_bounds(start_pt, bounds):
			start_pt = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
```

---
## Step 17 — perform a stochastic hill climbing search

```python
solution, solution_eval = hillclimbing(objective, bounds, n_iter, step_size, start_pt)
```

---
## Step 18 — check for new best

```python
if solution_eval < best_eval:
			best, best_eval = solution, solution_eval
			print('Restart %d, best: f(%s) = %.5f' % (n, best, best_eval))
	return [best, best_eval]
```

---
## Step 19 — seed the pseudorandom number generator

```python
seed(1)
```

---
## Step 20 — define range for input

```python
bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])
```

---
## Step 21 — define the total iterations

```python
n_iter = 1000
```

---
## Step 22 — define the maximum step size

```python
step_size = 0.05
```

---
## Step 23 — total number of random restarts

```python
n_restarts = 30
```

---
## Step 24 — perform the hill climbing search

```python
best, score = random_restarts(objective, bounds, n_iter, step_size, n_restarts)
print('Done!')
print('f(%s) = %f' % (best, score))
```

---
## Learning Notes / 学习笔记

- **概念**: hill climbing search with random restarts of the ackley objective function 是机器学习中的常用技术。  
  *hill climbing search with random restarts of the ackley objective function is a common technique in machine learning.*

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
# Random Restart Ackley / 12 Random Restart Ackley
# Complete Code / 完整代码
# ===============================

# hill climbing search with random restarts of the ackley objective function
from numpy import asarray
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed

# objective function
def objective(v):
	x, y = v
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# check if a point is within the bounds of the search
def in_bounds(point, bounds):
	# enumerate all dimensions of the point
	for d in range(len(bounds)):
		# check if out of bounds for this dimension
		if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
			return False
	return True

# hill climbing local search algorithm
def hillclimbing(objective, bounds, n_iterations, step_size, start_pt):
	# store the initial point
	solution = start_pt
	# evaluate the initial point
	solution_eval = objective(solution)
	# run the hill climb
	for i in range(n_iterations):
		# take a step
		candidate = None
		while candidate is None or not in_bounds(candidate, bounds):
			candidate = solution + randn(len(bounds)) * step_size
		# evaluate candidate point
		candidte_eval = objective(candidate)
		# check if we should keep the new point
		if candidte_eval <= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
	return [solution, solution_eval]

# hill climbing with random restarts algorithm
def random_restarts(objective, bounds, n_iter, step_size, n_restarts):
	best, best_eval = None, 1e+10
	# enumerate restarts
	for n in range(n_restarts):
		# generate a random initial point for the search
		start_pt = None
		while start_pt is None or not in_bounds(start_pt, bounds):
			start_pt = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
		# perform a stochastic hill climbing search
		solution, solution_eval = hillclimbing(objective, bounds, n_iter, step_size, start_pt)
		# check for new best
		if solution_eval < best_eval:
			best, best_eval = solution, solution_eval
			print('Restart %d, best: f(%s) = %.5f' % (n, best, best_eval))
	return [best, best_eval]

# seed the pseudorandom number generator
seed(1)
# define range for input
bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])
# define the total iterations
n_iter = 1000
# define the maximum step size
step_size = 0.05
# total number of random restarts
n_restarts = 30
# perform the hill climbing search
best, score = random_restarts(objective, bounds, n_iter, step_size, n_restarts)
print('Done!')
print('f(%s) = %f' % (best, score))
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Iterated Ackley

# 15 — Iterated Ackley / 15 Iterated Ackley

**Chapter 16 — File 4 of 4 / 第16章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **iterated local search of the ackley objective function**.

本脚本演示 **iterated local search of the ackley objective function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — iterated local search of the ackley objective function

```python
from numpy import asarray
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
```

---
## Step 2 — objective function

```python
def objective(v):
	x, y = v
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
```

---
## Step 3 — check if a point is within the bounds of the search

```python
def in_bounds(point, bounds):
```

---
## Step 4 — enumerate all dimensions of the point

```python
for d in range(len(bounds)):
```

---
## Step 5 — check if out of bounds for this dimension

```python
if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
			return False
	return True
```

---
## Step 6 — hill climbing local search algorithm

```python
def hillclimbing(objective, bounds, n_iterations, step_size, start_pt):
```

---
## Step 7 — store the initial point

```python
solution = start_pt
```

---
## Step 8 — evaluate the initial point

```python
solution_eval = objective(solution)
```

---
## Step 9 — run the hill climb

```python
for i in range(n_iterations):
```

---
## Step 10 — take a step

```python
candidate = None
		while candidate is None or not in_bounds(candidate, bounds):
			candidate = solution + randn(len(bounds)) * step_size
```

---
## Step 11 — evaluate candidate point

```python
candidte_eval = objective(candidate)
```

---
## Step 12 — check if we should keep the new point

```python
if candidte_eval <= solution_eval:
```

---
## Step 13 — store the new point

```python
solution, solution_eval = candidate, candidte_eval
	return [solution, solution_eval]
```

---
## Step 14 — iterated local search algorithm

```python
def iterated_local_search(objective, bounds, n_iter, step_size, n_restarts, p_size):
```

---
## Step 15 — define starting point

```python
best = None
	while best is None or not in_bounds(best, bounds):
		best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
```

---
## Step 16 — evaluate current best point

```python
best_eval = objective(best)
```

---
## Step 17 — enumerate restarts

```python
for n in range(n_restarts):
```

---
## Step 18 — generate an initial point as a perturbed version of the last best

```python
start_pt = None
		while start_pt is None or not in_bounds(start_pt, bounds):
			start_pt = best + randn(len(bounds)) * p_size
```

---
## Step 19 — perform a stochastic hill climbing search

```python
solution, solution_eval = hillclimbing(objective, bounds, n_iter, step_size, start_pt)
```

---
## Step 20 — check for new best

```python
if solution_eval < best_eval:
			best, best_eval = solution, solution_eval
			print('Restart %d, best: f(%s) = %.5f' % (n, best, best_eval))
	return [best, best_eval]
```

---
## Step 21 — seed the pseudorandom number generator

```python
seed(1)
```

---
## Step 22 — define range for input

```python
bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])
```

---
## Step 23 — define the total iterations

```python
n_iter = 1000
```

---
## Step 24 — define the maximum step size

```python
s_size = 0.05
```

---
## Step 25 — total number of random restarts

```python
n_restarts = 30
```

---
## Step 26 — perturbation step size

```python
p_size = 1.0
```

---
## Step 27 — perform the hill climbing search

```python
best, score = iterated_local_search(objective, bounds, n_iter, s_size, n_restarts, p_size)
print('Done!')
print('f(%s) = %f' % (best, score))
```

---
## Learning Notes / 学习笔记

- **概念**: iterated local search of the ackley objective function 是机器学习中的常用技术。  
  *iterated local search of the ackley objective function is a common technique in machine learning.*

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
# Iterated Ackley / 15 Iterated Ackley
# Complete Code / 完整代码
# ===============================

# iterated local search of the ackley objective function
from numpy import asarray
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed

# objective function
def objective(v):
	x, y = v
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# check if a point is within the bounds of the search
def in_bounds(point, bounds):
	# enumerate all dimensions of the point
	for d in range(len(bounds)):
		# check if out of bounds for this dimension
		if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
			return False
	return True

# hill climbing local search algorithm
def hillclimbing(objective, bounds, n_iterations, step_size, start_pt):
	# store the initial point
	solution = start_pt
	# evaluate the initial point
	solution_eval = objective(solution)
	# run the hill climb
	for i in range(n_iterations):
		# take a step
		candidate = None
		while candidate is None or not in_bounds(candidate, bounds):
			candidate = solution + randn(len(bounds)) * step_size
		# evaluate candidate point
		candidte_eval = objective(candidate)
		# check if we should keep the new point
		if candidte_eval <= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
	return [solution, solution_eval]

# iterated local search algorithm
def iterated_local_search(objective, bounds, n_iter, step_size, n_restarts, p_size):
	# define starting point
	best = None
	while best is None or not in_bounds(best, bounds):
		best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# evaluate current best point
	best_eval = objective(best)
	# enumerate restarts
	for n in range(n_restarts):
		# generate an initial point as a perturbed version of the last best
		start_pt = None
		while start_pt is None or not in_bounds(start_pt, bounds):
			start_pt = best + randn(len(bounds)) * p_size
		# perform a stochastic hill climbing search
		solution, solution_eval = hillclimbing(objective, bounds, n_iter, step_size, start_pt)
		# check for new best
		if solution_eval < best_eval:
			best, best_eval = solution, solution_eval
			print('Restart %d, best: f(%s) = %.5f' % (n, best, best_eval))
	return [best, best_eval]

# seed the pseudorandom number generator
seed(1)
# define range for input
bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])
# define the total iterations
n_iter = 1000
# define the maximum step size
s_size = 0.05
# total number of random restarts
n_restarts = 30
# perturbation step size
p_size = 1.0
# perform the hill climbing search
best, score = iterated_local_search(objective, bounds, n_iter, s_size, n_restarts, p_size)
print('Done!')
print('f(%s) = %f' % (best, score))
```

---

### Chapter Summary

# Chapter 16 Summary / 第16章总结

## Theme / 主题: Chapter 16 / Chapter 16

This chapter contains **4 code files** demonstrating chapter 16.

本章包含 **4 个代码文件**，演示Chapter 16。

---
## Evolution / 演化路线

  1. `01_ackley_function.ipynb` — Ackley Function
  2. `07_hillclimbing_ackley.ipynb` — Hillclimbing Ackley
  3. `12_random_restart_ackley.ipynb` — Random Restart Ackley
  4. `15_iterated_ackley.ipynb` — Iterated Ackley

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 16) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 16）是机器学习流水线中的基础构建块。

---
