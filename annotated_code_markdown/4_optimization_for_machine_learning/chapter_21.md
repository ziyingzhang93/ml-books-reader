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

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Function / 10 Plot Function
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

➡️ **Next / 下一步**: File 2 of 3

---

### Gradient Descent

# 13 — Gradient Descent / 梯度方法

**Chapter 21 — File 2 of 3 / 第21章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **example of gradient descent for a one-dimensional function**.

本脚本演示 **example of gradient descent for a one-dimensional function**。

---
## Step 1 — example of gradient descent for a one-dimensional function

```python
from numpy import asarray
from numpy.random import rand
```

---
## Step 2 — objective function

```python
def objective(x):
	return x**2.0
```

---
## Step 3 — derivative of objective function

```python
def derivative(x):
	return x * 2.0
```

---
## Step 4 — gradient descent algorithm

```python
def gradient_descent(objective, derivative, bounds, n_iter, step_size):
```

---
## Step 5 — generate an initial point

```python
solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
```

---
## Step 6 — run the gradient descent

```python
for i in range(n_iter):
```

---
## Step 7 — calculate gradient

```python
gradient = derivative(solution)
```

---
## Step 8 — take a step

```python
solution = solution - step_size * gradient
```

---
## Step 9 — evaluate candidate point

```python
solution_eval = objective(solution)
```

---
## Step 10 — report progress

```python
print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]
```

---
## Step 11 — define range for input

```python
bounds = asarray([[-1.0, 1.0]])
```

---
## Step 12 — define the total iterations

```python
n_iter = 30
```

---
## Step 13 — define the step size

```python
step_size = 0.1
```

---
## Step 14 — perform the gradient descent search

```python
best, score = gradient_descent(objective, derivative, bounds, n_iter, step_size)
print('Done!')
print('f(%s) = %f' % (best, score))
```

---
## Learning Notes / 学习笔记

- **概念**: example of gradient descent for a one-dimensional function 是机器学习中的常用技术。  
  *example of gradient descent for a one-dimensional function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gradient Descent / 梯度方法
# Complete Code / 完整代码
# ===============================

# example of gradient descent for a one-dimensional function
from numpy import asarray
from numpy.random import rand

# objective function
def objective(x):
	return x**2.0

# derivative of objective function
def derivative(x):
	return x * 2.0

# gradient descent algorithm
def gradient_descent(objective, derivative, bounds, n_iter, step_size):
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# run the gradient descent
	for i in range(n_iter):
		# calculate gradient
		gradient = derivative(solution)
		# take a step
		solution = solution - step_size * gradient
		# evaluate candidate point
		solution_eval = objective(solution)
		# report progress
		print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]

# define range for input
bounds = asarray([[-1.0, 1.0]])
# define the total iterations
n_iter = 30
# define the step size
step_size = 0.1
# perform the gradient descent search
best, score = gradient_descent(objective, derivative, bounds, n_iter, step_size)
print('Done!')
print('f(%s) = %f' % (best, score))
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Plot Gradient Descent

# 20 — Plot Gradient Descent / 梯度方法

**Chapter 21 — File 3 of 3 / 第21章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **example of plotting a gradient descent search on a one-dimensional function**.

本脚本演示 **example of plotting a gradient descent search on a one-dimensional function**。

---
## Step 1 — example of plotting a gradient descent search on a one-dimensional function

```python
from numpy import asarray
from numpy import arange
from numpy.random import rand
from matplotlib import pyplot
```

---
## Step 2 — objective function

```python
def objective(x):
	return x**2.0
```

---
## Step 3 — derivative of objective function

```python
def derivative(x):
	return x * 2.0
```

---
## Step 4 — gradient descent algorithm

```python
def gradient_descent(objective, derivative, bounds, n_iter, step_size):
```

---
## Step 5 — track all solutions

```python
solutions, scores = list(), list()
```

---
## Step 6 — generate an initial point

```python
solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
```

---
## Step 7 — run the gradient descent

```python
for i in range(n_iter):
```

---
## Step 8 — calculate gradient

```python
gradient = derivative(solution)
```

---
## Step 9 — take a step

```python
solution = solution - step_size * gradient
```

---
## Step 10 — evaluate candidate point

```python
solution_eval = objective(solution)
```

---
## Step 11 — store solution

```python
solutions.append(solution)
		scores.append(solution_eval)
```

---
## Step 12 — report progress

```python
print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solutions, scores]
```

---
## Step 13 — define range for input

```python
bounds = asarray([[-1.0, 1.0]])
```

---
## Step 14 — define the total iterations

```python
n_iter = 30
```

---
## Step 15 — define the step size

```python
step_size = 0.1
```

---
## Step 16 — perform the gradient descent search

```python
solutions, scores = gradient_descent(objective, derivative, bounds, n_iter, step_size)
```

---
## Step 17 — sample input range uniformly at 0.1 increments

```python
inputs = arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
```

---
## Step 18 — compute targets

```python
results = objective(inputs)
```

---
## Step 19 — create a line plot of input vs result

```python
pyplot.plot(inputs, results)
```

---
## Step 20 — plot the solutions found

```python
pyplot.plot(solutions, scores, '.-', color='red')
```

---
## Step 21 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of plotting a gradient descent search on a one-dimensional function 是机器学习中的常用技术。  
  *example of plotting a gradient descent search on a one-dimensional function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Gradient Descent / 梯度方法
# Complete Code / 完整代码
# ===============================

# example of plotting a gradient descent search on a one-dimensional function
from numpy import asarray
from numpy import arange
from numpy.random import rand
from matplotlib import pyplot

# objective function
def objective(x):
	return x**2.0

# derivative of objective function
def derivative(x):
	return x * 2.0

# gradient descent algorithm
def gradient_descent(objective, derivative, bounds, n_iter, step_size):
	# track all solutions
	solutions, scores = list(), list()
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# run the gradient descent
	for i in range(n_iter):
		# calculate gradient
		gradient = derivative(solution)
		# take a step
		solution = solution - step_size * gradient
		# evaluate candidate point
		solution_eval = objective(solution)
		# store solution
		solutions.append(solution)
		scores.append(solution_eval)
		# report progress
		print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solutions, scores]

# define range for input
bounds = asarray([[-1.0, 1.0]])
# define the total iterations
n_iter = 30
# define the step size
step_size = 0.1
# perform the gradient descent search
solutions, scores = gradient_descent(objective, derivative, bounds, n_iter, step_size)
# sample input range uniformly at 0.1 increments
inputs = arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
# compute targets
results = objective(inputs)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# plot the solutions found
pyplot.plot(solutions, scores, '.-', color='red')
# show the plot
pyplot.show()
```

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
