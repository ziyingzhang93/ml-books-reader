# 机器学习优化方法
## Chapter 22

---

### Plot Function

# 04 — Plot Function / 04 Plot Function

**Chapter 22 — File 1 of 5 / 第22章 — 第1个文件（共5个）**

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
# Plot Function / 04 Plot Function
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

➡️ **Next / 下一步**: File 2 of 5

---

### Gradient Descent

# 08 — Gradient Descent / 梯度方法

**Chapter 22 — File 2 of 5 / 第22章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of gradient descent for a one-dimensional function**.

本脚本演示 **example of gradient descent for a one-dimensional function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of gradient descent for a one-dimensional function

```python
from numpy import asarray
from numpy.random import rand
from numpy.random import seed
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
## Step 11 — seed the pseudo random number generator

```python
seed(4)
```

---
## Step 12 — define range for input

```python
bounds = asarray([[-1.0, 1.0]])
```

---
## Step 13 — define the total iterations

```python
n_iter = 30
```

---
## Step 14 — define the step size

```python
step_size = 0.1
```

---
## Step 15 — perform the gradient descent search

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
# Gradient Descent / 梯度方法
# Complete Code / 完整代码
# ===============================

# example of gradient descent for a one-dimensional function
from numpy import asarray
from numpy.random import rand
from numpy.random import seed

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

# seed the pseudo random number generator
seed(4)
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

➡️ **Next / 下一步**: File 3 of 5

---

### Plot Gradient Descent

# 13 — Plot Gradient Descent / 梯度方法

**Chapter 22 — File 3 of 5 / 第22章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of plotting a gradient descent search on a one-dimensional function**.

本脚本演示 **example of plotting a gradient descent search on a one-dimensional function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Step 1 — example of plotting a gradient descent search on a one-dimensional function

```python
from numpy import asarray
from numpy import arange
from numpy.random import rand
from numpy.random import seed
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
## Step 13 — seed the pseudo random number generator

```python
seed(4)
```

---
## Step 14 — define range for input

```python
bounds = asarray([[-1.0, 1.0]])
```

---
## Step 15 — define the total iterations

```python
n_iter = 30
```

---
## Step 16 — define the step size

```python
step_size = 0.1
```

---
## Step 17 — perform the gradient descent search

```python
solutions, scores = gradient_descent(objective, derivative, bounds, n_iter, step_size)
```

---
## Step 18 — sample input range uniformly at 0.1 increments

```python
inputs = arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
```

---
## Step 19 — compute targets

```python
results = objective(inputs)
```

---
## Step 20 — create a line plot of input vs result

```python
pyplot.plot(inputs, results)
```

---
## Step 21 — plot the solutions found

```python
pyplot.plot(solutions, scores, '.-', color='red')
```

---
## Step 22 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of plotting a gradient descent search on a one-dimensional function 是机器学习中的常用技术。  
  *example of plotting a gradient descent search on a one-dimensional function is a common technique in machine learning.*

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
# Plot Gradient Descent / 梯度方法
# Complete Code / 完整代码
# ===============================

# example of plotting a gradient descent search on a one-dimensional function
from numpy import asarray
from numpy import arange
from numpy.random import rand
from numpy.random import seed
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

# seed the pseudo random number generator
seed(4)
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

➡️ **Next / 下一步**: File 4 of 5

---

### Momentum

# 18 — Momentum / 18 Momentum

**Chapter 22 — File 4 of 5 / 第22章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of gradient descent with momentum for a one-dimensional function**.

本脚本演示 **example of gradient descent with momentum for a one-dimensional function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of gradient descent with momentum for a one-dimensional function

```python
from numpy import asarray
from numpy.random import rand
from numpy.random import seed
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
def gradient_descent(objective, derivative, bounds, n_iter, step_size, momentum):
```

---
## Step 5 — generate an initial point

```python
solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
```

---
## Step 6 — keep track of the change

```python
change = 0.0
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
## Step 9 — calculate update

```python
new_change = step_size * gradient + momentum * change
```

---
## Step 10 — take a step

```python
solution = solution - new_change
```

---
## Step 11 — save the change

```python
change = new_change
```

---
## Step 12 — evaluate candidate point

```python
solution_eval = objective(solution)
```

---
## Step 13 — report progress

```python
print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]
```

---
## Step 14 — seed the pseudo random number generator

```python
seed(4)
```

---
## Step 15 — define range for input

```python
bounds = asarray([[-1.0, 1.0]])
```

---
## Step 16 — define the total iterations

```python
n_iter = 30
```

---
## Step 17 — define the step size

```python
step_size = 0.1
```

---
## Step 18 — define momentum

```python
momentum = 0.3
```

---
## Step 19 — perform the gradient descent search with momentum

```python
best, score = gradient_descent(objective, derivative, bounds, n_iter, step_size, momentum)
print('Done!')
print('f(%s) = %f' % (best, score))
```

---
## Learning Notes / 学习笔记

- **概念**: example of gradient descent with momentum for a one-dimensional function 是机器学习中的常用技术。  
  *example of gradient descent with momentum for a one-dimensional function is a common technique in machine learning.*

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
# Momentum / 18 Momentum
# Complete Code / 完整代码
# ===============================

# example of gradient descent with momentum for a one-dimensional function
from numpy import asarray
from numpy.random import rand
from numpy.random import seed

# objective function
def objective(x):
	return x**2.0

# derivative of objective function
def derivative(x):
	return x * 2.0

# gradient descent algorithm
def gradient_descent(objective, derivative, bounds, n_iter, step_size, momentum):
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# keep track of the change
	change = 0.0
	# run the gradient descent
	for i in range(n_iter):
		# calculate gradient
		gradient = derivative(solution)
		# calculate update
		new_change = step_size * gradient + momentum * change
		# take a step
		solution = solution - new_change
		# save the change
		change = new_change
		# evaluate candidate point
		solution_eval = objective(solution)
		# report progress
		print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]

# seed the pseudo random number generator
seed(4)
# define range for input
bounds = asarray([[-1.0, 1.0]])
# define the total iterations
n_iter = 30
# define the step size
step_size = 0.1
# define momentum
momentum = 0.3
# perform the gradient descent search with momentum
best, score = gradient_descent(objective, derivative, bounds, n_iter, step_size, momentum)
print('Done!')
print('f(%s) = %f' % (best, score))
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Chapter Summary

# Chapter 22 Summary / 第22章总结

## Theme / 主题: Chapter 22 / Chapter 22

This chapter contains **5 code files** demonstrating chapter 22.

本章包含 **5 个代码文件**，演示Chapter 22。

---
## Evolution / 演化路线

  1. `04_plot_function.ipynb` — Plot Function
  2. `08_gradient_descent.ipynb` — Gradient Descent
  3. `13_plot_gradient_descent.ipynb` — Plot Gradient Descent
  4. `18_momentum.ipynb` — Momentum
  5. `19_plot_momentum.ipynb` — Plot Momentum

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 22) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 22）是机器学习流水线中的基础构建块。

---
