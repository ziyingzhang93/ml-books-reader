# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 15

---

### Plot Convex Function

# 09 — Plot Convex Function / 09 Plot Convex Function

**Chapter 15 — File 1 of 4 / 第15章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **convex unimodal optimization function**.

本脚本演示 **convex unimodal optimization function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — convex unimodal optimization function

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
	return x[0]**2.0
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
inputs = arange(r_min, r_max, 0.1)
```

---
## Step 5 — compute targets

```python
results = [objective([x]) for x in inputs]
```

---
## Step 6 — create a line plot of input vs result

```python
pyplot.plot(inputs, results)
```

---
## Step 7 — define optimal input value

```python
x_optima = 0.0
```

---
## Step 8 — draw a vertical line at the optimal input

```python
pyplot.axvline(x=x_optima, ls='--', color='red')
```

---
## Step 9 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: convex unimodal optimization function 是机器学习中的常用技术。  
  *convex unimodal optimization function is a common technique in machine learning.*

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
# Plot Convex Function / 09 Plot Convex Function
# Complete Code / 完整代码
# ===============================

# convex unimodal optimization function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# objective function
def objective(x):
	return x[0]**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
inputs = arange(r_min, r_max, 0.1)
# compute targets
results = [objective([x]) for x in inputs]
# create a line plot of input vs result
pyplot.plot(inputs, results)
# define optimal input value
x_optima = 0.0
# draw a vertical line at the optimal input
pyplot.axvline(x=x_optima, ls='--', color='red')
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Hillclimbing

# 13 — Hillclimbing / 13 Hillclimbing

**Chapter 15 — File 2 of 4 / 第15章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **hill climbing search of a one-dimensional objective function**.

本脚本演示 **hill climbing search of a one-dimensional objective function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — hill climbing search of a one-dimensional objective function

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
```

---
## Step 2 — objective function

```python
def objective(x):
	return x[0]**2.0
```

---
## Step 3 — hill climbing local search algorithm

```python
def hillclimbing(objective, bounds, n_iterations, step_size):
```

---
## Step 4 — generate an initial point

```python
# 获取长度 / Get length
solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
```

---
## Step 5 — evaluate the initial point

```python
solution_eval = objective(solution)
```

---
## Step 6 — run the hill climb

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_iterations):
```

---
## Step 7 — take a step

```python
# 获取长度 / Get length
candidate = solution + randn(len(bounds)) * step_size
```

---
## Step 8 — evaluate candidate point

```python
candidte_eval = objective(candidate)
```

---
## Step 9 — check if we should keep the new point

```python
if candidte_eval <= solution_eval:
```

---
## Step 10 — store the new point

```python
solution, solution_eval = candidate, candidte_eval
```

---
## Step 11 — report progress

```python
# 打印输出 / Print output
print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]
```

---
## Step 12 — seed the pseudorandom number generator

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(5)
```

---
## Step 13 — define range for input

```python
bounds = asarray([[-5.0, 5.0]])
```

---
## Step 14 — define the total iterations

```python
n_iterations = 1000
```

---
## Step 15 — define the maximum step size

```python
step_size = 0.1
```

---
## Step 16 — perform the hill climbing search

```python
best, score = hillclimbing(objective, bounds, n_iterations, step_size)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
```

---
## Learning Notes / 学习笔记

- **概念**: hill climbing search of a one-dimensional objective function 是机器学习中的常用技术。  
  *hill climbing search of a one-dimensional objective function is a common technique in machine learning.*

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
# Hillclimbing / 13 Hillclimbing
# Complete Code / 完整代码
# ===============================

# hill climbing search of a one-dimensional objective function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed

# objective function
def objective(x):
	return x[0]**2.0

# hill climbing local search algorithm
def hillclimbing(objective, bounds, n_iterations, step_size):
	# generate an initial point
 # 获取长度 / Get length
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# evaluate the initial point
	solution_eval = objective(solution)
	# run the hill climb
 # 生成整数序列 / Generate integer sequence
	for i in range(n_iterations):
		# take a step
  # 获取长度 / Get length
		candidate = solution + randn(len(bounds)) * step_size
		# evaluate candidate point
		candidte_eval = objective(candidate)
		# check if we should keep the new point
		if candidte_eval <= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
   # 打印输出 / Print output
			print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]

# seed the pseudorandom number generator
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(5)
# define range for input
bounds = asarray([[-5.0, 5.0]])
# define the total iterations
n_iterations = 1000
# define the maximum step size
step_size = 0.1
# perform the hill climbing search
best, score = hillclimbing(objective, bounds, n_iterations, step_size)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Plot Hillclimbing Scores

# 16 — Plot Hillclimbing Scores / 16 Plot Hillclimbing Scores

**Chapter 15 — File 3 of 4 / 第15章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **hill climbing search of a one-dimensional objective function**.

本脚本演示 **hill climbing search of a one-dimensional objective function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  📊 评估模型 / Evaluate Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — hill climbing search of a one-dimensional objective function

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — objective function

```python
def objective(x):
	return x[0]**2.0
```

---
## Step 3 — hill climbing local search algorithm

```python
def hillclimbing(objective, bounds, n_iterations, step_size):
```

---
## Step 4 — generate an initial point

```python
# 获取长度 / Get length
solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
```

---
## Step 5 — evaluate the initial point

```python
solution_eval = objective(solution)
```

---
## Step 6 — run the hill climb

```python
scores = list()
 # 添加元素到列表末尾 / Append element to list end
	scores.append(solution_eval)
 # 生成整数序列 / Generate integer sequence
	for i in range(n_iterations):
```

---
## Step 7 — take a step

```python
# 获取长度 / Get length
candidate = solution + randn(len(bounds)) * step_size
```

---
## Step 8 — evaluate candidate point

```python
candidte_eval = objective(candidate)
```

---
## Step 9 — check if we should keep the new point

```python
if candidte_eval <= solution_eval:
```

---
## Step 10 — store the new point

```python
solution, solution_eval = candidate, candidte_eval
```

---
## Step 11 — keep track of scores

```python
# 添加元素到列表末尾 / Append element to list end
scores.append(solution_eval)
```

---
## Step 12 — report progress

```python
# 打印输出 / Print output
print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval, scores]
```

---
## Step 13 — seed the pseudorandom number generator

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(5)
```

---
## Step 14 — define range for input

```python
bounds = asarray([[-5.0, 5.0]])
```

---
## Step 15 — define the total iterations

```python
n_iterations = 1000
```

---
## Step 16 — define the maximum step size

```python
step_size = 0.1
```

---
## Step 17 — perform the hill climbing search

```python
best, score, scores = hillclimbing(objective, bounds, n_iterations, step_size)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
```

---
## Step 18 — line plot of best scores

```python
pyplot.plot(scores, '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: hill climbing search of a one-dimensional objective function 是机器学习中的常用技术。  
  *hill climbing search of a one-dimensional objective function is a common technique in machine learning.*

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
# Plot Hillclimbing Scores / 16 Plot Hillclimbing Scores
# Complete Code / 完整代码
# ===============================

# hill climbing search of a one-dimensional objective function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# objective function
def objective(x):
	return x[0]**2.0

# hill climbing local search algorithm
def hillclimbing(objective, bounds, n_iterations, step_size):
	# generate an initial point
 # 获取长度 / Get length
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# evaluate the initial point
	solution_eval = objective(solution)
	# run the hill climb
	scores = list()
 # 添加元素到列表末尾 / Append element to list end
	scores.append(solution_eval)
 # 生成整数序列 / Generate integer sequence
	for i in range(n_iterations):
		# take a step
  # 获取长度 / Get length
		candidate = solution + randn(len(bounds)) * step_size
		# evaluate candidate point
		candidte_eval = objective(candidate)
		# check if we should keep the new point
		if candidte_eval <= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# keep track of scores
   # 添加元素到列表末尾 / Append element to list end
			scores.append(solution_eval)
			# report progress
   # 打印输出 / Print output
			print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval, scores]

# seed the pseudorandom number generator
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(5)
# define range for input
bounds = asarray([[-5.0, 5.0]])
# define the total iterations
n_iterations = 1000
# define the maximum step size
step_size = 0.1
# perform the hill climbing search
best, score, scores = hillclimbing(objective, bounds, n_iterations, step_size)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
# line plot of best scores
pyplot.plot(scores, '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Plot Hillclimbing Samples

# 20 — Plot Hillclimbing Samples / 20 Plot Hillclimbing Samples

**Chapter 15 — File 4 of 4 / 第15章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **hill climbing search of a one-dimensional objective function**.

本脚本演示 **hill climbing search of a one-dimensional objective function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  📊 评估模型 / Evaluate Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — hill climbing search of a one-dimensional objective function

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — objective function

```python
def objective(x):
	return x[0]**2.0
```

---
## Step 3 — hill climbing local search algorithm

```python
def hillclimbing(objective, bounds, n_iterations, step_size):
```

---
## Step 4 — generate an initial point

```python
# 获取长度 / Get length
solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
```

---
## Step 5 — evaluate the initial point

```python
solution_eval = objective(solution)
```

---
## Step 6 — run the hill climb

```python
solutions = list()
 # 添加元素到列表末尾 / Append element to list end
	solutions.append(solution)
 # 生成整数序列 / Generate integer sequence
	for i in range(n_iterations):
```

---
## Step 7 — take a step

```python
# 获取长度 / Get length
candidate = solution + randn(len(bounds)) * step_size
```

---
## Step 8 — evaluate candidate point

```python
candidte_eval = objective(candidate)
```

---
## Step 9 — check if we should keep the new point

```python
if candidte_eval <= solution_eval:
```

---
## Step 10 — store the new point

```python
solution, solution_eval = candidate, candidte_eval
```

---
## Step 11 — keep track of solutions

```python
# 添加元素到列表末尾 / Append element to list end
solutions.append(solution)
```

---
## Step 12 — report progress

```python
# 打印输出 / Print output
print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval, solutions]
```

---
## Step 13 — seed the pseudorandom number generator

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(5)
```

---
## Step 14 — define range for input

```python
bounds = asarray([[-5.0, 5.0]])
```

---
## Step 15 — define the total iterations

```python
n_iterations = 1000
```

---
## Step 16 — define the maximum step size

```python
step_size = 0.1
```

---
## Step 17 — perform the hill climbing search

```python
best, score, solutions = hillclimbing(objective, bounds, n_iterations, step_size)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
```

---
## Step 18 — sample input range uniformly at 0.1 increments

```python
# 生成整数序列 / Generate integer sequence
inputs = arange(bounds[0,0], bounds[0,1], 0.1)
```

---
## Step 19 — create a line plot of input vs result

```python
pyplot.plot(inputs, [objective([x]) for x in inputs], '--')
```

---
## Step 20 — draw a vertical line at the optimal input

```python
pyplot.axvline(x=[0.0], ls='--', color='red')
```

---
## Step 21 — plot the sample as black circles

```python
pyplot.plot(solutions, [objective(x) for x in solutions], 'o', color='black')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: hill climbing search of a one-dimensional objective function 是机器学习中的常用技术。  
  *hill climbing search of a one-dimensional objective function is a common technique in machine learning.*

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
# Plot Hillclimbing Samples / 20 Plot Hillclimbing Samples
# Complete Code / 完整代码
# ===============================

# hill climbing search of a one-dimensional objective function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# objective function
def objective(x):
	return x[0]**2.0

# hill climbing local search algorithm
def hillclimbing(objective, bounds, n_iterations, step_size):
	# generate an initial point
 # 获取长度 / Get length
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# evaluate the initial point
	solution_eval = objective(solution)
	# run the hill climb
	solutions = list()
 # 添加元素到列表末尾 / Append element to list end
	solutions.append(solution)
 # 生成整数序列 / Generate integer sequence
	for i in range(n_iterations):
		# take a step
  # 获取长度 / Get length
		candidate = solution + randn(len(bounds)) * step_size
		# evaluate candidate point
		candidte_eval = objective(candidate)
		# check if we should keep the new point
		if candidte_eval <= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# keep track of solutions
   # 添加元素到列表末尾 / Append element to list end
			solutions.append(solution)
			# report progress
   # 打印输出 / Print output
			print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval, solutions]

# seed the pseudorandom number generator
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(5)
# define range for input
bounds = asarray([[-5.0, 5.0]])
# define the total iterations
n_iterations = 1000
# define the maximum step size
step_size = 0.1
# perform the hill climbing search
best, score, solutions = hillclimbing(objective, bounds, n_iterations, step_size)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
inputs = arange(bounds[0,0], bounds[0,1], 0.1)
# create a line plot of input vs result
pyplot.plot(inputs, [objective([x]) for x in inputs], '--')
# draw a vertical line at the optimal input
pyplot.axvline(x=[0.0], ls='--', color='red')
# plot the sample as black circles
pyplot.plot(solutions, [objective(x) for x in solutions], 'o', color='black')
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 15 Summary / 第15章总结

## Theme / 主题: Chapter 15 / Chapter 15

This chapter contains **4 code files** demonstrating chapter 15.

本章包含 **4 个代码文件**，演示Chapter 15。

---
## Evolution / 演化路线

  1. `09_plot_convex_function.ipynb` — Plot Convex Function
  2. `13_hillclimbing.ipynb` — Hillclimbing
  3. `16_plot_hillclimbing_scores.ipynb` — Plot Hillclimbing Scores
  4. `20_plot_hillclimbing_samples.ipynb` — Plot Hillclimbing Samples

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 15) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 15）是机器学习流水线中的基础构建块。

---
