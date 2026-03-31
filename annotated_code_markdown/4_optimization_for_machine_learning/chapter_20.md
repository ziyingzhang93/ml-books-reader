# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 20

---

### Plot Function

# 14 — Plot Function / 14 Plot Function

**Chapter 20 — File 1 of 5 / 第20章 — 第1个文件（共5个）**

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
# Plot Function / 14 Plot Function
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

➡️ **Next / 下一步**: File 2 of 5

---

### Plot Temperature

# 15 — Plot Temperature / 15 Plot Temperature

**Chapter 20 — File 2 of 5 / 第20章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **explore temperature vs algorithm iteration for simulated annealing**.

本脚本演示 **explore temperature vs algorithm iteration for simulated annealing**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — explore temperature vs algorithm iteration for simulated annealing

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — total iterations of algorithm

```python
iterations = 100
```

---
## Step 3 — initial temperature

```python
initial_temp = 10
```

---
## Step 4 — array of iterations from 0 to iterations - 1

```python
# 生成整数序列 / Generate integer sequence
iterations = [i for i in range(iterations)]
```

---
## Step 5 — temperatures for each iterations

```python
temperatures = [initial_temp/float(i + 1) for i in iterations]
```

---
## Step 6 — plot iterations vs temperatures

```python
pyplot.plot(iterations, temperatures)
pyplot.xlabel('Iteration')
pyplot.ylabel('Temperature')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: explore temperature vs algorithm iteration for simulated annealing 是机器学习中的常用技术。  
  *explore temperature vs algorithm iteration for simulated annealing is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Temperature / 15 Plot Temperature
# Complete Code / 完整代码
# ===============================

# explore temperature vs algorithm iteration for simulated annealing
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# total iterations of algorithm
iterations = 100
# initial temperature
initial_temp = 10
# array of iterations from 0 to iterations - 1
# 生成整数序列 / Generate integer sequence
iterations = [i for i in range(iterations)]
# temperatures for each iterations
temperatures = [initial_temp/float(i + 1) for i in iterations]
# plot iterations vs temperatures
pyplot.plot(iterations, temperatures)
pyplot.xlabel('Iteration')
pyplot.ylabel('Temperature')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Plot Acceptance

# 16 — Plot Acceptance / 16 Plot Acceptance

**Chapter 20 — File 3 of 5 / 第20章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **explore metropolis acceptance criterion for simulated annealing**.

本脚本演示 **explore metropolis acceptance criterion for simulated annealing**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — explore metropolis acceptance criterion for simulated annealing

```python
from math import exp
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — total iterations of algorithm

```python
iterations = 100
```

---
## Step 3 — initial temperature

```python
initial_temp = 10
```

---
## Step 4 — array of iterations from 0 to iterations - 1

```python
# 生成整数序列 / Generate integer sequence
iterations = [i for i in range(iterations)]
```

---
## Step 5 — temperatures for each iterations

```python
temperatures = [initial_temp/float(i + 1) for i in iterations]
```

---
## Step 6 — metropolis acceptance criterion

```python
differences = [0.01, 0.1, 1.0]
for d in differences:
	metropolis = [exp(-d/t) for t in temperatures]
```

---
## Step 7 — plot iterations vs metropolis

```python
label = 'diff=%.2f' % d
	pyplot.plot(iterations, metropolis, label=label)
```

---
## Step 8 — inalize plot

```python
pyplot.xlabel('Iteration')
pyplot.ylabel('Metropolis Criterion')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: explore metropolis acceptance criterion for simulated annealing 是机器学习中的常用技术。  
  *explore metropolis acceptance criterion for simulated annealing is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Acceptance / 16 Plot Acceptance
# Complete Code / 完整代码
# ===============================

# explore metropolis acceptance criterion for simulated annealing
from math import exp
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# total iterations of algorithm
iterations = 100
# initial temperature
initial_temp = 10
# array of iterations from 0 to iterations - 1
# 生成整数序列 / Generate integer sequence
iterations = [i for i in range(iterations)]
# temperatures for each iterations
temperatures = [initial_temp/float(i + 1) for i in iterations]
# metropolis acceptance criterion
differences = [0.01, 0.1, 1.0]
for d in differences:
	metropolis = [exp(-d/t) for t in temperatures]
	# plot iterations vs metropolis
	label = 'diff=%.2f' % d
	pyplot.plot(iterations, metropolis, label=label)
# inalize plot
pyplot.xlabel('Iteration')
pyplot.ylabel('Metropolis Criterion')
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Simulated Annealing



---

### Plot Simulated Annealing



---

### Chapter Summary / 章节总结

# Chapter 20 Summary / 第20章总结

## Theme / 主题: Chapter 20 / Chapter 20

This chapter contains **5 code files** demonstrating chapter 20.

本章包含 **5 个代码文件**，演示Chapter 20。

---
## Evolution / 演化路线

  1. `14_plot_function.ipynb` — Plot Function
  2. `15_plot_temperature.ipynb` — Plot Temperature
  3. `16_plot_acceptance.ipynb` — Plot Acceptance
  4. `20_simulated_annealing.ipynb` — Simulated Annealing
  5. `23_plot_simulated_annealing.ipynb` — Plot Simulated Annealing

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 20) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 20）是机器学习流水线中的基础构建块。

---
