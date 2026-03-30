# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 12

---

### Optimize Convex Function

# 08 — Optimize Convex Function / 优化

**Chapter 12 — File 1 of 4 / 第12章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **nelder-mead optimization of a convex function**.

本脚本演示 **nelder-mead optimization of a convex function**。

---
## Step 1 — nelder-mead optimization of a convex function

```python
from scipy.optimize import minimize
from numpy.random import rand
```

---
## Step 2 — objective function

```python
def objective(x):
	return x[0]**2.0 + x[1]**2.0
```

---
## Step 3 — define range for input

```python
r_min, r_max = -5.0, 5.0
```

---
## Step 4 — define the starting point as a random sample from the domain

```python
pt = r_min + rand(2) * (r_max - r_min)
```

---
## Step 5 — perform the search

```python
result = minimize(objective, pt, method='nelder-mead')
```

---
## Step 6 — summarize the result

```python
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
```

---
## Step 7 — evaluate solution

```python
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```

---
## Learning Notes / 学习笔记

- **概念**: nelder-mead optimization of a convex function 是机器学习中的常用技术。  
  *nelder-mead optimization of a convex function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Optimize Convex Function / 优化
# Complete Code / 完整代码
# ===============================

# nelder-mead optimization of a convex function
from scipy.optimize import minimize
from numpy.random import rand

# objective function
def objective(x):
	return x[0]**2.0 + x[1]**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# define the starting point as a random sample from the domain
pt = r_min + rand(2) * (r_max - r_min)
# perform the search
result = minimize(objective, pt, method='nelder-mead')
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Optimize Noisy Function

# 10 — Optimize Noisy Function / 优化

**Chapter 12 — File 2 of 4 / 第12章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **nelder-mead optimization of noisy one-dimensional convex function**.

本脚本演示 **nelder-mead optimization of noisy one-dimensional convex function**。

---
## Step 1 — nelder-mead optimization of noisy one-dimensional convex function

```python
from scipy.optimize import minimize
from numpy.random import rand
from numpy.random import randn
```

---
## Step 2 — objective function

```python
def objective(x):
	return (x + randn(len(x))*0.3)**2.0
```

---
## Step 3 — define range for input

```python
r_min, r_max = -5.0, 5.0
```

---
## Step 4 — define the starting point as a random sample from the domain

```python
pt = r_min + rand(1) * (r_max - r_min)
```

---
## Step 5 — perform the search

```python
result = minimize(objective, pt, method='nelder-mead')
```

---
## Step 6 — summarize the result

```python
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
```

---
## Step 7 — evaluate solution

```python
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```

---
## Learning Notes / 学习笔记

- **概念**: nelder-mead optimization of noisy one-dimensional convex function 是机器学习中的常用技术。  
  *nelder-mead optimization of noisy one-dimensional convex function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Optimize Noisy Function / 优化
# Complete Code / 完整代码
# ===============================

# nelder-mead optimization of noisy one-dimensional convex function
from scipy.optimize import minimize
from numpy.random import rand
from numpy.random import randn

# objective function
def objective(x):
	return (x + randn(len(x))*0.3)**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# define the starting point as a random sample from the domain
pt = r_min + rand(1) * (r_max - r_min)
# perform the search
result = minimize(objective, pt, method='nelder-mead')
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Ackley Function

# 11 — Ackley Function / 11 Ackley Function

**Chapter 12 — File 3 of 4 / 第12章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **ackley multimodal function**.

本脚本演示 **ackley multimodal function**。

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

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ackley Function / 11 Ackley Function
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

➡️ **Next / 下一步**: File 4 of 4

---

### Optimize Ackley

# 12 — Optimize Ackley / 优化

**Chapter 12 — File 4 of 4 / 第12章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **nelder-mead for multimodal function optimization**.

本脚本演示 **nelder-mead for multimodal function optimization**。

---
## Step 1 — nelder-mead for multimodal function optimization

```python
from scipy.optimize import minimize
from numpy.random import rand
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
```

---
## Step 2 — objective function

```python
def objective(v):
	x, y = v
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
```

---
## Step 3 — define range for input

```python
r_min, r_max = -5.0, 5.0
```

---
## Step 4 — define the starting point as a random sample from the domain

```python
pt = r_min + rand(2) * (r_max - r_min)
```

---
## Step 5 — perform the search

```python
result = minimize(objective, pt, method='nelder-mead')
```

---
## Step 6 — summarize the result

```python
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
```

---
## Step 7 — evaluate solution

```python
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```

---
## Learning Notes / 学习笔记

- **概念**: nelder-mead for multimodal function optimization 是机器学习中的常用技术。  
  *nelder-mead for multimodal function optimization is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Optimize Ackley / 优化
# Complete Code / 完整代码
# ===============================

# nelder-mead for multimodal function optimization
from scipy.optimize import minimize
from numpy.random import rand
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi

# objective function
def objective(v):
	x, y = v
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# define range for input
r_min, r_max = -5.0, 5.0
# define the starting point as a random sample from the domain
pt = r_min + rand(2) * (r_max - r_min)
# perform the search
result = minimize(objective, pt, method='nelder-mead')
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```

---

### Chapter Summary / 章节总结

# Chapter 12 Summary / 第12章总结

## Theme / 主题: Chapter 12 / Chapter 12

This chapter contains **4 code files** demonstrating chapter 12.

本章包含 **4 个代码文件**，演示Chapter 12。

---
## Evolution / 演化路线

  1. `08_optimize_convex_function.ipynb` — Optimize Convex Function
  2. `10_optimize_noisy_function.ipynb` — Optimize Noisy Function
  3. `11_ackley_function.ipynb` — Ackley Function
  4. `12_optimize_ackley.ipynb` — Optimize Ackley

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 12) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 12）是机器学习流水线中的基础构建块。

---
