# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 11

---

### Plot Convex Function

# 03 — Plot Convex Function / 03 Plot Convex Function

**Chapter 11 — File 1 of 4 / 第11章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **plot a convex target function**.

本脚本演示 **plot a convex target function**。

---
## Step 1 — plot a convex target function

```python
from numpy import arange
from matplotlib import pyplot
```

---
## Step 2 — objective function

```python
def objective(x):
	return (5.0 + x)**2.0
```

---
## Step 3 — define range

```python
r_min, r_max = -10.0, 10.0
```

---
## Step 4 — prepare inputs

```python
inputs = arange(r_min, r_max, 0.1)
```

---
## Step 5 — compute targets

```python
targets = [objective(x) for x in inputs]
```

---
## Step 6 — plot inputs vs target

```python
pyplot.plot(inputs, targets, '--')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot a convex target function 是机器学习中的常用技术。  
  *plot a convex target function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Convex Function / 03 Plot Convex Function
# Complete Code / 完整代码
# ===============================

# plot a convex target function
from numpy import arange
from matplotlib import pyplot

# objective function
def objective(x):
	return (5.0 + x)**2.0

# define range
r_min, r_max = -10.0, 10.0
# prepare inputs
inputs = arange(r_min, r_max, 0.1)
# compute targets
targets = [objective(x) for x in inputs]
# plot inputs vs target
pyplot.plot(inputs, targets, '--')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Optimize Convex Function

# 07 — Optimize Convex Function / 优化

**Chapter 11 — File 2 of 4 / 第11章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **optimize convex objective function**.

本脚本演示 **optimize convex objective function**。

---
## Step 1 — optimize convex objective function

```python
from numpy import arange
from scipy.optimize import minimize_scalar
from matplotlib import pyplot
```

---
## Step 2 — objective function

```python
def objective(x):
	return (5.0 + x)**2.0
```

---
## Step 3 — minimize the function

```python
result = minimize_scalar(objective, method='brent')
```

---
## Step 4 — summarize the result

```python
opt_x, opt_y = result['x'], result['fun']
print('Optimal Input x: %.6f' % opt_x)
print('Optimal Output f(x): %.6f' % opt_y)
print('Total Evaluations n: %d' % result['nfev'])
```

---
## Step 5 — define the range

```python
r_min, r_max = -10.0, 10.0
```

---
## Step 6 — prepare inputs

```python
inputs = arange(r_min, r_max, 0.1)
```

---
## Step 7 — compute targets

```python
targets = [objective(x) for x in inputs]
```

---
## Step 8 — plot inputs vs target

```python
pyplot.plot(inputs, targets, '--')
```

---
## Step 9 — plot the optima

```python
pyplot.plot([opt_x], [opt_y], 's', color='r')
```

---
## Step 10 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: optimize convex objective function 是机器学习中的常用技术。  
  *optimize convex objective function is a common technique in machine learning.*

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

# optimize convex objective function
from numpy import arange
from scipy.optimize import minimize_scalar
from matplotlib import pyplot

# objective function
def objective(x):
	return (5.0 + x)**2.0

# minimize the function
result = minimize_scalar(objective, method='brent')
# summarize the result
opt_x, opt_y = result['x'], result['fun']
print('Optimal Input x: %.6f' % opt_x)
print('Optimal Output f(x): %.6f' % opt_y)
print('Total Evaluations n: %d' % result['nfev'])
# define the range
r_min, r_max = -10.0, 10.0
# prepare inputs
inputs = arange(r_min, r_max, 0.1)
# compute targets
targets = [objective(x) for x in inputs]
# plot inputs vs target
pyplot.plot(inputs, targets, '--')
# plot the optima
pyplot.plot([opt_x], [opt_y], 's', color='r')
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Plot Nonconvex Function

# 09 — Plot Nonconvex Function / 09 Plot Nonconvex Function

**Chapter 11 — File 3 of 4 / 第11章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **plot a non-convex univariate function**.

本脚本演示 **plot a non-convex univariate function**。

---
## Step 1 — plot a non-convex univariate function

```python
from numpy import arange
from matplotlib import pyplot
```

---
## Step 2 — objective function

```python
def objective(x):
	return (x - 2.0) * x * (x + 2.0)**2.0
```

---
## Step 3 — define range

```python
r_min, r_max = -3.0, 2.5
```

---
## Step 4 — prepare inputs

```python
inputs = arange(r_min, r_max, 0.1)
```

---
## Step 5 — compute targets

```python
targets = [objective(x) for x in inputs]
```

---
## Step 6 — plot inputs vs target

```python
pyplot.plot(inputs, targets, '--')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot a non-convex univariate function 是机器学习中的常用技术。  
  *plot a non-convex univariate function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Nonconvex Function / 09 Plot Nonconvex Function
# Complete Code / 完整代码
# ===============================

# plot a non-convex univariate function
from numpy import arange
from matplotlib import pyplot

# objective function
def objective(x):
	return (x - 2.0) * x * (x + 2.0)**2.0

# define range
r_min, r_max = -3.0, 2.5
# prepare inputs
inputs = arange(r_min, r_max, 0.1)
# compute targets
targets = [objective(x) for x in inputs]
# plot inputs vs target
pyplot.plot(inputs, targets, '--')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Optimize Nonconvex Function

# 10 — Optimize Nonconvex Function / 优化

**Chapter 11 — File 4 of 4 / 第11章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **optimize non-convex objective function**.

本脚本演示 **optimize non-convex objective function**。

---
## Step 1 — optimize non-convex objective function

```python
from numpy import arange
from scipy.optimize import minimize_scalar
from matplotlib import pyplot
```

---
## Step 2 — objective function

```python
def objective(x):
	return (x - 2.0) * x * (x + 2.0)**2.0
```

---
## Step 3 — minimize the function

```python
result = minimize_scalar(objective, method='brent')
```

---
## Step 4 — summarize the result

```python
opt_x, opt_y = result['x'], result['fun']
print('Optimal Input x: %.6f' % opt_x)
print('Optimal Output f(x): %.6f' % opt_y)
print('Total Evaluations n: %d' % result['nfev'])
```

---
## Step 5 — define the range

```python
r_min, r_max = -3.0, 2.5
```

---
## Step 6 — prepare inputs

```python
inputs = arange(r_min, r_max, 0.1)
```

---
## Step 7 — compute targets

```python
targets = [objective(x) for x in inputs]
```

---
## Step 8 — plot inputs vs target

```python
pyplot.plot(inputs, targets, '--')
```

---
## Step 9 — plot the optima

```python
pyplot.plot([opt_x], [opt_y], 's', color='r')
```

---
## Step 10 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: optimize non-convex objective function 是机器学习中的常用技术。  
  *optimize non-convex objective function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Optimize Nonconvex Function / 优化
# Complete Code / 完整代码
# ===============================

# optimize non-convex objective function
from numpy import arange
from scipy.optimize import minimize_scalar
from matplotlib import pyplot

# objective function
def objective(x):
	return (x - 2.0) * x * (x + 2.0)**2.0

# minimize the function
result = minimize_scalar(objective, method='brent')
# summarize the result
opt_x, opt_y = result['x'], result['fun']
print('Optimal Input x: %.6f' % opt_x)
print('Optimal Output f(x): %.6f' % opt_y)
print('Total Evaluations n: %d' % result['nfev'])
# define the range
r_min, r_max = -3.0, 2.5
# prepare inputs
inputs = arange(r_min, r_max, 0.1)
# compute targets
targets = [objective(x) for x in inputs]
# plot inputs vs target
pyplot.plot(inputs, targets, '--')
# plot the optima
pyplot.plot([opt_x], [opt_y], 's', color='r')
# show the plot
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **4 code files** demonstrating chapter 11.

本章包含 **4 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `03_plot_convex_function.ipynb` — Plot Convex Function
  2. `07_optimize_convex_function.ipynb` — Optimize Convex Function
  3. `09_plot_nonconvex_function.ipynb` — Plot Nonconvex Function
  4. `10_optimize_nonconvex_function.ipynb` — Optimize Nonconvex Function

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---
