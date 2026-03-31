# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 07

---

### Objective Function 1D



---

### Range Result



---

### Plot Function



---

### Scatter Plot

# 08 — Scatter Plot / 08 Scatter Plot

**Chapter 07 — File 4 of 13 / 第07章 — 第4个文件（共13个）**

---

## Summary / 总结

This script demonstrates **scatter plot of input vs result for a 1d objective function**.

本脚本演示 **scatter plot of input vs result for a 1d objective function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — scatter plot of input vs result for a 1d objective function

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
	return x**2.0
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
results = objective(inputs)
```

---
## Step 6 — create a scatter plot of input vs result

```python
pyplot.scatter(inputs, results)
```

---
## Step 7 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: scatter plot of input vs result for a 1d objective function 是机器学习中的常用技术。  
  *scatter plot of input vs result for a 1d objective function is a common technique in machine learning.*

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
# Scatter Plot / 08 Scatter Plot
# Complete Code / 完整代码
# ===============================

# scatter plot of input vs result for a 1d objective function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# objective function
def objective(x):
	return x**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
inputs = arange(r_min, r_max, 0.1)
# compute targets
results = objective(inputs)
# create a scatter plot of input vs result
pyplot.scatter(inputs, results)
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 13

---

### Plot With Optima

# 11 — Plot With Optima / 优化

**Chapter 07 — File 5 of 13 / 第07章 — 第5个文件（共13个）**

---

## Summary / 总结

This script demonstrates **line plot of input vs result for a 1d objective function and show optima**.

本脚本演示 **line plot of input vs result for a 1d objective function and show optima**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — line plot of input vs result for a 1d objective function and show optima

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
	return x**2.0
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
results = objective(inputs)
```

---
## Step 6 — create a line plot of input vs result

```python
pyplot.plot(inputs, results)
```

---
## Step 7 — define the known function optima

```python
optima_x = 0.0
optima_y = objective(optima_x)
```

---
## Step 8 — draw the function optima as a red square

```python
pyplot.plot([optima_x], [optima_y], 's', color='r')
```

---
## Step 9 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: line plot of input vs result for a 1d objective function and show optima 是机器学习中的常用技术。  
  *line plot of input vs result for a 1d objective function and show optima is a common technique in machine learning.*

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
# Plot With Optima / 优化
# Complete Code / 完整代码
# ===============================

# line plot of input vs result for a 1d objective function and show optima
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# objective function
def objective(x):
	return x**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
inputs = arange(r_min, r_max, 0.1)
# compute targets
results = objective(inputs)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# define the known function optima
optima_x = 0.0
optima_y = objective(optima_x)
# draw the function optima as a red square
pyplot.plot([optima_x], [optima_y], 's', color='r')
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 6 of 13

---

### Plot With Vertical

# 13 — Plot With Vertical / 13 Plot With Vertical

**Chapter 07 — File 6 of 13 / 第07章 — 第6个文件（共13个）**

---

## Summary / 总结

This script demonstrates **line plot of input vs result for a 1d objective function and show optima as line**.

本脚本演示 **line plot of input vs result for a 1d objective function and show optima as line**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — line plot of input vs result for a 1d objective function and show optima as line

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
	return x**2.0
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
results = objective(inputs)
```

---
## Step 6 — create a line plot of input vs result

```python
pyplot.plot(inputs, results)
```

---
## Step 7 — define the known function optima

```python
optima_x = 0.0
```

---
## Step 8 — draw a vertical line at the optimal input

```python
pyplot.axvline(x=optima_x, ls='--', color='red')
```

---
## Step 9 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: line plot of input vs result for a 1d objective function and show optima as line 是机器学习中的常用技术。  
  *line plot of input vs result for a 1d objective function and show optima as line is a common technique in machine learning.*

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
# Plot With Vertical / 13 Plot With Vertical
# Complete Code / 完整代码
# ===============================

# line plot of input vs result for a 1d objective function and show optima as line
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# objective function
def objective(x):
	return x**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
inputs = arange(r_min, r_max, 0.1)
# compute targets
results = objective(inputs)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# define the known function optima
optima_x = 0.0
# draw a vertical line at the optimal input
pyplot.axvline(x=optima_x, ls='--', color='red')
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 7 of 13

---

### Plot With Black Circles

# 16 — Plot With Black Circles / 16 Plot With Black Circles

**Chapter 07 — File 7 of 13 / 第07章 — 第7个文件（共13个）**

---

## Summary / 总结

This script demonstrates **line plot of domain for a 1d function with optima and algorithm sample**.

本脚本演示 **line plot of domain for a 1d function with optima and algorithm sample**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — line plot of domain for a 1d function with optima and algorithm sample

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
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
results = objective(inputs)
```

---
## Step 6 — simulate a sample made by an optimization algorithm

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
sample = r_min + rand(10) * (r_max - r_min)
```

---
## Step 7 — evaluate the sample

```python
sample_eval = objective(sample)
```

---
## Step 8 — create a line plot of input vs result

```python
pyplot.plot(inputs, results)
```

---
## Step 9 — define the known function optima

```python
optima_x = 0.0
```

---
## Step 10 — draw a vertical line at the optimal input

```python
pyplot.axvline(x=optima_x, ls='--', color='red')
```

---
## Step 11 — plot the sample as black circles

```python
pyplot.plot(sample, sample_eval, 'o', color='black')
```

---
## Step 12 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: line plot of domain for a 1d function with optima and algorithm sample 是机器学习中的常用技术。  
  *line plot of domain for a 1d function with optima and algorithm sample is a common technique in machine learning.*

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
# Plot With Black Circles / 16 Plot With Black Circles
# Complete Code / 完整代码
# ===============================

# line plot of domain for a 1d function with optima and algorithm sample
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# objective function
def objective(x):
	return x**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
inputs = arange(r_min, r_max, 0.1)
# compute targets
results = objective(inputs)
# simulate a sample made by an optimization algorithm
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
sample = r_min + rand(10) * (r_max - r_min)
# evaluate the sample
sample_eval = objective(sample)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# define the known function optima
optima_x = 0.0
# draw a vertical line at the optimal input
pyplot.axvline(x=optima_x, ls='--', color='red')
# plot the sample as black circles
pyplot.plot(sample, sample_eval, 'o', color='black')
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 8 of 13

---

### Objective Function 2D



---

### Meshgrid

# 21 — Meshgrid / 21 Meshgrid

**Chapter 07 — File 9 of 13 / 第07章 — 第9个文件（共13个）**

---

## Summary / 总结

This script demonstrates **sample 2d objective function**.

本脚本演示 **sample 2d objective function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — sample 2d objective function

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid
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
r_min, r_max = -5.0, 5.0
```

---
## Step 4 — sample input range uniformly at 0.1 increments

```python
# 生成整数序列 / Generate integer sequence
xaxis = arange(r_min, r_max, 0.1)
# 生成整数序列 / Generate integer sequence
yaxis = arange(r_min, r_max, 0.1)
```

---
## Step 5 — create a mesh from the axis

```python
x, y = meshgrid(xaxis, yaxis)
```

---
## Step 6 — summarize some of the input domain

```python
# 打印输出 / Print output
print(x[:5, :5])
```

---
## Step 7 — compute targets

```python
results = objective(x, y)
```

---
## Step 8 — summarize some of the results

```python
# 打印输出 / Print output
print(results[:5, :5])
```

---
## Step 9 — create a mapping of some inputs to some results

```python
# 生成整数序列 / Generate integer sequence
for i in range(5):
 # 打印输出 / Print output
	print('f(%.3f, %.3f) = %.3f' % (x[i,0], y[i,0], results[i,0]))
```

---
## Learning Notes / 学习笔记

- **概念**: sample 2d objective function 是机器学习中的常用技术。  
  *sample 2d objective function is a common technique in machine learning.*

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
# Meshgrid / 21 Meshgrid
# Complete Code / 完整代码
# ===============================

# sample 2d objective function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
xaxis = arange(r_min, r_max, 0.1)
# 生成整数序列 / Generate integer sequence
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# summarize some of the input domain
# 打印输出 / Print output
print(x[:5, :5])
# compute targets
results = objective(x, y)
# summarize some of the results
# 打印输出 / Print output
print(results[:5, :5])
# create a mapping of some inputs to some results
# 生成整数序列 / Generate integer sequence
for i in range(5):
 # 打印输出 / Print output
	print('f(%.3f, %.3f) = %.3f' % (x[i,0], y[i,0], results[i,0]))
```

---

➡️ **Next / 下一步**: File 10 of 13

---

### Contour Plot

# 23 — Contour Plot / 23 Contour Plot

**Chapter 07 — File 10 of 13 / 第07章 — 第10个文件（共13个）**

---

## Summary / 总结

This script demonstrates **contour plot for 2d objective function**.

本脚本演示 **contour plot for 2d objective function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — contour plot for 2d objective function

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
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
r_min, r_max = -5.0, 5.0
```

---
## Step 4 — sample input range uniformly at 0.1 increments

```python
# 生成整数序列 / Generate integer sequence
xaxis = arange(r_min, r_max, 0.1)
# 生成整数序列 / Generate integer sequence
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
## Step 7 — create a contour plot with 50 levels and jet color scheme

```python
pyplot.contour(x, y, results, 50, alpha=1.0, cmap='jet')
```

---
## Step 8 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: contour plot for 2d objective function 是机器学习中的常用技术。  
  *contour plot for 2d objective function is a common technique in machine learning.*

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
# Contour Plot / 23 Contour Plot
# Complete Code / 完整代码
# ===============================

# contour plot for 2d objective function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
xaxis = arange(r_min, r_max, 0.1)
# 生成整数序列 / Generate integer sequence
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a contour plot with 50 levels and jet color scheme
pyplot.contour(x, y, results, 50, alpha=1.0, cmap='jet')
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 11 of 13

---

### Contour With Optima

# 26 — Contour With Optima / 优化

**Chapter 07 — File 11 of 13 / 第07章 — 第11个文件（共13个）**

---

## Summary / 总结

This script demonstrates **filled contour plot for 2d objective function and show the optima**.

本脚本演示 **filled contour plot for 2d objective function and show the optima**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — filled contour plot for 2d objective function and show the optima

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
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
r_min, r_max = -5.0, 5.0
```

---
## Step 4 — sample input range uniformly at 0.1 increments

```python
# 生成整数序列 / Generate integer sequence
xaxis = arange(r_min, r_max, 0.1)
# 生成整数序列 / Generate integer sequence
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
## Step 7 — create a filled contour plot with 50 levels and jet color scheme

```python
pyplot.contourf(x, y, results, levels=50, cmap='jet')
```

---
## Step 8 — define the known function optima

```python
optima_x = [0.0, 0.0]
```

---
## Step 9 — draw the function optima as a white star

```python
pyplot.plot([optima_x[0]], [optima_x[1]], '*', color='white')
```

---
## Step 10 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: filled contour plot for 2d objective function and show the optima 是机器学习中的常用技术。  
  *filled contour plot for 2d objective function and show the optima is a common technique in machine learning.*

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
# Contour With Optima / 优化
# Complete Code / 完整代码
# ===============================

# filled contour plot for 2d objective function and show the optima
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
xaxis = arange(r_min, r_max, 0.1)
# 生成整数序列 / Generate integer sequence
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# define the known function optima
optima_x = [0.0, 0.0]
# draw the function optima as a white star
pyplot.plot([optima_x[0]], [optima_x[1]], '*', color='white')
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 12 of 13

---

### Contour With Sample

# 29 — Contour With Sample / 29 Contour With Sample

**Chapter 07 — File 12 of 13 / 第07章 — 第12个文件（共13个）**

---

## Summary / 总结

This script demonstrates **filled contour plot for 2d objective function and show the optima and sample**.

本脚本演示 **filled contour plot for 2d objective function and show the optima and sample**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — filled contour plot for 2d objective function and show the optima and sample

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
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
r_min, r_max = -5.0, 5.0
```

---
## Step 4 — sample input range uniformly at 0.1 increments

```python
# 生成整数序列 / Generate integer sequence
xaxis = arange(r_min, r_max, 0.1)
# 生成整数序列 / Generate integer sequence
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
## Step 7 — simulate a sample made by an optimization algorithm

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
sample_x = r_min + rand(10) * (r_max - r_min)
sample_y = r_min + rand(10) * (r_max - r_min)
```

---
## Step 8 — create a filled contour plot with 50 levels and jet color scheme

```python
pyplot.contourf(x, y, results, levels=50, cmap='jet')
```

---
## Step 9 — define the known function optima

```python
optima_x = [0.0, 0.0]
```

---
## Step 10 — draw the function optima as a white star

```python
pyplot.plot([optima_x[0]], [optima_x[1]], '*', color='white')
```

---
## Step 11 — plot the sample as black circles

```python
pyplot.plot(sample_x, sample_y, 'o', color='black')
```

---
## Step 12 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: filled contour plot for 2d objective function and show the optima and sample 是机器学习中的常用技术。  
  *filled contour plot for 2d objective function and show the optima and sample is a common technique in machine learning.*

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
# Contour With Sample / 29 Contour With Sample
# Complete Code / 完整代码
# ===============================

# filled contour plot for 2d objective function and show the optima and sample
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
xaxis = arange(r_min, r_max, 0.1)
# 生成整数序列 / Generate integer sequence
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# simulate a sample made by an optimization algorithm
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
sample_x = r_min + rand(10) * (r_max - r_min)
sample_y = r_min + rand(10) * (r_max - r_min)
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# define the known function optima
optima_x = [0.0, 0.0]
# draw the function optima as a white star
pyplot.plot([optima_x[0]], [optima_x[1]], '*', color='white')
# plot the sample as black circles
pyplot.plot(sample_x, sample_y, 'o', color='black')
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 13 of 13

---

### Surface Plot

# 31 — Surface Plot / 人脸识别

**Chapter 07 — File 13 of 13 / 第07章 — 第13个文件（共13个）**

---

## Summary / 总结

This script demonstrates **surface plot for 2d objective function**.

本脚本演示 **surface plot for 2d objective function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — surface plot for 2d objective function

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
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
## Step 3 — define range for input

```python
r_min, r_max = -5.0, 5.0
```

---
## Step 4 — sample input range uniformly at 0.1 increments

```python
# 生成整数序列 / Generate integer sequence
xaxis = arange(r_min, r_max, 0.1)
# 生成整数序列 / Generate integer sequence
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

- **概念**: surface plot for 2d objective function 是机器学习中的常用技术。  
  *surface plot for 2d objective function is a common technique in machine learning.*

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
# Surface Plot / 人脸识别
# Complete Code / 完整代码
# ===============================

# surface plot for 2d objective function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
xaxis = arange(r_min, r_max, 0.1)
# 生成整数序列 / Generate integer sequence
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

### Chapter Summary / 章节总结



---
