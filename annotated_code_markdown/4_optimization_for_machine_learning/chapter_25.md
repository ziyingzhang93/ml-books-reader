# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 25

---

### 3D Plot

# 02 — 3D Plot / 02 3D Plot

**Chapter 25 — File 1 of 4 / 第25章 — 第1个文件（共4个）**

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
r_min, r_max = -1.0, 1.0
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
r_min, r_max = -1.0, 1.0
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

➡️ **Next / 下一步**: File 2 of 4

---

### Contour Plot

# 03 — Contour Plot / 03 Contour Plot

**Chapter 25 — File 2 of 4 / 第25章 — 第2个文件（共4个）**

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
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
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
```

---
## Step 4 — sample input range uniformly at 0.1 increments

```python
# 生成整数序列 / Generate integer sequence
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
# 生成整数序列 / Generate integer sequence
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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
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
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
# 生成整数序列 / Generate integer sequence
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

### Adadelta

# 18 — Adadelta / 18 Adadelta

**Chapter 25 — File 3 of 4 / 第25章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **gradient descent optimization with adadelta for a two-dimensional test function**.

本脚本演示 **gradient descent optimization with adadelta for a two-dimensional test function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — gradient descent optimization with adadelta for a two-dimensional test function

```python
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
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
## Step 4 — gradient descent algorithm with adadelta

```python
def adadelta(objective, derivative, bounds, n_iter, rho, ep=1e-3):
```

---
## Step 5 — generate an initial point

```python
# 获取长度 / Get length
solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
```

---
## Step 6 — list of the average square gradients for each variable

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
```

---
## Step 7 — list of the average parameter updates

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
sq_para_avg = [0.0 for _ in range(bounds.shape[0])]
```

---
## Step 8 — run the gradient descent

```python
# 生成整数序列 / Generate integer sequence
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
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
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
## Step 13 — build a solution one variable at a time

```python
new_solution = list()
  # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
		for i in range(solution.shape[0]):
```

---
## Step 14 — calculate the step size for this variable

```python
alpha = (ep + sqrt(sq_para_avg[i])) / (ep + sqrt(sq_grad_avg[i]))
```

---
## Step 15 — calculate the change

```python
change = alpha * gradient[i]
```

---
## Step 16 — update the moving average of squared parameter changes

```python
sq_para_avg[i] = (sq_para_avg[i] * rho) + (change**2.0 * (1.0-rho))
```

---
## Step 17 — calculate the new position in this variable

```python
value = solution[i] - change
```

---
## Step 18 — store this variable

```python
# 添加元素到列表末尾 / Append element to list end
new_solution.append(value)
```

---
## Step 19 — evaluate candidate point

```python
solution = asarray(new_solution)
		solution_eval = objective(solution[0], solution[1])
```

---
## Step 20 — report progress

```python
# 打印输出 / Print output
print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
	return [solution, solution_eval]
```

---
## Step 21 — seed the pseudo random number generator

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 22 — define range for input

```python
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
```

---
## Step 23 — define the total iterations

```python
n_iter = 120
```

---
## Step 24 — momentum for adadelta

```python
rho = 0.99
```

---
## Step 25 — perform the gradient descent search with adadelta

```python
best, score = adadelta(objective, derivative, bounds, n_iter, rho)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
```

---
## Learning Notes / 学习笔记

- **概念**: gradient descent optimization with adadelta for a two-dimensional test function 是机器学习中的常用技术。  
  *gradient descent optimization with adadelta for a two-dimensional test function is a common technique in machine learning.*

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
# Adadelta / 18 Adadelta
# Complete Code / 完整代码
# ===============================

# gradient descent optimization with adadelta for a two-dimensional test function
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# derivative of objective function
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])

# gradient descent algorithm with adadelta
def adadelta(objective, derivative, bounds, n_iter, rho, ep=1e-3):
	# generate an initial point
 # 获取长度 / Get length
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# list of the average square gradients for each variable
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
	# list of the average parameter updates
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	sq_para_avg = [0.0 for _ in range(bounds.shape[0])]
	# run the gradient descent
 # 生成整数序列 / Generate integer sequence
	for it in range(n_iter):
		# calculate gradient
		gradient = derivative(solution[0], solution[1])
		# update the average of the squared partial derivatives
  # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
		for i in range(gradient.shape[0]):
			# calculate the squared gradient
			sg = gradient[i]**2.0
			# update the moving average of the squared gradient
			sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (sg * (1.0-rho))
		# build a solution one variable at a time
		new_solution = list()
  # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
		for i in range(solution.shape[0]):
			# calculate the step size for this variable
			alpha = (ep + sqrt(sq_para_avg[i])) / (ep + sqrt(sq_grad_avg[i]))
			# calculate the change
			change = alpha * gradient[i]
			# update the moving average of squared parameter changes
			sq_para_avg[i] = (sq_para_avg[i] * rho) + (change**2.0 * (1.0-rho))
			# calculate the new position in this variable
			value = solution[i] - change
			# store this variable
   # 添加元素到列表末尾 / Append element to list end
			new_solution.append(value)
		# evaluate candidate point
		solution = asarray(new_solution)
		solution_eval = objective(solution[0], solution[1])
		# report progress
  # 打印输出 / Print output
		print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
	return [solution, solution_eval]

# seed the pseudo random number generator
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 120
# momentum for adadelta
rho = 0.99
# perform the gradient descent search with adadelta
best, score = adadelta(objective, derivative, bounds, n_iter, rho)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Plot Adadelta



---

### Chapter Summary / 章节总结

# Chapter 25 Summary / 第25章总结

## Theme / 主题: Chapter 25 / Chapter 25

This chapter contains **4 code files** demonstrating chapter 25.

本章包含 **4 个代码文件**，演示Chapter 25。

---
## Evolution / 演化路线

  1. `02_3d_plot.ipynb` — 3D Plot
  2. `03_contour_plot.ipynb` — Contour Plot
  3. `18_adadelta.ipynb` — Adadelta
  4. `23_plot_adadelta.ipynb` — Plot Adadelta

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 25) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 25）是机器学习流水线中的基础构建块。

---
