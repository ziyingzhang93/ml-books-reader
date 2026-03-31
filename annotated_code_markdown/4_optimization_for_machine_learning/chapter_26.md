# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 26

---

### 3D Plot

# 02 — 3D Plot / 02 3D Plot

**Chapter 26 — File 1 of 4 / 第26章 — 第1个文件（共4个）**

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

**Chapter 26 — File 2 of 4 / 第26章 — 第2个文件（共4个）**

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

### Adam

# 18 — Adam / 18 Adam

**Chapter 26 — File 3 of 4 / 第26章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **gradient descent optimization with adam for a two-dimensional test function**.

本脚本演示 **gradient descent optimization with adam for a two-dimensional test function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — gradient descent optimization with adam for a two-dimensional test function

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
## Step 4 — gradient descent algorithm with adam

```python
def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
```

---
## Step 5 — generate an initial point

```python
# 获取长度 / Get length
x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	score = objective(x[0], x[1])
```

---
## Step 6 — initialize first and second moments

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
m = [0.0 for _ in range(bounds.shape[0])]
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	v = [0.0 for _ in range(bounds.shape[0])]
```

---
## Step 7 — run the gradient descent updates

```python
# 生成整数序列 / Generate integer sequence
for t in range(n_iter):
```

---
## Step 8 — calculate gradient g(t)

```python
g = derivative(x[0], x[1])
```

---
## Step 9 — build a solution one variable at a time

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for i in range(x.shape[0]):
```

---
## Step 10 — m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)

```python
m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
```

---
## Step 11 — v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2

```python
v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
```

---
## Step 12 — mhat(t) = m(t) / (1 - beta1(t))

```python
mhat = m[i] / (1.0 - beta1**(t+1))
```

---
## Step 13 — vhat(t) = v(t) / (1 - beta2(t))

```python
vhat = v[i] / (1.0 - beta2**(t+1))
```

---
## Step 14 — x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)

```python
x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)
```

---
## Step 15 — evaluate candidate point

```python
score = objective(x[0], x[1])
```

---
## Step 16 — report progress

```python
# 打印输出 / Print output
print('>%d f(%s) = %.5f' % (t, x, score))
	return [x, score]
```

---
## Step 17 — seed the pseudo random number generator

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 18 — define range for input

```python
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
```

---
## Step 19 — define the total iterations

```python
n_iter = 60
```

---
## Step 20 — steps size

```python
alpha = 0.02
```

---
## Step 21 — factor for average gradient

```python
beta1 = 0.8
```

---
## Step 22 — factor for average squared gradient

```python
beta2 = 0.999
```

---
## Step 23 — perform the gradient descent search with adam

```python
best, score = adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
```

---
## Learning Notes / 学习笔记

- **概念**: gradient descent optimization with adam for a two-dimensional test function 是机器学习中的常用技术。  
  *gradient descent optimization with adam for a two-dimensional test function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Adam / 18 Adam
# Complete Code / 完整代码
# ===============================

# gradient descent optimization with adam for a two-dimensional test function
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

# gradient descent algorithm with adam
def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
	# generate an initial point
 # 获取长度 / Get length
	x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	score = objective(x[0], x[1])
	# initialize first and second moments
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	m = [0.0 for _ in range(bounds.shape[0])]
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	v = [0.0 for _ in range(bounds.shape[0])]
	# run the gradient descent updates
 # 生成整数序列 / Generate integer sequence
	for t in range(n_iter):
		# calculate gradient g(t)
		g = derivative(x[0], x[1])
		# build a solution one variable at a time
  # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
		for i in range(x.shape[0]):
			# m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
			m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
			# v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
			v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
			# mhat(t) = m(t) / (1 - beta1(t))
			mhat = m[i] / (1.0 - beta1**(t+1))
			# vhat(t) = v(t) / (1 - beta2(t))
			vhat = v[i] / (1.0 - beta2**(t+1))
			# x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
			x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)
		# evaluate candidate point
		score = objective(x[0], x[1])
		# report progress
  # 打印输出 / Print output
		print('>%d f(%s) = %.5f' % (t, x, score))
	return [x, score]

# seed the pseudo random number generator
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 60
# steps size
alpha = 0.02
# factor for average gradient
beta1 = 0.8
# factor for average squared gradient
beta2 = 0.999
# perform the gradient descent search with adam
best, score = adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Plot Adam

# 23 — Plot Adam / 23 Plot Adam

**Chapter 26 — File 4 of 4 / 第26章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of plotting the adam search on a contour plot of the test function**.

本脚本演示 **example of plotting the adam search on a contour plot of the test function**。

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
## Step 1 — example of plotting the adam search on a contour plot of the test function

```python
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
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
## Step 3 — derivative of objective function

```python
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])
```

---
## Step 4 — gradient descent algorithm with adam

```python
def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
	solutions = list()
```

---
## Step 5 — generate an initial point

```python
# 获取长度 / Get length
x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	score = objective(x[0], x[1])
```

---
## Step 6 — initialize first and second moments

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
m = [0.0 for _ in range(bounds.shape[0])]
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	v = [0.0 for _ in range(bounds.shape[0])]
```

---
## Step 7 — run the gradient descent updates

```python
# 生成整数序列 / Generate integer sequence
for t in range(n_iter):
```

---
## Step 8 — calculate gradient g(t)

```python
g = derivative(x[0], x[1])
```

---
## Step 9 — build a solution one variable at a time

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for i in range(bounds.shape[0]):
```

---
## Step 10 — m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)

```python
m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
```

---
## Step 11 — v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2

```python
v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
```

---
## Step 12 — mhat(t) = m(t) / (1 - beta1(t))

```python
mhat = m[i] / (1.0 - beta1**(t+1))
```

---
## Step 13 — vhat(t) = v(t) / (1 - beta2(t))

```python
vhat = v[i] / (1.0 - beta2**(t+1))
```

---
## Step 14 — x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + ep)

```python
x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)
```

---
## Step 15 — evaluate candidate point

```python
score = objective(x[0], x[1])
```

---
## Step 16 — keep track of solutions

```python
# 添加元素到列表末尾 / Append element to list end
solutions.append(x.copy())
```

---
## Step 17 — report progress

```python
# 打印输出 / Print output
print('>%d f(%s) = %.5f' % (t, x, score))
	return solutions
```

---
## Step 18 — seed the pseudo random number generator

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 19 — define range for input

```python
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
```

---
## Step 20 — define the total iterations

```python
n_iter = 60
```

---
## Step 21 — steps size

```python
alpha = 0.02
```

---
## Step 22 — factor for average gradient

```python
beta1 = 0.8
```

---
## Step 23 — factor for average squared gradient

```python
beta2 = 0.999
```

---
## Step 24 — perform the gradient descent search with adam

```python
solutions = adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)
```

---
## Step 25 — sample input range uniformly at 0.1 increments

```python
# 生成整数序列 / Generate integer sequence
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
# 生成整数序列 / Generate integer sequence
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

- **概念**: example of plotting the adam search on a contour plot of the test function 是机器学习中的常用技术。  
  *example of plotting the adam search on a contour plot of the test function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Adam / 23 Plot Adam
# Complete Code / 完整代码
# ===============================

# example of plotting the adam search on a contour plot of the test function
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# derivative of objective function
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])

# gradient descent algorithm with adam
def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
	solutions = list()
	# generate an initial point
 # 获取长度 / Get length
	x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	score = objective(x[0], x[1])
	# initialize first and second moments
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	m = [0.0 for _ in range(bounds.shape[0])]
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	v = [0.0 for _ in range(bounds.shape[0])]
	# run the gradient descent updates
 # 生成整数序列 / Generate integer sequence
	for t in range(n_iter):
		# calculate gradient g(t)
		g = derivative(x[0], x[1])
		# build a solution one variable at a time
  # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
		for i in range(bounds.shape[0]):
			# m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
			m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
			# v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
			v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
			# mhat(t) = m(t) / (1 - beta1(t))
			mhat = m[i] / (1.0 - beta1**(t+1))
			# vhat(t) = v(t) / (1 - beta2(t))
			vhat = v[i] / (1.0 - beta2**(t+1))
			# x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + ep)
			x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)
		# evaluate candidate point
		score = objective(x[0], x[1])
		# keep track of solutions
  # 添加元素到列表末尾 / Append element to list end
		solutions.append(x.copy())
		# report progress
  # 打印输出 / Print output
		print('>%d f(%s) = %.5f' % (t, x, score))
	return solutions

# seed the pseudo random number generator
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 60
# steps size
alpha = 0.02
# factor for average gradient
beta1 = 0.8
# factor for average squared gradient
beta2 = 0.999
# perform the gradient descent search with adam
solutions = adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)
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
# plot the sample as black circles
solutions = asarray(solutions)
pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
# show the plot
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 26 Summary / 第26章总结

## Theme / 主题: Chapter 26 / Chapter 26

This chapter contains **4 code files** demonstrating chapter 26.

本章包含 **4 个代码文件**，演示Chapter 26。

---
## Evolution / 演化路线

  1. `02_3d_plot.ipynb` — 3D Plot
  2. `03_contour_plot.ipynb` — Contour Plot
  3. `18_adam.ipynb` — Adam
  4. `23_plot_adam.ipynb` — Plot Adam

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 26) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 26）是机器学习流水线中的基础构建块。

---
