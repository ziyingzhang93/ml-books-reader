# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 09

---

### Random Search

# 01 — Random Search / 01 Random Search

**Chapter 09 — File 1 of 4 / 第09章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of random search for function optimization**.

本脚本演示 **example of random search for function optimization**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of random search for function optimization

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
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
## Step 4 — generate a random sample from the domain

```python
sample = r_min + rand(100) * (r_max - r_min)
```

---
## Step 5 — evaluate the sample

```python
sample_eval = objective(sample)
```

---
## Step 6 — locate the best solution

```python
best_ix = 0
# 获取长度 / Get length
for i in range(len(sample)):
	if sample_eval[i] < sample_eval[best_ix]:
		best_ix = i
```

---
## Step 7 — summarize best solution

```python
# 打印输出 / Print output
print('Best: f(%.5f) = %.5f' % (sample[best_ix], sample_eval[best_ix]))
```

---
## Learning Notes / 学习笔记

- **概念**: example of random search for function optimization 是机器学习中的常用技术。  
  *example of random search for function optimization is a common technique in machine learning.*

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
# Random Search / 01 Random Search
# Complete Code / 完整代码
# ===============================

# example of random search for function optimization
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand

# objective function
def objective(x):
	return x**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# generate a random sample from the domain
sample = r_min + rand(100) * (r_max - r_min)
# evaluate the sample
sample_eval = objective(sample)
# locate the best solution
best_ix = 0
# 获取长度 / Get length
for i in range(len(sample)):
	if sample_eval[i] < sample_eval[best_ix]:
		best_ix = i
# summarize best solution
# 打印输出 / Print output
print('Best: f(%.5f) = %.5f' % (sample[best_ix], sample_eval[best_ix]))
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Plot Random Search

# 02 — Plot Random Search / 02 Plot Random Search

**Chapter 09 — File 2 of 4 / 第09章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of random search for function optimization with plot**.

本脚本演示 **example of random search for function optimization with plot**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — example of random search for function optimization with plot

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
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
## Step 4 — generate a random sample from the domain

```python
sample = r_min + rand(100) * (r_max - r_min)
```

---
## Step 5 — evaluate the sample

```python
sample_eval = objective(sample)
```

---
## Step 6 — locate the best solution

```python
best_ix = 0
# 获取长度 / Get length
for i in range(len(sample)):
	if sample_eval[i] < sample_eval[best_ix]:
		best_ix = i
```

---
## Step 7 — summarize best solution

```python
# 打印输出 / Print output
print('Best: f(%.5f) = %.5f' % (sample[best_ix], sample_eval[best_ix]))
```

---
## Step 8 — sample input range uniformly at 0.1 increments

```python
# 生成整数序列 / Generate integer sequence
inputs = arange(r_min, r_max, 0.1)
```

---
## Step 9 — compute targets

```python
results = objective(inputs)
```

---
## Step 10 — create a line plot of input vs result

```python
pyplot.plot(inputs, results)
```

---
## Step 11 — plot the sample

```python
pyplot.scatter(sample, sample_eval)
```

---
## Step 12 — draw a vertical line at the best input

```python
pyplot.axvline(x=sample[best_ix], ls='--', color='red')
```

---
## Step 13 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of random search for function optimization with plot 是机器学习中的常用技术。  
  *example of random search for function optimization with plot is a common technique in machine learning.*

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
# Plot Random Search / 02 Plot Random Search
# Complete Code / 完整代码
# ===============================

# example of random search for function optimization with plot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# objective function
def objective(x):
	return x**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# generate a random sample from the domain
sample = r_min + rand(100) * (r_max - r_min)
# evaluate the sample
sample_eval = objective(sample)
# locate the best solution
best_ix = 0
# 获取长度 / Get length
for i in range(len(sample)):
	if sample_eval[i] < sample_eval[best_ix]:
		best_ix = i
# summarize best solution
# 打印输出 / Print output
print('Best: f(%.5f) = %.5f' % (sample[best_ix], sample_eval[best_ix]))
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
inputs = arange(r_min, r_max, 0.1)
# compute targets
results = objective(inputs)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# plot the sample
pyplot.scatter(sample, sample_eval)
# draw a vertical line at the best input
pyplot.axvline(x=sample[best_ix], ls='--', color='red')
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Grid Search



---

### Plot Grid Search

# 04 — Plot Grid Search / 04 Plot Grid Search

**Chapter 09 — File 4 of 4 / 第09章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of grid search for function optimization with plot**.

本脚本演示 **example of grid search for function optimization with plot**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — example of grid search for function optimization with plot

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid
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
## Step 4 — generate a grid sample from the domain

```python
sample = list()
step = 0.5
# 生成整数序列 / Generate integer sequence
for x in arange(r_min, r_max+step, step):
 # 生成整数序列 / Generate integer sequence
	for y in arange(r_min, r_max+step, step):
  # 添加元素到列表末尾 / Append element to list end
		sample.append([x,y])
```

---
## Step 5 — evaluate the sample

```python
sample_eval = [objective(x,y) for x,y in sample]
```

---
## Step 6 — locate the best solution

```python
best_ix = 0
# 获取长度 / Get length
for i in range(len(sample)):
	if sample_eval[i] < sample_eval[best_ix]:
		best_ix = i
```

---
## Step 7 — summarize best solution

```python
# 打印输出 / Print output
print('Best: f(%.5f,%.5f) = %.5f' % (sample[best_ix][0], sample[best_ix][1], sample_eval[best_ix]))
```

---
## Step 8 — sample input range uniformly at 0.1 increments

```python
# 生成整数序列 / Generate integer sequence
xaxis = arange(r_min, r_max, 0.1)
# 生成整数序列 / Generate integer sequence
yaxis = arange(r_min, r_max, 0.1)
```

---
## Step 9 — create a mesh from the axis

```python
x, y = meshgrid(xaxis, yaxis)
```

---
## Step 10 — compute targets

```python
results = objective(x, y)
```

---
## Step 11 — create a filled contour plot

```python
pyplot.contourf(x, y, results, levels=50, cmap='jet')
```

---
## Step 12 — plot the sample as black circles

```python
pyplot.plot([x for x,_ in sample], [y for _,y in sample], '.', color='black')
```

---
## Step 13 — draw the best result as a white star

```python
pyplot.plot(sample[best_ix][0], sample[best_ix][1], '*', color='white')
```

---
## Step 14 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of grid search for function optimization with plot 是机器学习中的常用技术。  
  *example of grid search for function optimization with plot is a common technique in machine learning.*

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
# Plot Grid Search / 04 Plot Grid Search
# Complete Code / 完整代码
# ===============================

# example of grid search for function optimization with plot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meshgrid
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# generate a grid sample from the domain
sample = list()
step = 0.5
# 生成整数序列 / Generate integer sequence
for x in arange(r_min, r_max+step, step):
 # 生成整数序列 / Generate integer sequence
	for y in arange(r_min, r_max+step, step):
  # 添加元素到列表末尾 / Append element to list end
		sample.append([x,y])
# evaluate the sample
sample_eval = [objective(x,y) for x,y in sample]
# locate the best solution
best_ix = 0
# 获取长度 / Get length
for i in range(len(sample)):
	if sample_eval[i] < sample_eval[best_ix]:
		best_ix = i
# summarize best solution
# 打印输出 / Print output
print('Best: f(%.5f,%.5f) = %.5f' % (sample[best_ix][0], sample[best_ix][1], sample_eval[best_ix]))
# sample input range uniformly at 0.1 increments
# 生成整数序列 / Generate integer sequence
xaxis = arange(r_min, r_max, 0.1)
# 生成整数序列 / Generate integer sequence
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a filled contour plot
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# plot the sample as black circles
pyplot.plot([x for x,_ in sample], [y for _,y in sample], '.', color='black')
# draw the best result as a white star
pyplot.plot(sample[best_ix][0], sample[best_ix][1], '*', color='white')
# show the plot
pyplot.show()
```

---

### Chapter Summary / 章节总结



---
