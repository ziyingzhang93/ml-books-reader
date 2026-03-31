# 优化深度学习 / Better Deep Learning
## Chapter 07

---

### Rectifier

# 01 — Rectifier / 01 Rectifier

**Chapter 07 — File 1 of 8 / 第07章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **demonstrate the rectified linear function**.

本脚本演示 **demonstrate the rectified linear function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — demonstrate the rectified linear function
rectified linear function

```python
def rectified(x):
	return max(0.0, x)
```

---
## Step 2 — demonstrate with a positive input

```python
x = 1.0
# 打印输出 / Print output
print('rectified(%.1f) is %.1f' % (x, rectified(x)))
x = 1000.0
# 打印输出 / Print output
print('rectified(%.1f) is %.1f' % (x, rectified(x)))
```

---
## Step 3 — demonstrate with a zero input

```python
x = 0.0
# 打印输出 / Print output
print('rectified(%.1f) is %.1f' % (x, rectified(x)))
```

---
## Step 4 — demonstrate with a negative input

```python
x = -1.0
# 打印输出 / Print output
print('rectified(%.1f) is %.1f' % (x, rectified(x)))
x = -1000.0
# 打印输出 / Print output
print('rectified(%.1f) is %.1f' % (x, rectified(x)))
```

---
## Learning Notes / 学习笔记

- **概念**: demonstrate the rectified linear function 是机器学习中的常用技术。  
  *demonstrate the rectified linear function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Rectifier / 01 Rectifier
# Complete Code / 完整代码
# ===============================

# demonstrate the rectified linear function

# rectified linear function
def rectified(x):
	return max(0.0, x)

# demonstrate with a positive input
x = 1.0
# 打印输出 / Print output
print('rectified(%.1f) is %.1f' % (x, rectified(x)))
x = 1000.0
# 打印输出 / Print output
print('rectified(%.1f) is %.1f' % (x, rectified(x)))
# demonstrate with a zero input
x = 0.0
# 打印输出 / Print output
print('rectified(%.1f) is %.1f' % (x, rectified(x)))
# demonstrate with a negative input
x = -1.0
# 打印输出 / Print output
print('rectified(%.1f) is %.1f' % (x, rectified(x)))
x = -1000.0
# 打印输出 / Print output
print('rectified(%.1f) is %.1f' % (x, rectified(x)))
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Rectifier Plot

# 02 — Rectifier Plot / 02 Rectifier Plot

**Chapter 07 — File 2 of 8 / 第07章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **plot inputs and outputs**.

本脚本演示 **plot inputs and outputs**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — plot inputs and outputs

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — rectified linear function

```python
def rectified(x):
	return max(0.0, x)
```

---
## Step 3 — define a series of inputs

```python
# 生成整数序列 / Generate integer sequence
series_in = [x for x in range(-10, 11)]
```

---
## Step 4 — calculate outputs for our inputs

```python
series_out = [rectified(x) for x in series_in]
```

---
## Step 5 — line plot of raw inputs to rectified outputs

```python
pyplot.plot(series_in, series_out)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot inputs and outputs 是机器学习中的常用技术。  
  *plot inputs and outputs is a common technique in machine learning.*

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
# Rectifier Plot / 02 Rectifier Plot
# Complete Code / 完整代码
# ===============================

# plot inputs and outputs
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# rectified linear function
def rectified(x):
	return max(0.0, x)

# define a series of inputs
# 生成整数序列 / Generate integer sequence
series_in = [x for x in range(-10, 11)]
# calculate outputs for our inputs
series_out = [rectified(x) for x in series_in]
# line plot of raw inputs to rectified outputs
pyplot.plot(series_in, series_out)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Problem

# 03 — Problem / 03 Problem

**Chapter 07 — File 3 of 8 / 第07章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **scatter plot of the circles dataset with points colored by class**.

本脚本演示 **scatter plot of the circles dataset with points colored by class**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — scatter plot of the circles dataset with points colored by class

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — generate circles

```python
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
```

---
## Step 3 — select indices of points with each class label

```python
# 生成整数序列 / Generate integer sequence
for i in range(2):
	samples_ix = where(y == i)
	pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1], label=str(i))
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: scatter plot of the circles dataset with points colored by class 是机器学习中的常用技术。  
  *scatter plot of the circles dataset with points colored by class is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Problem / 03 Problem
# Complete Code / 完整代码
# ===============================

# scatter plot of the circles dataset with points colored by class
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate circles
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# select indices of points with each class label
# 生成整数序列 / Generate integer sequence
for i in range(2):
	samples_ix = where(y == i)
	pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1], label=str(i))
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Mlp Tanh



---

### Mlp Tanh Deeper



---

### Mlp Relu Deeper



---

### Mlp Relu Deeper 15



---

### Mlp Relu Deeper 20



---

### Chapter Summary / 章节总结

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **8 code files** demonstrating chapter 07.

本章包含 **8 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `01_rectifier.ipynb` — Rectifier
  2. `02_rectifier_plot.ipynb` — Rectifier Plot
  3. `03_problem.ipynb` — Problem
  4. `04_mlp_tanh.ipynb` — Mlp Tanh
  5. `05_mlp_tanh_deeper.ipynb` — Mlp Tanh Deeper
  6. `06_mlp_relu_deeper.ipynb` — Mlp Relu Deeper
  7. `07_mlp_relu_deeper_15.ipynb` — Mlp Relu Deeper 15
  8. `07_mlp_relu_deeper_20.ipynb` — Mlp Relu Deeper 20

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---
