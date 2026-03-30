# DL时间序列
## Chapter 06

---

### Transform Univariate 2D 3D

# 02 — Transform Univariate 2D 3D / 数据变换

**Chapter 06 — File 2 of 7 / 第06章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **transform univariate 2d to 3d**.

本脚本演示 **transform univariate 2d to 3d**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — transform univariate 2d to 3d

```python
from numpy import array
```

---
## Step 2 — split a univariate sequence into samples

```python
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps
```

---
## Step 4 — check if we are beyond the sequence

```python
if end_ix > len(sequence)-1:
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define univariate time series

```python
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(series.shape)
```

---
## Step 7 — transform to a supervised learning problem

```python
X, y = split_sequence(series, 3)
print(X.shape, y.shape)
```

---
## Step 8 — transform input from [samples, features] to [samples, timesteps, features]

```python
X = X.reshape((X.shape[0], X.shape[1], 1))
print(X.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: transform univariate 2d to 3d 是机器学习中的常用技术。  
  *transform univariate 2d to 3d is a common technique in machine learning.*

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
# Transform Univariate 2D 3D / 数据变换
# Complete Code / 完整代码
# ===============================

# transform univariate 2d to 3d
from numpy import array

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define univariate time series
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(series.shape)
# transform to a supervised learning problem
X, y = split_sequence(series, 3)
print(X.shape, y.shape)
# transform input from [samples, features] to [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
print(X.shape)
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Example Drop Time

# 04 — Example Drop Time / 04 Example Drop Time

**Chapter 06 — File 4 of 7 / 第06章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of dropping the time dimension from the dataset**.

本脚本演示 **example of dropping the time dimension from the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of dropping the time dimension from the dataset

```python
from numpy import array
```

---
## Step 2 — define the dataset

```python
data = list()
n = 5000
for i in range(n):
	data.append([i+1, (i+1)*10])
data = array(data)
```

---
## Step 3 — drop time

```python
data = data[:, 1]
print(data.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: example of dropping the time dimension from the dataset 是机器学习中的常用技术。  
  *example of dropping the time dimension from the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example Drop Time / 04 Example Drop Time
# Complete Code / 完整代码
# ===============================

# example of dropping the time dimension from the dataset
from numpy import array

# define the dataset
data = list()
n = 5000
for i in range(n):
	data.append([i+1, (i+1)*10])
data = array(data)
# drop time
data = data[:, 1]
print(data.shape)
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Chapter Summary

# Chapter 06 Summary / 第06章总结

## Theme / 主题: Chapter 06 / Chapter 06

This chapter contains **7 code files** demonstrating chapter 06.

本章包含 **7 个代码文件**，演示Chapter 06。

---
## Evolution / 演化路线

  1. `01_time_series_to_supervised.ipynb` — Time Series To Supervised
  2. `02_transform_univariate_2d_3d.ipynb` — Transform Univariate 2D 3D
  3. `03_example_load_data.ipynb` — Example Load Data
  4. `04_example_drop_time.ipynb` — Example Drop Time
  5. `05_example_split_subsequences.ipynb` — Example Split Subsequences
  6. `06_example_create_array.ipynb` — Example Create Array
  7. `07_example_reshape_3d.ipynb` — Example Reshape 3D

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 06) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 06）是机器学习流水线中的基础构建块。

---
