# 从零实现ML算法
## Chapter 02

---

### Chapter Summary

# Chapter 02 Summary / 第02章总结

## Theme / 主题: Chapter 02 / Chapter 02

This chapter contains **6 code files** demonstrating chapter 02.

本章包含 **6 个代码文件**，演示Chapter 02。

---
## Evolution / 演化路线

  1. `dataset_min_max.ipynb` — Dataset Min Max
  2. `normalize_contrived_dataset.ipynb` — Normalize Contrived Dataset
  3. `normalize_diabetes.ipynb` — Normalize Diabetes
  4. `standardize_contrived_dataset.ipynb` — Standardize Contrived Dataset
  5. `standardize_diabetes.ipynb` — Standardize Diabetes
  6. `statistics_contrived_dataset.ipynb` — Statistics Contrived Dataset

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 02) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 02）是机器学习流水线中的基础构建块。

---

### Dataset Min Max

# 01 — Dataset Min Max / Dataset Min Max

**Chapter 02 — File 1 of 6 / 第02章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Find the min and max values for each column**.

本脚本演示 **Find the min and max values for each column**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Find the min and max values for each column

```python
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax
```

---
## Step 2 — Contrive small dataset

```python
dataset = [[50, 30], [20, 90]]
print(dataset)
```

---
## Step 3 — Calculate min and max for each column

```python
minmax = dataset_minmax(dataset)
print(minmax)
```

---
## Learning Notes / 学习笔记

- **概念**: Find the min and max values for each column 是机器学习中的常用技术。  
  *Find the min and max values for each column is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dataset Min Max / Dataset Min Max
# Complete Code / 完整代码
# ===============================

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# Contrive small dataset
dataset = [[50, 30], [20, 90]]
print(dataset)
# Calculate min and max for each column
minmax = dataset_minmax(dataset)
print(minmax)
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Standardize Contrived Dataset

# 01 — Standardize Contrived Dataset / Standardize Contrived Dataset

**Chapter 02 — File 4 of 6 / 第02章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Example of standardizing a contrived dataset**.

本脚本演示 **Example of standardizing a contrived dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of standardizing a contrived dataset

```python
from math import sqrt
```

---
## Step 2 — calculate column means

```python
def column_means(dataset):
	means = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		means[i] = sum(col_values) / float(len(dataset))
	return means
```

---
## Step 3 — calculate column standard deviations

```python
def column_stdevs(dataset, means):
	stdevs = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		variance = [pow(row[i]-means[i], 2) for row in dataset]
		stdevs[i] = sum(variance)
	stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
	return stdevs
```

---
## Step 4 — standardize dataset

```python
def standardize_dataset(dataset, means, stdevs):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - means[i]) / stdevs[i]
```

---
## Step 5 — Standardize dataset

```python
dataset = [[50, 30], [20, 90], [30, 50]]
print(dataset)
```

---
## Step 6 — Estimate mean and standard deviation

```python
means = column_means(dataset)
stdevs = column_stdevs(dataset, means)
print(means)
print(stdevs)
```

---
## Step 7 — standardize dataset

```python
standardize_dataset(dataset, means, stdevs)
print(dataset)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of standardizing a contrived dataset 是机器学习中的常用技术。  
  *Example of standardizing a contrived dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Standardize Contrived Dataset / Standardize Contrived Dataset
# Complete Code / 完整代码
# ===============================

# Example of standardizing a contrived dataset
from math import sqrt

# calculate column means
def column_means(dataset):
	means = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		means[i] = sum(col_values) / float(len(dataset))
	return means

# calculate column standard deviations
def column_stdevs(dataset, means):
	stdevs = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		variance = [pow(row[i]-means[i], 2) for row in dataset]
		stdevs[i] = sum(variance)
	stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
	return stdevs

# standardize dataset
def standardize_dataset(dataset, means, stdevs):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - means[i]) / stdevs[i]

# Standardize dataset
dataset = [[50, 30], [20, 90], [30, 50]]
print(dataset)
# Estimate mean and standard deviation
means = column_means(dataset)
stdevs = column_stdevs(dataset, means)
print(means)
print(stdevs)
# standardize dataset
standardize_dataset(dataset, means, stdevs)
print(dataset)
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Standardize Diabetes

# 01 — Standardize Diabetes / Standardize Diabetes

**Chapter 02 — File 5 of 6 / 第02章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Standardize the Diabetes Dataset**.

本脚本演示 **Standardize the Diabetes Dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Standardize the Diabetes Dataset

```python
from csv import reader
from math import sqrt
```

---
## Step 2 — Load a CSV file

```python
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
```

---
## Step 3 — Convert string column to float

```python
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
```

---
## Step 4 — calculate column means

```python
def column_means(dataset):
	means = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		means[i] = sum(col_values) / float(len(dataset))
	return means
```

---
## Step 5 — calculate column standard deviations

```python
def column_stdevs(dataset, means):
	stdevs = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		variance = [pow(row[i]-means[i], 2) for row in dataset]
		stdevs[i] = sum(variance)
	stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
	return stdevs
```

---
## Step 6 — standardize dataset

```python
def standardize_dataset(dataset, means, stdevs):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - means[i]) / stdevs[i]
```

---
## Step 7 — Load pima-indians-diabetes dataset

```python
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
```

---
## Step 8 — convert string columns to float

```python
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
print(dataset[0])
```

---
## Step 9 — Estimate mean and standard deviation

```python
means = column_means(dataset)
stdevs = column_stdevs(dataset, means)
```

---
## Step 10 — standardize dataset

```python
standardize_dataset(dataset, means, stdevs)
print(dataset[0])
```

---
## Learning Notes / 学习笔记

- **概念**: Standardize the Diabetes Dataset 是机器学习中的常用技术。  
  *Standardize the Diabetes Dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Standardize Diabetes / Standardize Diabetes
# Complete Code / 完整代码
# ===============================

# Standardize the Diabetes Dataset
from csv import reader
from math import sqrt

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# calculate column means
def column_means(dataset):
	means = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		means[i] = sum(col_values) / float(len(dataset))
	return means

# calculate column standard deviations
def column_stdevs(dataset, means):
	stdevs = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		variance = [pow(row[i]-means[i], 2) for row in dataset]
		stdevs[i] = sum(variance)
	stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
	return stdevs

# standardize dataset
def standardize_dataset(dataset, means, stdevs):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - means[i]) / stdevs[i]

# Load pima-indians-diabetes dataset
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
# convert string columns to float
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
print(dataset[0])
# Estimate mean and standard deviation
means = column_means(dataset)
stdevs = column_stdevs(dataset, means)
# standardize dataset
standardize_dataset(dataset, means, stdevs)
print(dataset[0])
```

---

➡️ **Next / 下一步**: File 6 of 6

---
