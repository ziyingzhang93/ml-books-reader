# 从零实现机器学习算法 / ML Algorithms from Scratch
## Chapter 01

---

### Chapter Summary / 章节总结

# Chapter 01 Summary / 第01章总结

## Theme / 主题: Chapter 01 / Chapter 01

This chapter contains **4 code files** demonstrating chapter 01.

本章包含 **4 个代码文件**，演示Chapter 01。

---
## Evolution / 演化路线

  1. `convert_string_to_float.ipynb` — Convert String To Float
  2. `convert_string_to_int.ipynb` — Convert String To Int
  3. `load_csv.ipynb` — Load Csv
  4. `load_csv_better.ipynb` — Load Csv Better

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 01) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 01）是机器学习流水线中的基础构建块。

---

### Convert String To Float

# 01 — Convert String To Float / Convert String To Float

**Chapter 01 — File 1 of 4 / 第01章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Example of converting string variables to float**.

本脚本演示 **Example of converting string variables to float**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of converting string variables to float

```python
from csv import reader
```

---
## Step 2 — Load a CSV file

```python
def load_csv(filename):
	dataset = list()
 # 打开文件（自动关闭） / Open file (auto-close)
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
   # 添加元素到列表末尾 / Append element to list end
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
## Step 4 — Load pima-indians-diabetes dataset

```python
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
# 打印输出 / Print output
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
# 打印输出 / Print output
print(dataset[0])
```

---
## Step 5 — convert string columns to float

```python
# 获取长度 / Get length
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# 打印输出 / Print output
print(dataset[0])
```

---
## Learning Notes / 学习笔记

- **概念**: Example of converting string variables to float 是机器学习中的常用技术。  
  *Example of converting string variables to float is a common technique in machine learning.*

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
# Convert String To Float / Convert String To Float
# Complete Code / 完整代码
# ===============================

# Example of converting string variables to float
from csv import reader

# Load a CSV file
def load_csv(filename):
	dataset = list()
 # 打开文件（自动关闭） / Open file (auto-close)
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
   # 添加元素到列表末尾 / Append element to list end
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Load pima-indians-diabetes dataset
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
# 打印输出 / Print output
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
# 打印输出 / Print output
print(dataset[0])
# convert string columns to float
# 获取长度 / Get length
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# 打印输出 / Print output
print(dataset[0])
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Convert String To Int



---

### Load Csv

# 01 — Load Csv / Load Csv

**Chapter 01 — File 3 of 4 / 第01章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Example of loading Pima Indians CSV dataset**.

本脚本演示 **Example of loading Pima Indians CSV dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of loading Pima Indians CSV dataset

```python
from csv import reader
```

---
## Step 2 — Load a CSV file

```python
def load_csv(filename):
	file = open(filename, "r")
	lines = reader(file)
	dataset = list(lines)
	return dataset
```

---
## Step 3 — Load dataset

```python
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
# 打印输出 / Print output
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of loading Pima Indians CSV dataset 是机器学习中的常用技术。  
  *Example of loading Pima Indians CSV dataset is a common technique in machine learning.*

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
# Load Csv / Load Csv
# Complete Code / 完整代码
# ===============================

# Example of loading Pima Indians CSV dataset
from csv import reader

# Load a CSV file
def load_csv(filename):
	file = open(filename, "r")
	lines = reader(file)
	dataset = list(lines)
	return dataset

# Load dataset
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
# 打印输出 / Print output
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Load Csv Better

# 01 — Load Csv Better / Load Csv Better

**Chapter 01 — File 4 of 4 / 第01章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Example of loading Pima Indians CSV dataset**.

本脚本演示 **Example of loading Pima Indians CSV dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of loading Pima Indians CSV dataset

```python
from csv import reader
```

---
## Step 2 — Load a CSV file

```python
def load_csv(filename):
	dataset = list()
 # 打开文件（自动关闭） / Open file (auto-close)
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
   # 添加元素到列表末尾 / Append element to list end
			dataset.append(row)
	return dataset
```

---
## Step 3 — Load dataset

```python
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
# 打印输出 / Print output
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of loading Pima Indians CSV dataset 是机器学习中的常用技术。  
  *Example of loading Pima Indians CSV dataset is a common technique in machine learning.*

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
# Load Csv Better / Load Csv Better
# Complete Code / 完整代码
# ===============================

# Example of loading Pima Indians CSV dataset
from csv import reader

# Load a CSV file
def load_csv(filename):
	dataset = list()
 # 打开文件（自动关闭） / Open file (auto-close)
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
   # 添加元素到列表末尾 / Append element to list end
			dataset.append(row)
	return dataset

# Load dataset
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
# 打印输出 / Print output
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
```

---
