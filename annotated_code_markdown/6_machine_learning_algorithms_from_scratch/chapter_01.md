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
## Step 1 — Example of converting string variables to float

```python
from csv import reader
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
## Step 4 — Load pima-indians-diabetes dataset

```python
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
print(dataset[0])
```

---
## Step 5 — convert string columns to float

```python
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
print(dataset[0])
```

---
## Learning Notes / 学习笔记

- **概念**: Example of converting string variables to float 是机器学习中的常用技术。  
  *Example of converting string variables to float is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

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

# Load pima-indians-diabetes dataset
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
print(dataset[0])
# convert string columns to float
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
print(dataset[0])
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Convert String To Int

# 01 — Convert String To Int / Convert String To Int

**Chapter 01 — File 2 of 4 / 第01章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Example of integer encoding string class values**.

本脚本演示 **Example of integer encoding string class values**。

---
## Step 1 — Example of integer encoding string class values

```python
from csv import reader
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
## Step 4 — Convert string column to integer

```python
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
```

---
## Step 5 — Load iris dataset

```python
filename = 'iris.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
print(dataset[0])
```

---
## Step 6 — convert string columns to float

```python
for i in range(4):
	str_column_to_float(dataset, i)
```

---
## Step 7 — convert class column to int

```python
lookup = str_column_to_int(dataset, 4)
print(dataset[0])
print(lookup)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of integer encoding string class values 是机器学习中的常用技术。  
  *Example of integer encoding string class values is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Convert String To Int / Convert String To Int
# Complete Code / 完整代码
# ===============================

# Example of integer encoding string class values
from csv import reader

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

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Load iris dataset
filename = 'iris.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
print(dataset[0])
# convert string columns to float
for i in range(4):
	str_column_to_float(dataset, i)
# convert class column to int
lookup = str_column_to_int(dataset, 4)
print(dataset[0])
print(lookup)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Load Csv

# 01 — Load Csv / Load Csv

**Chapter 01 — File 3 of 4 / 第01章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Example of loading Pima Indians CSV dataset**.

本脚本演示 **Example of loading Pima Indians CSV dataset**。

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
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of loading Pima Indians CSV dataset 是机器学习中的常用技术。  
  *Example of loading Pima Indians CSV dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

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
## Step 1 — Example of loading Pima Indians CSV dataset

```python
from csv import reader
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
## Step 3 — Load dataset

```python
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of loading Pima Indians CSV dataset 是机器学习中的常用技术。  
  *Example of loading Pima Indians CSV dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

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
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Load dataset
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
```

---
