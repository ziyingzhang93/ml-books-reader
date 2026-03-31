# Python 机器学习实战 / ML Mastery with Python
## Chapter 04

---

### Chapter Summary / 章节总结

# Chapter 04 Summary / 第04章总结

## Theme / 主题: Chapter 04 / Chapter 04

This chapter contains **5 code files** demonstrating chapter 04.

本章包含 **5 个代码文件**，演示Chapter 04。

---
## Evolution / 演化路线

  1. `load_csv.ipynb` — Load Csv
  2. `load_csv_np.ipynb` — Load Csv Np
  3. `load_csv_np_url.ipynb` — Load Csv Np Url
  4. `load_csv_pandas.ipynb` — Load Csv Pandas
  5. `load_csv_pandas_url.ipynb` — Load Csv Pandas Url

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 04) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 04）是机器学习流水线中的基础构建块。

---

### Load Csv

# 01 — Load Csv / Load Csv

**Chapter 04 — File 1 of 5 / 第04章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Load CSV Using Python Standard Library**.

本脚本演示 **Load CSV Using Python Standard Library**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Load CSV Using Python Standard Library

```python
import csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
# 转换数据类型 / Convert data type
data = numpy.array(x).astype('float')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(data.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: Load CSV Using Python Standard Library 是机器学习中的常用技术。  
  *Load CSV Using Python Standard Library is a common technique in machine learning.*

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
# Load Csv / Load Csv
# Complete Code / 完整代码
# ===============================

# Load CSV Using Python Standard Library
import csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
# 转换数据类型 / Convert data type
data = numpy.array(x).astype('float')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(data.shape)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Load Csv Np

# 01 — Load Csv Np / Load Csv Np

**Chapter 04 — File 2 of 5 / 第04章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Load CSV using NumPy**.

本脚本演示 **Load CSV using NumPy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Load CSV using NumPy

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rt')
data = loadtxt(raw_data, delimiter=",")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(data.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: Load CSV using NumPy 是机器学习中的常用技术。  
  *Load CSV using NumPy is a common technique in machine learning.*

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
# Load Csv Np / Load Csv Np
# Complete Code / 完整代码
# ===============================

# Load CSV using NumPy
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rt')
data = loadtxt(raw_data, delimiter=",")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(data.shape)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Load Csv Np Url

# 01 — Load Csv Np Url / Load Csv Np Url

**Chapter 04 — File 3 of 5 / 第04章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Load CSV from URL using NumPy**.

本脚本演示 **Load CSV from URL using NumPy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Load CSV from URL using NumPy

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
from urllib.request import urlopen
url = 'https://goo.gl/bDdBiA'
raw_data = urlopen(url)
dataset = loadtxt(raw_data, delimiter=",")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: Load CSV from URL using NumPy 是机器学习中的常用技术。  
  *Load CSV from URL using NumPy is a common technique in machine learning.*

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
# Load Csv Np Url / Load Csv Np Url
# Complete Code / 完整代码
# ===============================

# Load CSV from URL using NumPy
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
from urllib.request import urlopen
url = 'https://goo.gl/bDdBiA'
raw_data = urlopen(url)
dataset = loadtxt(raw_data, delimiter=",")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Load Csv Pandas

# 01 — Load Csv Pandas / Load Csv Pandas

**Chapter 04 — File 4 of 5 / 第04章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Load CSV using Pandas**.

本脚本演示 **Load CSV using Pandas**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Load CSV using Pandas

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv(filename, names=names)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(data.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: Load CSV using Pandas 是机器学习中的常用技术。  
  *Load CSV using Pandas is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Csv Pandas / Load Csv Pandas
# Complete Code / 完整代码
# ===============================

# Load CSV using Pandas
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv(filename, names=names)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(data.shape)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Load Csv Pandas Url

# 01 — Load Csv Pandas Url / Load Csv Pandas Url

**Chapter 04 — File 5 of 5 / 第04章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Load CSV using Pandas from URL**.

本脚本演示 **Load CSV using Pandas from URL**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Load CSV using Pandas from URL

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
url = 'https://goo.gl/bDdBiA'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv(url, names=names)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(data.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: Load CSV using Pandas from URL 是机器学习中的常用技术。  
  *Load CSV using Pandas from URL is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Csv Pandas Url / Load Csv Pandas Url
# Complete Code / 完整代码
# ===============================

# Load CSV using Pandas from URL
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
url = 'https://goo.gl/bDdBiA'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv(url, names=names)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(data.shape)
```

---
