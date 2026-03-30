# Python ML实战
## Chapter 05

---

### Chapter Summary

# Chapter 05 Summary / 第05章总结

## Theme / 主题: Chapter 05 / Chapter 05

This chapter contains **7 code files** demonstrating chapter 05.

本章包含 **7 个代码文件**，演示Chapter 05。

---
## Evolution / 演化路线

  1. `class_distribution.ipynb` — Class Distribution
  2. `data_types.ipynb` — Data Types
  3. `describe.ipynb` — Describe
  4. `dimensions.ipynb` — Dimensions
  5. `head.ipynb` — Head
  6. `pearson_correlation.ipynb` — Pearson Correlation
  7. `skew.ipynb` — Skew

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 05) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 05）是机器学习流水线中的基础构建块。

---

### Class Distribution

# 01 — Class Distribution / Class Distribution

**Chapter 05 — File 1 of 7 / 第05章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Class Distribution**.

本脚本演示 **Class Distribution**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Class Distribution

```python
from pandas import read_csv
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
class_counts = data.groupby('class').size()
print(class_counts)
```

---
## Learning Notes / 学习笔记

- **概念**: Class Distribution 是机器学习中的常用技术。  
  *Class Distribution is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `groupby` | 分组聚合 | Group and aggregate |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Class Distribution / Class Distribution
# Complete Code / 完整代码
# ===============================

# Class Distribution
from pandas import read_csv
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
class_counts = data.groupby('class').size()
print(class_counts)
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Data Types

# 01 — Data Types / Data Types

**Chapter 05 — File 2 of 7 / 第05章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Data Types for Each Attribute**.

本脚本演示 **Data Types for Each Attribute**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Data Types for Each Attribute

```python
from pandas import read_csv
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
types = data.dtypes
print(types)
```

---
## Learning Notes / 学习笔记

- **概念**: Data Types for Each Attribute 是机器学习中的常用技术。  
  *Data Types for Each Attribute is a common technique in machine learning.*

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
# Data Types / Data Types
# Complete Code / 完整代码
# ===============================

# Data Types for Each Attribute
from pandas import read_csv
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
types = data.dtypes
print(types)
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Describe

# 01 — Describe / Describe

**Chapter 05 — File 3 of 7 / 第05章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Statistical Summary**.

本脚本演示 **Statistical Summary**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Statistical Summary

```python
from pandas import read_csv
from pandas import set_option
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
set_option('display.width', 100)
set_option('precision', 3)
description = data.describe()
print(description)
```

---
## Learning Notes / 学习笔记

- **概念**: Statistical Summary 是机器学习中的常用技术。  
  *Statistical Summary is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `describe()` | 统计摘要信息 | Statistical summary |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Describe / Describe
# Complete Code / 完整代码
# ===============================

# Statistical Summary
from pandas import read_csv
from pandas import set_option
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
set_option('display.width', 100)
set_option('precision', 3)
description = data.describe()
print(description)
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Dimensions

# 01 — Dimensions / Dimensions

**Chapter 05 — File 4 of 7 / 第05章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Dimensions of your data**.

本脚本演示 **Dimensions of your data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Dimensions of your data

```python
from pandas import read_csv
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
shape = data.shape
print(shape)
```

---
## Learning Notes / 学习笔记

- **概念**: Dimensions of your data 是机器学习中的常用技术。  
  *Dimensions of your data is a common technique in machine learning.*

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
# Dimensions / Dimensions
# Complete Code / 完整代码
# ===============================

# Dimensions of your data
from pandas import read_csv
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
shape = data.shape
print(shape)
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Head

# 01 — Head / Head

**Chapter 05 — File 5 of 7 / 第05章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **View first 20 rows**.

本脚本演示 **View first 20 rows**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — View first 20 rows

```python
from pandas import read_csv
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
peek = data.head(20)
print(peek)
```

---
## Learning Notes / 学习笔记

- **概念**: View first 20 rows 是机器学习中的常用技术。  
  *View first 20 rows is a common technique in machine learning.*

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
# Head / Head
# Complete Code / 完整代码
# ===============================

# View first 20 rows
from pandas import read_csv
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
peek = data.head(20)
print(peek)
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Pearson Correlation

# 01 — Pearson Correlation / Pearson Correlation

**Chapter 05 — File 6 of 7 / 第05章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Pairwise Pearson correlations**.

本脚本演示 **Pairwise Pearson correlations**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Pairwise Pearson correlations

```python
from pandas import read_csv
from pandas import set_option
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
set_option('display.width', 100)
set_option('precision', 3)
correlations = data.corr(method='pearson')
print(correlations)
```

---
## Learning Notes / 学习笔记

- **概念**: Pairwise Pearson correlations 是机器学习中的常用技术。  
  *Pairwise Pearson correlations is a common technique in machine learning.*

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
# Pearson Correlation / Pearson Correlation
# Complete Code / 完整代码
# ===============================

# Pairwise Pearson correlations
from pandas import read_csv
from pandas import set_option
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
set_option('display.width', 100)
set_option('precision', 3)
correlations = data.corr(method='pearson')
print(correlations)
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Skew

# 01 — Skew / Skew

**Chapter 05 — File 7 of 7 / 第05章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Skew for each attribute**.

本脚本演示 **Skew for each attribute**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Skew for each attribute

```python
from pandas import read_csv
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
skew = data.skew()
print(skew)
```

---
## Learning Notes / 学习笔记

- **概念**: Skew for each attribute 是机器学习中的常用技术。  
  *Skew for each attribute is a common technique in machine learning.*

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
# Skew / Skew
# Complete Code / 完整代码
# ===============================

# Skew for each attribute
from pandas import read_csv
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
skew = data.skew()
print(skew)
```

---
