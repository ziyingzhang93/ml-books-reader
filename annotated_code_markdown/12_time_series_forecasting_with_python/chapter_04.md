# 时间序列预测 / Time Series Forecasting with Python
## Chapter 04

---

### Chapter Summary / 章节总结

# Chapter 04 Summary / 第04章总结

## Theme / 主题: Chapter 04 / Chapter 04

This chapter contains **5 code files** demonstrating chapter 04.

本章包含 **5 个代码文件**，演示Chapter 04。

---
## Evolution / 演化路线

  1. `descriptive_stats.ipynb` — Descriptive Stats
  2. `head.ipynb` — Head
  3. `load_read_csv.ipynb` — Load Read Csv
  4. `query.ipynb` — Query
  5. `size.ipynb` — Size

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 04) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 04）是机器学习流水线中的基础构建块。

---

### Descriptive Stats

# 01 — Descriptive Stats / Descriptive Stats

**Chapter 04 — File 1 of 5 / 第04章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **calculate descriptive statistics**.

本脚本演示 **calculate descriptive statistics**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — calculate descriptive statistics

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(series.describe())
```

---
## Learning Notes / 学习笔记

- **概念**: calculate descriptive statistics 是机器学习中的常用技术。  
  *calculate descriptive statistics is a common technique in machine learning.*

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
# Descriptive Stats / Descriptive Stats
# Complete Code / 完整代码
# ===============================

# calculate descriptive statistics
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(series.describe())
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Head

# 01 — Head / Head

**Chapter 04 — File 2 of 5 / 第04章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **summarize first few lines of a file**.

本脚本演示 **summarize first few lines of a file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — summarize first few lines of a file

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head(10))
```

---
## Learning Notes / 学习笔记

- **概念**: summarize first few lines of a file 是机器学习中的常用技术。  
  *summarize first few lines of a file is a common technique in machine learning.*

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

# summarize first few lines of a file
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head(10))
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Load Read Csv

# 01 — Load Read Csv / Load Read Csv

**Chapter 04 — File 3 of 5 / 第04章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **load dataset using read_csv()**.

本脚本演示 **load dataset using read_csv()**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — load dataset using read_csv()

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 打印输出 / Print output
print(type(series))
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head())
```

---
## Learning Notes / 学习笔记

- **概念**: load dataset using read_csv() 是机器学习中的常用技术。  
  *load dataset using read_csv() is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `head()` | 查看前几行数据 | View first few rows |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Read Csv / Load Read Csv
# Complete Code / 完整代码
# ===============================

# load dataset using read_csv()
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 打印输出 / Print output
print(type(series))
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head())
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Query

# 01 — Query / Query

**Chapter 04 — File 4 of 5 / 第04章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **query a dataset using a date-time index**.

本脚本演示 **query a dataset using a date-time index**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — query a dataset using a date-time index

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 打印输出 / Print output
print(series['1959-01'])
```

---
## Learning Notes / 学习笔记

- **概念**: query a dataset using a date-time index 是机器学习中的常用技术。  
  *query a dataset using a date-time index is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Query / Query
# Complete Code / 完整代码
# ===============================

# query a dataset using a date-time index
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 打印输出 / Print output
print(series['1959-01'])
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Size

# 01 — Size / Size

**Chapter 04 — File 5 of 5 / 第04章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **summarize the dimensions of a time series**.

本脚本演示 **summarize the dimensions of a time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — summarize the dimensions of a time series

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 打印输出 / Print output
print(series.size)
```

---
## Learning Notes / 学习笔记

- **概念**: summarize the dimensions of a time series 是机器学习中的常用技术。  
  *summarize the dimensions of a time series is a common technique in machine learning.*

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
# Size / Size
# Complete Code / 完整代码
# ===============================

# summarize the dimensions of a time series
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 打印输出 / Print output
print(series.size)
```

---
