# 时间序列预测 / Time Series Forecasting with Python
## Chapter 20

---

### Chapter Summary / 章节总结

# Chapter 20 Summary / 第20章总结

## Theme / 主题: Chapter 20 / Chapter 20

This chapter contains **3 code files** demonstrating chapter 20.

本章包含 **3 个代码文件**，演示Chapter 20。

---
## Evolution / 演化路线

  1. `reframe_classification.ipynb` — Reframe Classification
  2. `reframe_horizon.ipynb` — Reframe Horizon
  3. `reframe_regression.ipynb` — Reframe Regression

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 20) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 20）是机器学习流水线中的基础构建块。

---

### Reframe Classification

# 01 — Reframe Classification / 分类

**Chapter 20 — File 1 of 3 / 第20章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **reframe regression as classification**.

本脚本演示 **reframe regression as classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — reframe regression as classification

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
```

---
## Step 2 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 3 — Create lagged dataset

```python
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1']
```

---
## Step 4 — make discrete

```python
# 获取长度 / Get length
for i in range(len(dataframe['t+1'])):
	value = dataframe['t+1'][i]
	if value < 10.0:
		dataframe['t+1'][i] = 0
	elif value >= 25.0:
		dataframe['t+1'][i] = 2
	else:
		dataframe['t+1'][i] = 1
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---
## Learning Notes / 学习笔记

- **概念**: reframe regression as classification 是机器学习中的常用技术。  
  *reframe regression as classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Reframe Classification / 分类
# Complete Code / 完整代码
# ===============================

# reframe regression as classification
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# Create lagged dataset
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1']
# make discrete
# 获取长度 / Get length
for i in range(len(dataframe['t+1'])):
	value = dataframe['t+1'][i]
	if value < 10.0:
		dataframe['t+1'][i] = 0
	elif value >= 25.0:
		dataframe['t+1'][i] = 2
	else:
		dataframe['t+1'][i] = 1
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Reframe Horizon

# 01 — Reframe Horizon / Reframe Horizon

**Chapter 20 — File 2 of 3 / 第20章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **reframe time horizon of forecast**.

本脚本演示 **reframe time horizon of forecast**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — reframe time horizon of forecast

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
```

---
## Step 2 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 3 — create lagged dataset

```python
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values, values.shift(-1),
	values.shift(-2), values.shift(-3), values.shift(-4), values.shift(-5),
	values.shift(-6)], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1', 't+2', 't+3', 't+4', 't+5', 't+6', 't+7']
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(14))
```

---
## Learning Notes / 学习笔记

- **概念**: reframe time horizon of forecast 是机器学习中的常用技术。  
  *reframe time horizon of forecast is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Reframe Horizon / Reframe Horizon
# Complete Code / 完整代码
# ===============================

# reframe time horizon of forecast
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# create lagged dataset
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values, values.shift(-1),
	values.shift(-2), values.shift(-3), values.shift(-4), values.shift(-5),
	values.shift(-6)], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1', 't+2', 't+3', 't+4', 't+5', 't+6', 't+7']
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(14))
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Reframe Regression

# 01 — Reframe Regression / 回归

**Chapter 20 — File 3 of 3 / 第20章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **reframe precision of regression forecast**.

本脚本演示 **reframe precision of regression forecast**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — reframe precision of regression forecast

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
```

---
## Step 2 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 3 — create lagged dataset

```python
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1']
```

---
## Step 4 — round forecast to nearest 5

```python
# 获取长度 / Get length
for i in range(len(dataframe['t+1'])):
	dataframe['t+1'][i] = int(dataframe['t+1'][i] / 5) * 5.0
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---
## Learning Notes / 学习笔记

- **概念**: reframe precision of regression forecast 是机器学习中的常用技术。  
  *reframe precision of regression forecast is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Reframe Regression / 回归
# Complete Code / 完整代码
# ===============================

# reframe precision of regression forecast
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# create lagged dataset
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1']
# round forecast to nearest 5
# 获取长度 / Get length
for i in range(len(dataframe['t+1'])):
	dataframe['t+1'][i] = int(dataframe['t+1'][i] / 5) * 5.0
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---
