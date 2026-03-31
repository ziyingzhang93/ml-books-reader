# 时间序列预测 / Time Series Forecasting with Python
## Chapter 05

---

### Chapter Summary / 章节总结



---

### Features Date Time

# 01 — Features Date Time / 特征工程

**Chapter 05 — File 1 of 6 / 第05章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **create date time features of a dataset**.

本脚本演示 **create date time features of a dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — create date time features of a dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
dataframe = DataFrame()
# 获取长度 / Get length
dataframe['month'] = [series.index[i].month for i in range(len(series))]
# 获取长度 / Get length
dataframe['day'] = [series.index[i].day for i in range(len(series))]
# 获取长度 / Get length
dataframe['temperature'] = [series[i] for i in range(len(series))]
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---
## Learning Notes / 学习笔记

- **概念**: create date time features of a dataset 是机器学习中的常用技术。  
  *create date time features of a dataset is a common technique in machine learning.*

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
# Features Date Time / 特征工程
# Complete Code / 完整代码
# ===============================

# create date time features of a dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
dataframe = DataFrame()
# 获取长度 / Get length
dataframe['month'] = [series.index[i].month for i in range(len(series))]
# 获取长度 / Get length
dataframe['day'] = [series.index[i].day for i in range(len(series))]
# 获取长度 / Get length
dataframe['temperature'] = [series[i] for i in range(len(series))]
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Features Expanding

# 01 — Features Expanding / 特征工程

**Chapter 05 — File 2 of 6 / 第05章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **create expanding window features**.

本脚本演示 **create expanding window features**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — create expanding window features

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
temps = DataFrame(series.values)
window = temps.expanding()
dataframe = concat([window.min(), window.mean(), window.max(), temps.shift(-1)], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['min', 'mean', 'max', 't+1']
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---
## Learning Notes / 学习笔记

- **概念**: create expanding window features 是机器学习中的常用技术。  
  *create expanding window features is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Features Expanding / 特征工程
# Complete Code / 完整代码
# ===============================

# create expanding window features
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
temps = DataFrame(series.values)
window = temps.expanding()
dataframe = concat([window.min(), window.mean(), window.max(), temps.shift(-1)], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['min', 'mean', 'max', 't+1']
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Features Lag1

# 01 — Features Lag1 / 特征工程

**Chapter 05 — File 3 of 6 / 第05章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **create a lag feature**.

本脚本演示 **create a lag feature**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — create a lag feature

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
temps = DataFrame(series.values)
dataframe = concat([temps.shift(1), temps], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1']
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---
## Learning Notes / 学习笔记

- **概念**: create a lag feature 是机器学习中的常用技术。  
  *create a lag feature is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Features Lag1 / 特征工程
# Complete Code / 完整代码
# ===============================

# create a lag feature
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
temps = DataFrame(series.values)
dataframe = concat([temps.shift(1), temps], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1']
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Features Lag3

# 01 — Features Lag3 / 特征工程

**Chapter 05 — File 4 of 6 / 第05章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **create lag features**.

本脚本演示 **create lag features**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — create lag features

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
temps = DataFrame(series.values)
dataframe = concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t-2', 't-1', 't', 't+1']
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---
## Learning Notes / 学习笔记

- **概念**: create lag features 是机器学习中的常用技术。  
  *create lag features is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Features Lag3 / 特征工程
# Complete Code / 完整代码
# ===============================

# create lag features
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
temps = DataFrame(series.values)
dataframe = concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t-2', 't-1', 't', 't+1']
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Features Rolling Mean

# 01 — Features Rolling Mean / 特征工程

**Chapter 05 — File 5 of 6 / 第05章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **create a rolling mean feature**.

本脚本演示 **create a rolling mean feature**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — create a rolling mean feature

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
temps = DataFrame(series.values)
shifted = temps.shift(1)
window = shifted.rolling(window=2)
means = window.mean()
dataframe = concat([means, temps], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['mean(t-1,t)', 't+1']
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---
## Learning Notes / 学习笔记

- **概念**: create a rolling mean feature 是机器学习中的常用技术。  
  *create a rolling mean feature is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Features Rolling Mean / 特征工程
# Complete Code / 完整代码
# ===============================

# create a rolling mean feature
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
temps = DataFrame(series.values)
shifted = temps.shift(1)
window = shifted.rolling(window=2)
means = window.mean()
dataframe = concat([means, temps], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['mean(t-1,t)', 't+1']
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Features Rolling Stats



---
