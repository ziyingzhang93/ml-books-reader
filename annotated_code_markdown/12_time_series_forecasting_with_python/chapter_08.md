# 时间序列预测
## Chapter 08

---

### Chapter Summary

# Chapter 08 Summary / 第08章总结

## Theme / 主题: Chapter 08 / Chapter 08

This chapter contains **9 code files** demonstrating chapter 08.

本章包含 **9 个代码文件**，演示Chapter 08。

---
## Evolution / 演化路线

  1. `airline_boxcox.ipynb` — Airline Boxcox
  2. `airline_boxcox_auto.ipynb` — Airline Boxcox Auto
  3. `airline_log.ipynb` — Airline Log
  4. `airline_sqrt.ipynb` — Airline Sqrt
  5. `plot_dataset.ipynb` — Plot Dataset
  6. `series_exponential.ipynb` — Series Exponential
  7. `series_exponential_log.ipynb` — Series Exponential Log
  8. `series_quadratic.ipynb` — Series Quadratic
  9. `series_quadratic_sqrt.ipynb` — Series Quadratic Sqrt

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 08) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 08）是机器学习流水线中的基础构建块。

---

### Airline Boxcox

# 01 — Airline Boxcox / Airline Boxcox

**Chapter 08 — File 1 of 9 / 第08章 — 第1个文件（共9个）**

---

## Summary / 总结

This script demonstrates **manually box-cox transform a time series**.

本脚本演示 **manually box-cox transform a time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — manually box-cox transform a time series

```python
from pandas import read_csv
from pandas import DataFrame
from scipy.stats import boxcox
from matplotlib import pyplot
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
dataframe = DataFrame(series.values)
dataframe.columns = ['passengers']
dataframe['passengers'] = boxcox(dataframe['passengers'], lmbda=0.0)
pyplot.figure(1)
```

---
## Step 2 — line plot

```python
pyplot.subplot(211)
pyplot.plot(dataframe['passengers'])
```

---
## Step 3 — histogram

```python
pyplot.subplot(212)
pyplot.hist(dataframe['passengers'])
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: manually box-cox transform a time series 是机器学习中的常用技术。  
  *manually box-cox transform a time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Airline Boxcox / Airline Boxcox
# Complete Code / 完整代码
# ===============================

# manually box-cox transform a time series
from pandas import read_csv
from pandas import DataFrame
from scipy.stats import boxcox
from matplotlib import pyplot
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
dataframe = DataFrame(series.values)
dataframe.columns = ['passengers']
dataframe['passengers'] = boxcox(dataframe['passengers'], lmbda=0.0)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['passengers'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['passengers'])
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 9

---

### Airline Boxcox Auto

# 01 — Airline Boxcox Auto / Airline Boxcox Auto

**Chapter 08 — File 2 of 9 / 第08章 — 第2个文件（共9个）**

---

## Summary / 总结

This script demonstrates **automatically box-cox transform a time series**.

本脚本演示 **automatically box-cox transform a time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — automatically box-cox transform a time series

```python
from pandas import read_csv
from pandas import DataFrame
from scipy.stats import boxcox
from matplotlib import pyplot
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
dataframe = DataFrame(series.values)
dataframe.columns = ['passengers']
dataframe['passengers'], lam = boxcox(dataframe['passengers'])
print('Lambda: %f' % lam)
pyplot.figure(1)
```

---
## Step 2 — line plot

```python
pyplot.subplot(211)
pyplot.plot(dataframe['passengers'])
```

---
## Step 3 — histogram

```python
pyplot.subplot(212)
pyplot.hist(dataframe['passengers'])
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: automatically box-cox transform a time series 是机器学习中的常用技术。  
  *automatically box-cox transform a time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Airline Boxcox Auto / Airline Boxcox Auto
# Complete Code / 完整代码
# ===============================

# automatically box-cox transform a time series
from pandas import read_csv
from pandas import DataFrame
from scipy.stats import boxcox
from matplotlib import pyplot
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
dataframe = DataFrame(series.values)
dataframe.columns = ['passengers']
dataframe['passengers'], lam = boxcox(dataframe['passengers'])
print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['passengers'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['passengers'])
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 9

---

### Airline Sqrt

# 01 — Airline Sqrt / Airline Sqrt

**Chapter 08 — File 4 of 9 / 第08章 — 第4个文件（共9个）**

---

## Summary / 总结

This script demonstrates **square root transform a time series**.

本脚本演示 **square root transform a time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — square root transform a time series

```python
from pandas import read_csv
from pandas import DataFrame
from numpy import sqrt
from matplotlib import pyplot
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
dataframe = DataFrame(series.values)
dataframe.columns = ['passengers']
dataframe['passengers'] = sqrt(dataframe['passengers'])
pyplot.figure(1)
```

---
## Step 2 — line plot

```python
pyplot.subplot(211)
pyplot.plot(dataframe['passengers'])
```

---
## Step 3 — histogram

```python
pyplot.subplot(212)
pyplot.hist(dataframe['passengers'])
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: square root transform a time series 是机器学习中的常用技术。  
  *square root transform a time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Airline Sqrt / Airline Sqrt
# Complete Code / 完整代码
# ===============================

# square root transform a time series
from pandas import read_csv
from pandas import DataFrame
from numpy import sqrt
from matplotlib import pyplot
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
dataframe = DataFrame(series.values)
dataframe.columns = ['passengers']
dataframe['passengers'] = sqrt(dataframe['passengers'])
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['passengers'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['passengers'])
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 9

---

### Series Quadratic Sqrt

# 01 — Series Quadratic Sqrt / Series Quadratic Sqrt

**Chapter 08 — File 9 of 9 / 第08章 — 第9个文件（共9个）**

---

## Summary / 总结

This script demonstrates **square root transform a contrived quadratic time series**.

本脚本演示 **square root transform a contrived quadratic time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 可视化结果 / Visualize results


---
## Step 1 — square root transform a contrived quadratic time series

```python
from matplotlib import pyplot
from numpy import sqrt
series = [i**2 for i in range(1,100)]
```

---
## Step 2 — sqrt transform

```python
transform = series = sqrt(series)
pyplot.figure(1)
```

---
## Step 3 — line plot

```python
pyplot.subplot(211)
pyplot.plot(transform)
```

---
## Step 4 — histogram

```python
pyplot.subplot(212)
pyplot.hist(transform)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: square root transform a contrived quadratic time series 是机器学习中的常用技术。  
  *square root transform a contrived quadratic time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Series Quadratic Sqrt / Series Quadratic Sqrt
# Complete Code / 完整代码
# ===============================

# square root transform a contrived quadratic time series
from matplotlib import pyplot
from numpy import sqrt
series = [i**2 for i in range(1,100)]
# sqrt transform
transform = series = sqrt(series)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(transform)
# histogram
pyplot.subplot(212)
pyplot.hist(transform)
pyplot.show()
```

---
