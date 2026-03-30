# 时间序列预测
## Chapter 15

---

### Chapter Summary

# Chapter 15 Summary / 第15章总结

## Theme / 主题: Chapter 15 / Chapter 15

This chapter contains **11 code files** demonstrating chapter 15.

本章包含 **11 个代码文件**，演示Chapter 15。

---
## Evolution / 演化路线

  1. `histogram_airline.ipynb` — Histogram Airline
  2. `histogram_births.ipynb` — Histogram Births
  3. `histogram_log_airline.ipynb` — Histogram Log Airline
  4. `load_airline.ipynb` — Load Airline
  5. `load_female_births.ipynb` — Load Female Births
  6. `stats_airline.ipynb` — Stats Airline
  7. `stats_births.ipynb` — Stats Births
  8. `stats_log_airline.ipynb` — Stats Log Airline
  9. `test_airline.ipynb` — Test Airline
  10. `test_births.ipynb` — Test Births
  11. `test_log_airline.ipynb` — Test Log Airline

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 15) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 15）是机器学习流水线中的基础构建块。

---

### Histogram Airline

# 01 — Histogram Airline / Histogram Airline

**Chapter 15 — File 1 of 11 / 第15章 — 第1个文件（共11个）**

---

## Summary / 总结

This script demonstrates **plot a histogram of a time series**.

本脚本演示 **plot a histogram of a time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — plot a histogram of a time series

```python
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.hist()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot a histogram of a time series 是机器学习中的常用技术。  
  *plot a histogram of a time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Histogram Airline / Histogram Airline
# Complete Code / 完整代码
# ===============================

# plot a histogram of a time series
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.hist()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 11

---

### Histogram Births

# 01 — Histogram Births / Histogram Births

**Chapter 15 — File 2 of 11 / 第15章 — 第2个文件（共11个）**

---

## Summary / 总结

This script demonstrates **plot a histogram of a time series**.

本脚本演示 **plot a histogram of a time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — plot a histogram of a time series

```python
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.hist()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot a histogram of a time series 是机器学习中的常用技术。  
  *plot a histogram of a time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Histogram Births / Histogram Births
# Complete Code / 完整代码
# ===============================

# plot a histogram of a time series
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.hist()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 11

---

### Histogram Log Airline

# 01 — Histogram Log Airline / Histogram Log Airline

**Chapter 15 — File 3 of 11 / 第15章 — 第3个文件（共11个）**

---

## Summary / 总结

This script demonstrates **histogram and line plot of log transformed time series**.

本脚本演示 **histogram and line plot of log transformed time series**。

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
## Step 1 — histogram and line plot of log transformed time series

```python
from pandas import read_csv
from matplotlib import pyplot
from numpy import log
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
X = log(X)
pyplot.hist(X)
pyplot.show()
pyplot.plot(X)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: histogram and line plot of log transformed time series 是机器学习中的常用技术。  
  *histogram and line plot of log transformed time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Histogram Log Airline / Histogram Log Airline
# Complete Code / 完整代码
# ===============================

# histogram and line plot of log transformed time series
from pandas import read_csv
from matplotlib import pyplot
from numpy import log
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
X = log(X)
pyplot.hist(X)
pyplot.show()
pyplot.plot(X)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 11

---

### Load Airline

# 01 — Load Airline / Load Airline

**Chapter 15 — File 4 of 11 / 第15章 — 第4个文件（共11个）**

---

## Summary / 总结

This script demonstrates **load time series data**.

本脚本演示 **load time series data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — load time series data

```python
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load time series data 是机器学习中的常用技术。  
  *load time series data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Airline / Load Airline
# Complete Code / 完整代码
# ===============================

# load time series data
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.plot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 11

---

### Load Female Births

# 01 — Load Female Births / Load Female Births

**Chapter 15 — File 5 of 11 / 第15章 — 第5个文件（共11个）**

---

## Summary / 总结

This script demonstrates **load time series data**.

本脚本演示 **load time series data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — load time series data

```python
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load time series data 是机器学习中的常用技术。  
  *load time series data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Female Births / Load Female Births
# Complete Code / 完整代码
# ===============================

# load time series data
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.plot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 6 of 11

---

### Stats Airline

# 01 — Stats Airline / Stats Airline

**Chapter 15 — File 6 of 11 / 第15章 — 第6个文件（共11个）**

---

## Summary / 总结

This script demonstrates **calculate statistics of partitioned time series data**.

本脚本演示 **calculate statistics of partitioned time series data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — calculate statistics of partitioned time series data

```python
from pandas import read_csv
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
```

---
## Learning Notes / 学习笔记

- **概念**: calculate statistics of partitioned time series data 是机器学习中的常用技术。  
  *calculate statistics of partitioned time series data is a common technique in machine learning.*

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
# Stats Airline / Stats Airline
# Complete Code / 完整代码
# ===============================

# calculate statistics of partitioned time series data
from pandas import read_csv
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
```

---

➡️ **Next / 下一步**: File 7 of 11

---

### Stats Births

# 01 — Stats Births / Stats Births

**Chapter 15 — File 7 of 11 / 第15章 — 第7个文件（共11个）**

---

## Summary / 总结

This script demonstrates **calculate statistics of partitioned time series data**.

本脚本演示 **calculate statistics of partitioned time series data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — calculate statistics of partitioned time series data

```python
from pandas import read_csv
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
```

---
## Learning Notes / 学习笔记

- **概念**: calculate statistics of partitioned time series data 是机器学习中的常用技术。  
  *calculate statistics of partitioned time series data is a common technique in machine learning.*

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
# Stats Births / Stats Births
# Complete Code / 完整代码
# ===============================

# calculate statistics of partitioned time series data
from pandas import read_csv
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
```

---

➡️ **Next / 下一步**: File 8 of 11

---

### Stats Log Airline

# 01 — Stats Log Airline / Stats Log Airline

**Chapter 15 — File 8 of 11 / 第15章 — 第8个文件（共11个）**

---

## Summary / 总结

This script demonstrates **calculate statistics of partitioned log transformed time series data**.

本脚本演示 **calculate statistics of partitioned log transformed time series data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Step 1 — calculate statistics of partitioned log transformed time series data

```python
from pandas import read_csv
from numpy import log
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
X = log(X)
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
```

---
## Learning Notes / 学习笔记

- **概念**: calculate statistics of partitioned log transformed time series data 是机器学习中的常用技术。  
  *calculate statistics of partitioned log transformed time series data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Stats Log Airline / Stats Log Airline
# Complete Code / 完整代码
# ===============================

# calculate statistics of partitioned log transformed time series data
from pandas import read_csv
from numpy import log
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
X = log(X)
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
```

---

➡️ **Next / 下一步**: File 9 of 11

---

### Test Airline

# 01 — Test Airline / Test Airline

**Chapter 15 — File 9 of 11 / 第15章 — 第9个文件（共11个）**

---

## Summary / 总结

This script demonstrates **calculate stationarity test of time series data**.

本脚本演示 **calculate stationarity test of time series data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — calculate stationarity test of time series data

```python
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
```

---
## Learning Notes / 学习笔记

- **概念**: calculate stationarity test of time series data 是机器学习中的常用技术。  
  *calculate stationarity test of time series data is a common technique in machine learning.*

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
# Test Airline / Test Airline
# Complete Code / 完整代码
# ===============================

# calculate stationarity test of time series data
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
```

---

➡️ **Next / 下一步**: File 10 of 11

---

### Test Births

# 01 — Test Births / Test Births

**Chapter 15 — File 10 of 11 / 第15章 — 第10个文件（共11个）**

---

## Summary / 总结

This script demonstrates **calculate stationarity test of time series data**.

本脚本演示 **calculate stationarity test of time series data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — calculate stationarity test of time series data

```python
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
```

---
## Learning Notes / 学习笔记

- **概念**: calculate stationarity test of time series data 是机器学习中的常用技术。  
  *calculate stationarity test of time series data is a common technique in machine learning.*

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
# Test Births / Test Births
# Complete Code / 完整代码
# ===============================

# calculate stationarity test of time series data
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
```

---

➡️ **Next / 下一步**: File 11 of 11

---

### Test Log Airline

# 01 — Test Log Airline / Test Log Airline

**Chapter 15 — File 11 of 11 / 第15章 — 第11个文件（共11个）**

---

## Summary / 总结

This script demonstrates **calculate stationarity test of log transformed time series data**.

本脚本演示 **calculate stationarity test of log transformed time series data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Step 1 — calculate stationarity test of log transformed time series data

```python
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
from numpy import log
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
X = log(X)
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
```

---
## Learning Notes / 学习笔记

- **概念**: calculate stationarity test of log transformed time series data 是机器学习中的常用技术。  
  *calculate stationarity test of log transformed time series data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Test Log Airline / Test Log Airline
# Complete Code / 完整代码
# ===============================

# calculate stationarity test of log transformed time series data
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
from numpy import log
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
X = log(X)
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
```

---
