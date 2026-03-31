# 时间序列预测 / Time Series Forecasting with Python
## Chapter 28

---

### Chapter Summary / 章节总结

# Chapter 28 Summary / 第28章总结

## Theme / 主题: Chapter 28 / Chapter 28

This chapter contains **2 code files** demonstrating chapter 28.

本章包含 **2 个代码文件**，演示Chapter 28。

---
## Evolution / 演化路线

  1. `confidence_interval.ipynb` — Confidence Interval
  2. `multiple_intervals.ipynb` — Multiple Intervals

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 28) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 28）是机器学习流水线中的基础构建块。

---

### Confidence Interval

# 01 — Confidence Interval / Confidence Interval

**Chapter 28 — File 1 of 2 / 第28章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **summarize the confidence interval on an ARIMA forecast**.

本脚本演示 **summarize the confidence interval on an ARIMA forecast**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 训练模型 / Train the model

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — summarize the confidence interval on an ARIMA forecast

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 3 — split into train and test sets

```python
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 获取长度 / Get length
size = len(X) - 1
train, test = X[0:size], X[size:]
```

---
## Step 4 — fit an ARIMA model

```python
model = ARIMA(train, order=(5,1,1))
# 训练模型 / Train the model
model_fit = model.fit()
```

---
## Step 5 — forecast

```python
result = model_fit.get_forecast()
```

---
## Step 6 — summarize forecast and confidence intervals

```python
# 打印输出 / Print output
print('Expected: %.3f' % result.predicted_mean)
# 打印输出 / Print output
print('Forecast: %.3f' % test[0])
# 打印输出 / Print output
print('Standard Error: %.3f' % result.se_mean)
ci = result.conf_int(0.05)
# 打印输出 / Print output
print('95%% Interval: %.3f to %.3f' % (ci[0,0], ci[0,1]))
```

---
## Learning Notes / 学习笔记

- **概念**: summarize the confidence interval on an ARIMA forecast 是机器学习中的常用技术。  
  *summarize the confidence interval on an ARIMA forecast is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Confidence Interval / Confidence Interval
# Complete Code / 完整代码
# ===============================

# summarize the confidence interval on an ARIMA forecast
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# split into train and test sets
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 获取长度 / Get length
size = len(X) - 1
train, test = X[0:size], X[size:]
# fit an ARIMA model
model = ARIMA(train, order=(5,1,1))
# 训练模型 / Train the model
model_fit = model.fit()
# forecast
result = model_fit.get_forecast()
# summarize forecast and confidence intervals
# 打印输出 / Print output
print('Expected: %.3f' % result.predicted_mean)
# 打印输出 / Print output
print('Forecast: %.3f' % test[0])
# 打印输出 / Print output
print('Standard Error: %.3f' % result.se_mean)
ci = result.conf_int(0.05)
# 打印输出 / Print output
print('95%% Interval: %.3f to %.3f' % (ci[0,0], ci[0,1]))
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Multiple Intervals

# 01 — Multiple Intervals / Multiple Intervals

**Chapter 28 — File 2 of 2 / 第28章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **summarize multiple confidence intervals on an ARIMA forecast**.

本脚本演示 **summarize multiple confidence intervals on an ARIMA forecast**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 训练模型 / Train the model

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — summarize multiple confidence intervals on an ARIMA forecast

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
```

---
## Step 2 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 3 — split data into train and test setes

```python
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 获取长度 / Get length
size = len(X) - 1
train, test = X[0:size], X[size:]
```

---
## Step 4 — fit an ARIMA model

```python
model = ARIMA(train, order=(5,1,1))
# 训练模型 / Train the model
model_fit = model.fit()
result = model_fit.get_forecast()
forecast = result.predicted_mean
```

---
## Step 5 — summarize confidence intervals

```python
intervals = [0.2, 0.1, 0.05, 0.01]
for a in intervals:
	ci = result.conf_int(alpha=a)
 # 打印输出 / Print output
	print('%.1f%% Confidence Interval: %.3f between %.3f and %.3f' % ((1-a)*100, forecast, ci[0,0], ci[0,1]))
```

---
## Learning Notes / 学习笔记

- **概念**: summarize multiple confidence intervals on an ARIMA forecast 是机器学习中的常用技术。  
  *summarize multiple confidence intervals on an ARIMA forecast is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `model.fit` | 训练模型 | Train the model |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Multiple Intervals / Multiple Intervals
# Complete Code / 完整代码
# ===============================

# summarize multiple confidence intervals on an ARIMA forecast
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# split data into train and test setes
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 获取长度 / Get length
size = len(X) - 1
train, test = X[0:size], X[size:]
# fit an ARIMA model
model = ARIMA(train, order=(5,1,1))
# 训练模型 / Train the model
model_fit = model.fit()
result = model_fit.get_forecast()
forecast = result.predicted_mean
# summarize confidence intervals
intervals = [0.2, 0.1, 0.05, 0.01]
for a in intervals:
	ci = result.conf_int(alpha=a)
 # 打印输出 / Print output
	print('%.1f%% Confidence Interval: %.3f between %.3f and %.3f' % ((1-a)*100, forecast, ci[0,0], ci[0,1]))
```

---
