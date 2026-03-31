# 时间序列预测 / Time Series Forecasting with Python
## Chapter 13

---

### Chapter Summary / 章节总结

# Chapter 13 Summary / 第13章总结

## Theme / 主题: Chapter 13 / Chapter 13

This chapter contains **2 code files** demonstrating chapter 13.

本章包含 **2 个代码文件**，演示Chapter 13。

---
## Evolution / 演化路线

  1. `differenced.ipynb` — Differenced
  2. `linear_model.ipynb` — Linear Model

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 13) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 13）是机器学习流水线中的基础构建块。

---

### Differenced

# 01 — Differenced / Differenced

**Chapter 13 — File 1 of 2 / 第13章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **detrend a time series using differencing**.

本脚本演示 **detrend a time series using differencing**。

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
## Step 1 — detrend a time series using differencing

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import datetime
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
diff = list()
# 获取长度 / Get length
for i in range(1, len(X)):
	value = X[i] - X[i - 1]
 # 添加元素到列表末尾 / Append element to list end
	diff.append(value)
pyplot.plot(diff)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: detrend a time series using differencing 是机器学习中的常用技术。  
  *detrend a time series using differencing is a common technique in machine learning.*

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
# Differenced / Differenced
# Complete Code / 完整代码
# ===============================

# detrend a time series using differencing
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import datetime
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
diff = list()
# 获取长度 / Get length
for i in range(1, len(X)):
	value = X[i] - X[i - 1]
 # 添加元素到列表末尾 / Append element to list end
	diff.append(value)
pyplot.plot(diff)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Linear Model

# 01 — Linear Model / 线性模型

**Chapter 13 — File 2 of 2 / 第13章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **use a linear model to detrend a time series**.

本脚本演示 **use a linear model to detrend a time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — use a linear model to detrend a time series

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import datetime
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
```

---
## Step 2 — fit linear model

```python
# 获取长度 / Get length
X = [i for i in range(0, len(series))]
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = numpy.reshape(X, (len(X), 1))
# 转换为NumPy数组 / Convert to NumPy array
y = series.values
model = LinearRegression()
# 训练模型 / Train the model
model.fit(X, y)
```

---
## Step 3 — calculate trend

```python
# 用模型做预测 / Make predictions with model
trend = model.predict(X)
```

---
## Step 4 — plot trend

```python
pyplot.plot(y)
pyplot.plot(trend)
pyplot.show()
```

---
## Step 5 — detrend

```python
# 获取长度 / Get length
detrended = [y[i]-trend[i] for i in range(0, len(series))]
```

---
## Step 6 — plot detrended

```python
pyplot.plot(detrended)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: use a linear model to detrend a time series 是机器学习中的常用技术。  
  *use a linear model to detrend a time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Linear Model / 线性模型
# Complete Code / 完整代码
# ===============================

# use a linear model to detrend a time series
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import datetime
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
# fit linear model
# 获取长度 / Get length
X = [i for i in range(0, len(series))]
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = numpy.reshape(X, (len(X), 1))
# 转换为NumPy数组 / Convert to NumPy array
y = series.values
model = LinearRegression()
# 训练模型 / Train the model
model.fit(X, y)
# calculate trend
# 用模型做预测 / Make predictions with model
trend = model.predict(X)
# plot trend
pyplot.plot(y)
pyplot.plot(trend)
pyplot.show()
# detrend
# 获取长度 / Get length
detrended = [y[i]-trend[i] for i in range(0, len(series))]
# plot detrended
pyplot.plot(detrended)
pyplot.show()
```

---
