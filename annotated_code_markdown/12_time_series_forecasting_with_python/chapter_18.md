# 时间序列预测 / Time Series Forecasting with Python
## Chapter 18

---

### Chapter Summary / 章节总结

# Chapter 18 Summary / 第18章总结

## Theme / 主题: Chapter 18 / Chapter 18

This chapter contains **1 code files** demonstrating chapter 18.

本章包含 **1 个代码文件**，演示Chapter 18。

---
## Evolution / 演化路线

  1. `persistence.ipynb` — Persistence

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 18) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 18）是机器学习流水线中的基础构建块。

---

### Persistence

# 01 — Persistence / Persistence

**Chapter 18 — File 1 of 1 / 第18章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **evaluate a persistence forecast model**.

本脚本演示 **evaluate a persistence forecast model**。

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
## Step 1 — evaluate a persistence forecast model

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import datetime
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
```

---
## Step 2 — load dataset

```python
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
```

---
## Step 3 — create lagged dataset

```python
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1']
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
```

---
## Step 4 — split into train and test sets

```python
# 转换为NumPy数组 / Convert to NumPy array
X = dataframe.values
# 获取长度 / Get length
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
```

---
## Step 5 — persistence model

```python
def model_persistence(x):
	return x
```

---
## Step 6 — walk-forward validation

```python
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test_y, predictions))
# 打印输出 / Print output
print('Test RMSE: %.3f' % rmse)
```

---
## Step 7 — plot predictions and expected results

```python
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate a persistence forecast model 是机器学习中的常用技术。  
  *evaluate a persistence forecast model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Persistence / Persistence
# Complete Code / 完整代码
# ===============================

# evaluate a persistence forecast model
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import datetime
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
# create lagged dataset
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1']
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataframe.head(5))
# split into train and test sets
# 转换为NumPy数组 / Convert to NumPy array
X = dataframe.values
# 获取长度 / Get length
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
# persistence model
def model_persistence(x):
	return x
# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test_y, predictions))
# 打印输出 / Print output
print('Test RMSE: %.3f' % rmse)
# plot predictions and expected results
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()
```

---
