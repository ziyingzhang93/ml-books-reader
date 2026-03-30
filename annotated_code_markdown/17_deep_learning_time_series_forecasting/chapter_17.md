# DL时间序列
## Chapter 17

---

### Naive Forecasts

# 04 — Naive Forecasts / 预测

**Chapter 17 — File 4 of 4 / 第17章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **naive forecast strategies for the power usage dataset**.

本脚本演示 **naive forecast strategies for the power usage dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 评估模型效果 / Evaluate model performance
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
## Step 1 — naive forecast strategies for the power usage dataset

```python
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
```

---
## Step 2 — split a univariate dataset into train/test sets

```python
def split_dataset(data):
```

---
## Step 3 — split into standard weeks

```python
train, test = data[1:-328], data[-328:-6]
```

---
## Step 4 — restructure into windows of weekly data

```python
train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test
```

---
## Step 5 — evaluate one or more weekly forecasts against expected values

```python
def evaluate_forecasts(actual, predicted):
	scores = list()
```

---
## Step 6 — calculate an RMSE score for each day

```python
for i in range(actual.shape[1]):
```

---
## Step 7 — calculate mse

```python
mse = mean_squared_error(actual[:, i], predicted[:, i])
```

---
## Step 8 — calculate rmse

```python
rmse = sqrt(mse)
```

---
## Step 9 — store

```python
scores.append(rmse)
```

---
## Step 10 — calculate overall RMSE

```python
s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores
```

---
## Step 11 — summarize scores

```python
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))
```

---
## Step 12 — evaluate a single model

```python
def evaluate_model(model_func, train, test):
```

---
## Step 13 — history is a list of weekly data

```python
history = [x for x in train]
```

---
## Step 14 — walk-forward validation over each week

```python
predictions = list()
	for i in range(len(test)):
```

---
## Step 15 — predict the week

```python
yhat_sequence = model_func(history)
```

---
## Step 16 — store the predictions

```python
predictions.append(yhat_sequence)
```

---
## Step 17 — get real observation and add to history for predicting the next week

```python
history.append(test[i, :])
	predictions = array(predictions)
```

---
## Step 18 — evaluate predictions days for each week

```python
score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores
```

---
## Step 19 — daily persistence model

```python
def daily_persistence(history):
```

---
## Step 20 — get the data for the prior week

```python
last_week = history[-1]
```

---
## Step 21 — get the total active power for the last day

```python
value = last_week[-1, 0]
```

---
## Step 22 — prepare 7 day forecast

```python
forecast = [value for _ in range(7)]
	return forecast
```

---
## Step 23 — weekly persistence model

```python
def weekly_persistence(history):
```

---
## Step 24 — get the data for the prior week

```python
last_week = history[-1]
	return last_week[:, 0]
```

---
## Step 25 — week one year ago persistence model

```python
def week_one_year_ago_persistence(history):
```

---
## Step 26 — get the data for the prior week

```python
last_week = history[-52]
	return last_week[:, 0]
```

---
## Step 27 — load the new file

```python
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
```

---
## Step 28 — split into train and test

```python
train, test = split_dataset(dataset.values)
```

---
## Step 29 — define the names and functions for the models we wish to evaluate

```python
models = dict()
models['daily'] = daily_persistence
models['weekly'] = weekly_persistence
models['week-oya'] = week_one_year_ago_persistence
```

---
## Step 30 — evaluate each model

```python
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
for name, func in models.items():
```

---
## Step 31 — evaluate and get scores

```python
score, scores = evaluate_model(func, train, test)
```

---
## Step 32 — summarize scores

```python
summarize_scores(name, score, scores)
```

---
## Step 33 — plot scores

```python
pyplot.plot(days, scores, marker='o', label=name)
```

---
## Step 34 — show plot

```python
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: naive forecast strategies for the power usage dataset 是机器学习中的常用技术。  
  *naive forecast strategies for the power usage dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Naive Forecasts / 预测
# Complete Code / 完整代码
# ===============================

# naive forecast strategies for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	train, test = data[1:-328], data[-328:-6]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# evaluate a single model
def evaluate_model(model_func, train, test):
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = model_func(history)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	predictions = array(predictions)
	# evaluate predictions days for each week
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores

# daily persistence model
def daily_persistence(history):
	# get the data for the prior week
	last_week = history[-1]
	# get the total active power for the last day
	value = last_week[-1, 0]
	# prepare 7 day forecast
	forecast = [value for _ in range(7)]
	return forecast

# weekly persistence model
def weekly_persistence(history):
	# get the data for the prior week
	last_week = history[-1]
	return last_week[:, 0]

# week one year ago persistence model
def week_one_year_ago_persistence(history):
	# get the data for the prior week
	last_week = history[-52]
	return last_week[:, 0]

# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
train, test = split_dataset(dataset.values)
# define the names and functions for the models we wish to evaluate
models = dict()
models['daily'] = daily_persistence
models['weekly'] = weekly_persistence
models['week-oya'] = week_one_year_ago_persistence
# evaluate each model
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
for name, func in models.items():
	# evaluate and get scores
	score, scores = evaluate_model(func, train, test)
	# summarize scores
	summarize_scores(name, score, scores)
	# plot scores
	pyplot.plot(days, scores, marker='o', label=name)
# show plot
pyplot.legend()
pyplot.show()
```

---

### Chapter Summary

# Chapter 17 Summary / 第17章总结

## Theme / 主题: Chapter 17 / Chapter 17

This chapter contains **4 code files** demonstrating chapter 17.

本章包含 **4 个代码文件**，演示Chapter 17。

---
## Evolution / 演化路线

  1. `01_prepare_data.ipynb` — Prepare Data
  2. `02_resample_dataset.ipynb` — Resample Dataset
  3. `03_train_test_split.ipynb` — Train Test Split
  4. `04_naive_forecasts.ipynb` — Naive Forecasts

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 17) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 17）是机器学习流水线中的基础构建块。

---
