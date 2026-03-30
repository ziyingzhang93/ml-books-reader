# DL时间序列
## Chapter 18

---

### Acf Pacf Plots

# 01 — Acf Pacf Plots / 01 Acf Pacf Plots

**Chapter 18 — File 1 of 3 / 第18章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **acf and pacf plots of total power usage**.

本脚本演示 **acf and pacf plots of total power usage**。

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
## Step 1 — acf and pacf plots of total power usage

```python
from numpy import split
from numpy import array
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
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
## Step 5 — convert windows of weekly multivariate data into a series of total power

```python
def to_series(data):
```

---
## Step 6 — extract just the total power from each week

```python
series = [week[:, 0] for week in data]
```

---
## Step 7 — flatten into a single series

```python
series = array(series).flatten()
	return series
```

---
## Step 8 — load the new file

```python
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
```

---
## Step 9 — split into train and test

```python
train, test = split_dataset(dataset.values)
```

---
## Step 10 — convert training data into a series

```python
series = to_series(train)
```

---
## Step 11 — plots

```python
pyplot.figure()
lags = 365
```

---
## Step 12 — acf

```python
axis = pyplot.subplot(2, 1, 1)
plot_acf(series, ax=axis, lags=lags)
```

---
## Step 13 — pacf

```python
axis = pyplot.subplot(2, 1, 2)
plot_pacf(series, ax=axis, lags=lags)
```

---
## Step 14 — show plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: acf and pacf plots of total power usage 是机器学习中的常用技术。  
  *acf and pacf plots of total power usage is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Acf Pacf Plots / 01 Acf Pacf Plots
# Complete Code / 完整代码
# ===============================

# acf and pacf plots of total power usage
from numpy import split
from numpy import array
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	train, test = data[1:-328], data[-328:-6]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test

# convert windows of weekly multivariate data into a series of total power
def to_series(data):
	# extract just the total power from each week
	series = [week[:, 0] for week in data]
	# flatten into a single series
	series = array(series).flatten()
	return series

# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
train, test = split_dataset(dataset.values)
# convert training data into a series
series = to_series(train)
# plots
pyplot.figure()
lags = 365
# acf
axis = pyplot.subplot(2, 1, 1)
plot_acf(series, ax=axis, lags=lags)
# pacf
axis = pyplot.subplot(2, 1, 2)
plot_pacf(series, ax=axis, lags=lags)
# show plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Acf Pacf Plots Zoomed

# 02 — Acf Pacf Plots Zoomed / 02 Acf Pacf Plots Zoomed

**Chapter 18 — File 2 of 3 / 第18章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **zoomed acf and pacf plots of total power usage**.

本脚本演示 **zoomed acf and pacf plots of total power usage**。

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
## Step 1 — zoomed acf and pacf plots of total power usage

```python
from numpy import split
from numpy import array
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
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
## Step 5 — convert windows of weekly multivariate data into a series of total power

```python
def to_series(data):
```

---
## Step 6 — extract just the total power from each week

```python
series = [week[:, 0] for week in data]
```

---
## Step 7 — flatten into a single series

```python
series = array(series).flatten()
	return series
```

---
## Step 8 — load the new file

```python
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
```

---
## Step 9 — split into train and test

```python
train, test = split_dataset(dataset.values)
```

---
## Step 10 — convert training data into a series

```python
series = to_series(train)
```

---
## Step 11 — plots

```python
pyplot.figure()
lags = 50
```

---
## Step 12 — acf

```python
axis = pyplot.subplot(2, 1, 1)
plot_acf(series, ax=axis, lags=lags)
```

---
## Step 13 — pacf

```python
axis = pyplot.subplot(2, 1, 2)
plot_pacf(series, ax=axis, lags=lags)
```

---
## Step 14 — show plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: zoomed acf and pacf plots of total power usage 是机器学习中的常用技术。  
  *zoomed acf and pacf plots of total power usage is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Acf Pacf Plots Zoomed / 02 Acf Pacf Plots Zoomed
# Complete Code / 完整代码
# ===============================

# zoomed acf and pacf plots of total power usage
from numpy import split
from numpy import array
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	train, test = data[1:-328], data[-328:-6]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test

# convert windows of weekly multivariate data into a series of total power
def to_series(data):
	# extract just the total power from each week
	series = [week[:, 0] for week in data]
	# flatten into a single series
	series = array(series).flatten()
	return series

# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
train, test = split_dataset(dataset.values)
# convert training data into a series
series = to_series(train)
# plots
pyplot.figure()
lags = 50
# acf
axis = pyplot.subplot(2, 1, 1)
plot_acf(series, ax=axis, lags=lags)
# pacf
axis = pyplot.subplot(2, 1, 2)
plot_pacf(series, ax=axis, lags=lags)
# show plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Arima Model

# 03 — Arima Model / ARIMA 模型

**Chapter 18 — File 3 of 3 / 第18章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **arima forecast for the power usage dataset**.

本脚本演示 **arima forecast for the power usage dataset**。

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
## Step 1 — arima forecast for the power usage dataset

```python
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
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
## Step 19 — convert windows of weekly multivariate data into a series of total power

```python
def to_series(data):
```

---
## Step 20 — extract just the total power from each week

```python
series = [week[:, 0] for week in data]
```

---
## Step 21 — flatten into a single series

```python
series = array(series).flatten()
	return series
```

---
## Step 22 — arima forecast

```python
def arima_forecast(history):
```

---
## Step 23 — convert history into a univariate series

```python
series = to_series(history)
```

---
## Step 24 — define the model

```python
model = ARIMA(series, order=(7,0,0))
```

---
## Step 25 — fit the model

```python
model_fit = model.fit()
```

---
## Step 26 — make forecast

```python
yhat = model_fit.predict(len(series), len(series)+6)
	return yhat
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
models['arima'] = arima_forecast
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

- **概念**: arima forecast for the power usage dataset 是机器学习中的常用技术。  
  *arima forecast for the power usage dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Arima Model / ARIMA 模型
# Complete Code / 完整代码
# ===============================

# arima forecast for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA

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

# convert windows of weekly multivariate data into a series of total power
def to_series(data):
	# extract just the total power from each week
	series = [week[:, 0] for week in data]
	# flatten into a single series
	series = array(series).flatten()
	return series

# arima forecast
def arima_forecast(history):
	# convert history into a univariate series
	series = to_series(history)
	# define the model
	model = ARIMA(series, order=(7,0,0))
	# fit the model
	model_fit = model.fit()
	# make forecast
	yhat = model_fit.predict(len(series), len(series)+6)
	return yhat

# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
train, test = split_dataset(dataset.values)
# define the names and functions for the models we wish to evaluate
models = dict()
models['arima'] = arima_forecast
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

# Chapter 18 Summary / 第18章总结

## Theme / 主题: Chapter 18 / Chapter 18

This chapter contains **3 code files** demonstrating chapter 18.

本章包含 **3 个代码文件**，演示Chapter 18。

---
## Evolution / 演化路线

  1. `01_acf_pacf_plots.ipynb` — Acf Pacf Plots
  2. `02_acf_pacf_plots_zoomed.ipynb` — Acf Pacf Plots Zoomed
  3. `03_arima_model.ipynb` — Arima Model

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 18) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 18）是机器学习流水线中的基础构建块。

---
