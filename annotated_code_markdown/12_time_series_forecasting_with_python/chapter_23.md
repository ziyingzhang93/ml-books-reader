# 时间序列预测 / Time Series Forecasting with Python
## Chapter 23

---

### Chapter Summary / 章节总结

# Chapter 23 Summary / 第23章总结

## Theme / 主题: Chapter 23 / Chapter 23

This chapter contains **4 code files** demonstrating chapter 23.

本章包含 **4 个代码文件**，演示Chapter 23。

---
## Evolution / 演化路线

  1. `autoregress_residuals.ipynb` — Autoregress Residuals
  2. `correct_predictions.ipynb` — Correct Predictions
  3. `persistence.ipynb` — Persistence
  4. `predict_residual_error.ipynb` — Predict Residual Error

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 23) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 23）是机器学习流水线中的基础构建块。

---

### Autoregress Residuals

# 01 — Autoregress Residuals / Autoregress Residuals

**Chapter 23 — File 1 of 4 / 第23章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **autoregressive model of residual errors**.

本脚本演示 **autoregressive model of residual errors**。

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
## Step 1 — autoregressive model of residual errors

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
from statsmodels.tsa.ar_model import AutoReg
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 2 — create lagged dataset

```python
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1']
```

---
## Step 3 — split into train and test sets

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
## Step 4 — persistence model on training set

```python
train_pred = [x for x in train_X]
```

---
## Step 5 — calculate residuals

```python
# 获取长度 / Get length
train_resid = [train_y[i]-train_pred[i] for i in range(len(train_pred))]
```

---
## Step 6 — model the training set residuals

```python
model = AutoReg(train_resid, lags=15)
# 训练模型 / Train the model
model_fit = model.fit()
# 打印输出 / Print output
print('Coef=%s' % (model_fit.params))
```

---
## Learning Notes / 学习笔记

- **概念**: autoregressive model of residual errors 是机器学习中的常用技术。  
  *autoregressive model of residual errors is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Autoregress Residuals / Autoregress Residuals
# Complete Code / 完整代码
# ===============================

# autoregressive model of residual errors
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
from statsmodels.tsa.ar_model import AutoReg
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# create lagged dataset
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1']
# split into train and test sets
# 转换为NumPy数组 / Convert to NumPy array
X = dataframe.values
# 获取长度 / Get length
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
# persistence model on training set
train_pred = [x for x in train_X]
# calculate residuals
# 获取长度 / Get length
train_resid = [train_y[i]-train_pred[i] for i in range(len(train_pred))]
# model the training set residuals
model = AutoReg(train_resid, lags=15)
# 训练模型 / Train the model
model_fit = model.fit()
# 打印输出 / Print output
print('Coef=%s' % (model_fit.params))
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Correct Predictions

# 01 — Correct Predictions / Correct Predictions

**Chapter 23 — File 2 of 4 / 第23章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **correct forecasts with a model of forecast residual errors**.

本脚本演示 **correct forecasts with a model of forecast residual errors**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 训练模型 / Train the model
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
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — correct forecasts with a model of forecast residual errors

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
from statsmodels.tsa.ar_model import AutoReg
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
```

---
## Step 2 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
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
## Step 5 — persistence model on training set

```python
train_pred = [x for x in train_X]
```

---
## Step 6 — calculate residuals

```python
# 获取长度 / Get length
train_resid = [train_y[i]-train_pred[i] for i in range(len(train_pred))]
```

---
## Step 7 — model the training set residuals

```python
window = 15
model = AutoReg(train_resid, lags=15)
# 训练模型 / Train the model
model_fit = model.fit()
coef = model_fit.params
```

---
## Step 8 — walk forward over time steps in test

```python
# 获取长度 / Get length
history = train_resid[len(train_resid)-window:]
# 获取长度 / Get length
history = [history[i] for i in range(len(history))]
predictions = list()
# 获取长度 / Get length
for t in range(len(test_y)):
```

---
## Step 9 — persistence

```python
yhat = test_X[t]
	error = test_y[t] - yhat
```

---
## Step 10 — predict error

```python
# 获取长度 / Get length
length = len(history)
 # 生成整数序列 / Generate integer sequence
	lag = [history[i] for i in range(length-window,length)]
	pred_error = coef[0]
 # 生成整数序列 / Generate integer sequence
	for d in range(window):
		pred_error += coef[d+1] * lag[window-d-1]
```

---
## Step 11 — correct the prediction

```python
yhat = yhat + pred_error
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
 # 添加元素到列表末尾 / Append element to list end
	history.append(error)
 # 打印输出 / Print output
	print('predicted=%f, expected=%f' % (yhat, test_y[t]))
```

---
## Step 12 — error

```python
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test_y, predictions))
# 打印输出 / Print output
print('Test RMSE: %.3f' % rmse)
```

---
## Step 13 — plot predicted error

```python
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: correct forecasts with a model of forecast residual errors 是机器学习中的常用技术。  
  *correct forecasts with a model of forecast residual errors is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Correct Predictions / Correct Predictions
# Complete Code / 完整代码
# ===============================

# correct forecasts with a model of forecast residual errors
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
from statsmodels.tsa.ar_model import AutoReg
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# create lagged dataset
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1']
# split into train and test sets
# 转换为NumPy数组 / Convert to NumPy array
X = dataframe.values
# 获取长度 / Get length
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
# persistence model on training set
train_pred = [x for x in train_X]
# calculate residuals
# 获取长度 / Get length
train_resid = [train_y[i]-train_pred[i] for i in range(len(train_pred))]
# model the training set residuals
window = 15
model = AutoReg(train_resid, lags=15)
# 训练模型 / Train the model
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
# 获取长度 / Get length
history = train_resid[len(train_resid)-window:]
# 获取长度 / Get length
history = [history[i] for i in range(len(history))]
predictions = list()
# 获取长度 / Get length
for t in range(len(test_y)):
	# persistence
	yhat = test_X[t]
	error = test_y[t] - yhat
	# predict error
 # 获取长度 / Get length
	length = len(history)
 # 生成整数序列 / Generate integer sequence
	lag = [history[i] for i in range(length-window,length)]
	pred_error = coef[0]
 # 生成整数序列 / Generate integer sequence
	for d in range(window):
		pred_error += coef[d+1] * lag[window-d-1]
	# correct the prediction
	yhat = yhat + pred_error
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
 # 添加元素到列表末尾 / Append element to list end
	history.append(error)
 # 打印输出 / Print output
	print('predicted=%f, expected=%f' % (yhat, test_y[t]))
# error
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test_y, predictions))
# 打印输出 / Print output
print('Test RMSE: %.3f' % rmse)
# plot predicted error
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Persistence

# 01 — Persistence / Persistence

**Chapter 23 — File 3 of 4 / 第23章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **calculate residual errors for a persistence forecast model**.

本脚本演示 **calculate residual errors for a persistence forecast model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
```

---
## Step 1 — calculate residual errors for a persistence forecast model

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
```

---
## Step 2 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
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
predictions = [x for x in test_X]
```

---
## Step 6 — skill of persistence model

```python
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test_y, predictions))
# 打印输出 / Print output
print('Test RMSE: %.3f' % rmse)
```

---
## Step 7 — calculate residuals

```python
# 获取长度 / Get length
residuals = [test_y[i]-predictions[i] for i in range(len(predictions))]
residuals = DataFrame(residuals)
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(residuals.head())
```

---
## Learning Notes / 学习笔记

- **概念**: calculate residual errors for a persistence forecast model 是机器学习中的常用技术。  
  *calculate residual errors for a persistence forecast model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `head()` | 查看前几行数据 | View first few rows |
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

# calculate residual errors for a persistence forecast model
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# create lagged dataset
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1']
# split into train and test sets
# 转换为NumPy数组 / Convert to NumPy array
X = dataframe.values
# 获取长度 / Get length
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
# persistence model
predictions = [x for x in test_X]
# skill of persistence model
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test_y, predictions))
# 打印输出 / Print output
print('Test RMSE: %.3f' % rmse)
# calculate residuals
# 获取长度 / Get length
residuals = [test_y[i]-predictions[i] for i in range(len(predictions))]
residuals = DataFrame(residuals)
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(residuals.head())
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Predict Residual Error

# 01 — Predict Residual Error / Predict Residual Error

**Chapter 23 — File 4 of 4 / 第23章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **forecast residual forecast error**.

本脚本演示 **forecast residual forecast error**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 训练模型 / Train the model
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
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — forecast residual forecast error

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
from statsmodels.tsa.ar_model import AutoReg
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 2 — create lagged dataset

```python
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1']
```

---
## Step 3 — split into train and test sets

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
## Step 4 — persistence model on training set

```python
train_pred = [x for x in train_X]
```

---
## Step 5 — calculate residuals

```python
# 获取长度 / Get length
train_resid = [train_y[i]-train_pred[i] for i in range(len(train_pred))]
```

---
## Step 6 — model the training set residuals

```python
window = 15
model = AutoReg(train_resid, lags=window)
# 训练模型 / Train the model
model_fit = model.fit()
coef = model_fit.params
```

---
## Step 7 — walk forward over time steps in test

```python
# 获取长度 / Get length
history = train_resid[len(train_resid)-window:]
# 获取长度 / Get length
history = [history[i] for i in range(len(history))]
predictions = list()
expected_error = list()
# 获取长度 / Get length
for t in range(len(test_y)):
```

---
## Step 8 — persistence

```python
yhat = test_X[t]
	error = test_y[t] - yhat
 # 添加元素到列表末尾 / Append element to list end
	expected_error.append(error)
```

---
## Step 9 — predict error

```python
# 获取长度 / Get length
length = len(history)
 # 生成整数序列 / Generate integer sequence
	lag = [history[i] for i in range(length-window,length)]
	pred_error = coef[0]
 # 生成整数序列 / Generate integer sequence
	for d in range(window):
		pred_error += coef[d+1] * lag[window-d-1]
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(pred_error)
 # 添加元素到列表末尾 / Append element to list end
	history.append(error)
 # 打印输出 / Print output
	print('predicted error=%f, expected error=%f' % (pred_error, error))
```

---
## Step 10 — plot predicted error

```python
pyplot.plot(expected_error)
pyplot.plot(predictions, color='red')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: forecast residual forecast error 是机器学习中的常用技术。  
  *forecast residual forecast error is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Predict Residual Error / Predict Residual Error
# Complete Code / 完整代码
# ===============================

# forecast residual forecast error
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
from statsmodels.tsa.ar_model import AutoReg
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# create lagged dataset
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
# 获取列名 / Get column names
dataframe.columns = ['t', 't+1']
# split into train and test sets
# 转换为NumPy数组 / Convert to NumPy array
X = dataframe.values
# 获取长度 / Get length
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
# persistence model on training set
train_pred = [x for x in train_X]
# calculate residuals
# 获取长度 / Get length
train_resid = [train_y[i]-train_pred[i] for i in range(len(train_pred))]
# model the training set residuals
window = 15
model = AutoReg(train_resid, lags=window)
# 训练模型 / Train the model
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
# 获取长度 / Get length
history = train_resid[len(train_resid)-window:]
# 获取长度 / Get length
history = [history[i] for i in range(len(history))]
predictions = list()
expected_error = list()
# 获取长度 / Get length
for t in range(len(test_y)):
	# persistence
	yhat = test_X[t]
	error = test_y[t] - yhat
 # 添加元素到列表末尾 / Append element to list end
	expected_error.append(error)
	# predict error
 # 获取长度 / Get length
	length = len(history)
 # 生成整数序列 / Generate integer sequence
	lag = [history[i] for i in range(length-window,length)]
	pred_error = coef[0]
 # 生成整数序列 / Generate integer sequence
	for d in range(window):
		pred_error += coef[d+1] * lag[window-d-1]
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(pred_error)
 # 添加元素到列表末尾 / Append element to list end
	history.append(error)
 # 打印输出 / Print output
	print('predicted error=%f, expected error=%f' % (pred_error, error))
# plot predicted error
pyplot.plot(expected_error)
pyplot.plot(predictions, color='red')
pyplot.show()
```

---
