# 时间序列预测 / Time Series Forecasting with Python
## Chapter 24

---

### Chapter Summary / 章节总结

# Chapter 24 Summary / 第24章总结

## Theme / 主题: Chapter 24 / Chapter 24

This chapter contains **4 code files** demonstrating chapter 24.

本章包含 **4 个代码文件**，演示Chapter 24。

---
## Evolution / 演化路线

  1. `arima.ipynb` — Arima
  2. `arima_rolling_forecast.ipynb` — Arima Rolling Forecast
  3. `autocorrelation_plot.ipynb` — Autocorrelation Plot
  4. `load_and_plot.ipynb` — Load And Plot

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 24) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 24）是机器学习流水线中的基础构建块。

---

### Arima

# 01 — Arima / ARIMA 模型

**Chapter 24 — File 1 of 4 / 第24章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **fit an ARIMA model and plot residual errors**.

本脚本演示 **fit an ARIMA model and plot residual errors**。

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
## Step 1 — fit an ARIMA model and plot residual errors

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import datetime
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load dataset

```python
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
series.index = series.index.to_period('M')
```

---
## Step 3 — fit model

```python
model = ARIMA(series, order=(5,1,0))
# 训练模型 / Train the model
model_fit = model.fit()
```

---
## Step 4 — summary of fit model

```python
# 打印输出 / Print output
print(model_fit.summary())
```

---
## Step 5 — line plot of residuals

```python
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
```

---
## Step 6 — density plot of residuals

```python
residuals.plot(kind='kde')
pyplot.show()
```

---
## Step 7 — summary stats of residuals

```python
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(residuals.describe())
```

---
## Learning Notes / 学习笔记

- **概念**: fit an ARIMA model and plot residual errors 是机器学习中的常用技术。  
  *fit an ARIMA model and plot residual errors is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `describe()` | 统计摘要信息 | Statistical summary |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Arima / ARIMA 模型
# Complete Code / 完整代码
# ===============================

# fit an ARIMA model and plot residual errors
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import datetime
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
series.index = series.index.to_period('M')
# fit model
model = ARIMA(series, order=(5,1,0))
# 训练模型 / Train the model
model_fit = model.fit()
# summary of fit model
# 打印输出 / Print output
print(model_fit.summary())
# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()
# summary stats of residuals
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(residuals.describe())
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Arima Rolling Forecast

# 01 — Arima Rolling Forecast / ARIMA 模型

**Chapter 24 — File 2 of 4 / 第24章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **evaluate an ARIMA model using a walk-forward validation**.

本脚本演示 **evaluate an ARIMA model using a walk-forward validation**。

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
## Step 1 — evaluate an ARIMA model using a walk-forward validation

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import datetime
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
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
series.index = series.index.to_period('M')
```

---
## Step 3 — split into train and test sets

```python
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 获取长度 / Get length
size = int(len(X) * 0.66)
# 获取长度 / Get length
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
```

---
## Step 4 — walk-forward validation

```python
# 获取长度 / Get length
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
 # 训练模型 / Train the model
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
	obs = test[t]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
 # 打印输出 / Print output
	print('predicted=%f, expected=%f' % (yhat, obs))
```

---
## Step 5 — evaluate forecasts

```python
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test, predictions))
# 打印输出 / Print output
print('Test RMSE: %.3f' % rmse)
```

---
## Step 6 — plot forecasts against actual outcomes

```python
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate an ARIMA model using a walk-forward validation 是机器学习中的常用技术。  
  *evaluate an ARIMA model using a walk-forward validation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
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
# Arima Rolling Forecast / ARIMA 模型
# Complete Code / 完整代码
# ===============================

# evaluate an ARIMA model using a walk-forward validation
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import datetime
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
series.index = series.index.to_period('M')
# split into train and test sets
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 获取长度 / Get length
size = int(len(X) * 0.66)
# 获取长度 / Get length
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
# 获取长度 / Get length
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
 # 训练模型 / Train the model
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
	obs = test[t]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
 # 打印输出 / Print output
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test, predictions))
# 打印输出 / Print output
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Autocorrelation Plot

# 01 — Autocorrelation Plot / Autocorrelation Plot

**Chapter 24 — File 3 of 4 / 第24章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **autocorrelation plot of time series**.

本脚本演示 **autocorrelation plot of time series**。

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
## Step 1 — autocorrelation plot of time series

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import datetime
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import autocorrelation_plot
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
## Step 3 — autocorrelation plot

```python
autocorrelation_plot(series)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: autocorrelation plot of time series 是机器学习中的常用技术。  
  *autocorrelation plot of time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Autocorrelation Plot / Autocorrelation Plot
# Complete Code / 完整代码
# ===============================

# autocorrelation plot of time series
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import datetime
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import autocorrelation_plot
# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
# autocorrelation plot
autocorrelation_plot(series)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Load And Plot

# 01 — Load And Plot / Load And Plot

**Chapter 24 — File 4 of 4 / 第24章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **load and plot dataset**.

本脚本演示 **load and plot dataset**。

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
## Step 1 — load and plot dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import datetime
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
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
## Step 3 — summarize first few rows

```python
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head())
```

---
## Step 4 — line plot

```python
series.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load and plot dataset 是机器学习中的常用技术。  
  *load and plot dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `head()` | 查看前几行数据 | View first few rows |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load And Plot / Load And Plot
# Complete Code / 完整代码
# ===============================

# load and plot dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import datetime
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
# summarize first few rows
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head())
# line plot
series.plot()
pyplot.show()
```

---
