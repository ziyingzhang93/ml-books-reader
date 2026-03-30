# 时间序列预测
## Chapter 30

---

### Chapter Summary

# Chapter 30 Summary / 第30章总结

## Theme / 主题: Chapter 30 / Chapter 30

This chapter contains **17 code files** demonstrating chapter 30.

本章包含 **17 个代码文件**，演示Chapter 30。

---
## Evolution / 演化路线

  1. `analysis_boxplots.ipynb` — Analysis Boxplots
  2. `analysis_density_plots.ipynb` — Analysis Density Plots
  3. `analysis_line_plot.ipynb` — Analysis Line Plot
  4. `analysis_summary.ipynb` — Analysis Summary
  5. `finalize_load_predict.ipynb` — Finalize Load Predict
  6. `finalize_save.ipynb` — Finalize Save
  7. `finalize_validate.ipynb` — Finalize Validate
  8. `models_acf_pacf.ipynb` — Models Acf Pacf
  9. `models_arima_boxcox.ipynb` — Models Arima Boxcox
  10. `models_boxcox.ipynb` — Models Boxcox
  11. `models_grid_search_arima.ipynb` — Models Grid Search Arima
  12. `models_manual_arima.ipynb` — Models Manual Arima
  13. `models_plot_residuals.ipynb` — Models Plot Residuals
  14. `models_plot_residuals_acf.ipynb` — Models Plot Residuals Acf
  15. `models_stationary.ipynb` — Models Stationary
  16. `persistence.ipynb` — Persistence
  17. `split_dataset.ipynb` — Split Dataset

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 30) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 30）是机器学习流水线中的基础构建块。

---

### Analysis Boxplots

# 01 — Analysis Boxplots / Analysis Boxplots

**Chapter 30 — File 1 of 17 / 第30章 — 第1个文件（共17个）**

---

## Summary / 总结

This script demonstrates **boxplots of time series**.

本脚本演示 **boxplots of time series**。

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
## Step 1 — boxplots of time series

```python
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
print(series)
groups = series['1966':'1973'].groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
	years[name.year] = group.values
years.boxplot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: boxplots of time series 是机器学习中的常用技术。  
  *boxplots of time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `groupby` | 分组聚合 | Group and aggregate |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Analysis Boxplots / Analysis Boxplots
# Complete Code / 完整代码
# ===============================

# boxplots of time series
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
print(series)
groups = series['1966':'1973'].groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
	years[name.year] = group.values
years.boxplot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 17

---

### Analysis Density Plots

# 01 — Analysis Density Plots / Analysis Density Plots

**Chapter 30 — File 2 of 17 / 第30章 — 第2个文件（共17个）**

---

## Summary / 总结

This script demonstrates **density plots of time series**.

本脚本演示 **density plots of time series**。

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
## Step 1 — density plots of time series

```python
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
pyplot.figure(1)
pyplot.subplot(211)
series.hist()
pyplot.subplot(212)
series.plot(kind='kde')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: density plots of time series 是机器学习中的常用技术。  
  *density plots of time series is a common technique in machine learning.*

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
# Analysis Density Plots / Analysis Density Plots
# Complete Code / 完整代码
# ===============================

# density plots of time series
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
pyplot.figure(1)
pyplot.subplot(211)
series.hist()
pyplot.subplot(212)
series.plot(kind='kde')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 17

---

### Analysis Line Plot

# 01 — Analysis Line Plot / Analysis Line Plot

**Chapter 30 — File 3 of 17 / 第30章 — 第3个文件（共17个）**

---

## Summary / 总结

This script demonstrates **line plots of time series**.

本脚本演示 **line plots of time series**。

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
## Step 1 — line plots of time series

```python
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
series.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: line plots of time series 是机器学习中的常用技术。  
  *line plots of time series is a common technique in machine learning.*

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
# Analysis Line Plot / Analysis Line Plot
# Complete Code / 完整代码
# ===============================

# line plots of time series
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
series.plot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 17

---

### Analysis Summary

# 01 — Analysis Summary / Analysis Summary

**Chapter 30 — File 4 of 17 / 第30章 — 第4个文件（共17个）**

---

## Summary / 总结

This script demonstrates **summary statistics of time series**.

本脚本演示 **summary statistics of time series**。

---
## Step 1 — summary statistics of time series

```python
from pandas import read_csv
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
print(series.describe())
```

---
## Learning Notes / 学习笔记

- **概念**: summary statistics of time series 是机器学习中的常用技术。  
  *summary statistics of time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Analysis Summary / Analysis Summary
# Complete Code / 完整代码
# ===============================

# summary statistics of time series
from pandas import read_csv
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
print(series.describe())
```

---

➡️ **Next / 下一步**: File 5 of 17

---

### Finalize Load Predict

# 01 — Finalize Load Predict / Finalize Load Predict

**Chapter 30 — File 5 of 17 / 第30章 — 第5个文件（共17个）**

---

## Summary / 总结

This script demonstrates **load the finalized model and make a prediction**.

本脚本演示 **load the finalized model and make a prediction**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — load the finalized model and make a prediction

```python
from statsmodels.tsa.arima.model import ARIMAResults
from math import exp
from math import log
import numpy
```

---
## Step 2 — invert box-cox transform

```python
def boxcox_inverse(value, lam):
	if lam == 0:
		return exp(value)
	return exp(log(lam * value + 1) / lam)

model_fit = ARIMAResults.load('model.pkl')
lam = numpy.load('model_lambda.npy')
yhat = model_fit.forecast()[0]
yhat = boxcox_inverse(yhat, lam)
print('Predicted: %.3f' % yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: load the finalized model and make a prediction 是机器学习中的常用技术。  
  *load the finalized model and make a prediction is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Finalize Load Predict / Finalize Load Predict
# Complete Code / 完整代码
# ===============================

# load the finalized model and make a prediction
from statsmodels.tsa.arima.model import ARIMAResults
from math import exp
from math import log
import numpy

# invert box-cox transform
def boxcox_inverse(value, lam):
	if lam == 0:
		return exp(value)
	return exp(log(lam * value + 1) / lam)

model_fit = ARIMAResults.load('model.pkl')
lam = numpy.load('model_lambda.npy')
yhat = model_fit.forecast()[0]
yhat = boxcox_inverse(yhat, lam)
print('Predicted: %.3f' % yhat)
```

---

➡️ **Next / 下一步**: File 6 of 17

---

### Finalize Save

# 01 — Finalize Save / 保存/加载模型

**Chapter 30 — File 6 of 17 / 第30章 — 第6个文件（共17个）**

---

## Summary / 总结

This script demonstrates **finalize model and save to file**.

本脚本演示 **finalize model and save to file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — finalize model and save to file

```python
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import boxcox
import numpy
```

---
## Step 2 — load data

```python
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 3 — prepare data

```python
X = series.values
X = X.astype('float32')
```

---
## Step 4 — transform data

```python
transformed, lam = boxcox(X)
```

---
## Step 5 — fit model

```python
model = ARIMA(transformed, order=(0,1,2))
model_fit = model.fit()
```

---
## Step 6 — save model

```python
model_fit.save('model.pkl')
numpy.save('model_lambda.npy', [lam])
```

---
## Learning Notes / 学习笔记

- **概念**: finalize model and save to file 是机器学习中的常用技术。  
  *finalize model and save to file is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Finalize Save / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# finalize model and save to file
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import boxcox
import numpy
# load data
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values
X = X.astype('float32')
# transform data
transformed, lam = boxcox(X)
# fit model
model = ARIMA(transformed, order=(0,1,2))
model_fit = model.fit()
# save model
model_fit.save('model.pkl')
numpy.save('model_lambda.npy', [lam])
```

---

➡️ **Next / 下一步**: File 7 of 17

---

### Models Acf Pacf

# 01 — Models Acf Pacf / Models Acf Pacf

**Chapter 30 — File 8 of 17 / 第30章 — 第8个文件（共17个）**

---

## Summary / 总结

This script demonstrates **ACF and PACF plots of time series**.

本脚本演示 **ACF and PACF plots of time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
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
## Step 1 — ACF and PACF plots of time series

```python
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
pyplot.figure()
pyplot.subplot(211)
plot_acf(series, lags=50, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(series, lags=50, ax=pyplot.gca())
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: ACF and PACF plots of time series 是机器学习中的常用技术。  
  *ACF and PACF plots of time series is a common technique in machine learning.*

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
# Models Acf Pacf / Models Acf Pacf
# Complete Code / 完整代码
# ===============================

# ACF and PACF plots of time series
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
pyplot.figure()
pyplot.subplot(211)
plot_acf(series, lags=50, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(series, lags=50, ax=pyplot.gca())
pyplot.show()
```

---

➡️ **Next / 下一步**: File 9 of 17

---

### Models Arima Boxcox

# 01 — Models Arima Boxcox / ARIMA 模型

**Chapter 30 — File 9 of 17 / 第30章 — 第9个文件（共17个）**

---

## Summary / 总结

This script demonstrates **evaluate ARIMA models with box-cox transformed time series**.

本脚本演示 **evaluate ARIMA models with box-cox transformed time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
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
## Step 1 — evaluate ARIMA models with box-cox transformed time series

```python
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from math import log
from math import exp
from scipy.stats import boxcox
```

---
## Step 2 — invert box-cox transform

```python
def boxcox_inverse(value, lam):
	if lam == 0:
		return exp(value)
	return exp(log(lam * value + 1) / lam)
```

---
## Step 3 — load data

```python
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 4 — prepare data

```python
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
```

---
## Step 5 — walk-forward validation

```python
history = [x for x in train]
predictions = list()
for i in range(len(test)):
```

---
## Step 6 — transform

```python
transformed, lam = boxcox(history)
	if lam < -5:
		transformed, lam = history, 1
```

---
## Step 7 — predict

```python
model = ARIMA(transformed, order=(0,1,2))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
```

---
## Step 8 — invert transformed prediction

```python
yhat = boxcox_inverse(yhat, lam)
	predictions.append(yhat)
```

---
## Step 9 — observation

```python
obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
```

---
## Step 10 — report performance

```python
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate ARIMA models with box-cox transformed time series 是机器学习中的常用技术。  
  *evaluate ARIMA models with box-cox transformed time series is a common technique in machine learning.*

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
# Models Arima Boxcox / ARIMA 模型
# Complete Code / 完整代码
# ===============================

# evaluate ARIMA models with box-cox transformed time series
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from math import log
from math import exp
from scipy.stats import boxcox

# invert box-cox transform
def boxcox_inverse(value, lam):
	if lam == 0:
		return exp(value)
	return exp(log(lam * value + 1) / lam)

# load data
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# transform
	transformed, lam = boxcox(history)
	if lam < -5:
		transformed, lam = history, 1
	# predict
	model = ARIMA(transformed, order=(0,1,2))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	# invert transformed prediction
	yhat = boxcox_inverse(yhat, lam)
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
```

---

➡️ **Next / 下一步**: File 10 of 17

---

### Models Boxcox

# 01 — Models Boxcox / Models Boxcox

**Chapter 30 — File 10 of 17 / 第30章 — 第10个文件（共17个）**

---

## Summary / 总结

This script demonstrates **plots of box-cox transformed dataset**.

本脚本演示 **plots of box-cox transformed dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
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
## Step 1 — plots of box-cox transformed dataset

```python
from pandas import read_csv
from scipy.stats import boxcox
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
X = series.values
transformed, lam = boxcox(X)
print('Lambda: %f' % lam)
pyplot.figure(1)
```

---
## Step 2 — line plot

```python
pyplot.subplot(311)
pyplot.plot(transformed)
```

---
## Step 3 — histogram

```python
pyplot.subplot(312)
pyplot.hist(transformed)
```

---
## Step 4 — q-q plot

```python
pyplot.subplot(313)
qqplot(transformed, line='r', ax=pyplot.gca())
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plots of box-cox transformed dataset 是机器学习中的常用技术。  
  *plots of box-cox transformed dataset is a common technique in machine learning.*

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
# Models Boxcox / Models Boxcox
# Complete Code / 完整代码
# ===============================

# plots of box-cox transformed dataset
from pandas import read_csv
from scipy.stats import boxcox
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
X = series.values
transformed, lam = boxcox(X)
print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(311)
pyplot.plot(transformed)
# histogram
pyplot.subplot(312)
pyplot.hist(transformed)
# q-q plot
pyplot.subplot(313)
qqplot(transformed, line='r', ax=pyplot.gca())
pyplot.show()
```

---

➡️ **Next / 下一步**: File 11 of 17

---

### Models Grid Search Arima

# 01 — Models Grid Search Arima / ARIMA 模型

**Chapter 30 — File 11 of 17 / 第30章 — 第11个文件（共17个）**

---

## Summary / 总结

This script demonstrates **grid search ARIMA parameters for time series**.

本脚本演示 **grid search ARIMA parameters for time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

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
## Step 1 — grid search ARIMA parameters for time series

```python
import warnings
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
```

---
## Step 2 — evaluate an ARIMA model for a given order (p,d,q) and return RMSE

```python
def evaluate_arima_model(X, arima_order):
```

---
## Step 3 — prepare training dataset

```python
X = X.astype('float32')
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
```

---
## Step 4 — make predictions

```python
predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
```

---
## Step 5 — calculate out of sample error

```python
rmse = sqrt(mean_squared_error(test, predictions))
	return rmse
```

---
## Step 6 — evaluate combinations of p, d and q values for an ARIMA model

```python
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
```

---
## Step 7 — load dataset

```python
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 8 — evaluate parameters

```python
p_values = range(0,13)
d_values = range(0, 4)
q_values = range(0, 13)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```

---
## Learning Notes / 学习笔记

- **概念**: grid search ARIMA parameters for time series 是机器学习中的常用技术。  
  *grid search ARIMA parameters for time series is a common technique in machine learning.*

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
# Models Grid Search Arima / ARIMA 模型
# Complete Code / 完整代码
# ===============================

# grid search ARIMA parameters for time series
import warnings
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = sqrt(mean_squared_error(test, predictions))
	return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# load dataset
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# evaluate parameters
p_values = range(0,13)
d_values = range(0, 4)
q_values = range(0, 13)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```

---

➡️ **Next / 下一步**: File 12 of 17

---

### Models Manual Arima

# 01 — Models Manual Arima / ARIMA 模型

**Chapter 30 — File 12 of 17 / 第30章 — 第12个文件（共17个）**

---

## Summary / 总结

This script demonstrates **evaluate manually configured ARIMA model**.

本脚本演示 **evaluate manually configured ARIMA model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
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
## Step 1 — evaluate manually configured ARIMA model

```python
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
```

---
## Step 2 — load data

```python
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 3 — prepare data

```python
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
```

---
## Step 4 — walk-forward validation

```python
history = [x for x in train]
predictions = list()
for i in range(len(test)):
```

---
## Step 5 — predict

```python
model = ARIMA(history, order=(0,1,2))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
```

---
## Step 6 — observation

```python
obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
```

---
## Step 7 — report performance

```python
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate manually configured ARIMA model 是机器学习中的常用技术。  
  *evaluate manually configured ARIMA model is a common technique in machine learning.*

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
# Models Manual Arima / ARIMA 模型
# Complete Code / 完整代码
# ===============================

# evaluate manually configured ARIMA model
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
# load data
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	model = ARIMA(history, order=(0,1,2))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
```

---

➡️ **Next / 下一步**: File 13 of 17

---

### Models Plot Residuals

# 01 — Models Plot Residuals / Models Plot Residuals

**Chapter 30 — File 13 of 17 / 第30章 — 第13个文件（共17个）**

---

## Summary / 总结

This script demonstrates **plot residual errors for ARIMA model**.

本脚本演示 **plot residual errors for ARIMA model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
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
## Step 1 — plot residual errors for ARIMA model

```python
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
```

---
## Step 2 — load data

```python
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 3 — prepare data

```python
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
```

---
## Step 4 — walk-forward validation

```python
history = [x for x in train]
predictions = list()
for i in range(len(test)):
```

---
## Step 5 — predict

```python
model = ARIMA(history, order=(0,1,2))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
```

---
## Step 6 — observation

```python
obs = test[i]
	history.append(obs)
```

---
## Step 7 — errors

```python
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot residual errors for ARIMA model 是机器学习中的常用技术。  
  *plot residual errors for ARIMA model is a common technique in machine learning.*

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
# Models Plot Residuals / Models Plot Residuals
# Complete Code / 完整代码
# ===============================

# plot residual errors for ARIMA model
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
# load data
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	model = ARIMA(history, order=(0,1,2))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()
```

---

➡️ **Next / 下一步**: File 14 of 17

---

### Models Plot Residuals Acf

# 01 — Models Plot Residuals Acf / Models Plot Residuals Acf

**Chapter 30 — File 14 of 17 / 第30章 — 第14个文件（共17个）**

---

## Summary / 总结

This script demonstrates **ACF and PACF plots of forecast residual errors**.

本脚本演示 **ACF and PACF plots of forecast residual errors**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
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
## Step 1 — ACF and PACF plots of forecast residual errors

```python
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
```

---
## Step 2 — load data

```python
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 3 — prepare data

```python
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
```

---
## Step 4 — walk-forward validation

```python
history = [x for x in train]
predictions = list()
for i in range(len(test)):
```

---
## Step 5 — predict

```python
model = ARIMA(history, order=(0,1,2))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
```

---
## Step 6 — observation

```python
obs = test[i]
	history.append(obs)
```

---
## Step 7 — errors

```python
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
pyplot.figure()
pyplot.subplot(211)
plot_acf(residuals, lags=25, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(residuals, lags=25, ax=pyplot.gca())
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: ACF and PACF plots of forecast residual errors 是机器学习中的常用技术。  
  *ACF and PACF plots of forecast residual errors is a common technique in machine learning.*

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
# Models Plot Residuals Acf / Models Plot Residuals Acf
# Complete Code / 完整代码
# ===============================

# ACF and PACF plots of forecast residual errors
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
# load data
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	model = ARIMA(history, order=(0,1,2))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
pyplot.figure()
pyplot.subplot(211)
plot_acf(residuals, lags=25, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(residuals, lags=25, ax=pyplot.gca())
pyplot.show()
```

---

➡️ **Next / 下一步**: File 15 of 17

---

### Models Stationary

# 01 — Models Stationary / Models Stationary

**Chapter 30 — File 15 of 17 / 第30章 — 第15个文件（共17个）**

---

## Summary / 总结

This script demonstrates **statistical test for the stationarity of the time series**.

本脚本演示 **statistical test for the stationarity of the time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture


---
## Step 1 — statistical test for the stationarity of the time series

```python
from pandas import read_csv
from pandas import Series
from statsmodels.tsa.stattools import adfuller
```

---
## Step 2 — create a differenced time series

```python
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return Series(diff)

series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
X = series.values
```

---
## Step 3 — difference data

```python
stationary = difference(X)
stationary.index = series.index[1:]
```

---
## Step 4 — check if stationary

```python
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
```

---
## Step 5 — save

```python
stationary.to_csv('stationary.csv', header=False)
```

---
## Learning Notes / 学习笔记

- **概念**: statistical test for the stationarity of the time series 是机器学习中的常用技术。  
  *statistical test for the stationarity of the time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Models Stationary / Models Stationary
# Complete Code / 完整代码
# ===============================

# statistical test for the stationarity of the time series
from pandas import read_csv
from pandas import Series
from statsmodels.tsa.stattools import adfuller

# create a differenced time series
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return Series(diff)

series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
X = series.values
# difference data
stationary = difference(X)
stationary.index = series.index[1:]
# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv', header=False)
```

---

➡️ **Next / 下一步**: File 16 of 17

---

### Persistence

# 01 — Persistence / Persistence

**Chapter 30 — File 16 of 17 / 第30章 — 第16个文件（共17个）**

---

## Summary / 总结

This script demonstrates **evaluate a persistence model**.

本脚本演示 **evaluate a persistence model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — evaluate a persistence model

```python
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
```

---
## Step 2 — load data

```python
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 3 — prepare data

```python
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
```

---
## Step 4 — walk-forward validation

```python
history = [x for x in train]
predictions = list()
for i in range(len(test)):
```

---
## Step 5 — predict

```python
yhat = history[-1]
	predictions.append(yhat)
```

---
## Step 6 — observation

```python
obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
```

---
## Step 7 — report performance

```python
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate a persistence model 是机器学习中的常用技术。  
  *evaluate a persistence model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
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

# evaluate a persistence model
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	yhat = history[-1]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
```

---

➡️ **Next / 下一步**: File 17 of 17

---

### Split Dataset

# 01 — Split Dataset / Split Dataset

**Chapter 30 — File 17 of 17 / 第30章 — 第17个文件（共17个）**

---

## Summary / 总结

This script demonstrates **split into a training and validation dataset**.

本脚本演示 **split into a training and validation dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — split into a training and validation dataset

```python
from pandas import read_csv
series = read_csv('robberies.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', header=False)
validation.to_csv('validation.csv', header=False)
```

---
## Learning Notes / 学习笔记

- **概念**: split into a training and validation dataset 是机器学习中的常用技术。  
  *split into a training and validation dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Split Dataset / Split Dataset
# Complete Code / 完整代码
# ===============================

# split into a training and validation dataset
from pandas import read_csv
series = read_csv('robberies.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', header=False)
validation.to_csv('validation.csv', header=False)
```

---
