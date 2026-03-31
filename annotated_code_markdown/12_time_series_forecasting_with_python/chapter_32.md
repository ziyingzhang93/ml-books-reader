# 时间序列预测 / Time Series Forecasting with Python
## Chapter 32

---

### Chapter Summary / 章节总结



---

### Analysis Boxplots

# 01 — Analysis Boxplots / Analysis Boxplots

**Chapter 32 — File 1 of 17 / 第32章 — 第1个文件（共17个）**

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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Grouper
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
groups = series['1964':'1970'].groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
 # 转换为NumPy数组 / Convert to NumPy array
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Grouper
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
groups = series['1964':'1970'].groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
 # 转换为NumPy数组 / Convert to NumPy array
	years[name.year] = group.values
years.boxplot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 17

---

### Analysis Density Plots

# 01 — Analysis Density Plots / Analysis Density Plots

**Chapter 32 — File 2 of 17 / 第32章 — 第2个文件（共17个）**

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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
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

**Chapter 32 — File 3 of 17 / 第32章 — 第3个文件（共17个）**

---

## Summary / 总结

This script demonstrates **line plot of time series**.

本脚本演示 **line plot of time series**。

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
## Step 1 — line plot of time series

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
series.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: line plot of time series 是机器学习中的常用技术。  
  *line plot of time series is a common technique in machine learning.*

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

# line plot of time series
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
series.plot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 17

---

### Analysis Multiple Line Plots

# 01 — Analysis Multiple Line Plots / Analysis Multiple Line Plots

**Chapter 32 — File 4 of 17 / 第32章 — 第4个文件（共17个）**

---

## Summary / 总结

This script demonstrates **multiple line plots of time series**.

本脚本演示 **multiple line plots of time series**。

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
## Step 1 — multiple line plots of time series

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Grouper
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
groups = series['1964':'1970'].groupby(Grouper(freq='A'))
years = DataFrame()
pyplot.figure()
i = 1
# 获取长度 / Get length
n_groups = len(groups)
for name, group in groups:
	pyplot.subplot((n_groups*100) + 10 + i)
	i += 1
	pyplot.plot(group)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: multiple line plots of time series 是机器学习中的常用技术。  
  *multiple line plots of time series is a common technique in machine learning.*

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
# Analysis Multiple Line Plots / Analysis Multiple Line Plots
# Complete Code / 完整代码
# ===============================

# multiple line plots of time series
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Grouper
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
groups = series['1964':'1970'].groupby(Grouper(freq='A'))
years = DataFrame()
pyplot.figure()
i = 1
# 获取长度 / Get length
n_groups = len(groups)
for name, group in groups:
	pyplot.subplot((n_groups*100) + 10 + i)
	i += 1
	pyplot.plot(group)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 17

---

### Analysis Summary

# 01 — Analysis Summary / Analysis Summary

**Chapter 32 — File 5 of 17 / 第32章 — 第5个文件（共17个）**

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

➡️ **Next / 下一步**: File 6 of 17

---

### Finalize Load Predict



---

### Finalize Save

# 01 — Finalize Save / 保存/加载模型

**Chapter 32 — File 7 of 17 / 第32章 — 第7个文件（共17个）**

---

## Summary / 总结

This script demonstrates **save finalized model**.

本脚本演示 **save finalized model**。

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
## Step 1 — save finalized model

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
```

---
## Step 2 — create a differenced series

```python
def difference(dataset, interval=1):
	diff = list()
 # 获取长度 / Get length
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
  # 添加元素到列表末尾 / Append element to list end
		diff.append(value)
	return diff
```

---
## Step 3 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 4 — prepare data

```python
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
```

---
## Step 5 — difference data

```python
months_in_year = 12
diff = difference(X, months_in_year)
```

---
## Step 6 — fit model

```python
model = ARIMA(diff, order=(0,0,1))
# 训练模型 / Train the model
model_fit = model.fit()
```

---
## Step 7 — bias constant, could be calculated from in-sample mean residual

```python
bias = 165.904728
```

---
## Step 8 — save model

```python
model_fit.save('model.pkl')
numpy.save('model_bias.npy', [bias])
```

---
## Learning Notes / 学习笔记

- **概念**: save finalized model 是机器学习中的常用技术。  
  *save finalized model is a common technique in machine learning.*

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

# save finalized model
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
 # 获取长度 / Get length
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
  # 添加元素到列表末尾 / Append element to list end
		diff.append(value)
	return diff

# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# difference data
months_in_year = 12
diff = difference(X, months_in_year)
# fit model
model = ARIMA(diff, order=(0,0,1))
# 训练模型 / Train the model
model_fit = model.fit()
# bias constant, could be calculated from in-sample mean residual
bias = 165.904728
# save model
model_fit.save('model.pkl')
numpy.save('model_bias.npy', [bias])
```

---

➡️ **Next / 下一步**: File 8 of 17

---

### Finalize Validate

# 01 — Finalize Validate / Finalize Validate

**Chapter 32 — File 8 of 17 / 第32章 — 第8个文件（共17个）**

---

## Summary / 总结

This script demonstrates **load and evaluate the finalized model on the validation dataset**.

本脚本演示 **load and evaluate the finalized model on the validation dataset**。

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
## Step 1 — load and evaluate the finalized model on the validation dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
```

---
## Step 2 — create a differenced series

```python
def difference(dataset, interval=1):
	diff = list()
 # 获取长度 / Get length
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
  # 添加元素到列表末尾 / Append element to list end
		diff.append(value)
	return diff
```

---
## Step 3 — invert differenced value

```python
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
```

---
## Step 4 — load and prepare datasets

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = dataset.values.astype('float32')
history = [x for x in X]
months_in_year = 12
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
validation = read_csv('validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
y = validation.values.astype('float32')
```

---
## Step 5 — load model

```python
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
```

---
## Step 6 — make first prediction

```python
predictions = list()
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(history, yhat, months_in_year)
# 添加元素到列表末尾 / Append element to list end
predictions.append(yhat)
# 添加元素到列表末尾 / Append element to list end
history.append(y[0])
# 打印输出 / Print output
print('>Predicted=%.3f, Expected=%.3f' % (yhat, y[0]))
```

---
## Step 7 — rolling forecasts

```python
# 获取长度 / Get length
for i in range(1, len(y)):
```

---
## Step 8 — difference data

```python
months_in_year = 12
	diff = difference(history, months_in_year)
```

---
## Step 9 — predict

```python
model = ARIMA(diff, order=(0,0,1))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = bias + inverse_difference(history, yhat, months_in_year)
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
```

---
## Step 10 — observation

```python
obs = y[i]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
 # 打印输出 / Print output
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
```

---
## Step 11 — report performance

```python
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(y, predictions))
# 打印输出 / Print output
print('RMSE: %.3f' % rmse)
pyplot.plot(y)
pyplot.plot(predictions, color='red')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load and evaluate the finalized model on the validation dataset 是机器学习中的常用技术。  
  *load and evaluate the finalized model on the validation dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
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
# Finalize Validate / Finalize Validate
# Complete Code / 完整代码
# ===============================

# load and evaluate the finalized model on the validation dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
 # 获取长度 / Get length
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
  # 添加元素到列表末尾 / Append element to list end
		diff.append(value)
	return diff

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load and prepare datasets
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = dataset.values.astype('float32')
history = [x for x in X]
months_in_year = 12
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
validation = read_csv('validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
# make first prediction
predictions = list()
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(history, yhat, months_in_year)
# 添加元素到列表末尾 / Append element to list end
predictions.append(yhat)
# 添加元素到列表末尾 / Append element to list end
history.append(y[0])
# 打印输出 / Print output
print('>Predicted=%.3f, Expected=%.3f' % (yhat, y[0]))
# rolling forecasts
# 获取长度 / Get length
for i in range(1, len(y)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(0,0,1))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = bias + inverse_difference(history, yhat, months_in_year)
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
	# observation
	obs = y[i]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
 # 打印输出 / Print output
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(y, predictions))
# 打印输出 / Print output
print('RMSE: %.3f' % rmse)
pyplot.plot(y)
pyplot.plot(predictions, color='red')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 9 of 17

---

### Harness Split



---

### Models Acf Pacf

# 01 — Models Acf Pacf / Models Acf Pacf

**Chapter 32 — File 10 of 17 / 第32章 — 第10个文件（共17个）**

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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('stationary.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
pyplot.figure()
pyplot.subplot(211)
plot_acf(series, lags=25, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(series, lags=25, ax=pyplot.gca())
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('stationary.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
pyplot.figure()
pyplot.subplot(211)
plot_acf(series, lags=25, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(series, lags=25, ax=pyplot.gca())
pyplot.show()
```

---

➡️ **Next / 下一步**: File 11 of 17

---

### Models Bias Corrected Acf

# 01 — Models Bias Corrected Acf / Models Bias Corrected Acf

**Chapter 32 — File 11 of 17 / 第32章 — 第11个文件（共17个）**

---

## Summary / 总结

This script demonstrates **ACF and PACF plots of residual errors of bias corrected forecasts**.

本脚本演示 **ACF and PACF plots of residual errors of bias corrected forecasts**。

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
## Step 1 — ACF and PACF plots of residual errors of bias corrected forecasts

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
```

---
## Step 2 — create a differenced series

```python
def difference(dataset, interval=1):
	diff = list()
 # 获取长度 / Get length
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
  # 添加元素到列表末尾 / Append element to list end
		diff.append(value)
	return diff
```

---
## Step 3 — invert differenced value

```python
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
```

---
## Step 4 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 5 — prepare data

```python
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 获取长度 / Get length
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
```

---
## Step 6 — walk-forward validation

```python
history = [x for x in train]
predictions = list()
# 获取长度 / Get length
for i in range(len(test)):
```

---
## Step 7 — difference data

```python
months_in_year = 12
	diff = difference(history, months_in_year)
```

---
## Step 8 — predict

```python
model = ARIMA(diff, order=(0,0,1))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history, yhat, months_in_year)
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
```

---
## Step 9 — observation

```python
obs = test[i]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
```

---
## Step 10 — errors

```python
# 获取长度 / Get length
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(residuals.describe())
```

---
## Step 11 — plot

```python
pyplot.figure()
pyplot.subplot(211)
plot_acf(residuals, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(residuals, ax=pyplot.gca())
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: ACF and PACF plots of residual errors of bias corrected forecasts 是机器学习中的常用技术。  
  *ACF and PACF plots of residual errors of bias corrected forecasts is a common technique in machine learning.*

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
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Models Bias Corrected Acf / Models Bias Corrected Acf
# Complete Code / 完整代码
# ===============================

# ACF and PACF plots of residual errors of bias corrected forecasts
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
 # 获取长度 / Get length
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
  # 添加元素到列表末尾 / Append element to list end
		diff.append(value)
	return diff

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 获取长度 / Get length
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
# 获取长度 / Get length
for i in range(len(test)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(0,0,1))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history, yhat, months_in_year)
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
	# observation
	obs = test[i]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
# errors
# 获取长度 / Get length
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
plot_acf(residuals, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(residuals, ax=pyplot.gca())
pyplot.show()
```

---

➡️ **Next / 下一步**: File 12 of 17

---

### Models Bias Corrected Residuals

# 01 — Models Bias Corrected Residuals / Models Bias Corrected Residuals

**Chapter 32 — File 12 of 17 / 第32章 — 第12个文件（共17个）**

---

## Summary / 总结

This script demonstrates **plots of residual errors of bias corrected forecasts**.

本脚本演示 **plots of residual errors of bias corrected forecasts**。

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
## Step 1 — plots of residual errors of bias corrected forecasts

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
```

---
## Step 2 — create a differenced series

```python
def difference(dataset, interval=1):
	diff = list()
 # 获取长度 / Get length
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
  # 添加元素到列表末尾 / Append element to list end
		diff.append(value)
	return diff
```

---
## Step 3 — invert differenced value

```python
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
```

---
## Step 4 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 5 — prepare data

```python
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 获取长度 / Get length
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
```

---
## Step 6 — walk-forward validation

```python
history = [x for x in train]
predictions = list()
bias = 165.904728
# 获取长度 / Get length
for i in range(len(test)):
```

---
## Step 7 — difference data

```python
months_in_year = 12
	diff = difference(history, months_in_year)
```

---
## Step 8 — predict

```python
model = ARIMA(diff, order=(0,0,1))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = bias + inverse_difference(history, yhat, months_in_year)
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
```

---
## Step 9 — observation

```python
obs = test[i]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
```

---
## Step 10 — report performance

```python
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test, predictions))
# 打印输出 / Print output
print('RMSE: %.3f' % rmse)
```

---
## Step 11 — errors

```python
# 获取长度 / Get length
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(residuals.describe())
```

---
## Step 12 — plot

```python
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plots of residual errors of bias corrected forecasts 是机器学习中的常用技术。  
  *plots of residual errors of bias corrected forecasts is a common technique in machine learning.*

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
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Models Bias Corrected Residuals / Models Bias Corrected Residuals
# Complete Code / 完整代码
# ===============================

# plots of residual errors of bias corrected forecasts
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
 # 获取长度 / Get length
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
  # 添加元素到列表末尾 / Append element to list end
		diff.append(value)
	return diff

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 获取长度 / Get length
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
bias = 165.904728
# 获取长度 / Get length
for i in range(len(test)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(0,0,1))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = bias + inverse_difference(history, yhat, months_in_year)
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
	# observation
	obs = test[i]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
# report performance
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test, predictions))
# 打印输出 / Print output
print('RMSE: %.3f' % rmse)
# errors
# 获取长度 / Get length
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()
```

---

➡️ **Next / 下一步**: File 13 of 17

---

### Models Grid Search Arima

# 01 — Models Grid Search Arima / ARIMA 模型

**Chapter 32 — File 13 of 17 / 第32章 — 第13个文件（共17个）**

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
# 导入警告控制模块 / Import warnings control module
import warnings
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
```

---
## Step 2 — create a differenced series

```python
def difference(dataset, interval=1):
	diff = list()
 # 获取长度 / Get length
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
  # 添加元素到列表末尾 / Append element to list end
		diff.append(value)
	return numpy.array(diff)
```

---
## Step 3 — invert differenced value

```python
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
```

---
## Step 4 — evaluate an ARIMA model for a given order (p,d,q) and return RMSE

```python
def evaluate_arima_model(X, arima_order):
```

---
## Step 5 — prepare training dataset

```python
# 转换数据类型 / Convert data type
X = X.astype('float32')
 # 获取长度 / Get length
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
```

---
## Step 6 — make predictions

```python
predictions = list()
 # 获取长度 / Get length
	for t in range(len(test)):
```

---
## Step 7 — difference data

```python
months_in_year = 12
		diff = difference(history, months_in_year)
		model = ARIMA(diff, order=arima_order)
  # 训练模型 / Train the model
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		yhat = inverse_difference(history, yhat, months_in_year)
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(yhat)
  # 添加元素到列表末尾 / Append element to list end
		history.append(test[t])
```

---
## Step 8 — calculate out of sample error

```python
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test, predictions))
	return rmse
```

---
## Step 9 — evaluate combinations of p, d and q values for an ARIMA model

```python
def evaluate_models(dataset, p_values, d_values, q_values):
 # 转换数据类型 / Convert data type
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
     # 打印输出 / Print output
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
 # 打印输出 / Print output
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
```

---
## Step 10 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 11 — evaluate parameters

```python
# 生成整数序列 / Generate integer sequence
p_values = range(0, 7)
# 生成整数序列 / Generate integer sequence
d_values = range(0, 3)
# 生成整数序列 / Generate integer sequence
q_values = range(0, 7)
# 过滤警告信息 / Filter warning messages
warnings.filterwarnings("ignore")
# 转换为NumPy数组 / Convert to NumPy array
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
| `numpy` | 数值计算库 | Numerical computing library |
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
# 导入警告控制模块 / Import warnings control module
import warnings
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
 # 获取长度 / Get length
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
  # 添加元素到列表末尾 / Append element to list end
		diff.append(value)
	return numpy.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
 # 转换数据类型 / Convert data type
	X = X.astype('float32')
 # 获取长度 / Get length
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
 # 获取长度 / Get length
	for t in range(len(test)):
		# difference data
		months_in_year = 12
		diff = difference(history, months_in_year)
		model = ARIMA(diff, order=arima_order)
  # 训练模型 / Train the model
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		yhat = inverse_difference(history, yhat, months_in_year)
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(yhat)
  # 添加元素到列表末尾 / Append element to list end
		history.append(test[t])
	# calculate out of sample error
 # 计算均方误差 / Calculate Mean Squared Error
	rmse = sqrt(mean_squared_error(test, predictions))
	return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
 # 转换数据类型 / Convert data type
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
     # 打印输出 / Print output
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
 # 打印输出 / Print output
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# evaluate parameters
# 生成整数序列 / Generate integer sequence
p_values = range(0, 7)
# 生成整数序列 / Generate integer sequence
d_values = range(0, 3)
# 生成整数序列 / Generate integer sequence
q_values = range(0, 7)
# 过滤警告信息 / Filter warning messages
warnings.filterwarnings("ignore")
# 转换为NumPy数组 / Convert to NumPy array
evaluate_models(series.values, p_values, d_values, q_values)
```

---

➡️ **Next / 下一步**: File 14 of 17

---

### Models Manual Arima



---

### Models Residuals

# 01 — Models Residuals / Models Residuals

**Chapter 32 — File 15 of 17 / 第32章 — 第15个文件（共17个）**

---

## Summary / 总结

This script demonstrates **summarize ARIMA forecast residuals**.

本脚本演示 **summarize ARIMA forecast residuals**。

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
## Step 1 — summarize ARIMA forecast residuals

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — create a differenced series

```python
def difference(dataset, interval=1):
	diff = list()
 # 获取长度 / Get length
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
  # 添加元素到列表末尾 / Append element to list end
		diff.append(value)
	return diff
```

---
## Step 3 — invert differenced value

```python
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
```

---
## Step 4 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 5 — prepare data

```python
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 获取长度 / Get length
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
```

---
## Step 6 — walk-forward validation

```python
history = [x for x in train]
predictions = list()
# 获取长度 / Get length
for i in range(len(test)):
```

---
## Step 7 — difference data

```python
months_in_year = 12
	diff = difference(history, months_in_year)
```

---
## Step 8 — predict

```python
model = ARIMA(diff, order=(0,0,1))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history, yhat, months_in_year)
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
```

---
## Step 9 — observation

```python
obs = test[i]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
```

---
## Step 10 — errors

```python
# 获取长度 / Get length
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(residuals.describe())
```

---
## Step 11 — plot

```python
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: summarize ARIMA forecast residuals 是机器学习中的常用技术。  
  *summarize ARIMA forecast residuals is a common technique in machine learning.*

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
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Models Residuals / Models Residuals
# Complete Code / 完整代码
# ===============================

# summarize ARIMA forecast residuals
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
 # 获取长度 / Get length
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
  # 添加元素到列表末尾 / Append element to list end
		diff.append(value)
	return diff

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 获取长度 / Get length
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
# 获取长度 / Get length
for i in range(len(test)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(0,0,1))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history, yhat, months_in_year)
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
	# observation
	obs = test[i]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
# errors
# 获取长度 / Get length
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()
```

---

➡️ **Next / 下一步**: File 16 of 17

---

### Models Stationary

# 01 — Models Stationary / Models Stationary

**Chapter 32 — File 16 of 17 / 第32章 — 第16个文件（共17个）**

---

## Summary / 总结

This script demonstrates **create and summarize stationary version of time series**.

本脚本演示 **create and summarize stationary version of time series**。

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
## Step 1 — create and summarize stationary version of time series

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Series
from statsmodels.tsa.stattools import adfuller
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — create a differenced series

```python
def difference(dataset, interval=1):
	diff = list()
 # 获取长度 / Get length
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
  # 添加元素到列表末尾 / Append element to list end
		diff.append(value)
	return Series(diff)

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
```

---
## Step 3 — difference data

```python
months_in_year = 12
stationary = difference(X, months_in_year)
stationary.index = series.index[months_in_year:]
```

---
## Step 4 — check if stationary

```python
result = adfuller(stationary)
# 打印输出 / Print output
print('ADF Statistic: %f' % result[0])
# 打印输出 / Print output
print('p-value: %f' % result[1])
# 打印输出 / Print output
print('Critical Values:')
# 获取字典的键值对 / Get dict key-value pairs
for key, value in result[4].items():
 # 打印输出 / Print output
	print('\t%s: %.3f' % (key, value))
```

---
## Step 5 — save

```python
stationary.to_csv('stationary.csv', header=False)
```

---
## Step 6 — plot

```python
stationary.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create and summarize stationary version of time series 是机器学习中的常用技术。  
  *create and summarize stationary version of time series is a common technique in machine learning.*

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
# Models Stationary / Models Stationary
# Complete Code / 完整代码
# ===============================

# create and summarize stationary version of time series
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Series
from statsmodels.tsa.stattools import adfuller
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
 # 获取长度 / Get length
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
  # 添加元素到列表末尾 / Append element to list end
		diff.append(value)
	return Series(diff)

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# difference data
months_in_year = 12
stationary = difference(X, months_in_year)
stationary.index = series.index[months_in_year:]
# check if stationary
result = adfuller(stationary)
# 打印输出 / Print output
print('ADF Statistic: %f' % result[0])
# 打印输出 / Print output
print('p-value: %f' % result[1])
# 打印输出 / Print output
print('Critical Values:')
# 获取字典的键值对 / Get dict key-value pairs
for key, value in result[4].items():
 # 打印输出 / Print output
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv', header=False)
# plot
stationary.plot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 17 of 17

---

### Persistence

# 01 — Persistence / Persistence

**Chapter 32 — File 17 of 17 / 第32章 — 第17个文件（共17个）**

---

## Summary / 总结

This script demonstrates **evaluate persistence model on time series**.

本脚本演示 **evaluate persistence model on time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — evaluate persistence model on time series

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
```

---
## Step 2 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 3 — prepare data

```python
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 获取长度 / Get length
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
```

---
## Step 4 — walk-forward validation

```python
history = [x for x in train]
predictions = list()
# 获取长度 / Get length
for i in range(len(test)):
```

---
## Step 5 — predict

```python
yhat = history[-1]
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
```

---
## Step 6 — observation

```python
obs = test[i]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
 # 打印输出 / Print output
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
```

---
## Step 7 — report performance

```python
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test, predictions))
# 打印输出 / Print output
print('RMSE: %.3f' % rmse)
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate persistence model on time series 是机器学习中的常用技术。  
  *evaluate persistence model on time series is a common technique in machine learning.*

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

# evaluate persistence model on time series
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 获取长度 / Get length
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
# 获取长度 / Get length
for i in range(len(test)):
	# predict
	yhat = history[-1]
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
	# observation
	obs = test[i]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
 # 打印输出 / Print output
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test, predictions))
# 打印输出 / Print output
print('RMSE: %.3f' % rmse)
```

---
