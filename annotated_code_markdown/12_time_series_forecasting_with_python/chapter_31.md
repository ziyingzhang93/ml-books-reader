# 时间序列预测 / Time Series Forecasting with Python
## Chapter 31

---

### Chapter Summary / 章节总结

# Chapter 31 Summary / 第31章总结

## Theme / 主题: Chapter 31 / Chapter 31

This chapter contains **15 code files** demonstrating chapter 31.

本章包含 **15 个代码文件**，演示Chapter 31。

---
## Evolution / 演化路线

  1. `analysis_boxplots.ipynb` — Analysis Boxplots
  2. `analysis_density_plots.ipynb` — Analysis Density Plots
  3. `analysis_line_plot.ipynb` — Analysis Line Plot
  4. `analysis_summary.ipynb` — Analysis Summary
  5. `finalize_load_predict.ipynb` — Finalize Load Predict
  6. `finalize_save.ipynb` — Finalize Save
  7. `finalize_validate.ipynb` — Finalize Validate
  8. `harness_split.ipynb` — Harness Split
  9. `models_acf_pacf.ipynb` — Models Acf Pacf
  10. `models_bias_corrected_residuals.ipynb` — Models Bias Corrected Residuals
  11. `models_grid_search_arima.ipynb` — Models Grid Search Arima
  12. `models_manual_arima.ipynb` — Models Manual Arima
  13. `models_residuals.ipynb` — Models Residuals
  14. `models_stationary.ipynb` — Models Stationary
  15. `persistence.ipynb` — Persistence

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 31) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 31）是机器学习流水线中的基础构建块。

---

### Analysis Boxplots

# 01 — Analysis Boxplots / Analysis Boxplots

**Chapter 31 — File 1 of 15 / 第31章 — 第1个文件（共15个）**

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
groups = series.groupby(Grouper(freq='10YS'))
decades = DataFrame()
for name, group in groups:
 # 转换为NumPy数组 / Convert to NumPy array
	if len(group.values) is 10:
  # 转换为NumPy数组 / Convert to NumPy array
		decades[name.year] = group.values
decades.boxplot()
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
groups = series.groupby(Grouper(freq='10YS'))
decades = DataFrame()
for name, group in groups:
 # 转换为NumPy数组 / Convert to NumPy array
	if len(group.values) is 10:
  # 转换为NumPy数组 / Convert to NumPy array
		decades[name.year] = group.values
decades.boxplot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 15

---

### Analysis Density Plots

# 01 — Analysis Density Plots / Analysis Density Plots

**Chapter 31 — File 2 of 15 / 第31章 — 第2个文件（共15个）**

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

➡️ **Next / 下一步**: File 3 of 15

---

### Analysis Line Plot

# 01 — Analysis Line Plot / Analysis Line Plot

**Chapter 31 — File 3 of 15 / 第31章 — 第3个文件（共15个）**

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

➡️ **Next / 下一步**: File 4 of 15

---

### Analysis Summary

# 01 — Analysis Summary / Analysis Summary

**Chapter 31 — File 4 of 15 / 第31章 — 第4个文件（共15个）**

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

➡️ **Next / 下一步**: File 5 of 15

---

### Finalize Load Predict

# 01 — Finalize Load Predict / Finalize Load Predict

**Chapter 31 — File 5 of 15 / 第31章 — 第5个文件（共15个）**

---

## Summary / 总结

This script demonstrates **load finalized model and make a prediction**.

本脚本演示 **load finalized model and make a prediction**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — load finalized model and make a prediction

```python
from statsmodels.tsa.arima.model import ARIMAResults
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
yhat = bias + float(model_fit.forecast()[0])
# 打印输出 / Print output
print('Predicted: %.3f' % yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: load finalized model and make a prediction 是机器学习中的常用技术。  
  *load finalized model and make a prediction is a common technique in machine learning.*

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

# load finalized model and make a prediction
from statsmodels.tsa.arima.model import ARIMAResults
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
yhat = bias + float(model_fit.forecast()[0])
# 打印输出 / Print output
print('Predicted: %.3f' % yhat)
```

---

➡️ **Next / 下一步**: File 6 of 15

---

### Finalize Save

# 01 — Finalize Save / 保存/加载模型

**Chapter 31 — File 6 of 15 / 第31章 — 第6个文件（共15个）**

---

## Summary / 总结

This script demonstrates **save finalized model to file**.

本脚本演示 **save finalized model to file**。

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
## Step 1 — save finalized model to file

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
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
```

---
## Step 4 — fit model

```python
model = ARIMA(X, order=(2,1,0))
# 训练模型 / Train the model
model_fit = model.fit()
```

---
## Step 5 — bias constant, could be calculated from in-sample mean residual

```python
bias = 1.081624
```

---
## Step 6 — save model

```python
model_fit.save('model.pkl')
numpy.save('model_bias.npy', [bias])
```

---
## Learning Notes / 学习笔记

- **概念**: save finalized model to file 是机器学习中的常用技术。  
  *save finalized model to file is a common technique in machine learning.*

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

# save finalized model to file
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 转换数据类型 / Convert data type
X = X.astype('float32')
# fit model
model = ARIMA(X, order=(2,1,0))
# 训练模型 / Train the model
model_fit = model.fit()
# bias constant, could be calculated from in-sample mean residual
bias = 1.081624
# save model
model_fit.save('model.pkl')
numpy.save('model_bias.npy', [bias])
```

---

➡️ **Next / 下一步**: File 7 of 15

---

### Finalize Validate

# 01 — Finalize Validate / Finalize Validate

**Chapter 31 — File 7 of 15 / 第31章 — 第7个文件（共15个）**

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
## Step 2 — load and prepare datasets

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = dataset.values.astype('float32')
history = [x for x in X]
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
validation = read_csv('validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
y = validation.values.astype('float32')
```

---
## Step 3 — load model

```python
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
```

---
## Step 4 — make first prediction

```python
predictions = list()
yhat = bias + float(model_fit.forecast()[0])
# 添加元素到列表末尾 / Append element to list end
predictions.append(yhat)
# 添加元素到列表末尾 / Append element to list end
history.append(y[0])
# 打印输出 / Print output
print('>Predicted=%.3f, Expected=%.3f' % (yhat, y[0]))
```

---
## Step 5 — rolling forecasts

```python
# 获取长度 / Get length
for i in range(1, len(y)):
```

---
## Step 6 — predict

```python
model = ARIMA(history, order=(2,1,0))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = bias + float(model_fit.forecast()[0])
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
```

---
## Step 7 — observation

```python
obs = y[i]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
 # 打印输出 / Print output
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
```

---
## Step 8 — report performance

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
# load and prepare datasets
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = dataset.values.astype('float32')
history = [x for x in X]
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
validation = read_csv('validation.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
# make first prediction
predictions = list()
yhat = bias + float(model_fit.forecast()[0])
# 添加元素到列表末尾 / Append element to list end
predictions.append(yhat)
# 添加元素到列表末尾 / Append element to list end
history.append(y[0])
# 打印输出 / Print output
print('>Predicted=%.3f, Expected=%.3f' % (yhat, y[0]))
# rolling forecasts
# 获取长度 / Get length
for i in range(1, len(y)):
	# predict
	model = ARIMA(history, order=(2,1,0))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = bias + float(model_fit.forecast()[0])
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

➡️ **Next / 下一步**: File 8 of 15

---

### Harness Split

# 01 — Harness Split / Harness Split

**Chapter 31 — File 8 of 15 / 第31章 — 第8个文件（共15个）**

---

## Summary / 总结

This script demonstrates **separate out a validation dataset**.

本脚本演示 **separate out a validation dataset**。

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
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — separate out a validation dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('water.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 获取长度 / Get length
split_point = len(series) - 10
dataset, validation = series[0:split_point], series[split_point:]
# 打印输出 / Print output
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', header=False)
validation.to_csv('validation.csv', header=False)
```

---
## Learning Notes / 学习笔记

- **概念**: separate out a validation dataset 是机器学习中的常用技术。  
  *separate out a validation dataset is a common technique in machine learning.*

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
# Harness Split / Harness Split
# Complete Code / 完整代码
# ===============================

# separate out a validation dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('water.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 获取长度 / Get length
split_point = len(series) - 10
dataset, validation = series[0:split_point], series[split_point:]
# 打印输出 / Print output
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', header=False)
validation.to_csv('validation.csv', header=False)
```

---

➡️ **Next / 下一步**: File 9 of 15

---

### Models Acf Pacf

# 01 — Models Acf Pacf / Models Acf Pacf

**Chapter 31 — File 9 of 15 / 第31章 — 第9个文件（共15个）**

---

## Summary / 总结

This script demonstrates **ACF and PACF plots of the time series**.

本脚本演示 **ACF and PACF plots of the time series**。

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
## Step 1 — ACF and PACF plots of the time series

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
pyplot.figure()
pyplot.subplot(211)
plot_acf(series, lags=20, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(series, lags=20, ax=pyplot.gca())
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: ACF and PACF plots of the time series 是机器学习中的常用技术。  
  *ACF and PACF plots of the time series is a common technique in machine learning.*

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

# ACF and PACF plots of the time series
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
pyplot.figure()
pyplot.subplot(211)
plot_acf(series, lags=20, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(series, lags=20, ax=pyplot.gca())
pyplot.show()
```

---

➡️ **Next / 下一步**: File 10 of 15

---

### Models Bias Corrected Residuals

# 01 — Models Bias Corrected Residuals / Models Bias Corrected Residuals

**Chapter 31 — File 10 of 15 / 第31章 — 第10个文件（共15个）**

---

## Summary / 总结

This script demonstrates **summarize residual errors from bias corrected forecasts**.

本脚本演示 **summarize residual errors from bias corrected forecasts**。

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
## Step 1 — summarize residual errors from bias corrected forecasts

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
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
bias = 1.081624
# 获取长度 / Get length
for i in range(len(test)):
```

---
## Step 5 — predict

```python
model = ARIMA(history, order=(2,1,0))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = bias + float(model_fit.forecast()[0])
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
```

---
## Step 6 — observation

```python
obs = test[i]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
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
## Step 8 — summarize residual errors

```python
# 获取长度 / Get length
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(residuals.describe())
```

---
## Step 9 — plot residual errors

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

- **概念**: summarize residual errors from bias corrected forecasts 是机器学习中的常用技术。  
  *summarize residual errors from bias corrected forecasts is a common technique in machine learning.*

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

# summarize residual errors from bias corrected forecasts
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
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
bias = 1.081624
# 获取长度 / Get length
for i in range(len(test)):
	# predict
	model = ARIMA(history, order=(2,1,0))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = bias + float(model_fit.forecast()[0])
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
# summarize residual errors
# 获取长度 / Get length
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(residuals.describe())
# plot residual errors
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()
```

---

➡️ **Next / 下一步**: File 11 of 15

---

### Models Grid Search Arima

# 01 — Models Grid Search Arima / ARIMA 模型

**Chapter 31 — File 11 of 15 / 第31章 — 第11个文件（共15个）**

---

## Summary / 总结

This script demonstrates **grid search ARIMA parameters for a time series**.

本脚本演示 **grid search ARIMA parameters for a time series**。

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
## Step 1 — grid search ARIMA parameters for a time series

```python
# 导入警告控制模块 / Import warnings control module
import warnings
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
# 转换数据类型 / Convert data type
X = X.astype('float32')
 # 获取长度 / Get length
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
```

---
## Step 4 — make predictions

```python
predictions = list()
 # 获取长度 / Get length
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
  # 训练模型 / Train the model
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(yhat)
  # 添加元素到列表末尾 / Append element to list end
		history.append(test[t])
```

---
## Step 5 — calculate out of sample error

```python
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test, predictions))
	return rmse
```

---
## Step 6 — evaluate combinations of p, d and q values for an ARIMA model

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
## Step 7 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 8 — evaluate parameters

```python
# 生成整数序列 / Generate integer sequence
p_values = range(0, 5)
# 生成整数序列 / Generate integer sequence
d_values = range(0, 3)
# 生成整数序列 / Generate integer sequence
q_values = range(0, 5)
# 过滤警告信息 / Filter warning messages
warnings.filterwarnings("ignore")
# 转换为NumPy数组 / Convert to NumPy array
evaluate_models(series.values, p_values, d_values, q_values)
```

---
## Learning Notes / 学习笔记

- **概念**: grid search ARIMA parameters for a time series 是机器学习中的常用技术。  
  *grid search ARIMA parameters for a time series is a common technique in machine learning.*

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

# grid search ARIMA parameters for a time series
# 导入警告控制模块 / Import warnings control module
import warnings
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt

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
		model = ARIMA(history, order=arima_order)
  # 训练模型 / Train the model
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
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
p_values = range(0, 5)
# 生成整数序列 / Generate integer sequence
d_values = range(0, 3)
# 生成整数序列 / Generate integer sequence
q_values = range(0, 5)
# 过滤警告信息 / Filter warning messages
warnings.filterwarnings("ignore")
# 转换为NumPy数组 / Convert to NumPy array
evaluate_models(series.values, p_values, d_values, q_values)
```

---

➡️ **Next / 下一步**: File 12 of 15

---

### Models Manual Arima

# 01 — Models Manual Arima / ARIMA 模型

**Chapter 31 — File 12 of 15 / 第31章 — 第12个文件（共15个）**

---

## Summary / 总结

This script demonstrates **evaluate a manually configured ARIMA model**.

本脚本演示 **evaluate a manually configured ARIMA model**。

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
## Step 1 — evaluate a manually configured ARIMA model

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
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
model = ARIMA(history, order=(4,1,1))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
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
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
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

- **概念**: evaluate a manually configured ARIMA model 是机器学习中的常用技术。  
  *evaluate a manually configured ARIMA model is a common technique in machine learning.*

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

# evaluate a manually configured ARIMA model
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
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
	model = ARIMA(history, order=(4,1,1))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
	# observation
	obs = test[i]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
 # 打印输出 / Print output
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test, predictions))
# 打印输出 / Print output
print('RMSE: %.3f' % rmse)
```

---

➡️ **Next / 下一步**: File 13 of 15

---

### Models Residuals

# 01 — Models Residuals / Models Residuals

**Chapter 31 — File 13 of 15 / 第31章 — 第13个文件（共15个）**

---

## Summary / 总结

This script demonstrates **summarize residual errors for an ARIMA model**.

本脚本演示 **summarize residual errors for an ARIMA model**。

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
## Step 1 — summarize residual errors for an ARIMA model

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
model = ARIMA(history, order=(2,1,0))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
```

---
## Step 6 — observation

```python
obs = test[i]
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
```

---
## Step 7 — errors

```python
# 获取长度 / Get length
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(residuals.describe())
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: summarize residual errors for an ARIMA model 是机器学习中的常用技术。  
  *summarize residual errors for an ARIMA model is a common technique in machine learning.*

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

# summarize residual errors for an ARIMA model
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
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
	model = ARIMA(history, order=(2,1,0))
 # 训练模型 / Train the model
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
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
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()
```

---

➡️ **Next / 下一步**: File 14 of 15

---

### Models Stationary

# 01 — Models Stationary / Models Stationary

**Chapter 31 — File 14 of 15 / 第31章 — 第14个文件（共15个）**

---

## Summary / 总结

This script demonstrates **create and summarize a stationary version of the time series**.

本脚本演示 **create and summarize a stationary version of the time series**。

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
## Step 1 — create and summarize a stationary version of the time series

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
def difference(dataset):
	diff = list()
 # 获取长度 / Get length
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
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
stationary = difference(X)
stationary.index = series.index[1:]
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
## Step 5 — plot differenced data

```python
stationary.plot()
pyplot.show()
```

---
## Step 6 — save

```python
stationary.to_csv('stationary.csv', header=False)
```

---
## Learning Notes / 学习笔记

- **概念**: create and summarize a stationary version of the time series 是机器学习中的常用技术。  
  *create and summarize a stationary version of the time series is a common technique in machine learning.*

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

# create and summarize a stationary version of the time series
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Series
from statsmodels.tsa.stattools import adfuller
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# create a differenced series
def difference(dataset):
	diff = list()
 # 获取长度 / Get length
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
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
stationary = difference(X)
stationary.index = series.index[1:]
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
# plot differenced data
stationary.plot()
pyplot.show()
# save
stationary.to_csv('stationary.csv', header=False)
```

---

➡️ **Next / 下一步**: File 15 of 15

---

### Persistence

# 01 — Persistence / Persistence

**Chapter 31 — File 15 of 15 / 第31章 — 第15个文件（共15个）**

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
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
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
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test, predictions))
# 打印输出 / Print output
print('RMSE: %.3f' % rmse)
```

---
