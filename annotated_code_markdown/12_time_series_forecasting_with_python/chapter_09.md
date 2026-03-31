# 时间序列预测 / Time Series Forecasting with Python
## Chapter 09

---

### Chapter Summary / 章节总结

# Chapter 09 Summary / 第09章总结

## Theme / 主题: Chapter 09 / Chapter 09

This chapter contains **3 code files** demonstrating chapter 09.

本章包含 **3 个代码文件**，演示Chapter 09。

---
## Evolution / 演化路线

  1. `ma_data_prep.ipynb` — Ma Data Prep
  2. `ma_feature_eng.ipynb` — Ma Feature Eng
  3. `ma_prediction.ipynb` — Ma Prediction

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 09) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 09）是机器学习流水线中的基础构建块。

---

### Ma Data Prep

# 01 — Ma Data Prep / Ma Data Prep

**Chapter 09 — File 1 of 3 / 第09章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **moving average smoothing as data preparation**.

本脚本演示 **moving average smoothing as data preparation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — moving average smoothing as data preparation

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 2 — tail-rolling average transform

```python
rolling = series.rolling(window=3)
rolling_mean = rolling.mean()
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(rolling_mean.head(10))
```

---
## Step 3 — plot original and transformed dataset

```python
series.plot()
rolling_mean.plot(color='red')
pyplot.show()
```

---
## Step 4 — zoomed plot original and transformed dataset

```python
series[:100].plot()
rolling_mean[:100].plot(color='red')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: moving average smoothing as data preparation 是机器学习中的常用技术。  
  *moving average smoothing as data preparation is a common technique in machine learning.*

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
# Ma Data Prep / Ma Data Prep
# Complete Code / 完整代码
# ===============================

# moving average smoothing as data preparation
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# tail-rolling average transform
rolling = series.rolling(window=3)
rolling_mean = rolling.mean()
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(rolling_mean.head(10))
# plot original and transformed dataset
series.plot()
rolling_mean.plot(color='red')
pyplot.show()
# zoomed plot original and transformed dataset
series[:100].plot()
rolling_mean[:100].plot(color='red')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Ma Feature Eng



---

### Ma Prediction

# 01 — Ma Prediction / Ma Prediction

**Chapter 09 — File 3 of 3 / 第09章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **moving average smoothing as a forecast model**.

本脚本演示 **moving average smoothing as a forecast model**。

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
## Step 1 — moving average smoothing as a forecast model

```python
from math import sqrt
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 2 — prepare situation

```python
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
window = 3
# 生成整数序列 / Generate integer sequence
history = [X[i] for i in range(window)]
# 获取长度 / Get length
test = [X[i] for i in range(window, len(X))]
predictions = list()
```

---
## Step 3 — walk forward over time steps in test

```python
# 获取长度 / Get length
for t in range(len(test)):
 # 获取长度 / Get length
	length = len(history)
 # 生成整数序列 / Generate integer sequence
	yhat = mean([history[i] for i in range(length-window,length)])
	obs = test[t]
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
 # 打印输出 / Print output
	print('predicted=%f, expected=%f' % (yhat, obs))
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test, predictions))
# 打印输出 / Print output
print('Test RMSE: %.3f' % rmse)
```

---
## Step 4 — plot

```python
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
```

---
## Step 5 — zoom plot

```python
pyplot.plot(test[:100])
pyplot.plot(predictions[:100], color='red')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: moving average smoothing as a forecast model 是机器学习中的常用技术。  
  *moving average smoothing as a forecast model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
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
# Ma Prediction / Ma Prediction
# Complete Code / 完整代码
# ===============================

# moving average smoothing as a forecast model
from math import sqrt
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# prepare situation
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
window = 3
# 生成整数序列 / Generate integer sequence
history = [X[i] for i in range(window)]
# 获取长度 / Get length
test = [X[i] for i in range(window, len(X))]
predictions = list()
# walk forward over time steps in test
# 获取长度 / Get length
for t in range(len(test)):
 # 获取长度 / Get length
	length = len(history)
 # 生成整数序列 / Generate integer sequence
	yhat = mean([history[i] for i in range(length-window,length)])
	obs = test[t]
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
 # 添加元素到列表末尾 / Append element to list end
	history.append(obs)
 # 打印输出 / Print output
	print('predicted=%f, expected=%f' % (yhat, obs))
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test, predictions))
# 打印输出 / Print output
print('Test RMSE: %.3f' % rmse)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
# zoom plot
pyplot.plot(test[:100])
pyplot.plot(predictions[:100], color='red')
pyplot.show()
```

---
