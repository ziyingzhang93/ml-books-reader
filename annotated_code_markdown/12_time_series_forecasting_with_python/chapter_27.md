# 时间序列预测
## Chapter 27

---

### Chapter Summary

# Chapter 27 Summary / 第27章总结

## Theme / 主题: Chapter 27 / Chapter 27

This chapter contains **9 code files** demonstrating chapter 27.

本章包含 **9 个代码文件**，演示Chapter 27。

---
## Evolution / 演化路线

  1. `ar_model.ipynb` — Ar Model
  2. `load_ar.ipynb` — Load Ar
  3. `load_manual.ipynb` — Load Manual
  4. `prediction_ar.ipynb` — Prediction Ar
  5. `prediction_manual.ipynb` — Prediction Manual
  6. `save_ar.ipynb` — Save Ar
  7. `save_manual.ipynb` — Save Manual
  8. `update_ar.ipynb` — Update Ar
  9. `update_manual.ipynb` — Update Manual

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 27) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 27）是机器学习流水线中的基础构建块。

---

### Ar Model

# 01 — Ar Model / Ar Model

**Chapter 27 — File 1 of 9 / 第27章 — 第1个文件（共9个）**

---

## Summary / 总结

This script demonstrates **fit and evaluate an AR model**.

本脚本演示 **fit and evaluate an AR model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — fit and evaluate an AR model

```python
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy
from math import sqrt
```

---
## Step 2 — create a difference transform of the dataset

```python
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)
```

---
## Step 3 — Make a prediction give regression coefficients and lag obs

```python
def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat

series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 4 — split dataset

```python
X = difference(series.values)
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:]
```

---
## Step 5 — train autoregression

```python
window = 6
model = AutoReg(train, lags=6)
model_fit = model.fit()
coef = model_fit.params
```

---
## Step 6 — walk forward over time steps in test

```python
history = [train[i] for i in range(len(train))]
predictions = list()
for t in range(len(test)):
	yhat = predict(coef, history)
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
```

---
## Step 7 — plot

```python
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: fit and evaluate an AR model 是机器学习中的常用技术。  
  *fit and evaluate an AR model is a common technique in machine learning.*

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
# Ar Model / Ar Model
# Complete Code / 完整代码
# ===============================

# fit and evaluate an AR model
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy
from math import sqrt

# create a difference transform of the dataset
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

# Make a prediction give regression coefficients and lag obs
def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat

series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# split dataset
X = difference(series.values)
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:]
# train autoregression
window = 6
model = AutoReg(train, lags=6)
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
history = [train[i] for i in range(len(train))]
predictions = list()
for t in range(len(test)):
	yhat = predict(coef, history)
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 9

---

### Load Ar

# 01 — Load Ar / Load Ar

**Chapter 27 — File 2 of 9 / 第27章 — 第2个文件（共9个）**

---

## Summary / 总结

This script demonstrates **load the AR model from file**.

本脚本演示 **load the AR model from file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — load the AR model from file

```python
from statsmodels.tsa.ar_model import AutoRegResults
import numpy
loaded = AutoRegResults.load('ar_model.pkl')
print(loaded.params)
data = numpy.load('ar_data.npy')
last_ob = numpy.load('ar_obs.npy')
print(last_ob)
```

---
## Learning Notes / 学习笔记

- **概念**: load the AR model from file 是机器学习中的常用技术。  
  *load the AR model from file is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Ar / Load Ar
# Complete Code / 完整代码
# ===============================

# load the AR model from file
from statsmodels.tsa.ar_model import AutoRegResults
import numpy
loaded = AutoRegResults.load('ar_model.pkl')
print(loaded.params)
data = numpy.load('ar_data.npy')
last_ob = numpy.load('ar_obs.npy')
print(last_ob)
```

---

➡️ **Next / 下一步**: File 3 of 9

---

### Load Manual

# 01 — Load Manual / Load Manual

**Chapter 27 — File 3 of 9 / 第27章 — 第3个文件（共9个）**

---

## Summary / 总结

This script demonstrates **load the manually saved model from file**.

本脚本演示 **load the manually saved model from file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — load the manually saved model from file

```python
import numpy
coef = numpy.load('man_model.npy')
print(coef)
lag = numpy.load('man_data.npy')
print(lag)
last_ob = numpy.load('man_obs.npy')
print(last_ob)
```

---
## Learning Notes / 学习笔记

- **概念**: load the manually saved model from file 是机器学习中的常用技术。  
  *load the manually saved model from file is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Manual / Load Manual
# Complete Code / 完整代码
# ===============================

# load the manually saved model from file
import numpy
coef = numpy.load('man_model.npy')
print(coef)
lag = numpy.load('man_data.npy')
print(lag)
last_ob = numpy.load('man_obs.npy')
print(last_ob)
```

---

➡️ **Next / 下一步**: File 4 of 9

---

### Prediction Ar

# 01 — Prediction Ar / Prediction Ar

**Chapter 27 — File 4 of 9 / 第27章 — 第4个文件（共9个）**

---

## Summary / 总结

This script demonstrates **load AR model from file and make a one-step prediction**.

本脚本演示 **load AR model from file and make a one-step prediction**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 评估模型效果 / Evaluate model performance


---
## Step 1 — load AR model from file and make a one-step prediction

```python
from statsmodels.tsa.ar_model import AutoRegResults
import numpy
```

---
## Step 2 — load model

```python
model = AutoRegResults.load('ar_model.pkl')
data = numpy.load('ar_data.npy')
last_ob = numpy.load('ar_obs.npy')
```

---
## Step 3 — make prediction

```python
predictions = model.predict(start=len(data), end=len(data))
```

---
## Step 4 — transform prediction

```python
yhat = predictions[0] + last_ob[0]
print('Prediction: %f' % yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: load AR model from file and make a one-step prediction 是机器学习中的常用技术。  
  *load AR model from file and make a one-step prediction is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Prediction Ar / Prediction Ar
# Complete Code / 完整代码
# ===============================

# load AR model from file and make a one-step prediction
from statsmodels.tsa.ar_model import AutoRegResults
import numpy
# load model
model = AutoRegResults.load('ar_model.pkl')
data = numpy.load('ar_data.npy')
last_ob = numpy.load('ar_obs.npy')
# make prediction
predictions = model.predict(start=len(data), end=len(data))
# transform prediction
yhat = predictions[0] + last_ob[0]
print('Prediction: %f' % yhat)
```

---

➡️ **Next / 下一步**: File 5 of 9

---

### Prediction Manual

# 01 — Prediction Manual / Prediction Manual

**Chapter 27 — File 5 of 9 / 第27章 — 第5个文件（共9个）**

---

## Summary / 总结

This script demonstrates **load a coefficients and from file and make a manual prediction**.

本脚本演示 **load a coefficients and from file and make a manual prediction**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — load a coefficients and from file and make a manual prediction

```python
import numpy

def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat
```

---
## Step 2 — load model

```python
coef = numpy.load('man_model.npy')
lag = numpy.load('man_data.npy')
last_ob = numpy.load('man_obs.npy')
```

---
## Step 3 — make prediction

```python
prediction = predict(coef, lag)
```

---
## Step 4 — transform prediction

```python
yhat = prediction + last_ob[0]
print('Prediction: %f' % yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: load a coefficients and from file and make a manual prediction 是机器学习中的常用技术。  
  *load a coefficients and from file and make a manual prediction is a common technique in machine learning.*

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
# Prediction Manual / Prediction Manual
# Complete Code / 完整代码
# ===============================

# load a coefficients and from file and make a manual prediction
import numpy

def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat

# load model
coef = numpy.load('man_model.npy')
lag = numpy.load('man_data.npy')
last_ob = numpy.load('man_obs.npy')
# make prediction
prediction = predict(coef, lag)
# transform prediction
yhat = prediction + last_ob[0]
print('Prediction: %f' % yhat)
```

---

➡️ **Next / 下一步**: File 6 of 9

---

### Save Ar

# 01 — Save Ar / 保存/加载模型

**Chapter 27 — File 6 of 9 / 第27章 — 第6个文件（共9个）**

---

## Summary / 总结

This script demonstrates **fit an AR model and save the whole model to file**.

本脚本演示 **fit an AR model and save the whole model to file**。

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
## Step 1 — fit an AR model and save the whole model to file

```python
from pandas import read_csv
from statsmodels.tsa.ar_model import AutoReg
import numpy
```

---
## Step 2 — create a difference transform of the dataset

```python
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)
```

---
## Step 3 — load dataset

```python
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = difference(series.values)
```

---
## Step 4 — fit model

```python
model = AutoReg(X, lags=6)
model_fit = model.fit()
```

---
## Step 5 — save model to file

```python
model_fit.save('ar_model.pkl')
```

---
## Step 6 — save the differenced dataset

```python
numpy.save('ar_data.npy', X)
```

---
## Step 7 — save the last ob

```python
numpy.save('ar_obs.npy', [series.values[-1]])
```

---
## Learning Notes / 学习笔记

- **概念**: fit an AR model and save the whole model to file 是机器学习中的常用技术。  
  *fit an AR model and save the whole model to file is a common technique in machine learning.*

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
# Save Ar / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# fit an AR model and save the whole model to file
from pandas import read_csv
from statsmodels.tsa.ar_model import AutoReg
import numpy

# create a difference transform of the dataset
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

# load dataset
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = difference(series.values)
# fit model
model = AutoReg(X, lags=6)
model_fit = model.fit()
# save model to file
model_fit.save('ar_model.pkl')
# save the differenced dataset
numpy.save('ar_data.npy', X)
# save the last ob
numpy.save('ar_obs.npy', [series.values[-1]])
```

---

➡️ **Next / 下一步**: File 7 of 9

---

### Save Manual

# 01 — Save Manual / 保存/加载模型

**Chapter 27 — File 7 of 9 / 第27章 — 第7个文件（共9个）**

---

## Summary / 总结

This script demonstrates **fit an AR model and manually save coefficients to file**.

本脚本演示 **fit an AR model and manually save coefficients to file**。

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
## Step 1 — fit an AR model and manually save coefficients to file

```python
from pandas import read_csv
from statsmodels.tsa.ar_model import AutoReg
import numpy
```

---
## Step 2 — create a difference transform of the dataset

```python
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)
```

---
## Step 3 — load dataset

```python
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = difference(series.values)
```

---
## Step 4 — fit model

```python
window_size = 6
model = AutoReg(X, lags=window_size)
model_fit = model.fit()
```

---
## Step 5 — save coefficients

```python
coef = model_fit.params
numpy.save('man_model.npy', coef)
```

---
## Step 6 — save lag

```python
lag = X[-window_size:]
numpy.save('man_data.npy', lag)
```

---
## Step 7 — save the last ob

```python
numpy.save('man_obs.npy', [series.values[-1]])
```

---
## Learning Notes / 学习笔记

- **概念**: fit an AR model and manually save coefficients to file 是机器学习中的常用技术。  
  *fit an AR model and manually save coefficients to file is a common technique in machine learning.*

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
# Save Manual / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# fit an AR model and manually save coefficients to file
from pandas import read_csv
from statsmodels.tsa.ar_model import AutoReg
import numpy

# create a difference transform of the dataset
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

# load dataset
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = difference(series.values)
# fit model
window_size = 6
model = AutoReg(X, lags=window_size)
model_fit = model.fit()
# save coefficients
coef = model_fit.params
numpy.save('man_model.npy', coef)
# save lag
lag = X[-window_size:]
numpy.save('man_data.npy', lag)
# save the last ob
numpy.save('man_obs.npy', [series.values[-1]])
```

---

➡️ **Next / 下一步**: File 8 of 9

---

### Update Ar

# 01 — Update Ar / Update Ar

**Chapter 27 — File 8 of 9 / 第27章 — 第8个文件（共9个）**

---

## Summary / 总结

This script demonstrates **update the data for the AR model with a new obs**.

本脚本演示 **update the data for the AR model with a new obs**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — update the data for the AR model with a new obs

```python
import numpy
```

---
## Step 2 — get real observation

```python
observation = 48
```

---
## Step 3 — load the saved data

```python
data = numpy.load('ar_data.npy')
last_ob = numpy.load('ar_obs.npy')
```

---
## Step 4 — update and save differenced observation

```python
diffed = observation - last_ob[0]
data = numpy.append(data, [diffed], axis=0)
numpy.save('ar_data.npy', data)
```

---
## Step 5 — update and save real observation

```python
last_ob[0] = observation
numpy.save('ar_obs.npy', last_ob)
```

---
## Learning Notes / 学习笔记

- **概念**: update the data for the AR model with a new obs 是机器学习中的常用技术。  
  *update the data for the AR model with a new obs is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Update Ar / Update Ar
# Complete Code / 完整代码
# ===============================

# update the data for the AR model with a new obs
import numpy
# get real observation
observation = 48
# load the saved data
data = numpy.load('ar_data.npy')
last_ob = numpy.load('ar_obs.npy')
# update and save differenced observation
diffed = observation - last_ob[0]
data = numpy.append(data, [diffed], axis=0)
numpy.save('ar_data.npy', data)
# update and save real observation
last_ob[0] = observation
numpy.save('ar_obs.npy', last_ob)
```

---

➡️ **Next / 下一步**: File 9 of 9

---

### Update Manual

# 01 — Update Manual / Update Manual

**Chapter 27 — File 9 of 9 / 第27章 — 第9个文件（共9个）**

---

## Summary / 总结

This script demonstrates **update the data for the manual model with a new obs**.

本脚本演示 **update the data for the manual model with a new obs**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — update the data for the manual model with a new obs

```python
import numpy
```

---
## Step 2 — get real observation

```python
observation = 48
```

---
## Step 3 — update and save differenced observation

```python
lag = numpy.load('man_data.npy')
last_ob = numpy.load('man_obs.npy')
diffed = observation - last_ob[0]
lag = numpy.append(lag[1:], [diffed], axis=0)
numpy.save('man_data.npy', lag)
```

---
## Step 4 — update and save real observation

```python
last_ob[0] = observation
numpy.save('man_obs.npy', last_ob)
```

---
## Learning Notes / 学习笔记

- **概念**: update the data for the manual model with a new obs 是机器学习中的常用技术。  
  *update the data for the manual model with a new obs is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Update Manual / Update Manual
# Complete Code / 完整代码
# ===============================

# update the data for the manual model with a new obs
import numpy
# get real observation
observation = 48
# update and save differenced observation
lag = numpy.load('man_data.npy')
last_ob = numpy.load('man_obs.npy')
diffed = observation - last_ob[0]
lag = numpy.append(lag[1:], [diffed], axis=0)
numpy.save('man_data.npy', lag)
# update and save real observation
last_ob[0] = observation
numpy.save('man_obs.npy', last_ob)
```

---
