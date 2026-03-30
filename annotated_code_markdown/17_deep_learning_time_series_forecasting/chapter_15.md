# DL时间序列
## Chapter 15

---

### Grid Search Persistence

# 02 — Grid Search Persistence / 02 Grid Search Persistence

**Chapter 15 — File 2 of 5 / 第15章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **grid search persistence models for monthly airline passengers dataset**.

本脚本演示 **grid search persistence models for monthly airline passengers dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
```

---
## Step 1 — grid search persistence models for monthly airline passengers dataset

```python
from math import sqrt
from numpy import mean
from pandas import read_csv
from sklearn.metrics import mean_squared_error
```

---
## Step 2 — split a univariate dataset into train/test sets

```python
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]
```

---
## Step 3 — root mean squared error or rmse

```python
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))
```

---
## Step 4 — fit a model

```python
def model_fit(train, config):
	return None
```

---
## Step 5 — forecast with a pre-fit model

```python
def model_predict(model, history, offset):
	return history[-offset]
```

---
## Step 6 — walk-forward validation for univariate data

```python
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
```

---
## Step 7 — split dataset

```python
train, test = train_test_split(data, n_test)
```

---
## Step 8 — fit model

```python
model = model_fit(train, cfg)
```

---
## Step 9 — seed history with training dataset

```python
history = [x for x in train]
```

---
## Step 10 — step over each time-step in the test set

```python
for i in range(len(test)):
```

---
## Step 11 — fit model and make forecast for history

```python
yhat = model_predict(model, history, cfg)
```

---
## Step 12 — store forecast in list of predictions

```python
predictions.append(yhat)
```

---
## Step 13 — add actual observation to history for the next loop

```python
history.append(test[i])
```

---
## Step 14 — estimate prediction error

```python
error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return error
```

---
## Step 15 — score a model, return None on failure

```python
def repeat_evaluate(data, config, n_test, n_repeats=10):
```

---
## Step 16 — convert config to a key

```python
key = str(config)
```

---
## Step 17 — fit and evaluate the model n times

```python
scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
```

---
## Step 18 — summarize score

```python
result = mean(scores)
	print('> Model[%s] %.3f' % (key, result))
	return (key, result)
```

---
## Step 19 — grid search configs

```python
def grid_search(data, cfg_list, n_test):
```

---
## Step 20 — evaluate configs

```python
scores = scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
```

---
## Step 21 — sort configs by error, asc

```python
scores.sort(key=lambda tup: tup[1])
	return scores
```

---
## Step 22 — define dataset

```python
series = read_csv('monthly-airline-passengers.csv', header=0, index_col=0)
data = series.values
```

---
## Step 23 — data split

```python
n_test = 12
```

---
## Step 24 — model configs

```python
cfg_list = [1, 6, 12, 24, 36]
```

---
## Step 25 — grid search

```python
scores = grid_search(data, cfg_list, n_test)
print('done')
```

---
## Step 26 — list top 10 configs

```python
for cfg, error in scores[:10]:
	print(cfg, error)
```

---
## Learning Notes / 学习笔记

- **概念**: grid search persistence models for monthly airline passengers dataset 是机器学习中的常用技术。  
  *grid search persistence models for monthly airline passengers dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grid Search Persistence / 02 Grid Search Persistence
# Complete Code / 完整代码
# ===============================

# grid search persistence models for monthly airline passengers dataset
from math import sqrt
from numpy import mean
from pandas import read_csv
from sklearn.metrics import mean_squared_error

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# fit a model
def model_fit(train, config):
	return None

# forecast with a pre-fit model
def model_predict(model, history, offset):
	return history[-offset]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# fit model
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return error

# score a model, return None on failure
def repeat_evaluate(data, config, n_test, n_repeats=10):
	# convert config to a key
	key = str(config)
	# fit and evaluate the model n times
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	# summarize score
	result = mean(scores)
	print('> Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test):
	# evaluate configs
	scores = scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# define dataset
series = read_csv('monthly-airline-passengers.csv', header=0, index_col=0)
data = series.values
# data split
n_test = 12
# model configs
cfg_list = [1, 6, 12, 24, 36]
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 10 configs
for cfg, error in scores[:10]:
	print(cfg, error)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Grid Search Mlp

# 03 — Grid Search Mlp / 03 Grid Search Mlp

**Chapter 15 — File 3 of 5 / 第15章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **grid search mlps for monthly airline passengers dataset**.

本脚本演示 **grid search mlps for monthly airline passengers dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
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
```

---
## Step 1 — grid search mlps for monthly airline passengers dataset

```python
from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
```

---
## Step 2 — split a univariate dataset into train/test sets

```python
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]
```

---
## Step 3 — transform list into supervised learning format

```python
def series_to_supervised(data, n_in, n_out=1):
	df = DataFrame(data)
	cols = list()
```

---
## Step 4 — input sequence (t-n, ... t-1)

```python
for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
```

---
## Step 5 — forecast sequence (t, t+1, ... t+n)

```python
for i in range(0, n_out):
		cols.append(df.shift(-i))
```

---
## Step 6 — put it all together

```python
agg = concat(cols, axis=1)
```

---
## Step 7 — drop rows with NaN values

```python
agg.dropna(inplace=True)
	return agg.values
```

---
## Step 8 — root mean squared error or rmse

```python
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))
```

---
## Step 9 — difference dataset

```python
def difference(data, order):
	return [data[i] - data[i - order] for i in range(order, len(data))]
```

---
## Step 10 — fit a model

```python
def model_fit(train, config):
```

---
## Step 11 — unpack config

```python
n_input, n_nodes, n_epochs, n_batch, n_diff = config
```

---
## Step 12 — prepare data

```python
if n_diff > 0:
		train = difference(train, n_diff)
```

---
## Step 13 — transform series into supervised format

```python
data = series_to_supervised(train, n_in=n_input)
```

---
## Step 14 — separate inputs and outputs

```python
train_x, train_y = data[:, :-1], data[:, -1]
```

---
## Step 15 — define model

```python
model = Sequential()
	model.add(Dense(n_nodes, activation='relu', input_dim=n_input))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
```

---
## Step 16 — fit model

```python
model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model
```

---
## Step 17 — forecast with the fit model

```python
def model_predict(model, history, config):
```

---
## Step 18 — unpack config

```python
n_input, _, _, _, n_diff = config
```

---
## Step 19 — prepare data

```python
correction = 0.0
	if n_diff > 0:
		correction = history[-n_diff]
		history = difference(history, n_diff)
```

---
## Step 20 — shape input for model

```python
x_input = array(history[-n_input:]).reshape((1, n_input))
```

---
## Step 21 — make forecast

```python
yhat = model.predict(x_input, verbose=0)
```

---
## Step 22 — correct forecast if it was differenced

```python
return correction + yhat[0]
```

---
## Step 23 — walk-forward validation for univariate data

```python
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
```

---
## Step 24 — split dataset

```python
train, test = train_test_split(data, n_test)
```

---
## Step 25 — fit model

```python
model = model_fit(train, cfg)
```

---
## Step 26 — seed history with training dataset

```python
history = [x for x in train]
```

---
## Step 27 — step over each time-step in the test set

```python
for i in range(len(test)):
```

---
## Step 28 — fit model and make forecast for history

```python
yhat = model_predict(model, history, cfg)
```

---
## Step 29 — store forecast in list of predictions

```python
predictions.append(yhat)
```

---
## Step 30 — add actual observation to history for the next loop

```python
history.append(test[i])
```

---
## Step 31 — estimate prediction error

```python
error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return error
```

---
## Step 32 — score a model, return None on failure

```python
def repeat_evaluate(data, config, n_test, n_repeats=10):
```

---
## Step 33 — convert config to a key

```python
key = str(config)
```

---
## Step 34 — fit and evaluate the model n times

```python
scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
```

---
## Step 35 — summarize score

```python
result = mean(scores)
	print('> Model[%s] %.3f' % (key, result))
	return (key, result)
```

---
## Step 36 — grid search configs

```python
def grid_search(data, cfg_list, n_test):
```

---
## Step 37 — evaluate configs

```python
scores = scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
```

---
## Step 38 — sort configs by error, asc

```python
scores.sort(key=lambda tup: tup[1])
	return scores
```

---
## Step 39 — create a list of configs to try

```python
def model_configs():
```

---
## Step 40 — define scope of configs

```python
n_input = [12]
	n_nodes = [50, 100]
	n_epochs = [100]
	n_batch = [1, 150]
	n_diff = [0, 12]
```

---
## Step 41 — create configs

```python
configs = list()
	for i in n_input:
		for j in n_nodes:
			for k in n_epochs:
				for l in n_batch:
					for m in n_diff:
						cfg = [i, j, k, l, m]
						configs.append(cfg)
	print('Total configs: %d' % len(configs))
	return configs
```

---
## Step 42 — define dataset

```python
series = read_csv('monthly-airline-passengers.csv', header=0, index_col=0)
data = series.values
```

---
## Step 43 — data split

```python
n_test = 12
```

---
## Step 44 — model configs

```python
cfg_list = model_configs()
```

---
## Step 45 — grid search

```python
scores = grid_search(data, cfg_list, n_test)
print('done')
```

---
## Step 46 — list top 3 configs

```python
for cfg, error in scores[:3]:
	print(cfg, error)
```

---
## Learning Notes / 学习笔记

- **概念**: grid search mlps for monthly airline passengers dataset 是机器学习中的常用技术。  
  *grid search mlps for monthly airline passengers dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropna` | 删除缺失值 | Drop missing values |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grid Search Mlp / 03 Grid Search Mlp
# Complete Code / 完整代码
# ===============================

# grid search mlps for monthly airline passengers dataset
from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	agg.dropna(inplace=True)
	return agg.values

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# difference dataset
def difference(data, order):
	return [data[i] - data[i - order] for i in range(order, len(data))]

# fit a model
def model_fit(train, config):
	# unpack config
	n_input, n_nodes, n_epochs, n_batch, n_diff = config
	# prepare data
	if n_diff > 0:
		train = difference(train, n_diff)
	# transform series into supervised format
	data = series_to_supervised(train, n_in=n_input)
	# separate inputs and outputs
	train_x, train_y = data[:, :-1], data[:, -1]
	# define model
	model = Sequential()
	model.add(Dense(n_nodes, activation='relu', input_dim=n_input))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	# fit model
	model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model

# forecast with the fit model
def model_predict(model, history, config):
	# unpack config
	n_input, _, _, _, n_diff = config
	# prepare data
	correction = 0.0
	if n_diff > 0:
		correction = history[-n_diff]
		history = difference(history, n_diff)
	# shape input for model
	x_input = array(history[-n_input:]).reshape((1, n_input))
	# make forecast
	yhat = model.predict(x_input, verbose=0)
	# correct forecast if it was differenced
	return correction + yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# fit model
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return error

# score a model, return None on failure
def repeat_evaluate(data, config, n_test, n_repeats=10):
	# convert config to a key
	key = str(config)
	# fit and evaluate the model n times
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	# summarize score
	result = mean(scores)
	print('> Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test):
	# evaluate configs
	scores = scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a list of configs to try
def model_configs():
	# define scope of configs
	n_input = [12]
	n_nodes = [50, 100]
	n_epochs = [100]
	n_batch = [1, 150]
	n_diff = [0, 12]
	# create configs
	configs = list()
	for i in n_input:
		for j in n_nodes:
			for k in n_epochs:
				for l in n_batch:
					for m in n_diff:
						cfg = [i, j, k, l, m]
						configs.append(cfg)
	print('Total configs: %d' % len(configs))
	return configs

# define dataset
series = read_csv('monthly-airline-passengers.csv', header=0, index_col=0)
data = series.values
# data split
n_test = 12
# model configs
cfg_list = model_configs()
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 3 configs
for cfg, error in scores[:3]:
	print(cfg, error)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Grid Search Cnn

# 04 — Grid Search Cnn / 卷积神经网络

**Chapter 15 — File 4 of 5 / 第15章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **grid search cnn for monthly airline passengers dataset**.

本脚本演示 **grid search cnn for monthly airline passengers dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
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
```

---
## Step 1 — grid search cnn for monthly airline passengers dataset

```python
from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
```

---
## Step 2 — split a univariate dataset into train/test sets

```python
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]
```

---
## Step 3 — transform list into supervised learning format

```python
def series_to_supervised(data, n_in, n_out=1):
	df = DataFrame(data)
	cols = list()
```

---
## Step 4 — input sequence (t-n, ... t-1)

```python
for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
```

---
## Step 5 — forecast sequence (t, t+1, ... t+n)

```python
for i in range(0, n_out):
		cols.append(df.shift(-i))
```

---
## Step 6 — put it all together

```python
agg = concat(cols, axis=1)
```

---
## Step 7 — drop rows with NaN values

```python
agg.dropna(inplace=True)
	return agg.values
```

---
## Step 8 — root mean squared error or rmse

```python
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))
```

---
## Step 9 — difference dataset

```python
def difference(data, order):
	return [data[i] - data[i - order] for i in range(order, len(data))]
```

---
## Step 10 — fit a model

```python
def model_fit(train, config):
```

---
## Step 11 — unpack config

```python
n_input, n_filters, n_kernel, n_epochs, n_batch, n_diff = config
```

---
## Step 12 — prepare data

```python
if n_diff > 0:
		train = difference(train, n_diff)
```

---
## Step 13 — transform series into supervised format

```python
data = series_to_supervised(train, n_in=n_input)
```

---
## Step 14 — separate inputs and outputs

```python
train_x, train_y = data[:, :-1], data[:, -1]
```

---
## Step 15 — reshape input data into [samples, timesteps, features]

```python
n_features = 1
	train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))
```

---
## Step 16 — define model

```python
model = Sequential()
	model.add(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(n_input, n_features)))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
```

---
## Step 17 — fit

```python
model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model
```

---
## Step 18 — forecast with the fit model

```python
def model_predict(model, history, config):
```

---
## Step 19 — unpack config

```python
n_input, _, _, _, _, n_diff = config
```

---
## Step 20 — prepare data

```python
correction = 0.0
	if n_diff > 0:
		correction = history[-n_diff]
		history = difference(history, n_diff)
	x_input = array(history[-n_input:]).reshape((1, n_input, 1))
```

---
## Step 21 — forecast

```python
yhat = model.predict(x_input, verbose=0)
	return correction + yhat[0]
```

---
## Step 22 — walk-forward validation for univariate data

```python
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
```

---
## Step 23 — split dataset

```python
train, test = train_test_split(data, n_test)
```

---
## Step 24 — fit model

```python
model = model_fit(train, cfg)
```

---
## Step 25 — seed history with training dataset

```python
history = [x for x in train]
```

---
## Step 26 — step over each time-step in the test set

```python
for i in range(len(test)):
```

---
## Step 27 — fit model and make forecast for history

```python
yhat = model_predict(model, history, cfg)
```

---
## Step 28 — store forecast in list of predictions

```python
predictions.append(yhat)
```

---
## Step 29 — add actual observation to history for the next loop

```python
history.append(test[i])
```

---
## Step 30 — estimate prediction error

```python
error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return error
```

---
## Step 31 — score a model, return None on failure

```python
def repeat_evaluate(data, config, n_test, n_repeats=10):
```

---
## Step 32 — convert config to a key

```python
key = str(config)
```

---
## Step 33 — fit and evaluate the model n times

```python
scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
```

---
## Step 34 — summarize score

```python
result = mean(scores)
	print('> Model[%s] %.3f' % (key, result))
	return (key, result)
```

---
## Step 35 — grid search configs

```python
def grid_search(data, cfg_list, n_test):
```

---
## Step 36 — evaluate configs

```python
scores = scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
```

---
## Step 37 — sort configs by error, asc

```python
scores.sort(key=lambda tup: tup[1])
	return scores
```

---
## Step 38 — create a list of configs to try

```python
def model_configs():
```

---
## Step 39 — define scope of configs

```python
n_input = [12]
	n_filters = [64]
	n_kernels = [3, 5]
	n_epochs = [100]
	n_batch = [1, 150]
	n_diff = [0, 12]
```

---
## Step 40 — create configs

```python
configs = list()
	for a in n_input:
		for b in n_filters:
			for c in n_kernels:
				for d in n_epochs:
					for e in n_batch:
						for f in n_diff:
							cfg = [a,b,c,d,e,f]
							configs.append(cfg)
	print('Total configs: %d' % len(configs))
	return configs
```

---
## Step 41 — define dataset

```python
series = read_csv('monthly-airline-passengers.csv', header=0, index_col=0)
data = series.values
```

---
## Step 42 — data split

```python
n_test = 12
```

---
## Step 43 — model configs

```python
cfg_list = model_configs()
```

---
## Step 44 — grid search

```python
scores = grid_search(data, cfg_list, n_test)
print('done')
```

---
## Step 45 — list top 10 configs

```python
for cfg, error in scores[:3]:
	print(cfg, error)
```

---
## Learning Notes / 学习笔记

- **概念**: grid search cnn for monthly airline passengers dataset 是机器学习中的常用技术。  
  *grid search cnn for monthly airline passengers dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropna` | 删除缺失值 | Drop missing values |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grid Search Cnn / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# grid search cnn for monthly airline passengers dataset
from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	agg.dropna(inplace=True)
	return agg.values

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# difference dataset
def difference(data, order):
	return [data[i] - data[i - order] for i in range(order, len(data))]

# fit a model
def model_fit(train, config):
	# unpack config
	n_input, n_filters, n_kernel, n_epochs, n_batch, n_diff = config
	# prepare data
	if n_diff > 0:
		train = difference(train, n_diff)
	# transform series into supervised format
	data = series_to_supervised(train, n_in=n_input)
	# separate inputs and outputs
	train_x, train_y = data[:, :-1], data[:, -1]
	# reshape input data into [samples, timesteps, features]
	n_features = 1
	train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))
	# define model
	model = Sequential()
	model.add(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(n_input, n_features)))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	# fit
	model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model

# forecast with the fit model
def model_predict(model, history, config):
	# unpack config
	n_input, _, _, _, _, n_diff = config
	# prepare data
	correction = 0.0
	if n_diff > 0:
		correction = history[-n_diff]
		history = difference(history, n_diff)
	x_input = array(history[-n_input:]).reshape((1, n_input, 1))
	# forecast
	yhat = model.predict(x_input, verbose=0)
	return correction + yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# fit model
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return error

# score a model, return None on failure
def repeat_evaluate(data, config, n_test, n_repeats=10):
	# convert config to a key
	key = str(config)
	# fit and evaluate the model n times
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	# summarize score
	result = mean(scores)
	print('> Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test):
	# evaluate configs
	scores = scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a list of configs to try
def model_configs():
	# define scope of configs
	n_input = [12]
	n_filters = [64]
	n_kernels = [3, 5]
	n_epochs = [100]
	n_batch = [1, 150]
	n_diff = [0, 12]
	# create configs
	configs = list()
	for a in n_input:
		for b in n_filters:
			for c in n_kernels:
				for d in n_epochs:
					for e in n_batch:
						for f in n_diff:
							cfg = [a,b,c,d,e,f]
							configs.append(cfg)
	print('Total configs: %d' % len(configs))
	return configs

# define dataset
series = read_csv('monthly-airline-passengers.csv', header=0, index_col=0)
data = series.values
# data split
n_test = 12
# model configs
cfg_list = model_configs()
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 10 configs
for cfg, error in scores[:3]:
	print(cfg, error)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Grid Search Lstm

# 05 — Grid Search Lstm / LSTM 网络

**Chapter 15 — File 5 of 5 / 第15章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **grid search lstm for monthly airline passengers dataset**.

本脚本演示 **grid search lstm for monthly airline passengers dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
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
```

---
## Step 1 — grid search lstm for monthly airline passengers dataset

```python
from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
```

---
## Step 2 — split a univariate dataset into train/test sets

```python
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]
```

---
## Step 3 — transform list into supervised learning format

```python
def series_to_supervised(data, n_in, n_out=1):
	df = DataFrame(data)
	cols = list()
```

---
## Step 4 — input sequence (t-n, ... t-1)

```python
for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
```

---
## Step 5 — forecast sequence (t, t+1, ... t+n)

```python
for i in range(0, n_out):
		cols.append(df.shift(-i))
```

---
## Step 6 — put it all together

```python
agg = concat(cols, axis=1)
```

---
## Step 7 — drop rows with NaN values

```python
agg.dropna(inplace=True)
	return agg.values
```

---
## Step 8 — root mean squared error or rmse

```python
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))
```

---
## Step 9 — difference dataset

```python
def difference(data, order):
	return [data[i] - data[i - order] for i in range(order, len(data))]
```

---
## Step 10 — fit a model

```python
def model_fit(train, config):
```

---
## Step 11 — unpack config

```python
n_input, n_nodes, n_epochs, n_batch, n_diff = config
```

---
## Step 12 — prepare data

```python
if n_diff > 0:
		train = difference(train, n_diff)
```

---
## Step 13 — transform series into supervised format

```python
data = series_to_supervised(train, n_in=n_input)
```

---
## Step 14 — separate inputs and outputs

```python
train_x, train_y = data[:, :-1], data[:, -1]
```

---
## Step 15 — reshape input data into [samples, timesteps, features]

```python
n_features = 1
	train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))
```

---
## Step 16 — define model

```python
model = Sequential()
	model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, n_features)))
	model.add(Dense(n_nodes, activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
```

---
## Step 17 — fit model

```python
model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model
```

---
## Step 18 — forecast with the fit model

```python
def model_predict(model, history, config):
```

---
## Step 19 — unpack config

```python
n_input, _, _, _, n_diff = config
```

---
## Step 20 — prepare data

```python
correction = 0.0
	if n_diff > 0:
		correction = history[-n_diff]
		history = difference(history, n_diff)
```

---
## Step 21 — reshape sample into [samples, timesteps, features]

```python
x_input = array(history[-n_input:]).reshape((1, n_input, 1))
```

---
## Step 22 — forecast

```python
yhat = model.predict(x_input, verbose=0)
	return correction + yhat[0]
```

---
## Step 23 — walk-forward validation for univariate data

```python
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
```

---
## Step 24 — split dataset

```python
train, test = train_test_split(data, n_test)
```

---
## Step 25 — fit model

```python
model = model_fit(train, cfg)
```

---
## Step 26 — seed history with training dataset

```python
history = [x for x in train]
```

---
## Step 27 — step over each time-step in the test set

```python
for i in range(len(test)):
```

---
## Step 28 — fit model and make forecast for history

```python
yhat = model_predict(model, history, cfg)
```

---
## Step 29 — store forecast in list of predictions

```python
predictions.append(yhat)
```

---
## Step 30 — add actual observation to history for the next loop

```python
history.append(test[i])
```

---
## Step 31 — estimate prediction error

```python
error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return error
```

---
## Step 32 — score a model, return None on failure

```python
def repeat_evaluate(data, config, n_test, n_repeats=10):
```

---
## Step 33 — convert config to a key

```python
key = str(config)
```

---
## Step 34 — fit and evaluate the model n times

```python
scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
```

---
## Step 35 — summarize score

```python
result = mean(scores)
	print('> Model[%s] %.3f' % (key, result))
	return (key, result)
```

---
## Step 36 — grid search configs

```python
def grid_search(data, cfg_list, n_test):
```

---
## Step 37 — evaluate configs

```python
scores = scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
```

---
## Step 38 — sort configs by error, asc

```python
scores.sort(key=lambda tup: tup[1])
	return scores
```

---
## Step 39 — create a list of configs to try

```python
def model_configs():
```

---
## Step 40 — define scope of configs

```python
n_input = [12]
	n_nodes = [100]
	n_epochs = [50]
	n_batch = [1, 150]
	n_diff = [12]
```

---
## Step 41 — create configs

```python
configs = list()
	for i in n_input:
		for j in n_nodes:
			for k in n_epochs:
				for l in n_batch:
					for m in n_diff:
						cfg = [i, j, k, l, m]
						configs.append(cfg)
	print('Total configs: %d' % len(configs))
	return configs
```

---
## Step 42 — define dataset

```python
series = read_csv('monthly-airline-passengers.csv', header=0, index_col=0)
data = series.values
```

---
## Step 43 — data split

```python
n_test = 12
```

---
## Step 44 — model configs

```python
cfg_list = model_configs()
```

---
## Step 45 — grid search

```python
scores = grid_search(data, cfg_list, n_test)
print('done')
```

---
## Step 46 — list top 10 configs

```python
for cfg, error in scores[:3]:
	print(cfg, error)
```

---
## Learning Notes / 学习笔记

- **概念**: grid search lstm for monthly airline passengers dataset 是机器学习中的常用技术。  
  *grid search lstm for monthly airline passengers dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropna` | 删除缺失值 | Drop missing values |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grid Search Lstm / LSTM 网络
# Complete Code / 完整代码
# ===============================

# grid search lstm for monthly airline passengers dataset
from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	agg.dropna(inplace=True)
	return agg.values

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# difference dataset
def difference(data, order):
	return [data[i] - data[i - order] for i in range(order, len(data))]

# fit a model
def model_fit(train, config):
	# unpack config
	n_input, n_nodes, n_epochs, n_batch, n_diff = config
	# prepare data
	if n_diff > 0:
		train = difference(train, n_diff)
	# transform series into supervised format
	data = series_to_supervised(train, n_in=n_input)
	# separate inputs and outputs
	train_x, train_y = data[:, :-1], data[:, -1]
	# reshape input data into [samples, timesteps, features]
	n_features = 1
	train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))
	# define model
	model = Sequential()
	model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, n_features)))
	model.add(Dense(n_nodes, activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	# fit model
	model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model

# forecast with the fit model
def model_predict(model, history, config):
	# unpack config
	n_input, _, _, _, n_diff = config
	# prepare data
	correction = 0.0
	if n_diff > 0:
		correction = history[-n_diff]
		history = difference(history, n_diff)
	# reshape sample into [samples, timesteps, features]
	x_input = array(history[-n_input:]).reshape((1, n_input, 1))
	# forecast
	yhat = model.predict(x_input, verbose=0)
	return correction + yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# fit model
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return error

# score a model, return None on failure
def repeat_evaluate(data, config, n_test, n_repeats=10):
	# convert config to a key
	key = str(config)
	# fit and evaluate the model n times
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	# summarize score
	result = mean(scores)
	print('> Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test):
	# evaluate configs
	scores = scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a list of configs to try
def model_configs():
	# define scope of configs
	n_input = [12]
	n_nodes = [100]
	n_epochs = [50]
	n_batch = [1, 150]
	n_diff = [12]
	# create configs
	configs = list()
	for i in n_input:
		for j in n_nodes:
			for k in n_epochs:
				for l in n_batch:
					for m in n_diff:
						cfg = [i, j, k, l, m]
						configs.append(cfg)
	print('Total configs: %d' % len(configs))
	return configs

# define dataset
series = read_csv('monthly-airline-passengers.csv', header=0, index_col=0)
data = series.values
# data split
n_test = 12
# model configs
cfg_list = model_configs()
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 10 configs
for cfg, error in scores[:3]:
	print(cfg, error)
```

---
