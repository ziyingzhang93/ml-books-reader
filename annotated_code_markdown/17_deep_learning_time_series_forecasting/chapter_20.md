# DL时间序列
## Chapter 20

---

### Encoder Decoder Lstm

# 02 — Encoder Decoder Lstm / LSTM 网络

**Chapter 20 — File 2 of 5 / 第20章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **univariate multi-step encoder-decoder lstm for the power usage dataset**.

本脚本演示 **univariate multi-step encoder-decoder lstm for the power usage dataset**。

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
## Step 1 — univariate multi-step encoder-decoder lstm for the power usage dataset

```python
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
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
## Step 12 — convert history into inputs and outputs

```python
def to_supervised(train, n_input, n_out=7):
```

---
## Step 13 — flatten data

```python
data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
```

---
## Step 14 — step over the entire history one time step at a time

```python
for _ in range(len(data)):
```

---
## Step 15 — define the end of the input sequence

```python
in_end = in_start + n_input
		out_end = in_end + n_out
```

---
## Step 16 — ensure we have enough data for this instance

```python
if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
```

---
## Step 17 — move along one time step

```python
in_start += 1
	return array(X), array(y)
```

---
## Step 18 — train the model

```python
def build_model(train, n_input):
```

---
## Step 19 — prepare data

```python
train_x, train_y = to_supervised(train, n_input)
```

---
## Step 20 — define parameters

```python
verbose, epochs, batch_size = 0, 20, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
```

---
## Step 21 — reshape output into [samples, timesteps, features]

```python
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
```

---
## Step 22 — define model

```python
model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
```

---
## Step 23 — fit network

```python
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model
```

---
## Step 24 — make a forecast

```python
def forecast(model, history, n_input):
```

---
## Step 25 — flatten data

```python
data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
```

---
## Step 26 — retrieve last observations for input data

```python
input_x = data[-n_input:, 0]
```

---
## Step 27 — reshape into [1, n_input, 1]

```python
input_x = input_x.reshape((1, len(input_x), 1))
```

---
## Step 28 — forecast the next week

```python
yhat = model.predict(input_x, verbose=0)
```

---
## Step 29 — we only want the vector forecast

```python
yhat = yhat[0]
	return yhat
```

---
## Step 30 — evaluate a single model

```python
def evaluate_model(train, test, n_input):
```

---
## Step 31 — fit model

```python
model = build_model(train, n_input)
```

---
## Step 32 — history is a list of weekly data

```python
history = [x for x in train]
```

---
## Step 33 — walk-forward validation over each week

```python
predictions = list()
	for i in range(len(test)):
```

---
## Step 34 — predict the week

```python
yhat_sequence = forecast(model, history, n_input)
```

---
## Step 35 — store the predictions

```python
predictions.append(yhat_sequence)
```

---
## Step 36 — get real observation and add to history for predicting the next week

```python
history.append(test[i, :])
```

---
## Step 37 — evaluate predictions days for each week

```python
predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores
```

---
## Step 38 — load the new file

```python
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
```

---
## Step 39 — split into train and test

```python
train, test = split_dataset(dataset.values)
```

---
## Step 40 — evaluate model and get scores

```python
n_input = 14
score, scores = evaluate_model(train, test, n_input)
```

---
## Step 41 — summarize scores

```python
summarize_scores('lstm', score, scores)
```

---
## Step 42 — plot scores

```python
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores, marker='o', label='lstm')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: univariate multi-step encoder-decoder lstm for the power usage dataset 是机器学习中的常用技术。  
  *univariate multi-step encoder-decoder lstm for the power usage dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Encoder Decoder Lstm / LSTM 网络
# Complete Code / 完整代码
# ===============================

# univariate multi-step encoder-decoder lstm for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

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

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

# train the model
def build_model(train, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 0, 20, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

# evaluate a single model
def evaluate_model(train, test, n_input):
	# fit model
	model = build_model(train, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores

# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
train, test = split_dataset(dataset.values)
# evaluate model and get scores
n_input = 14
score, scores = evaluate_model(train, test, n_input)
# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores, marker='o', label='lstm')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Cnn Encoder Decoder Lstm

# 04 — Cnn Encoder Decoder Lstm / 卷积神经网络

**Chapter 20 — File 4 of 5 / 第20章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **univariate multi-step encoder-decoder cnn-lstm for the power usage dataset**.

本脚本演示 **univariate multi-step encoder-decoder cnn-lstm for the power usage dataset**。

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
## Step 1 — univariate multi-step encoder-decoder cnn-lstm for the power usage dataset

```python
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
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
## Step 12 — convert history into inputs and outputs

```python
def to_supervised(train, n_input, n_out=7):
```

---
## Step 13 — flatten data

```python
data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
```

---
## Step 14 — step over the entire history one time step at a time

```python
for _ in range(len(data)):
```

---
## Step 15 — define the end of the input sequence

```python
in_end = in_start + n_input
		out_end = in_end + n_out
```

---
## Step 16 — ensure we have enough data for this instance

```python
if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
```

---
## Step 17 — move along one time step

```python
in_start += 1
	return array(X), array(y)
```

---
## Step 18 — train the model

```python
def build_model(train, n_input):
```

---
## Step 19 — prepare data

```python
train_x, train_y = to_supervised(train, n_input)
```

---
## Step 20 — define parameters

```python
verbose, epochs, batch_size = 0, 20, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
```

---
## Step 21 — reshape output into [samples, timesteps, features]

```python
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
```

---
## Step 22 — define model

```python
model = Sequential()
	model.add(Conv1D(64, 3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(64, 3, activation='relu'))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
```

---
## Step 23 — fit network

```python
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model
```

---
## Step 24 — make a forecast

```python
def forecast(model, history, n_input):
```

---
## Step 25 — flatten data

```python
data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
```

---
## Step 26 — retrieve last observations for input data

```python
input_x = data[-n_input:, 0]
```

---
## Step 27 — reshape into [1, n_input, 1]

```python
input_x = input_x.reshape((1, len(input_x), 1))
```

---
## Step 28 — forecast the next week

```python
yhat = model.predict(input_x, verbose=0)
```

---
## Step 29 — we only want the vector forecast

```python
yhat = yhat[0]
	return yhat
```

---
## Step 30 — evaluate a single model

```python
def evaluate_model(train, test, n_input):
```

---
## Step 31 — fit model

```python
model = build_model(train, n_input)
```

---
## Step 32 — history is a list of weekly data

```python
history = [x for x in train]
```

---
## Step 33 — walk-forward validation over each week

```python
predictions = list()
	for i in range(len(test)):
```

---
## Step 34 — predict the week

```python
yhat_sequence = forecast(model, history, n_input)
```

---
## Step 35 — store the predictions

```python
predictions.append(yhat_sequence)
```

---
## Step 36 — get real observation and add to history for predicting the next week

```python
history.append(test[i, :])
```

---
## Step 37 — evaluate predictions days for each week

```python
predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores
```

---
## Step 38 — load the new file

```python
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
```

---
## Step 39 — split into train and test

```python
train, test = split_dataset(dataset.values)
```

---
## Step 40 — evaluate model and get scores

```python
n_input = 14
score, scores = evaluate_model(train, test, n_input)
```

---
## Step 41 — summarize scores

```python
summarize_scores('lstm', score, scores)
```

---
## Step 42 — plot scores

```python
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores, marker='o', label='lstm')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: univariate multi-step encoder-decoder cnn-lstm for the power usage dataset 是机器学习中的常用技术。  
  *univariate multi-step encoder-decoder cnn-lstm for the power usage dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cnn Encoder Decoder Lstm / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# univariate multi-step encoder-decoder cnn-lstm for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

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

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

# train the model
def build_model(train, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 0, 20, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(Conv1D(64, 3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(64, 3, activation='relu'))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

# evaluate a single model
def evaluate_model(train, test, n_input):
	# fit model
	model = build_model(train, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores

# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
train, test = split_dataset(dataset.values)
# evaluate model and get scores
n_input = 14
score, scores = evaluate_model(train, test, n_input)
# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores, marker='o', label='lstm')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Convlstm Encoder Decoder Lstm

# 05 — Convlstm Encoder Decoder Lstm / LSTM 网络

**Chapter 20 — File 5 of 5 / 第20章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **univariate multi-step encoder-decoder convlstm for the power usage dataset**.

本脚本演示 **univariate multi-step encoder-decoder convlstm for the power usage dataset**。

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
## Step 1 — univariate multi-step encoder-decoder convlstm for the power usage dataset

```python
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
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
## Step 12 — convert history into inputs and outputs

```python
def to_supervised(train, n_input, n_out=7):
```

---
## Step 13 — flatten data

```python
data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
```

---
## Step 14 — step over the entire history one time step at a time

```python
for _ in range(len(data)):
```

---
## Step 15 — define the end of the input sequence

```python
in_end = in_start + n_input
		out_end = in_end + n_out
```

---
## Step 16 — ensure we have enough data for this instance

```python
if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
```

---
## Step 17 — move along one time step

```python
in_start += 1
	return array(X), array(y)
```

---
## Step 18 — train the model

```python
def build_model(train, n_steps, n_length, n_input):
```

---
## Step 19 — prepare data

```python
train_x, train_y = to_supervised(train, n_input)
```

---
## Step 20 — define parameters

```python
verbose, epochs, batch_size = 0, 20, 16
	n_features, n_outputs = train_x.shape[2], train_y.shape[1]
```

---
## Step 21 — reshape into subsequences [samples, timesteps, rows, cols, channels]

```python
train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
```

---
## Step 22 — reshape output into [samples, timesteps, features]

```python
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
```

---
## Step 23 — define model

```python
model = Sequential()
	model.add(ConvLSTM2D(64, (1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
	model.add(Flatten())
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
```

---
## Step 24 — fit network

```python
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model
```

---
## Step 25 — make a forecast

```python
def forecast(model, history, n_steps, n_length, n_input):
```

---
## Step 26 — flatten data

```python
data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
```

---
## Step 27 — retrieve last observations for input data

```python
input_x = data[-n_input:, 0]
```

---
## Step 28 — reshape into [samples, timesteps, rows, cols, channels]

```python
input_x = input_x.reshape((1, n_steps, 1, n_length, 1))
```

---
## Step 29 — forecast the next week

```python
yhat = model.predict(input_x, verbose=0)
```

---
## Step 30 — we only want the vector forecast

```python
yhat = yhat[0]
	return yhat
```

---
## Step 31 — evaluate a single model

```python
def evaluate_model(train, test, n_steps, n_length, n_input):
```

---
## Step 32 — fit model

```python
model = build_model(train, n_steps, n_length, n_input)
```

---
## Step 33 — history is a list of weekly data

```python
history = [x for x in train]
```

---
## Step 34 — walk-forward validation over each week

```python
predictions = list()
	for i in range(len(test)):
```

---
## Step 35 — predict the week

```python
yhat_sequence = forecast(model, history, n_steps, n_length, n_input)
```

---
## Step 36 — store the predictions

```python
predictions.append(yhat_sequence)
```

---
## Step 37 — get real observation and add to history for predicting the next week

```python
history.append(test[i, :])
```

---
## Step 38 — evaluate predictions days for each week

```python
predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores
```

---
## Step 39 — load the new file

```python
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
```

---
## Step 40 — split into train and test

```python
train, test = split_dataset(dataset.values)
```

---
## Step 41 — define the number of subsequences and the length of subsequences

```python
n_steps, n_length = 2, 7
```

---
## Step 42 — define the total days to use as input

```python
n_input = n_length * n_steps
score, scores = evaluate_model(train, test, n_steps, n_length, n_input)
```

---
## Step 43 — summarize scores

```python
summarize_scores('lstm', score, scores)
```

---
## Step 44 — plot scores

```python
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores, marker='o', label='lstm')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: univariate multi-step encoder-decoder convlstm for the power usage dataset 是机器学习中的常用技术。  
  *univariate multi-step encoder-decoder convlstm for the power usage dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Convlstm Encoder Decoder Lstm / LSTM 网络
# Complete Code / 完整代码
# ===============================

# univariate multi-step encoder-decoder convlstm for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D

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

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

# train the model
def build_model(train, n_steps, n_length, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 0, 20, 16
	n_features, n_outputs = train_x.shape[2], train_y.shape[1]
	# reshape into subsequences [samples, timesteps, rows, cols, channels]
	train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(ConvLSTM2D(64, (1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
	model.add(Flatten())
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# make a forecast
def forecast(model, history, n_steps, n_length, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [samples, timesteps, rows, cols, channels]
	input_x = input_x.reshape((1, n_steps, 1, n_length, 1))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

# evaluate a single model
def evaluate_model(train, test, n_steps, n_length, n_input):
	# fit model
	model = build_model(train, n_steps, n_length, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_steps, n_length, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores

# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
train, test = split_dataset(dataset.values)
# define the number of subsequences and the length of subsequences
n_steps, n_length = 2, 7
# define the total days to use as input
n_input = n_length * n_steps
score, scores = evaluate_model(train, test, n_steps, n_length, n_input)
# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores, marker='o', label='lstm')
pyplot.show()
```

---

### Chapter Summary

# Chapter 20 Summary / 第20章总结

## Theme / 主题: Chapter 20 / Chapter 20

This chapter contains **5 code files** demonstrating chapter 20.

本章包含 **5 个代码文件**，演示Chapter 20。

---
## Evolution / 演化路线

  1. `01_univariate_lstm.ipynb` — Univariate Lstm
  2. `02_encoder_decoder_lstm.ipynb` — Encoder Decoder Lstm
  3. `03_multivariate_encoder_decoder_lstm.ipynb` — Multivariate Encoder Decoder Lstm
  4. `04_cnn_encoder_decoder_lstm.ipynb` — Cnn Encoder Decoder Lstm
  5. `05_convlstm_encoder_decoder_lstm.ipynb` — Convlstm Encoder Decoder Lstm

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 20) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 20）是机器学习流水线中的基础构建块。

---
