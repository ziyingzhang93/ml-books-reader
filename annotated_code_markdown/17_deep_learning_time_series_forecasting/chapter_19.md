# DL时间序列
## Chapter 19

---

### Multiheaded Cnn

# 03 — Multiheaded Cnn / 卷积神经网络

**Chapter 19 — File 3 of 3 / 第19章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **multi headed multi-step cnn for the power usage dataset**.

本脚本演示 **multi headed multi-step cnn for the power usage dataset**。

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
## Step 1 — multi headed multi-step cnn for the power usage dataset

```python
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate
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
			X.append(data[in_start:in_end, :])
			y.append(data[in_end:out_end, 0])
```

---
## Step 17 — move along one time step

```python
in_start += 1
	return array(X), array(y)
```

---
## Step 18 — plot training history

```python
def plot_history(history):
```

---
## Step 19 — plot loss

```python
pyplot.subplot(2, 1, 1)
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.title('loss', y=0, loc='center')
	pyplot.legend()
```

---
## Step 20 — plot rmse

```python
pyplot.subplot(2, 1, 2)
	pyplot.plot(history.history['rmse'], label='train')
	pyplot.plot(history.history['val_rmse'], label='test')
	pyplot.title('rmse', y=0, loc='center')
	pyplot.legend()
	pyplot.show()
```

---
## Step 21 — train the model

```python
def build_model(train, n_input):
```

---
## Step 22 — prepare data

```python
train_x, train_y = to_supervised(train, n_input)
```

---
## Step 23 — define parameters

```python
verbose, epochs, batch_size = 0, 25, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
```

---
## Step 24 — create a channel for each variable

```python
in_layers, out_layers = list(), list()
	for _ in range(n_features):
		inputs = Input(shape=(n_timesteps,1))
		conv1 = Conv1D(32, 3, activation='relu')(inputs)
		conv2 = Conv1D(32, 3, activation='relu')(conv1)
		pool1 = MaxPooling1D()(conv2)
		flat = Flatten()(pool1)
```

---
## Step 25 — store layers

```python
in_layers.append(inputs)
		out_layers.append(flat)
```

---
## Step 26 — merge heads

```python
merged = concatenate(out_layers)
```

---
## Step 27 — interpretation

```python
dense1 = Dense(200, activation='relu')(merged)
	dense2 = Dense(100, activation='relu')(dense1)
	outputs = Dense(n_outputs)(dense2)
	model = Model(inputs=in_layers, outputs=outputs)
```

---
## Step 28 — compile model

```python
model.compile(loss='mse', optimizer='adam')
```

---
## Step 29 — plot the model

```python
plot_model(model, show_shapes=True, to_file='multiheaded_cnn.png')
```

---
## Step 30 — fit network

```python
input_data = [train_x[:,:,i].reshape((train_x.shape[0],n_timesteps,1)) for i in range(n_features)]
	model.fit(input_data, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model
```

---
## Step 31 — make a forecast

```python
def forecast(model, history, n_input):
```

---
## Step 32 — flatten data

```python
data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
```

---
## Step 33 — retrieve last observations for input data

```python
input_x = data[-n_input:, :]
```

---
## Step 34 — reshape into n input arrays

```python
input_x = [input_x[:,i].reshape((1,input_x.shape[0],1)) for i in range(input_x.shape[1])]
```

---
## Step 35 — forecast the next week

```python
yhat = model.predict(input_x, verbose=0)
```

---
## Step 36 — we only want the vector forecast

```python
yhat = yhat[0]
	return yhat
```

---
## Step 37 — evaluate a single model

```python
def evaluate_model(train, test, n_input):
```

---
## Step 38 — fit model

```python
model = build_model(train, n_input)
```

---
## Step 39 — history is a list of weekly data

```python
history = [x for x in train]
```

---
## Step 40 — walk-forward validation over each week

```python
predictions = list()
	for i in range(len(test)):
```

---
## Step 41 — predict the week

```python
yhat_sequence = forecast(model, history, n_input)
```

---
## Step 42 — store the predictions

```python
predictions.append(yhat_sequence)
```

---
## Step 43 — get real observation and add to history for predicting the next week

```python
history.append(test[i, :])
```

---
## Step 44 — evaluate predictions days for each week

```python
predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores
```

---
## Step 45 — load the new file

```python
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
```

---
## Step 46 — split into train and test

```python
train, test = split_dataset(dataset.values)
```

---
## Step 47 — evaluate model and get scores

```python
n_input = 14
score, scores = evaluate_model(train, test, n_input)
```

---
## Step 48 — summarize scores

```python
summarize_scores('cnn', score, scores)
```

---
## Step 49 — plot scores

```python
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores, marker='o', label='cnn')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: multi headed multi-step cnn for the power usage dataset 是机器学习中的常用技术。  
  *multi headed multi-step cnn for the power usage dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
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
# Multiheaded Cnn / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# multi headed multi-step cnn for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate

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
			X.append(data[in_start:in_end, :])
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

# plot training history
def plot_history(history):
	# plot loss
	pyplot.subplot(2, 1, 1)
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.title('loss', y=0, loc='center')
	pyplot.legend()
	# plot rmse
	pyplot.subplot(2, 1, 2)
	pyplot.plot(history.history['rmse'], label='train')
	pyplot.plot(history.history['val_rmse'], label='test')
	pyplot.title('rmse', y=0, loc='center')
	pyplot.legend()
	pyplot.show()

# train the model
def build_model(train, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 0, 25, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# create a channel for each variable
	in_layers, out_layers = list(), list()
	for _ in range(n_features):
		inputs = Input(shape=(n_timesteps,1))
		conv1 = Conv1D(32, 3, activation='relu')(inputs)
		conv2 = Conv1D(32, 3, activation='relu')(conv1)
		pool1 = MaxPooling1D()(conv2)
		flat = Flatten()(pool1)
		# store layers
		in_layers.append(inputs)
		out_layers.append(flat)
	# merge heads
	merged = concatenate(out_layers)
	# interpretation
	dense1 = Dense(200, activation='relu')(merged)
	dense2 = Dense(100, activation='relu')(dense1)
	outputs = Dense(n_outputs)(dense2)
	model = Model(inputs=in_layers, outputs=outputs)
	# compile model
	model.compile(loss='mse', optimizer='adam')
	# plot the model
	plot_model(model, show_shapes=True, to_file='multiheaded_cnn.png')
	# fit network
	input_data = [train_x[:,:,i].reshape((train_x.shape[0],n_timesteps,1)) for i in range(n_features)]
	model.fit(input_data, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, :]
	# reshape into n input arrays
	input_x = [input_x[:,i].reshape((1,input_x.shape[0],1)) for i in range(input_x.shape[1])]
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
summarize_scores('cnn', score, scores)
# plot scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores, marker='o', label='cnn')
pyplot.show()
```

---
