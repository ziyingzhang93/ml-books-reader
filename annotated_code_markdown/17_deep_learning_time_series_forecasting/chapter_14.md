# 深度学习时间序列预测 / DL Time Series Forecasting
## Chapter 14

---

### Simple Forecast

# 01 — Simple Forecast / 预测

**Chapter 14 — File 1 of 6 / 第14章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **persistence forecast for monthly car sales dataset**.

本脚本演示 **persistence forecast for monthly car sales dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — persistence forecast for monthly car sales dataset

```python
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import median
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — split a univariate dataset into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]
```

---
## Step 3 — root mean squared error or rmse

```python
def measure_rmse(actual, predicted):
 # 计算均方误差 / Calculate Mean Squared Error
	return sqrt(mean_squared_error(actual, predicted))
```

---
## Step 4 — difference dataset

```python
def difference(data, interval):
 # 获取长度 / Get length
	return [data[i] - data[i - interval] for i in range(interval, len(data))]
```

---
## Step 5 — fit a model

```python
def model_fit(train, config):
	return None
```

---
## Step 6 — forecast with a pre-fit model

```python
def model_predict(model, history, config):
	values = list()
	for offset in config:
  # 添加元素到列表末尾 / Append element to list end
		values.append(history[-offset])
	return median(values)
```

---
## Step 7 — walk-forward validation for univariate data

```python
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
```

---
## Step 8 — split dataset

```python
# 划分训练集和测试集 / Split into train and test sets
train, test = train_test_split(data, n_test)
```

---
## Step 9 — fit model

```python
model = model_fit(train, cfg)
```

---
## Step 10 — seed history with training dataset

```python
history = [x for x in train]
```

---
## Step 11 — step over each time-step in the test set

```python
# 获取长度 / Get length
for i in range(len(test)):
```

---
## Step 12 — fit model and make forecast for history

```python
yhat = model_predict(model, history, cfg)
```

---
## Step 13 — store forecast in list of predictions

```python
# 添加元素到列表末尾 / Append element to list end
predictions.append(yhat)
```

---
## Step 14 — add actual observation to history for the next loop

```python
# 添加元素到列表末尾 / Append element to list end
history.append(test[i])
```

---
## Step 15 — estimate prediction error

```python
error = measure_rmse(test, predictions)
 # 打印输出 / Print output
	print(' > %.3f' % error)
	return error
```

---
## Step 16 — repeat evaluation of a config

```python
def repeat_evaluate(data, config, n_test, n_repeats=30):
```

---
## Step 17 — fit and evaluate the model n times

```python
# 生成整数序列 / Generate integer sequence
scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return scores
```

---
## Step 18 — summarize model performance

```python
def summarize_scores(name, scores):
```

---
## Step 19 — print a summary

```python
scores_m, score_std = mean(scores), std(scores)
 # 打印输出 / Print output
	print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
```

---
## Step 20 — box and whisker plot

```python
pyplot.boxplot(scores)
	pyplot.show()

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
# 转换为NumPy数组 / Convert to NumPy array
data = series.values
```

---
## Step 21 — data split

```python
n_test = 12
```

---
## Step 22 — define config

```python
config = [12, 24, 36]
```

---
## Step 23 — grid search

```python
scores = repeat_evaluate(data, config, n_test)
```

---
## Step 24 — summarize scores

```python
summarize_scores('persistence', scores)
```

---
## Learning Notes / 学习笔记

- **概念**: persistence forecast for monthly car sales dataset 是机器学习中的常用技术。  
  *persistence forecast for monthly car sales dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
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
# Simple Forecast / 预测
# Complete Code / 完整代码
# ===============================

# persistence forecast for monthly car sales dataset
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import median
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# split a univariate dataset into train/test sets
# 划分训练集和测试集 / Split into train and test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
 # 计算均方误差 / Calculate Mean Squared Error
	return sqrt(mean_squared_error(actual, predicted))

# difference dataset
def difference(data, interval):
 # 获取长度 / Get length
	return [data[i] - data[i - interval] for i in range(interval, len(data))]

# fit a model
def model_fit(train, config):
	return None

# forecast with a pre-fit model
def model_predict(model, history, config):
	values = list()
	for offset in config:
  # 添加元素到列表末尾 / Append element to list end
		values.append(history[-offset])
	return median(values)

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
 # 划分训练集和测试集 / Split into train and test sets
	train, test = train_test_split(data, n_test)
	# fit model
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
 # 获取长度 / Get length
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(yhat)
		# add actual observation to history for the next loop
  # 添加元素到列表末尾 / Append element to list end
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
 # 打印输出 / Print output
	print(' > %.3f' % error)
	return error

# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats=30):
	# fit and evaluate the model n times
 # 生成整数序列 / Generate integer sequence
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return scores

# summarize model performance
def summarize_scores(name, scores):
	# print a summary
	scores_m, score_std = mean(scores), std(scores)
 # 打印输出 / Print output
	print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
	# box and whisker plot
	pyplot.boxplot(scores)
	pyplot.show()

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
# 转换为NumPy数组 / Convert to NumPy array
data = series.values
# data split
n_test = 12
# define config
config = [12, 24, 36]
# grid search
scores = repeat_evaluate(data, config, n_test)
# summarize scores
summarize_scores('persistence', scores)
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Mlp Forecast Model



---

### Cnn Forecast Model



---

### Lstm Forecast Model

# 04 — Lstm Forecast Model / LSTM 网络

**Chapter 14 — File 4 of 6 / 第14章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **evaluate lstm for monthly car sales dataset**.

本脚本演示 **evaluate lstm for monthly car sales dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — evaluate lstm for monthly car sales dataset

```python
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — split a univariate dataset into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
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
# 生成整数序列 / Generate integer sequence
for i in range(n_in, 0, -1):
  # 添加元素到列表末尾 / Append element to list end
		cols.append(df.shift(i))
```

---
## Step 5 — forecast sequence (t, t+1, ... t+n)

```python
# 生成整数序列 / Generate integer sequence
for i in range(0, n_out):
  # 添加元素到列表末尾 / Append element to list end
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
# 删除含缺失值的行 / Drop rows with missing values
agg.dropna(inplace=True)
 # 转换为NumPy数组 / Convert to NumPy array
	return agg.values
```

---
## Step 8 — root mean squared error or rmse

```python
def measure_rmse(actual, predicted):
 # 计算均方误差 / Calculate Mean Squared Error
	return sqrt(mean_squared_error(actual, predicted))
```

---
## Step 9 — difference dataset

```python
def difference(data, interval):
 # 获取长度 / Get length
	return [data[i] - data[i - interval] for i in range(interval, len(data))]
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
	data = series_to_supervised(train, n_input)
	train_x, train_y = data[:, :-1], data[:, -1]
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
```

---
## Step 13 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, 1)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='mse', optimizer='adam')
```

---
## Step 14 — fit

```python
# 训练模型 / Train the model
model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model
```

---
## Step 15 — forecast with a pre-fit model

```python
def model_predict(model, history, config):
```

---
## Step 16 — unpack config

```python
n_input, _, _, _, n_diff = config
```

---
## Step 17 — prepare data

```python
correction = 0.0
	if n_diff > 0:
		correction = history[-n_diff]
		history = difference(history, n_diff)
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	x_input = array(history[-n_input:]).reshape((1, n_input, 1))
```

---
## Step 18 — forecast

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
	return correction + yhat[0]
```

---
## Step 19 — walk-forward validation for univariate data

```python
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
```

---
## Step 20 — split dataset

```python
# 划分训练集和测试集 / Split into train and test sets
train, test = train_test_split(data, n_test)
```

---
## Step 21 — fit model

```python
model = model_fit(train, cfg)
```

---
## Step 22 — seed history with training dataset

```python
history = [x for x in train]
```

---
## Step 23 — step over each time-step in the test set

```python
# 获取长度 / Get length
for i in range(len(test)):
```

---
## Step 24 — fit model and make forecast for history

```python
yhat = model_predict(model, history, cfg)
```

---
## Step 25 — store forecast in list of predictions

```python
# 添加元素到列表末尾 / Append element to list end
predictions.append(yhat)
```

---
## Step 26 — add actual observation to history for the next loop

```python
# 添加元素到列表末尾 / Append element to list end
history.append(test[i])
```

---
## Step 27 — estimate prediction error

```python
error = measure_rmse(test, predictions)
 # 打印输出 / Print output
	print(' > %.3f' % error)
	return error
```

---
## Step 28 — repeat evaluation of a config

```python
def repeat_evaluate(data, config, n_test, n_repeats=30):
```

---
## Step 29 — fit and evaluate the model n times

```python
# 生成整数序列 / Generate integer sequence
scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return scores
```

---
## Step 30 — summarize model performance

```python
def summarize_scores(name, scores):
```

---
## Step 31 — print a summary

```python
scores_m, score_std = mean(scores), std(scores)
 # 打印输出 / Print output
	print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
```

---
## Step 32 — box and whisker plot

```python
pyplot.boxplot(scores)
	pyplot.show()

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
# 转换为NumPy数组 / Convert to NumPy array
data = series.values
```

---
## Step 33 — data split

```python
n_test = 12
```

---
## Step 34 — define config

```python
config = [36, 50, 100, 100, 12]
```

---
## Step 35 — grid search

```python
scores = repeat_evaluate(data, config, n_test)
```

---
## Step 36 — summarize scores

```python
summarize_scores('lstm', scores)
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate lstm for monthly car sales dataset 是机器学习中的常用技术。  
  *evaluate lstm for monthly car sales dataset is a common technique in machine learning.*

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
| `matplotlib` | 绑图库 | Plotting library |
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
# Lstm Forecast Model / LSTM 网络
# Complete Code / 完整代码
# ===============================

# evaluate lstm for monthly car sales dataset
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# split a univariate dataset into train/test sets
# 划分训练集和测试集 / Split into train and test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
 # 生成整数序列 / Generate integer sequence
	for i in range(n_in, 0, -1):
  # 添加元素到列表末尾 / Append element to list end
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
 # 生成整数序列 / Generate integer sequence
	for i in range(0, n_out):
  # 添加元素到列表末尾 / Append element to list end
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
 # 删除含缺失值的行 / Drop rows with missing values
	agg.dropna(inplace=True)
 # 转换为NumPy数组 / Convert to NumPy array
	return agg.values

# root mean squared error or rmse
def measure_rmse(actual, predicted):
 # 计算均方误差 / Calculate Mean Squared Error
	return sqrt(mean_squared_error(actual, predicted))

# difference dataset
def difference(data, interval):
 # 获取长度 / Get length
	return [data[i] - data[i - interval] for i in range(interval, len(data))]

# fit a model
def model_fit(train, config):
	# unpack config
	n_input, n_nodes, n_epochs, n_batch, n_diff = config
	# prepare data
	if n_diff > 0:
		train = difference(train, n_diff)
	data = series_to_supervised(train, n_input)
	train_x, train_y = data[:, :-1], data[:, -1]
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, 1)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='mse', optimizer='adam')
	# fit
 # 训练模型 / Train the model
	model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model

# forecast with a pre-fit model
def model_predict(model, history, config):
	# unpack config
	n_input, _, _, _, n_diff = config
	# prepare data
	correction = 0.0
	if n_diff > 0:
		correction = history[-n_diff]
		history = difference(history, n_diff)
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	x_input = array(history[-n_input:]).reshape((1, n_input, 1))
	# forecast
 # 用模型做预测 / Make predictions with model
	yhat = model.predict(x_input, verbose=0)
	return correction + yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
 # 划分训练集和测试集 / Split into train and test sets
	train, test = train_test_split(data, n_test)
	# fit model
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
 # 获取长度 / Get length
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(yhat)
		# add actual observation to history for the next loop
  # 添加元素到列表末尾 / Append element to list end
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
 # 打印输出 / Print output
	print(' > %.3f' % error)
	return error

# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats=30):
	# fit and evaluate the model n times
 # 生成整数序列 / Generate integer sequence
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return scores

# summarize model performance
def summarize_scores(name, scores):
	# print a summary
	scores_m, score_std = mean(scores), std(scores)
 # 打印输出 / Print output
	print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
	# box and whisker plot
	pyplot.boxplot(scores)
	pyplot.show()

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
# 转换为NumPy数组 / Convert to NumPy array
data = series.values
# data split
n_test = 12
# define config
config = [36, 50, 100, 100, 12]
# grid search
scores = repeat_evaluate(data, config, n_test)
# summarize scores
summarize_scores('lstm', scores)
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Cnn Lstm Forecast Model

# 05 — Cnn Lstm Forecast Model / 卷积神经网络

**Chapter 14 — File 5 of 6 / 第14章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **evaluate cnn-lstm for monthly car sales dataset**.

本脚本演示 **evaluate cnn-lstm for monthly car sales dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — evaluate cnn-lstm for monthly car sales dataset

```python
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import TimeDistributed
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — split a univariate dataset into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
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
# 生成整数序列 / Generate integer sequence
for i in range(n_in, 0, -1):
  # 添加元素到列表末尾 / Append element to list end
		cols.append(df.shift(i))
```

---
## Step 5 — forecast sequence (t, t+1, ... t+n)

```python
# 生成整数序列 / Generate integer sequence
for i in range(0, n_out):
  # 添加元素到列表末尾 / Append element to list end
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
# 删除含缺失值的行 / Drop rows with missing values
agg.dropna(inplace=True)
 # 转换为NumPy数组 / Convert to NumPy array
	return agg.values
```

---
## Step 8 — root mean squared error or rmse

```python
def measure_rmse(actual, predicted):
 # 计算均方误差 / Calculate Mean Squared Error
	return sqrt(mean_squared_error(actual, predicted))
```

---
## Step 9 — fit a model

```python
def model_fit(train, config):
```

---
## Step 10 — unpack config

```python
n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch = config
	n_input = n_seq * n_steps
```

---
## Step 11 — prepare data

```python
data = series_to_supervised(train, n_input)
	train_x, train_y = data[:, :-1], data[:, -1]
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	train_x = train_x.reshape((train_x.shape[0], n_seq, n_steps, 1))
```

---
## Step 12 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(None,n_steps,1))))
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu')))
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(MaxPooling1D()))
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(Flatten()))
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(n_nodes, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='mse', optimizer='adam')
```

---
## Step 13 — fit

```python
# 训练模型 / Train the model
model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model
```

---
## Step 14 — forecast with a pre-fit model

```python
def model_predict(model, history, config):
```

---
## Step 15 — unpack config

```python
n_seq, n_steps, _, _, _, _, _ = config
	n_input = n_seq * n_steps
```

---
## Step 16 — prepare data

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = array(history[-n_input:]).reshape((1, n_seq, n_steps, 1))
```

---
## Step 17 — forecast

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
	return yhat[0]
```

---
## Step 18 — walk-forward validation for univariate data

```python
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
```

---
## Step 19 — split dataset

```python
# 划分训练集和测试集 / Split into train and test sets
train, test = train_test_split(data, n_test)
```

---
## Step 20 — fit model

```python
model = model_fit(train, cfg)
```

---
## Step 21 — seed history with training dataset

```python
history = [x for x in train]
```

---
## Step 22 — step over each time-step in the test set

```python
# 获取长度 / Get length
for i in range(len(test)):
```

---
## Step 23 — fit model and make forecast for history

```python
yhat = model_predict(model, history, cfg)
```

---
## Step 24 — store forecast in list of predictions

```python
# 添加元素到列表末尾 / Append element to list end
predictions.append(yhat)
```

---
## Step 25 — add actual observation to history for the next loop

```python
# 添加元素到列表末尾 / Append element to list end
history.append(test[i])
```

---
## Step 26 — estimate prediction error

```python
error = measure_rmse(test, predictions)
 # 打印输出 / Print output
	print(' > %.3f' % error)
	return error
```

---
## Step 27 — repeat evaluation of a config

```python
def repeat_evaluate(data, config, n_test, n_repeats=30):
```

---
## Step 28 — fit and evaluate the model n times

```python
# 生成整数序列 / Generate integer sequence
scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return scores
```

---
## Step 29 — summarize model performance

```python
def summarize_scores(name, scores):
```

---
## Step 30 — print a summary

```python
scores_m, score_std = mean(scores), std(scores)
 # 打印输出 / Print output
	print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
```

---
## Step 31 — box and whisker plot

```python
pyplot.boxplot(scores)
	pyplot.show()

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
# 转换为NumPy数组 / Convert to NumPy array
data = series.values
```

---
## Step 32 — data split

```python
n_test = 12
```

---
## Step 33 — define config

```python
config = [3, 12, 64, 3, 100, 200, 100]
```

---
## Step 34 — grid search

```python
scores = repeat_evaluate(data, config, n_test)
```

---
## Step 35 — summarize scores

```python
summarize_scores('cnn-lstm', scores)
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate cnn-lstm for monthly car sales dataset 是机器学习中的常用技术。  
  *evaluate cnn-lstm for monthly car sales dataset is a common technique in machine learning.*

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
| `matplotlib` | 绑图库 | Plotting library |
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
# Cnn Lstm Forecast Model / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# evaluate cnn-lstm for monthly car sales dataset
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import TimeDistributed
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# split a univariate dataset into train/test sets
# 划分训练集和测试集 / Split into train and test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
 # 生成整数序列 / Generate integer sequence
	for i in range(n_in, 0, -1):
  # 添加元素到列表末尾 / Append element to list end
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
 # 生成整数序列 / Generate integer sequence
	for i in range(0, n_out):
  # 添加元素到列表末尾 / Append element to list end
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
 # 删除含缺失值的行 / Drop rows with missing values
	agg.dropna(inplace=True)
 # 转换为NumPy数组 / Convert to NumPy array
	return agg.values

# root mean squared error or rmse
def measure_rmse(actual, predicted):
 # 计算均方误差 / Calculate Mean Squared Error
	return sqrt(mean_squared_error(actual, predicted))

# fit a model
def model_fit(train, config):
	# unpack config
	n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch = config
	n_input = n_seq * n_steps
	# prepare data
	data = series_to_supervised(train, n_input)
	train_x, train_y = data[:, :-1], data[:, -1]
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	train_x = train_x.reshape((train_x.shape[0], n_seq, n_steps, 1))
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(None,n_steps,1))))
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu')))
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(MaxPooling1D()))
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(Flatten()))
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(n_nodes, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='mse', optimizer='adam')
	# fit
 # 训练模型 / Train the model
	model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model

# forecast with a pre-fit model
def model_predict(model, history, config):
	# unpack config
	n_seq, n_steps, _, _, _, _, _ = config
	n_input = n_seq * n_steps
	# prepare data
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	x_input = array(history[-n_input:]).reshape((1, n_seq, n_steps, 1))
	# forecast
 # 用模型做预测 / Make predictions with model
	yhat = model.predict(x_input, verbose=0)
	return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
 # 划分训练集和测试集 / Split into train and test sets
	train, test = train_test_split(data, n_test)
	# fit model
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
 # 获取长度 / Get length
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(yhat)
		# add actual observation to history for the next loop
  # 添加元素到列表末尾 / Append element to list end
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
 # 打印输出 / Print output
	print(' > %.3f' % error)
	return error

# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats=30):
	# fit and evaluate the model n times
 # 生成整数序列 / Generate integer sequence
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return scores

# summarize model performance
def summarize_scores(name, scores):
	# print a summary
	scores_m, score_std = mean(scores), std(scores)
 # 打印输出 / Print output
	print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
	# box and whisker plot
	pyplot.boxplot(scores)
	pyplot.show()

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
# 转换为NumPy数组 / Convert to NumPy array
data = series.values
# data split
n_test = 12
# define config
config = [3, 12, 64, 3, 100, 200, 100]
# grid search
scores = repeat_evaluate(data, config, n_test)
# summarize scores
summarize_scores('cnn-lstm', scores)
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Convlstm Forecast Model

# 06 — Convlstm Forecast Model / LSTM 网络

**Chapter 14 — File 6 of 6 / 第14章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **evaluate convlstm for monthly car sales dataset**.

本脚本演示 **evaluate convlstm for monthly car sales dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — evaluate convlstm for monthly car sales dataset

```python
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import ConvLSTM2D
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — split a univariate dataset into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
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
# 生成整数序列 / Generate integer sequence
for i in range(n_in, 0, -1):
  # 添加元素到列表末尾 / Append element to list end
		cols.append(df.shift(i))
```

---
## Step 5 — forecast sequence (t, t+1, ... t+n)

```python
# 生成整数序列 / Generate integer sequence
for i in range(0, n_out):
  # 添加元素到列表末尾 / Append element to list end
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
# 删除含缺失值的行 / Drop rows with missing values
agg.dropna(inplace=True)
 # 转换为NumPy数组 / Convert to NumPy array
	return agg.values
```

---
## Step 8 — root mean squared error or rmse

```python
def measure_rmse(actual, predicted):
 # 计算均方误差 / Calculate Mean Squared Error
	return sqrt(mean_squared_error(actual, predicted))
```

---
## Step 9 — difference dataset

```python
def difference(data, interval):
 # 获取长度 / Get length
	return [data[i] - data[i - interval] for i in range(interval, len(data))]
```

---
## Step 10 — fit a model

```python
def model_fit(train, config):
```

---
## Step 11 — unpack config

```python
n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch = config
	n_input = n_seq * n_steps
```

---
## Step 12 — prepare data

```python
data = series_to_supervised(train, n_input)
	train_x, train_y = data[:, :-1], data[:, -1]
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	train_x = train_x.reshape((train_x.shape[0], n_seq, 1, n_steps, 1))
```

---
## Step 13 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(ConvLSTM2D(n_filters, (1,n_kernel), activation='relu', input_shape=(n_seq, 1, n_steps, 1)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='mse', optimizer='adam')
```

---
## Step 14 — fit

```python
# 训练模型 / Train the model
model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model
```

---
## Step 15 — forecast with a pre-fit model

```python
def model_predict(model, history, config):
```

---
## Step 16 — unpack config

```python
n_seq, n_steps, _, _, _, _, _ = config
	n_input = n_seq * n_steps
```

---
## Step 17 — prepare data

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = array(history[-n_input:]).reshape((1, n_seq, 1, n_steps, 1))
```

---
## Step 18 — forecast

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
	return yhat[0]
```

---
## Step 19 — walk-forward validation for univariate data

```python
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
```

---
## Step 20 — split dataset

```python
# 划分训练集和测试集 / Split into train and test sets
train, test = train_test_split(data, n_test)
```

---
## Step 21 — fit model

```python
model = model_fit(train, cfg)
```

---
## Step 22 — seed history with training dataset

```python
history = [x for x in train]
```

---
## Step 23 — step over each time-step in the test set

```python
# 获取长度 / Get length
for i in range(len(test)):
```

---
## Step 24 — fit model and make forecast for history

```python
yhat = model_predict(model, history, cfg)
```

---
## Step 25 — store forecast in list of predictions

```python
# 添加元素到列表末尾 / Append element to list end
predictions.append(yhat)
```

---
## Step 26 — add actual observation to history for the next loop

```python
# 添加元素到列表末尾 / Append element to list end
history.append(test[i])
```

---
## Step 27 — estimate prediction error

```python
error = measure_rmse(test, predictions)
 # 打印输出 / Print output
	print(' > %.3f' % error)
	return error
```

---
## Step 28 — repeat evaluation of a config

```python
def repeat_evaluate(data, config, n_test, n_repeats=30):
```

---
## Step 29 — fit and evaluate the model n times

```python
# 生成整数序列 / Generate integer sequence
scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return scores
```

---
## Step 30 — summarize model performance

```python
def summarize_scores(name, scores):
```

---
## Step 31 — print a summary

```python
scores_m, score_std = mean(scores), std(scores)
 # 打印输出 / Print output
	print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
```

---
## Step 32 — box and whisker plot

```python
pyplot.boxplot(scores)
	pyplot.show()

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
# 转换为NumPy数组 / Convert to NumPy array
data = series.values
```

---
## Step 33 — data split

```python
n_test = 12
```

---
## Step 34 — define config

```python
config = [3, 12, 256, 3, 200, 200, 100]
```

---
## Step 35 — grid search

```python
scores = repeat_evaluate(data, config, n_test)
```

---
## Step 36 — summarize scores

```python
summarize_scores('convlstm', scores)
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate convlstm for monthly car sales dataset 是机器学习中的常用技术。  
  *evaluate convlstm for monthly car sales dataset is a common technique in machine learning.*

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
| `matplotlib` | 绑图库 | Plotting library |
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
# Convlstm Forecast Model / LSTM 网络
# Complete Code / 完整代码
# ===============================

# evaluate convlstm for monthly car sales dataset
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import ConvLSTM2D
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# split a univariate dataset into train/test sets
# 划分训练集和测试集 / Split into train and test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
 # 生成整数序列 / Generate integer sequence
	for i in range(n_in, 0, -1):
  # 添加元素到列表末尾 / Append element to list end
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
 # 生成整数序列 / Generate integer sequence
	for i in range(0, n_out):
  # 添加元素到列表末尾 / Append element to list end
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
 # 删除含缺失值的行 / Drop rows with missing values
	agg.dropna(inplace=True)
 # 转换为NumPy数组 / Convert to NumPy array
	return agg.values

# root mean squared error or rmse
def measure_rmse(actual, predicted):
 # 计算均方误差 / Calculate Mean Squared Error
	return sqrt(mean_squared_error(actual, predicted))

# difference dataset
def difference(data, interval):
 # 获取长度 / Get length
	return [data[i] - data[i - interval] for i in range(interval, len(data))]

# fit a model
def model_fit(train, config):
	# unpack config
	n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch = config
	n_input = n_seq * n_steps
	# prepare data
	data = series_to_supervised(train, n_input)
	train_x, train_y = data[:, :-1], data[:, -1]
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	train_x = train_x.reshape((train_x.shape[0], n_seq, 1, n_steps, 1))
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(ConvLSTM2D(n_filters, (1,n_kernel), activation='relu', input_shape=(n_seq, 1, n_steps, 1)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='mse', optimizer='adam')
	# fit
 # 训练模型 / Train the model
	model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model

# forecast with a pre-fit model
def model_predict(model, history, config):
	# unpack config
	n_seq, n_steps, _, _, _, _, _ = config
	n_input = n_seq * n_steps
	# prepare data
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	x_input = array(history[-n_input:]).reshape((1, n_seq, 1, n_steps, 1))
	# forecast
 # 用模型做预测 / Make predictions with model
	yhat = model.predict(x_input, verbose=0)
	return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
 # 划分训练集和测试集 / Split into train and test sets
	train, test = train_test_split(data, n_test)
	# fit model
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
 # 获取长度 / Get length
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(yhat)
		# add actual observation to history for the next loop
  # 添加元素到列表末尾 / Append element to list end
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
 # 打印输出 / Print output
	print(' > %.3f' % error)
	return error

# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats=30):
	# fit and evaluate the model n times
 # 生成整数序列 / Generate integer sequence
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return scores

# summarize model performance
def summarize_scores(name, scores):
	# print a summary
	scores_m, score_std = mean(scores), std(scores)
 # 打印输出 / Print output
	print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
	# box and whisker plot
	pyplot.boxplot(scores)
	pyplot.show()

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
# 转换为NumPy数组 / Convert to NumPy array
data = series.values
# data split
n_test = 12
# define config
config = [3, 12, 256, 3, 200, 200, 100]
# grid search
scores = repeat_evaluate(data, config, n_test)
# summarize scores
summarize_scores('convlstm', scores)
```

---

### Chapter Summary / 章节总结

# Chapter 14 Summary / 第14章总结

## Theme / 主题: Chapter 14 / Chapter 14

This chapter contains **6 code files** demonstrating chapter 14.

本章包含 **6 个代码文件**，演示Chapter 14。

---
## Evolution / 演化路线

  1. `01_simple_forecast.ipynb` — Simple Forecast
  2. `02_mlp_forecast_model.ipynb` — Mlp Forecast Model
  3. `03_cnn_forecast_model.ipynb` — Cnn Forecast Model
  4. `04_lstm_forecast_model.ipynb` — Lstm Forecast Model
  5. `05_cnn_lstm_forecast_model.ipynb` — Cnn Lstm Forecast Model
  6. `06_convlstm_forecast_model.ipynb` — Convlstm Forecast Model

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 14) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 14）是机器学习流水线中的基础构建块。

---
