# 深度学习时间序列预测 / DL Time Series Forecasting
## Chapter 25

---

### Lstm Model



---

### Cnn Lstm Model

# 02 — Cnn Lstm Model / 卷积神经网络

**Chapter 25 — File 2 of 3 / 第25章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **cnn lstm model for the har dataset**.

本脚本演示 **cnn lstm model for the har dataset**。

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
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — cnn lstm model for the har dataset

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import dstack
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dropout
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import TimeDistributed
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
 # 转换为NumPy数组 / Convert to NumPy array
	return dataframe.values
```

---
## Step 3 — load a list of files and return as a 3d numpy array

```python
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
  # 添加元素到列表末尾 / Append element to list end
		loaded.append(data)
```

---
## Step 4 — stack group so that features are the 3rd dimension

```python
loaded = dstack(loaded)
	return loaded
```

---
## Step 5 — load a dataset group, such as train or test

```python
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
```

---
## Step 6 — load all 9 files as a single array

```python
filenames = list()
```

---
## Step 7 — total acceleration

```python
filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
```

---
## Step 8 — body acceleration

```python
filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
```

---
## Step 9 — body gyroscope

```python
filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
```

---
## Step 10 — load input data

```python
X = load_group(filenames, filepath)
```

---
## Step 11 — load class output

```python
y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y
```

---
## Step 12 — load the dataset, returns train and test X and y elements

```python
def load_dataset(prefix=''):
```

---
## Step 13 — load all train

```python
trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
```

---
## Step 14 — load all test

```python
testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
```

---
## Step 15 — zero-offset class values

```python
trainy = trainy - 1
	testy = testy - 1
```

---
## Step 16 — one hot encode y

```python
trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	return trainX, trainy, testX, testy
```

---
## Step 17 — fit and evaluate a model

```python
def evaluate_model(trainX, trainy, testX, testy):
```

---
## Step 18 — define model

```python
verbose, epochs, batch_size = 0, 25, 64
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	n_features, n_outputs = trainX.shape[2], trainy.shape[1]
```

---
## Step 19 — reshape data into time steps of sub-sequences

```python
n_steps, n_length = 4, 32
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
```

---
## Step 20 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(Conv1D(64, 3, activation='relu'), input_shape=(None,n_length,n_features)))
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(Conv1D(64, 3, activation='relu')))
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(Dropout(0.5)))
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(MaxPooling1D()))
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(Flatten()))
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(100))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dropout(0.5))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(100, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_outputs, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 21 — fit network

```python
# 训练模型 / Train the model
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
```

---
## Step 22 — evaluate model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy
```

---
## Step 23 — summarize scores

```python
def summarize_results(scores):
 # 打印输出 / Print output
	print(scores)
	m, s = mean(scores), std(scores)
 # 打印输出 / Print output
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
```

---
## Step 24 — run an experiment

```python
def run_experiment(repeats=10):
```

---
## Step 25 — load data

```python
trainX, trainy, testX, testy = load_dataset()
```

---
## Step 26 — repeat experiment

```python
scores = list()
 # 生成整数序列 / Generate integer sequence
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
  # 打印输出 / Print output
		print('>#%d: %.3f' % (r+1, score))
  # 添加元素到列表末尾 / Append element to list end
		scores.append(score)
```

---
## Step 27 — summarize results

```python
summarize_results(scores)
```

---
## Step 28 — run the experiment

```python
run_experiment()
```

---
## Learning Notes / 学习笔记

- **概念**: cnn lstm model for the har dataset 是机器学习中的常用技术。  
  *cnn lstm model for the har dataset is a common technique in machine learning.*

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
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cnn Lstm Model / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# cnn lstm model for the har dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import dstack
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dropout
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import TimeDistributed
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical

# load a single file as a numpy array
def load_file(filepath):
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
 # 转换为NumPy数组 / Convert to NumPy array
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
  # 添加元素到列表末尾 / Append element to list end
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	# define model
	verbose, epochs, batch_size = 0, 25, 64
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	n_features, n_outputs = trainX.shape[2], trainy.shape[1]
	# reshape data into time steps of sub-sequences
	n_steps, n_length = 4, 32
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(Conv1D(64, 3, activation='relu'), input_shape=(None,n_length,n_features)))
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(Conv1D(64, 3, activation='relu')))
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(Dropout(0.5)))
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(MaxPooling1D()))
 # 向模型添加一层 / Add a layer to the model
	model.add(TimeDistributed(Flatten()))
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(100))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dropout(0.5))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(100, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_outputs, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
 # 训练模型 / Train the model
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores):
 # 打印输出 / Print output
	print(scores)
	m, s = mean(scores), std(scores)
 # 打印输出 / Print output
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
 # 生成整数序列 / Generate integer sequence
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
  # 打印输出 / Print output
		print('>#%d: %.3f' % (r+1, score))
  # 添加元素到列表末尾 / Append element to list end
		scores.append(score)
	# summarize results
	summarize_results(scores)

# run the experiment
run_experiment()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Convlstm Model

# 03 — Convlstm Model / LSTM 网络

**Chapter 25 — File 3 of 3 / 第25章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **convlstm model for the har dataset**.

本脚本演示 **convlstm model for the har dataset**。

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
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — convlstm model for the har dataset

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import dstack
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dropout
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import ConvLSTM2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
 # 转换为NumPy数组 / Convert to NumPy array
	return dataframe.values
```

---
## Step 3 — load a list of files and return as a 3d numpy array

```python
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
  # 添加元素到列表末尾 / Append element to list end
		loaded.append(data)
```

---
## Step 4 — stack group so that features are the 3rd dimension

```python
loaded = dstack(loaded)
	return loaded
```

---
## Step 5 — load a dataset group, such as train or test

```python
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
```

---
## Step 6 — load all 9 files as a single array

```python
filenames = list()
```

---
## Step 7 — total acceleration

```python
filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
```

---
## Step 8 — body acceleration

```python
filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
```

---
## Step 9 — body gyroscope

```python
filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
```

---
## Step 10 — load input data

```python
X = load_group(filenames, filepath)
```

---
## Step 11 — load class output

```python
y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y
```

---
## Step 12 — load the dataset, returns train and test X and y elements

```python
def load_dataset(prefix=''):
```

---
## Step 13 — load all train

```python
trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
```

---
## Step 14 — load all test

```python
testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
```

---
## Step 15 — zero-offset class values

```python
trainy = trainy - 1
	testy = testy - 1
```

---
## Step 16 — one hot encode y

```python
trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	return trainX, trainy, testX, testy
```

---
## Step 17 — fit and evaluate a model

```python
def evaluate_model(trainX, trainy, testX, testy):
```

---
## Step 18 — define model

```python
verbose, epochs, batch_size = 0, 25, 64
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	n_features, n_outputs = trainX.shape[2], trainy.shape[1]
```

---
## Step 19 — reshape into subsequences (samples, time steps, rows, cols, channels)

```python
n_steps, n_length = 4, 32
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
```

---
## Step 20 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(ConvLSTM2D(64, (1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dropout(0.5))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(100, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_outputs, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 21 — fit network

```python
# 训练模型 / Train the model
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
```

---
## Step 22 — evaluate model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy
```

---
## Step 23 — summarize scores

```python
def summarize_results(scores):
 # 打印输出 / Print output
	print(scores)
	m, s = mean(scores), std(scores)
 # 打印输出 / Print output
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
```

---
## Step 24 — run an experiment

```python
def run_experiment(repeats=10):
```

---
## Step 25 — load data

```python
trainX, trainy, testX, testy = load_dataset()
```

---
## Step 26 — repeat experiment

```python
scores = list()
 # 生成整数序列 / Generate integer sequence
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
  # 打印输出 / Print output
		print('>#%d: %.3f' % (r+1, score))
  # 添加元素到列表末尾 / Append element to list end
		scores.append(score)
```

---
## Step 27 — summarize results

```python
summarize_results(scores)
```

---
## Step 28 — run the experiment

```python
run_experiment()
```

---
## Learning Notes / 学习笔记

- **概念**: convlstm model for the har dataset 是机器学习中的常用技术。  
  *convlstm model for the har dataset is a common technique in machine learning.*

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
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Convlstm Model / LSTM 网络
# Complete Code / 完整代码
# ===============================

# convlstm model for the har dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import dstack
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dropout
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import ConvLSTM2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical

# load a single file as a numpy array
def load_file(filepath):
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
 # 转换为NumPy数组 / Convert to NumPy array
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
  # 添加元素到列表末尾 / Append element to list end
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	# define model
	verbose, epochs, batch_size = 0, 25, 64
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	n_features, n_outputs = trainX.shape[2], trainy.shape[1]
	# reshape into subsequences (samples, time steps, rows, cols, channels)
	n_steps, n_length = 4, 32
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(ConvLSTM2D(64, (1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dropout(0.5))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(100, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_outputs, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
 # 训练模型 / Train the model
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores):
 # 打印输出 / Print output
	print(scores)
	m, s = mean(scores), std(scores)
 # 打印输出 / Print output
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
 # 生成整数序列 / Generate integer sequence
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
  # 打印输出 / Print output
		print('>#%d: %.3f' % (r+1, score))
  # 添加元素到列表末尾 / Append element to list end
		scores.append(score)
	# summarize results
	summarize_results(scores)

# run the experiment
run_experiment()
```

---

### Chapter Summary / 章节总结

# Chapter 25 Summary / 第25章总结

## Theme / 主题: Chapter 25 / Chapter 25

This chapter contains **3 code files** demonstrating chapter 25.

本章包含 **3 个代码文件**，演示Chapter 25。

---
## Evolution / 演化路线

  1. `01_lstm_model.ipynb` — Lstm Model
  2. `02_cnn_lstm_model.ipynb` — Cnn Lstm Model
  3. `03_convlstm_model.ipynb` — Convlstm Model

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 25) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 25）是机器学习流水线中的基础构建块。

---
