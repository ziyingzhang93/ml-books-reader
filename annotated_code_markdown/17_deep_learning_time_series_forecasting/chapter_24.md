# DL时间序列
## Chapter 24

---

### 1D Cnn Model

# 01 — 1D Cnn Model / 卷积神经网络

**Chapter 24 — File 1 of 6 / 第24章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **cnn model for the har dataset**.

本脚本演示 **cnn model for the har dataset**。

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
## Step 1 — cnn model for the har dataset

```python
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

---
## Step 3 — load a list of files and return as a 3d numpy array

```python
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
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
	verbose, epochs, batch_size = 0, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(Conv1D(64, 3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(64, 3, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 18 — fit network

```python
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
```

---
## Step 19 — evaluate model

```python
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy
```

---
## Step 20 — summarize scores

```python
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
```

---
## Step 21 — run an experiment

```python
def run_experiment(repeats=10):
```

---
## Step 22 — load data

```python
trainX, trainy, testX, testy = load_dataset()
```

---
## Step 23 — repeat experiment

```python
scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
```

---
## Step 24 — summarize results

```python
summarize_results(scores)
```

---
## Step 25 — run the experiment

```python
run_experiment()
```

---
## Learning Notes / 学习笔记

- **概念**: cnn model for the har dataset 是机器学习中的常用技术。  
  *cnn model for the har dataset is a common technique in machine learning.*

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
# 1D Cnn Model / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# cnn model for the har dataset
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
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
	verbose, epochs, batch_size = 0, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(Conv1D(64, 3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(64, 3, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

# run the experiment
run_experiment()
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Histograms All Variables

# 02 — Histograms All Variables / 02 Histograms All Variables

**Chapter 24 — File 2 of 6 / 第24章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **plot distributions for the har dataset**.

本脚本演示 **plot distributions for the har dataset**。

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
## Step 1 — plot distributions for the har dataset

```python
from numpy import dstack
from pandas import read_csv
from keras.utils import to_categorical
from matplotlib import pyplot
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

---
## Step 3 — load a list of files and return as a 3d numpy array

```python
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
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
## Step 17 — plot a histogram of each variable in the dataset

```python
def plot_variable_distributions(trainX):
```

---
## Step 18 — remove overlap

```python
cut = int(trainX.shape[1] / 2)
	longX = trainX[:, -cut:, :]
```

---
## Step 19 — flatten windows

```python
longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
	pyplot.figure()
	for i in range(longX.shape[1]):
```

---
## Step 20 — create figure

```python
ax = pyplot.subplot(longX.shape[1], 1, i+1)
		ax.set_xlim(-1, 1)
```

---
## Step 21 — create histogram

```python
pyplot.hist(longX[:, i], bins=100)
```

---
## Step 22 — simplify axis remove clutter

```python
pyplot.yticks([])
		pyplot.xticks([-1,0,1])
	pyplot.show()
```

---
## Step 23 — load data

```python
trainX, trainy, testX, testy = load_dataset()
```

---
## Step 24 — plot histograms

```python
plot_variable_distributions(trainX)
```

---
## Learning Notes / 学习笔记

- **概念**: plot distributions for the har dataset 是机器学习中的常用技术。  
  *plot distributions for the har dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Histograms All Variables / 02 Histograms All Variables
# Complete Code / 完整代码
# ===============================

# plot distributions for the har dataset
from numpy import dstack
from pandas import read_csv
from keras.utils import to_categorical
from matplotlib import pyplot

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
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

# plot a histogram of each variable in the dataset
def plot_variable_distributions(trainX):
	# remove overlap
	cut = int(trainX.shape[1] / 2)
	longX = trainX[:, -cut:, :]
	# flatten windows
	longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
	pyplot.figure()
	for i in range(longX.shape[1]):
		# create figure
		ax = pyplot.subplot(longX.shape[1], 1, i+1)
		ax.set_xlim(-1, 1)
		# create histogram
		pyplot.hist(longX[:, i], bins=100)
		# simplify axis remove clutter
		pyplot.yticks([])
		pyplot.xticks([-1,0,1])
	pyplot.show()

# load data
trainX, trainy, testX, testy = load_dataset()
# plot histograms
plot_variable_distributions(trainX)
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Cnn Standardization

# 03 — Cnn Standardization / 卷积神经网络

**Chapter 24 — File 3 of 6 / 第24章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **cnn model with standardization for the har dataset**.

本脚本演示 **cnn model with standardization for the har dataset**。

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
## Step 1 — cnn model with standardization for the har dataset

```python
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

---
## Step 3 — load a list of files and return as a 3d numpy array

```python
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
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
## Step 17 — standardize data

```python
def scale_data(trainX, testX, standardize):
```

---
## Step 18 — remove overlap

```python
cut = int(trainX.shape[1] / 2)
	longX = trainX[:, -cut:, :]
```

---
## Step 19 — flatten windows

```python
longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
```

---
## Step 20 — flatten train and test

```python
flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
	flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
```

---
## Step 21 — standardize

```python
if standardize:
		s = StandardScaler()
```

---
## Step 22 — fit on training data

```python
s.fit(longX)
```

---
## Step 23 — apply to training and test data

```python
longX = s.transform(longX)
		flatTrainX = s.transform(flatTrainX)
		flatTestX = s.transform(flatTestX)
```

---
## Step 24 — reshape

```python
flatTrainX = flatTrainX.reshape((trainX.shape))
	flatTestX = flatTestX.reshape((testX.shape))
	return flatTrainX, flatTestX
```

---
## Step 25 — fit and evaluate a model

```python
def evaluate_model(trainX, trainy, testX, testy, param):
	verbose, epochs, batch_size = 0, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
```

---
## Step 26 — scale data

```python
trainX, testX = scale_data(trainX, testX, param)
	model = Sequential()
	model.add(Conv1D(64, 3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(64, 3, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 27 — fit network

```python
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
```

---
## Step 28 — evaluate model

```python
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy
```

---
## Step 29 — summarize scores

```python
def summarize_results(scores, params):
	print(scores, params)
```

---
## Step 30 — summarize mean and standard deviation

```python
for i in range(len(scores)):
		m, s = mean(scores[i]), std(scores[i])
		print('Param=%s: %.3f%% (+/-%.3f)' % (params[i], m, s))
```

---
## Step 31 — boxplot of scores

```python
pyplot.boxplot(scores, labels=params)
	pyplot.savefig('exp_cnn_standardize.png')
```

---
## Step 32 — run an experiment

```python
def run_experiment(params, repeats=10):
```

---
## Step 33 — load data

```python
trainX, trainy, testX, testy = load_dataset()
```

---
## Step 34 — test each parameter

```python
all_scores = list()
	for p in params:
```

---
## Step 35 — repeat experiment

```python
scores = list()
		for r in range(repeats):
			score = evaluate_model(trainX, trainy, testX, testy, p)
			score = score * 100.0
			print('>p=%s #%d: %.3f' % (p, r+1, score))
			scores.append(score)
		all_scores.append(scores)
```

---
## Step 36 — summarize results

```python
summarize_results(all_scores, params)
```

---
## Step 37 — run the experiment

```python
n_params = [False, True]
run_experiment(n_params)
```

---
## Learning Notes / 学习笔记

- **概念**: cnn model with standardization for the har dataset 是机器学习中的常用技术。  
  *cnn model with standardization for the har dataset is a common technique in machine learning.*

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
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
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
# Cnn Standardization / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# cnn model with standardization for the har dataset
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
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

# standardize data
def scale_data(trainX, testX, standardize):
	# remove overlap
	cut = int(trainX.shape[1] / 2)
	longX = trainX[:, -cut:, :]
	# flatten windows
	longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
	# flatten train and test
	flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
	flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
	# standardize
	if standardize:
		s = StandardScaler()
		# fit on training data
		s.fit(longX)
		# apply to training and test data
		longX = s.transform(longX)
		flatTrainX = s.transform(flatTrainX)
		flatTestX = s.transform(flatTestX)
	# reshape
	flatTrainX = flatTrainX.reshape((trainX.shape))
	flatTestX = flatTestX.reshape((testX.shape))
	return flatTrainX, flatTestX

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy, param):
	verbose, epochs, batch_size = 0, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# scale data
	trainX, testX = scale_data(trainX, testX, param)
	model = Sequential()
	model.add(Conv1D(64, 3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(64, 3, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores, params):
	print(scores, params)
	# summarize mean and standard deviation
	for i in range(len(scores)):
		m, s = mean(scores[i]), std(scores[i])
		print('Param=%s: %.3f%% (+/-%.3f)' % (params[i], m, s))
	# boxplot of scores
	pyplot.boxplot(scores, labels=params)
	pyplot.savefig('exp_cnn_standardize.png')

# run an experiment
def run_experiment(params, repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# test each parameter
	all_scores = list()
	for p in params:
		# repeat experiment
		scores = list()
		for r in range(repeats):
			score = evaluate_model(trainX, trainy, testX, testy, p)
			score = score * 100.0
			print('>p=%s #%d: %.3f' % (p, r+1, score))
			scores.append(score)
		all_scores.append(scores)
	# summarize results
	summarize_results(all_scores, params)

# run the experiment
n_params = [False, True]
run_experiment(n_params)
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Cnn Tune Filter Maps

# 04 — Cnn Tune Filter Maps / 卷积神经网络

**Chapter 24 — File 4 of 6 / 第24章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **cnn model with filters for the har dataset**.

本脚本演示 **cnn model with filters for the har dataset**。

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
## Step 1 — cnn model with filters for the har dataset

```python
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

---
## Step 3 — load a list of files and return as a 3d numpy array

```python
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
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
def evaluate_model(trainX, trainy, testX, testy, n_filters):
	verbose, epochs, batch_size = 0, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(Conv1D(n_filters, 3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(n_filters, 3, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 18 — fit network

```python
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
```

---
## Step 19 — evaluate model

```python
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy
```

---
## Step 20 — summarize scores

```python
def summarize_results(scores, params):
	print(scores, params)
```

---
## Step 21 — summarize mean and standard deviation

```python
for i in range(len(scores)):
		m, s = mean(scores[i]), std(scores[i])
		print('Param=%d: %.3f%% (+/-%.3f)' % (params[i], m, s))
```

---
## Step 22 — boxplot of scores

```python
pyplot.boxplot(scores, labels=params)
	pyplot.savefig('exp_cnn_filters.png')
```

---
## Step 23 — run an experiment

```python
def run_experiment(params, repeats=10):
```

---
## Step 24 — load data

```python
trainX, trainy, testX, testy = load_dataset()
```

---
## Step 25 — test each parameter

```python
all_scores = list()
	for p in params:
```

---
## Step 26 — repeat experiment

```python
scores = list()
		for r in range(repeats):
			score = evaluate_model(trainX, trainy, testX, testy, p)
			score = score * 100.0
			print('>p=%d #%d: %.3f' % (p, r+1, score))
			scores.append(score)
		all_scores.append(scores)
```

---
## Step 27 — summarize results

```python
summarize_results(all_scores, params)
```

---
## Step 28 — run the experiment

```python
n_params = [8, 16, 32, 64, 128, 256]
run_experiment(n_params)
```

---
## Learning Notes / 学习笔记

- **概念**: cnn model with filters for the har dataset 是机器学习中的常用技术。  
  *cnn model with filters for the har dataset is a common technique in machine learning.*

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
| `matplotlib` | 绑图库 | Plotting library |
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
# Cnn Tune Filter Maps / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# cnn model with filters for the har dataset
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
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
def evaluate_model(trainX, trainy, testX, testy, n_filters):
	verbose, epochs, batch_size = 0, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(Conv1D(n_filters, 3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(n_filters, 3, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores, params):
	print(scores, params)
	# summarize mean and standard deviation
	for i in range(len(scores)):
		m, s = mean(scores[i]), std(scores[i])
		print('Param=%d: %.3f%% (+/-%.3f)' % (params[i], m, s))
	# boxplot of scores
	pyplot.boxplot(scores, labels=params)
	pyplot.savefig('exp_cnn_filters.png')

# run an experiment
def run_experiment(params, repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# test each parameter
	all_scores = list()
	for p in params:
		# repeat experiment
		scores = list()
		for r in range(repeats):
			score = evaluate_model(trainX, trainy, testX, testy, p)
			score = score * 100.0
			print('>p=%d #%d: %.3f' % (p, r+1, score))
			scores.append(score)
		all_scores.append(scores)
	# summarize results
	summarize_results(all_scores, params)

# run the experiment
n_params = [8, 16, 32, 64, 128, 256]
run_experiment(n_params)
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Cnn Tune Kernel Size

# 05 — Cnn Tune Kernel Size / 卷积神经网络

**Chapter 24 — File 5 of 6 / 第24章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **cnn model vary kernel size for the har dataset**.

本脚本演示 **cnn model vary kernel size for the har dataset**。

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
## Step 1 — cnn model vary kernel size for the har dataset

```python
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

---
## Step 3 — load a list of files and return as a 3d numpy array

```python
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
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
def evaluate_model(trainX, trainy, testX, testy, n_kernel):
	verbose, epochs, batch_size = 0, 15, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(Conv1D(64, n_kernel, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(64, n_kernel, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 18 — fit network

```python
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
```

---
## Step 19 — evaluate model

```python
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy
```

---
## Step 20 — summarize scores

```python
def summarize_results(scores, params):
	print(scores, params)
```

---
## Step 21 — summarize mean and standard deviation

```python
for i in range(len(scores)):
		m, s = mean(scores[i]), std(scores[i])
		print('Param=%d: %.3f%% (+/-%.3f)' % (params[i], m, s))
```

---
## Step 22 — boxplot of scores

```python
pyplot.boxplot(scores, labels=params)
	pyplot.savefig('exp_cnn_kernel.png')
```

---
## Step 23 — run an experiment

```python
def run_experiment(params, repeats=10):
```

---
## Step 24 — load data

```python
trainX, trainy, testX, testy = load_dataset()
```

---
## Step 25 — test each parameter

```python
all_scores = list()
	for p in params:
```

---
## Step 26 — repeat experiment

```python
scores = list()
		for r in range(repeats):
			score = evaluate_model(trainX, trainy, testX, testy, p)
			score = score * 100.0
			print('>p=%d #%d: %.3f' % (p, r+1, score))
			scores.append(score)
		all_scores.append(scores)
```

---
## Step 27 — summarize results

```python
summarize_results(all_scores, params)
```

---
## Step 28 — run the experiment

```python
n_params = [2, 3, 5, 7, 11]
run_experiment(n_params)
```

---
## Learning Notes / 学习笔记

- **概念**: cnn model vary kernel size for the har dataset 是机器学习中的常用技术。  
  *cnn model vary kernel size for the har dataset is a common technique in machine learning.*

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
| `matplotlib` | 绑图库 | Plotting library |
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
# Cnn Tune Kernel Size / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# cnn model vary kernel size for the har dataset
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
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
def evaluate_model(trainX, trainy, testX, testy, n_kernel):
	verbose, epochs, batch_size = 0, 15, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(Conv1D(64, n_kernel, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(64, n_kernel, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores, params):
	print(scores, params)
	# summarize mean and standard deviation
	for i in range(len(scores)):
		m, s = mean(scores[i]), std(scores[i])
		print('Param=%d: %.3f%% (+/-%.3f)' % (params[i], m, s))
	# boxplot of scores
	pyplot.boxplot(scores, labels=params)
	pyplot.savefig('exp_cnn_kernel.png')

# run an experiment
def run_experiment(params, repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# test each parameter
	all_scores = list()
	for p in params:
		# repeat experiment
		scores = list()
		for r in range(repeats):
			score = evaluate_model(trainX, trainy, testX, testy, p)
			score = score * 100.0
			print('>p=%d #%d: %.3f' % (p, r+1, score))
			scores.append(score)
		all_scores.append(scores)
	# summarize results
	summarize_results(all_scores, params)

# run the experiment
n_params = [2, 3, 5, 7, 11]
run_experiment(n_params)
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Multiheaded Cnn

# 06 — Multiheaded Cnn / 卷积神经网络

**Chapter 24 — File 6 of 6 / 第24章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **multi-headed cnn model for the har dataset**.

本脚本演示 **multi-headed cnn model for the har dataset**。

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
## Step 1 — multi-headed cnn model for the har dataset

```python
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

---
## Step 3 — load a list of files and return as a 3d numpy array

```python
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
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
	verbose, epochs, batch_size = 0, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
```

---
## Step 18 — head 1

```python
inputs1 = Input(shape=(n_timesteps,n_features))
	conv1 = Conv1D(64, 3, activation='relu')(inputs1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D()(drop1)
	flat1 = Flatten()(pool1)
```

---
## Step 19 — head 2

```python
inputs2 = Input(shape=(n_timesteps,n_features))
	conv2 = Conv1D(64, 5, activation='relu')(inputs2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D()(drop2)
	flat2 = Flatten()(pool2)
```

---
## Step 20 — head 3

```python
inputs3 = Input(shape=(n_timesteps,n_features))
	conv3 = Conv1D(64, 11, activation='relu')(inputs3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D()(drop3)
	flat3 = Flatten()(pool3)
```

---
## Step 21 — merge

```python
merged = concatenate([flat1, flat2, flat3])
```

---
## Step 22 — interpretation

```python
dense1 = Dense(100, activation='relu')(merged)
	outputs = Dense(n_outputs, activation='softmax')(dense1)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
```

---
## Step 23 — save a plot of the model

```python
plot_model(model, show_shapes=True, to_file='multiheaded.png')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 24 — fit network

```python
model.fit([trainX,trainX,trainX], trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
```

---
## Step 25 — evaluate model

```python
_, accuracy = model.evaluate([testX,testX,testX], testy, batch_size=batch_size, verbose=0)
	return accuracy
```

---
## Step 26 — summarize scores

```python
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
```

---
## Step 27 — run an experiment

```python
def run_experiment(repeats=10):
```

---
## Step 28 — load data

```python
trainX, trainy, testX, testy = load_dataset()
```

---
## Step 29 — repeat experiment

```python
scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
```

---
## Step 30 — summarize results

```python
summarize_results(scores)
```

---
## Step 31 — run the experiment

```python
run_experiment()
```

---
## Learning Notes / 学习笔记

- **概念**: multi-headed cnn model for the har dataset 是机器学习中的常用技术。  
  *multi-headed cnn model for the har dataset is a common technique in machine learning.*

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
# Multiheaded Cnn / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# multi-headed cnn model for the har dataset
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
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
	verbose, epochs, batch_size = 0, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
 	# head 1
	inputs1 = Input(shape=(n_timesteps,n_features))
	conv1 = Conv1D(64, 3, activation='relu')(inputs1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D()(drop1)
	flat1 = Flatten()(pool1)
	# head 2
	inputs2 = Input(shape=(n_timesteps,n_features))
	conv2 = Conv1D(64, 5, activation='relu')(inputs2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D()(drop2)
	flat2 = Flatten()(pool2)
	# head 3
	inputs3 = Input(shape=(n_timesteps,n_features))
	conv3 = Conv1D(64, 11, activation='relu')(inputs3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D()(drop3)
	flat3 = Flatten()(pool3)
	# merge
	merged = concatenate([flat1, flat2, flat3])
	# interpretation
	dense1 = Dense(100, activation='relu')(merged)
	outputs = Dense(n_outputs, activation='softmax')(dense1)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
	# save a plot of the model
	plot_model(model, show_shapes=True, to_file='multiheaded.png')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit([trainX,trainX,trainX], trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate([testX,testX,testX], testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

# run the experiment
run_experiment()
```

---

### Chapter Summary

# Chapter 24 Summary / 第24章总结

## Theme / 主题: Chapter 24 / Chapter 24

This chapter contains **6 code files** demonstrating chapter 24.

本章包含 **6 个代码文件**，演示Chapter 24。

---
## Evolution / 演化路线

  1. `01_1d_cnn_model.ipynb` — 1D Cnn Model
  2. `02_histograms_all_variables.ipynb` — Histograms All Variables
  3. `03_cnn_standardization.ipynb` — Cnn Standardization
  4. `04_cnn_tune_filter_maps.ipynb` — Cnn Tune Filter Maps
  5. `05_cnn_tune_kernel_size.ipynb` — Cnn Tune Kernel Size
  6. `06_multiheaded_cnn.ipynb` — Multiheaded Cnn

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 24) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 24）是机器学习流水线中的基础构建块。

---
