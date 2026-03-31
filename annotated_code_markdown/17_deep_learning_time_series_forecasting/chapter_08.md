# 深度学习时间序列预测 / DL Time Series Forecasting
## Chapter 08

---

### Univariate Dataset

# 01 — Univariate Dataset / 单变量

**Chapter 08 — File 1 of 15 / 第08章 — 第1个文件（共15个）**

---

## Summary / 总结

This script demonstrates **univariate data preparation**.

本脚本演示 **univariate data preparation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — univariate data preparation

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
```

---
## Step 2 — split a univariate sequence into samples

```python
def split_sequence(sequence, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequence)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps
```

---
## Step 4 — check if we are beyond the sequence

```python
# 获取长度 / Get length
if end_ix > len(sequence)-1:
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define input sequence

```python
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
```

---
## Step 7 — choose a number of time steps

```python
n_steps = 3
```

---
## Step 8 — split into samples

```python
X, y = split_sequence(raw_seq, n_steps)
```

---
## Step 9 — summarize the data

```python
# 获取长度 / Get length
for i in range(len(X)):
 # 打印输出 / Print output
	print(X[i], y[i])
```

---
## Learning Notes / 学习笔记

- **概念**: univariate data preparation 是机器学习中的常用技术。  
  *univariate data preparation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Univariate Dataset / 单变量
# Complete Code / 完整代码
# ===============================

# univariate data preparation
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
  # 获取长度 / Get length
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# summarize the data
# 获取长度 / Get length
for i in range(len(X)):
 # 打印输出 / Print output
	print(X[i], y[i])
```

---

➡️ **Next / 下一步**: File 2 of 15

---

### Cnn Univariate

# 02 — Cnn Univariate / 卷积神经网络

**Chapter 08 — File 2 of 15 / 第08章 — 第2个文件（共15个）**

---

## Summary / 总结

This script demonstrates **univariate cnn example**.

本脚本演示 **univariate cnn example**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — univariate cnn example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D
```

---
## Step 2 — split a univariate sequence into samples

```python
def split_sequence(sequence, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequence)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps
```

---
## Step 4 — check if we are beyond the sequence

```python
# 获取长度 / Get length
if end_ix > len(sequence)-1:
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define input sequence

```python
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
```

---
## Step 7 — choose a number of time steps

```python
n_steps = 3
```

---
## Step 8 — split into samples

```python
X, y = split_sequence(raw_seq, n_steps)
```

---
## Step 9 — reshape from [samples, timesteps] into [samples, timesteps, features]

```python
n_features = 1
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], X.shape[1], n_features))
```

---
## Step 10 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling1D())
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 11 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=1000, verbose=0)
```

---
## Step 12 — demonstrate prediction

```python
x_input = array([70, 80, 90])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_steps, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: univariate cnn example 是机器学习中的常用技术。  
  *univariate cnn example is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cnn Univariate / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# univariate cnn example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
  # 获取长度 / Get length
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling1D())
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=1000, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_steps, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 3 of 15

---

### Dependent Time Series Dataset

# 03 — Dependent Time Series Dataset / 03 Dependent Time Series Dataset

**Chapter 08 — File 3 of 15 / 第08章 — 第3个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate data preparation**.

本脚本演示 **multivariate data preparation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — multivariate data preparation

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
```

---
## Step 2 — define input sequence

```python
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
```

---
## Step 3 — convert to [rows, columns] structure

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
```

---
## Step 4 — horizontally stack columns

```python
dataset = hstack((in_seq1, in_seq2, out_seq))
# 打印输出 / Print output
print(dataset)
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate data preparation 是机器学习中的常用技术。  
  *multivariate data preparation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dependent Time Series Dataset / 03 Dependent Time Series Dataset
# Complete Code / 完整代码
# ===============================

# multivariate data preparation
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# 打印输出 / Print output
print(dataset)
```

---

➡️ **Next / 下一步**: File 4 of 15

---

### Split Samples Dependent Time Series

# 04 — Split Samples Dependent Time Series / 04 Split Samples Dependent Time Series

**Chapter 08 — File 4 of 15 / 第08章 — 第4个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate data preparation**.

本脚本演示 **multivariate data preparation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — multivariate data preparation

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
```

---
## Step 2 — split a multivariate sequence into samples

```python
def split_sequences(sequences, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps
```

---
## Step 4 — check if we are beyond the dataset

```python
# 获取长度 / Get length
if end_ix > len(sequences):
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define input sequence

```python
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
```

---
## Step 7 — convert to [rows, columns] structure

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
```

---
## Step 8 — horizontally stack columns

```python
dataset = hstack((in_seq1, in_seq2, out_seq))
```

---
## Step 9 — choose a number of time steps

```python
n_steps = 3
```

---
## Step 10 — convert into input/output

```python
X, y = split_sequences(dataset, n_steps)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---
## Step 11 — summarize the data

```python
# 获取长度 / Get length
for i in range(len(X)):
 # 打印输出 / Print output
	print(X[i], y[i])
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate data preparation 是机器学习中的常用技术。  
  *multivariate data preparation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Split Samples Dependent Time Series / 04 Split Samples Dependent Time Series
# Complete Code / 完整代码
# ===============================

# multivariate data preparation
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
  # 获取长度 / Get length
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
# summarize the data
# 获取长度 / Get length
for i in range(len(X)):
 # 打印输出 / Print output
	print(X[i], y[i])
```

---

➡️ **Next / 下一步**: File 5 of 15

---

### Cnn Multivariate Dependent Series

# 05 — Cnn Multivariate Dependent Series / 卷积神经网络

**Chapter 08 — File 5 of 15 / 第08章 — 第5个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate cnn example**.

本脚本演示 **multivariate cnn example**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — multivariate cnn example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D
```

---
## Step 2 — split a multivariate sequence into samples

```python
def split_sequences(sequences, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps
```

---
## Step 4 — check if we are beyond the dataset

```python
# 获取长度 / Get length
if end_ix > len(sequences):
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define input sequence

```python
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
```

---
## Step 7 — convert to [rows, columns] structure

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
```

---
## Step 8 — horizontally stack columns

```python
dataset = hstack((in_seq1, in_seq2, out_seq))
```

---
## Step 9 — choose a number of time steps

```python
n_steps = 3
```

---
## Step 10 — convert into input/output

```python
X, y = split_sequences(dataset, n_steps)
```

---
## Step 11 — the dataset knows the number of features, e.g. 2

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_features = X.shape[2]
```

---
## Step 12 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling1D())
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 13 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=1000, verbose=0)
```

---
## Step 14 — demonstrate prediction

```python
x_input = array([[80, 85], [90, 95], [100, 105]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_steps, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate cnn example 是机器学习中的常用技术。  
  *multivariate cnn example is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cnn Multivariate Dependent Series / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# multivariate cnn example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
  # 获取长度 / Get length
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# the dataset knows the number of features, e.g. 2
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_features = X.shape[2]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling1D())
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=1000, verbose=0)
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_steps, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 6 of 15

---

### Multiheaded Cnn Multivariate Dependent Series

# 06 — Multiheaded Cnn Multivariate Dependent Series / 卷积神经网络

**Chapter 08 — File 6 of 15 / 第08章 — 第6个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate multi-headed 1d cnn example**.

本脚本演示 **multivariate multi-headed 1d cnn example**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — multivariate multi-headed 1d cnn example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.merge import concatenate
```

---
## Step 2 — split a multivariate sequence into samples

```python
def split_sequences(sequences, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps
```

---
## Step 4 — check if we are beyond the dataset

```python
# 获取长度 / Get length
if end_ix > len(sequences):
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define input sequence

```python
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
```

---
## Step 7 — convert to [rows, columns] structure

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
```

---
## Step 8 — horizontally stack columns

```python
dataset = hstack((in_seq1, in_seq2, out_seq))
```

---
## Step 9 — choose a number of time steps

```python
n_steps = 3
```

---
## Step 10 — convert into input/output

```python
X, y = split_sequences(dataset, n_steps)
```

---
## Step 11 — one time series per head

```python
n_features = 1
```

---
## Step 12 — separate input data

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X1 = X[:, :, 0].reshape(X.shape[0], X.shape[1], n_features)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X2 = X[:, :, 1].reshape(X.shape[0], X.shape[1], n_features)
```

---
## Step 13 — first input model

```python
visible1 = Input(shape=(n_steps, n_features))
cnn1 = Conv1D(64, 2, activation='relu')(visible1)
cnn1 = MaxPooling1D()(cnn1)
# 展平层：多维→一维 / Flatten: multi-dim → 1D
cnn1 = Flatten()(cnn1)
```

---
## Step 14 — second input model

```python
visible2 = Input(shape=(n_steps, n_features))
cnn2 = Conv1D(64, 2, activation='relu')(visible2)
cnn2 = MaxPooling1D()(cnn2)
# 展平层：多维→一维 / Flatten: multi-dim → 1D
cnn2 = Flatten()(cnn2)
```

---
## Step 15 — merge input models

```python
merge = concatenate([cnn1, cnn2])
# 全连接层（Keras） / Fully connected layer (Keras)
dense = Dense(50, activation='relu')(merge)
# 全连接层（Keras） / Fully connected layer (Keras)
output = Dense(1)(dense)
model = Model(inputs=[visible1, visible2], outputs=output)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 16 — fit model

```python
# 训练模型 / Train the model
model.fit([X1, X2], y, epochs=1000, verbose=0)
```

---
## Step 17 — demonstrate prediction

```python
x_input = array([[80, 85], [90, 95], [100, 105]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x1 = x_input[:, 0].reshape((1, n_steps, n_features))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x2 = x_input[:, 1].reshape((1, n_steps, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict([x1, x2], verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate multi-headed 1d cnn example 是机器学习中的常用技术。  
  *multivariate multi-headed 1d cnn example is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Multiheaded Cnn Multivariate Dependent Series / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# multivariate multi-headed 1d cnn example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.merge import concatenate

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
  # 获取长度 / Get length
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# one time series per head
n_features = 1
# separate input data
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X1 = X[:, :, 0].reshape(X.shape[0], X.shape[1], n_features)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X2 = X[:, :, 1].reshape(X.shape[0], X.shape[1], n_features)
# first input model
visible1 = Input(shape=(n_steps, n_features))
cnn1 = Conv1D(64, 2, activation='relu')(visible1)
cnn1 = MaxPooling1D()(cnn1)
# 展平层：多维→一维 / Flatten: multi-dim → 1D
cnn1 = Flatten()(cnn1)
# second input model
visible2 = Input(shape=(n_steps, n_features))
cnn2 = Conv1D(64, 2, activation='relu')(visible2)
cnn2 = MaxPooling1D()(cnn2)
# 展平层：多维→一维 / Flatten: multi-dim → 1D
cnn2 = Flatten()(cnn2)
# merge input models
merge = concatenate([cnn1, cnn2])
# 全连接层（Keras） / Fully connected layer (Keras)
dense = Dense(50, activation='relu')(merge)
# 全连接层（Keras） / Fully connected layer (Keras)
output = Dense(1)(dense)
model = Model(inputs=[visible1, visible2], outputs=output)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit([X1, X2], y, epochs=1000, verbose=0)
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x1 = x_input[:, 0].reshape((1, n_steps, n_features))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x2 = x_input[:, 1].reshape((1, n_steps, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict([x1, x2], verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 7 of 15

---

### Split Parallel Series

# 07 — Split Parallel Series / 07 Split Parallel Series

**Chapter 08 — File 7 of 15 / 第08章 — 第7个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate output data prep**.

本脚本演示 **multivariate output data prep**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — multivariate output data prep

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
```

---
## Step 2 — split a multivariate sequence into samples

```python
def split_sequences(sequences, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps
```

---
## Step 4 — check if we are beyond the dataset

```python
# 获取长度 / Get length
if end_ix > len(sequences)-1:
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define input sequence

```python
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
```

---
## Step 7 — convert to [rows, columns] structure

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
```

---
## Step 8 — horizontally stack columns

```python
dataset = hstack((in_seq1, in_seq2, out_seq))
```

---
## Step 9 — choose a number of time steps

```python
n_steps = 3
```

---
## Step 10 — convert into input/output

```python
X, y = split_sequences(dataset, n_steps)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---
## Step 11 — summarize the data

```python
# 获取长度 / Get length
for i in range(len(X)):
 # 打印输出 / Print output
	print(X[i], y[i])
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate output data prep 是机器学习中的常用技术。  
  *multivariate output data prep is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Split Parallel Series / 07 Split Parallel Series
# Complete Code / 完整代码
# ===============================

# multivariate output data prep
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
  # 获取长度 / Get length
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
# summarize the data
# 获取长度 / Get length
for i in range(len(X)):
 # 打印输出 / Print output
	print(X[i], y[i])
```

---

➡️ **Next / 下一步**: File 8 of 15

---

### Cnn Multivariate Parallel Series

# 08 — Cnn Multivariate Parallel Series / 卷积神经网络

**Chapter 08 — File 8 of 15 / 第08章 — 第8个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate output 1d cnn example**.

本脚本演示 **multivariate output 1d cnn example**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — multivariate output 1d cnn example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D
```

---
## Step 2 — split a multivariate sequence into samples

```python
def split_sequences(sequences, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps
```

---
## Step 4 — check if we are beyond the dataset

```python
# 获取长度 / Get length
if end_ix > len(sequences)-1:
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define input sequence

```python
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
```

---
## Step 7 — convert to [rows, columns] structure

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
```

---
## Step 8 — horizontally stack columns

```python
dataset = hstack((in_seq1, in_seq2, out_seq))
```

---
## Step 9 — choose a number of time steps

```python
n_steps = 3
```

---
## Step 10 — convert into input/output

```python
X, y = split_sequences(dataset, n_steps)
```

---
## Step 11 — the dataset knows the number of features, e.g. 2

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_features = X.shape[2]
```

---
## Step 12 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling1D())
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_features))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 13 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=3000, verbose=0)
```

---
## Step 14 — demonstrate prediction

```python
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_steps, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate output 1d cnn example 是机器学习中的常用技术。  
  *multivariate output 1d cnn example is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cnn Multivariate Parallel Series / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# multivariate output 1d cnn example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
  # 获取长度 / Get length
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# the dataset knows the number of features, e.g. 2
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_features = X.shape[2]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling1D())
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_features))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=3000, verbose=0)
# demonstrate prediction
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_steps, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 9 of 15

---

### Multi Output Cnn Multivariate Parallel Series

# 09 — Multi Output Cnn Multivariate Parallel Series / 卷积神经网络

**Chapter 08 — File 9 of 15 / 第08章 — 第9个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate output 1d cnn example**.

本脚本演示 **multivariate output 1d cnn example**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — multivariate output 1d cnn example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D
```

---
## Step 2 — split a multivariate sequence into samples

```python
def split_sequences(sequences, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps
```

---
## Step 4 — check if we are beyond the dataset

```python
# 获取长度 / Get length
if end_ix > len(sequences)-1:
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define input sequence

```python
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
```

---
## Step 7 — convert to [rows, columns] structure

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
```

---
## Step 8 — horizontally stack columns

```python
dataset = hstack((in_seq1, in_seq2, out_seq))
```

---
## Step 9 — choose a number of time steps

```python
n_steps = 3
```

---
## Step 10 — convert into input/output

```python
X, y = split_sequences(dataset, n_steps)
```

---
## Step 11 — the dataset knows the number of features, e.g. 2

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_features = X.shape[2]
```

---
## Step 12 — separate output

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y1 = y[:, 0].reshape((y.shape[0], 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y2 = y[:, 1].reshape((y.shape[0], 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y3 = y[:, 2].reshape((y.shape[0], 1))
```

---
## Step 13 — define model

```python
visible = Input(shape=(n_steps, n_features))
cnn = Conv1D(64, 2, activation='relu')(visible)
cnn = MaxPooling1D()(cnn)
# 展平层：多维→一维 / Flatten: multi-dim → 1D
cnn = Flatten()(cnn)
# 全连接层（Keras） / Fully connected layer (Keras)
cnn = Dense(50, activation='relu')(cnn)
```

---
## Step 14 — define output 1

```python
# 全连接层（Keras） / Fully connected layer (Keras)
output1 = Dense(1)(cnn)
```

---
## Step 15 — define output 2

```python
# 全连接层（Keras） / Fully connected layer (Keras)
output2 = Dense(1)(cnn)
```

---
## Step 16 — define output 3

```python
# 全连接层（Keras） / Fully connected layer (Keras)
output3 = Dense(1)(cnn)
```

---
## Step 17 — tie together

```python
model = Model(inputs=visible, outputs=[output1, output2, output3])
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 18 — fit model

```python
# 训练模型 / Train the model
model.fit(X, [y1,y2,y3], epochs=2000, verbose=0)
```

---
## Step 19 — demonstrate prediction

```python
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_steps, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate output 1d cnn example 是机器学习中的常用技术。  
  *multivariate output 1d cnn example is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Multi Output Cnn Multivariate Parallel Series / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# multivariate output 1d cnn example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
  # 获取长度 / Get length
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# the dataset knows the number of features, e.g. 2
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_features = X.shape[2]
# separate output
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y1 = y[:, 0].reshape((y.shape[0], 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y2 = y[:, 1].reshape((y.shape[0], 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y3 = y[:, 2].reshape((y.shape[0], 1))
# define model
visible = Input(shape=(n_steps, n_features))
cnn = Conv1D(64, 2, activation='relu')(visible)
cnn = MaxPooling1D()(cnn)
# 展平层：多维→一维 / Flatten: multi-dim → 1D
cnn = Flatten()(cnn)
# 全连接层（Keras） / Fully connected layer (Keras)
cnn = Dense(50, activation='relu')(cnn)
# define output 1
# 全连接层（Keras） / Fully connected layer (Keras)
output1 = Dense(1)(cnn)
# define output 2
# 全连接层（Keras） / Fully connected layer (Keras)
output2 = Dense(1)(cnn)
# define output 3
# 全连接层（Keras） / Fully connected layer (Keras)
output3 = Dense(1)(cnn)
# tie together
model = Model(inputs=visible, outputs=[output1, output2, output3])
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, [y1,y2,y3], epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_steps, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 10 of 15

---

### Multi Step Dataset

# 10 — Multi Step Dataset / 10 Multi Step Dataset

**Chapter 08 — File 10 of 15 / 第08章 — 第10个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multi-step data preparation**.

本脚本演示 **multi-step data preparation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — multi-step data preparation

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
```

---
## Step 2 — split a univariate sequence into samples

```python
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequence)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
```

---
## Step 4 — check if we are beyond the sequence

```python
# 获取长度 / Get length
if out_end_ix > len(sequence):
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define input sequence

```python
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
```

---
## Step 7 — choose a number of time steps

```python
n_steps_in, n_steps_out = 3, 2
```

---
## Step 8 — split into samples

```python
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
```

---
## Step 9 — summarize the data

```python
# 获取长度 / Get length
for i in range(len(X)):
 # 打印输出 / Print output
	print(X[i], y[i])
```

---
## Learning Notes / 学习笔记

- **概念**: multi-step data preparation 是机器学习中的常用技术。  
  *multi-step data preparation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Multi Step Dataset / 10 Multi Step Dataset
# Complete Code / 完整代码
# ===============================

# multi-step data preparation
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
  # 获取长度 / Get length
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# summarize the data
# 获取长度 / Get length
for i in range(len(X)):
 # 打印输出 / Print output
	print(X[i], y[i])
```

---

➡️ **Next / 下一步**: File 11 of 15

---

### Vector Cnn Multi Step

# 11 — Vector Cnn Multi Step / 卷积神经网络

**Chapter 08 — File 11 of 15 / 第08章 — 第11个文件（共15个）**

---

## Summary / 总结

This script demonstrates **univariate multi-step vector-output 1d cnn example**.

本脚本演示 **univariate multi-step vector-output 1d cnn example**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — univariate multi-step vector-output 1d cnn example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D
```

---
## Step 2 — split a univariate sequence into samples

```python
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequence)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
```

---
## Step 4 — check if we are beyond the sequence

```python
# 获取长度 / Get length
if out_end_ix > len(sequence):
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define input sequence

```python
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
```

---
## Step 7 — choose a number of time steps

```python
n_steps_in, n_steps_out = 3, 2
```

---
## Step 8 — split into samples

```python
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
```

---
## Step 9 — reshape from [samples, timesteps] into [samples, timesteps, features]

```python
n_features = 1
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], X.shape[1], n_features))
```

---
## Step 10 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps_in, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling1D())
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_steps_out))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 11 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=2000, verbose=0)
```

---
## Step 12 — demonstrate prediction

```python
x_input = array([70, 80, 90])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_steps_in, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: univariate multi-step vector-output 1d cnn example 是机器学习中的常用技术。  
  *univariate multi-step vector-output 1d cnn example is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Vector Cnn Multi Step / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# univariate multi-step vector-output 1d cnn example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
  # 获取长度 / Get length
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps_in, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling1D())
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_steps_out))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_steps_in, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 12 of 15

---

### Multivariate Multistep Dependent Dataset

# 12 — Multivariate Multistep Dependent Dataset / 多变量

**Chapter 08 — File 12 of 15 / 第08章 — 第12个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate multi-step data preparation**.

本脚本演示 **multivariate multi-step data preparation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — multivariate multi-step data preparation

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
```

---
## Step 2 — split a multivariate sequence into samples

```python
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
```

---
## Step 4 — check if we are beyond the dataset

```python
# 获取长度 / Get length
if out_end_ix > len(sequences):
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define input sequence

```python
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
```

---
## Step 7 — convert to [rows, columns] structure

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
```

---
## Step 8 — horizontally stack columns

```python
dataset = hstack((in_seq1, in_seq2, out_seq))
```

---
## Step 9 — choose a number of time steps

```python
n_steps_in, n_steps_out = 3, 2
```

---
## Step 10 — convert into input/output

```python
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---
## Step 11 — summarize the data

```python
# 获取长度 / Get length
for i in range(len(X)):
 # 打印输出 / Print output
	print(X[i], y[i])
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate multi-step data preparation 是机器学习中的常用技术。  
  *multivariate multi-step data preparation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Multivariate Multistep Dependent Dataset / 多变量
# Complete Code / 完整代码
# ===============================

# multivariate multi-step data preparation
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
  # 获取长度 / Get length
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# convert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
# summarize the data
# 获取长度 / Get length
for i in range(len(X)):
 # 打印输出 / Print output
	print(X[i], y[i])
```

---

➡️ **Next / 下一步**: File 13 of 15

---

### Cnn Multivariate Dependent Multistep

# 13 — Cnn Multivariate Dependent Multistep / 卷积神经网络

**Chapter 08 — File 13 of 15 / 第08章 — 第13个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate multi-step 1d cnn example**.

本脚本演示 **multivariate multi-step 1d cnn example**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — multivariate multi-step 1d cnn example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D
```

---
## Step 2 — split a multivariate sequence into samples

```python
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
```

---
## Step 4 — check if we are beyond the dataset

```python
# 获取长度 / Get length
if out_end_ix > len(sequences):
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define input sequence

```python
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
```

---
## Step 7 — convert to [rows, columns] structure

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
```

---
## Step 8 — horizontally stack columns

```python
dataset = hstack((in_seq1, in_seq2, out_seq))
```

---
## Step 9 — choose a number of time steps

```python
n_steps_in, n_steps_out = 3, 2
```

---
## Step 10 — convert into input/output

```python
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
```

---
## Step 11 — the dataset knows the number of features, e.g. 2

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_features = X.shape[2]
```

---
## Step 12 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps_in, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling1D())
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_steps_out))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 13 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=2000, verbose=0)
```

---
## Step 14 — demonstrate prediction

```python
x_input = array([[70, 75], [80, 85], [90, 95]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_steps_in, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate multi-step 1d cnn example 是机器学习中的常用技术。  
  *multivariate multi-step 1d cnn example is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cnn Multivariate Dependent Multistep / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# multivariate multi-step 1d cnn example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
  # 获取长度 / Get length
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# convert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# the dataset knows the number of features, e.g. 2
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_features = X.shape[2]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps_in, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling1D())
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_steps_out))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([[70, 75], [80, 85], [90, 95]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_steps_in, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 14 of 15

---

### Multivariate Multistep Parallel Dataset

# 14 — Multivariate Multistep Parallel Dataset / 多变量

**Chapter 08 — File 14 of 15 / 第08章 — 第14个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate multi-step data preparation**.

本脚本演示 **multivariate multi-step data preparation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — multivariate multi-step data preparation

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
```

---
## Step 2 — split a multivariate sequence into samples

```python
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
```

---
## Step 4 — check if we are beyond the dataset

```python
# 获取长度 / Get length
if out_end_ix > len(sequences):
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define input sequence

```python
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
```

---
## Step 7 — convert to [rows, columns] structure

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
```

---
## Step 8 — horizontally stack columns

```python
dataset = hstack((in_seq1, in_seq2, out_seq))
```

---
## Step 9 — choose a number of time steps

```python
n_steps_in, n_steps_out = 3, 2
```

---
## Step 10 — convert into input/output

```python
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---
## Step 11 — summarize the data

```python
# 获取长度 / Get length
for i in range(len(X)):
 # 打印输出 / Print output
	print(X[i], y[i])
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate multi-step data preparation 是机器学习中的常用技术。  
  *multivariate multi-step data preparation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Multivariate Multistep Parallel Dataset / 多变量
# Complete Code / 完整代码
# ===============================

# multivariate multi-step data preparation
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
  # 获取长度 / Get length
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# convert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
# summarize the data
# 获取长度 / Get length
for i in range(len(X)):
 # 打印输出 / Print output
	print(X[i], y[i])
```

---

➡️ **Next / 下一步**: File 15 of 15

---

### Cnn Multivariate Parallel Multistep

# 15 — Cnn Multivariate Parallel Multistep / 卷积神经网络

**Chapter 08 — File 15 of 15 / 第08章 — 第15个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate output multi-step 1d cnn example**.

本脚本演示 **multivariate output multi-step 1d cnn example**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — multivariate output multi-step 1d cnn example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D
```

---
## Step 2 — split a multivariate sequence into samples

```python
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
```

---
## Step 3 — find the end of this pattern

```python
end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
```

---
## Step 4 — check if we are beyond the dataset

```python
# 获取长度 / Get length
if out_end_ix > len(sequences):
			break
```

---
## Step 5 — gather input and output parts of the pattern

```python
seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)
```

---
## Step 6 — define input sequence

```python
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
```

---
## Step 7 — convert to [rows, columns] structure

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
```

---
## Step 8 — horizontally stack columns

```python
dataset = hstack((in_seq1, in_seq2, out_seq))
```

---
## Step 9 — choose a number of time steps

```python
n_steps_in, n_steps_out = 3, 2
```

---
## Step 10 — convert into input/output

```python
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
```

---
## Step 11 — flatten output

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_output = y.shape[1] * y.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y = y.reshape((y.shape[0], n_output))
```

---
## Step 12 — the dataset knows the number of features, e.g. 2

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_features = X.shape[2]
```

---
## Step 13 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps_in, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling1D())
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_output))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 14 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=7000, verbose=0)
```

---
## Step 15 — demonstrate prediction

```python
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_steps_in, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate output multi-step 1d cnn example 是机器学习中的常用技术。  
  *multivariate output multi-step 1d cnn example is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cnn Multivariate Parallel Multistep / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# multivariate output multi-step 1d cnn example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import Conv1D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.convolutional import MaxPooling1D

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
 # 获取长度 / Get length
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
  # 获取长度 / Get length
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
  # 添加元素到列表末尾 / Append element to list end
		X.append(seq_x)
  # 添加元素到列表末尾 / Append element to list end
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# 获取长度 / Get length
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# convert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# flatten output
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_output = y.shape[1] * y.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y = y.reshape((y.shape[0], n_output))
# the dataset knows the number of features, e.g. 2
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_features = X.shape[2]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps_in, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling1D())
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_output))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=7000, verbose=0)
# demonstrate prediction
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_steps_in, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

### Chapter Summary / 章节总结



---
