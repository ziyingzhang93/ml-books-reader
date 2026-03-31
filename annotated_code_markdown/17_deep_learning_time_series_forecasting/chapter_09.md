# 深度学习时间序列预测 / DL Time Series Forecasting
## Chapter 09

---

### Univariate Dataset

# 01 — Univariate Dataset / 单变量

**Chapter 09 — File 1 of 18 / 第09章 — 第1个文件（共18个）**

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

➡️ **Next / 下一步**: File 2 of 18

---

### Vanilla Lstm Univariate

# 02 — Vanilla Lstm Univariate / LSTM 网络

**Chapter 09 — File 2 of 18 / 第09章 — 第2个文件（共18个）**

---

## Summary / 总结

This script demonstrates **univariate lstm example**.

本脚本演示 **univariate lstm example**。

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
## Step 1 — univariate lstm example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
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
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 11 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=200, verbose=0)
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

- **概念**: univariate lstm example 是机器学习中的常用技术。  
  *univariate lstm example is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
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
# Vanilla Lstm Univariate / LSTM 网络
# Complete Code / 完整代码
# ===============================

# univariate lstm example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense

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
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=200, verbose=0)
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

➡️ **Next / 下一步**: File 3 of 18

---

### Stacked Lstm Univariate

# 03 — Stacked Lstm Univariate / LSTM 网络

**Chapter 09 — File 3 of 18 / 第09章 — 第3个文件（共18个）**

---

## Summary / 总结

This script demonstrates **univariate stacked lstm example**.

本脚本演示 **univariate stacked lstm example**。

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
## Step 1 — univariate stacked lstm example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
```

---
## Step 2 — split a univariate sequence

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
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 11 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=200, verbose=0)
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

- **概念**: univariate stacked lstm example 是机器学习中的常用技术。  
  *univariate stacked lstm example is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
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
# Stacked Lstm Univariate / LSTM 网络
# Complete Code / 完整代码
# ===============================

# univariate stacked lstm example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense

# split a univariate sequence
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
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=200, verbose=0)
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

➡️ **Next / 下一步**: File 4 of 18

---

### Bidirectional Lstm Univariate

# 04 — Bidirectional Lstm Univariate / LSTM 网络

**Chapter 09 — File 4 of 18 / 第09章 — 第4个文件（共18个）**

---

## Summary / 总结

This script demonstrates **univariate bidirectional lstm example**.

本脚本演示 **univariate bidirectional lstm example**。

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
## Step 1 — univariate bidirectional lstm example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Bidirectional
```

---
## Step 2 — split a univariate sequence

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
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 11 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=200, verbose=0)
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

- **概念**: univariate bidirectional lstm example 是机器学习中的常用技术。  
  *univariate bidirectional lstm example is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
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
# Bidirectional Lstm Univariate / LSTM 网络
# Complete Code / 完整代码
# ===============================

# univariate bidirectional lstm example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Bidirectional

# split a univariate sequence
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
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=200, verbose=0)
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

➡️ **Next / 下一步**: File 5 of 18

---

### Cnn Lstm Univariate

# 05 — Cnn Lstm Univariate / 卷积神经网络

**Chapter 09 — File 5 of 18 / 第09章 — 第5个文件（共18个）**

---

## Summary / 总结

This script demonstrates **univariate cnn lstm example**.

本脚本演示 **univariate cnn lstm example**。

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
## Step 1 — univariate cnn lstm example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import TimeDistributed
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
n_steps = 4
```

---
## Step 8 — split into samples

```python
X, y = split_sequence(raw_seq, n_steps)
```

---
## Step 9 — reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]

```python
n_features = 1
n_seq = 2
n_steps = 2
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
```

---
## Step 10 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(TimeDistributed(Conv1D(64, 1, activation='relu'), input_shape=(None, n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(TimeDistributed(MaxPooling1D()))
# 向模型添加一层 / Add a layer to the model
model.add(TimeDistributed(Flatten()))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 11 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=500, verbose=0)
```

---
## Step 12 — demonstrate prediction

```python
x_input = array([60, 70, 80, 90])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: univariate cnn lstm example 是机器学习中的常用技术。  
  *univariate cnn lstm example is a common technique in machine learning.*

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
# Cnn Lstm Univariate / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# univariate cnn lstm example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import TimeDistributed
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
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(TimeDistributed(Conv1D(64, 1, activation='relu'), input_shape=(None, n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(TimeDistributed(MaxPooling1D()))
# 向模型添加一层 / Add a layer to the model
model.add(TimeDistributed(Flatten()))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(50, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=500, verbose=0)
# demonstrate prediction
x_input = array([60, 70, 80, 90])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 6 of 18

---

### Convlstm Univariate

# 06 — Convlstm Univariate / LSTM 网络

**Chapter 09 — File 6 of 18 / 第09章 — 第6个文件（共18个）**

---

## Summary / 总结

This script demonstrates **univariate convlstm example**.

本脚本演示 **univariate convlstm example**。

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
## Step 1 — univariate convlstm example

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
from keras.layers import ConvLSTM2D
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
n_steps = 4
```

---
## Step 8 — split into samples

```python
X, y = split_sequence(raw_seq, n_steps)
```

---
## Step 9 — reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]

```python
n_features = 1
n_seq = 2
n_steps = 2
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
```

---
## Step 10 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(ConvLSTM2D(64, (1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 11 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=500, verbose=0)
```

---
## Step 12 — demonstrate prediction

```python
x_input = array([60, 70, 80, 90])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: univariate convlstm example 是机器学习中的常用技术。  
  *univariate convlstm example is a common technique in machine learning.*

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
# Convlstm Univariate / LSTM 网络
# Complete Code / 完整代码
# ===============================

# univariate convlstm example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import ConvLSTM2D

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
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
n_seq = 2
n_steps = 2
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(ConvLSTM2D(64, (1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=500, verbose=0)
# demonstrate prediction
x_input = array([60, 70, 80, 90])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 7 of 18

---

### Dependent Series Dataset



---

### Dependent Series To Samples



---

### Vanilla Lstm Multivariate Dependent Series

# 09 — Vanilla Lstm Multivariate Dependent Series / LSTM 网络

**Chapter 09 — File 9 of 18 / 第09章 — 第9个文件（共18个）**

---

## Summary / 总结

This script demonstrates **multivariate lstm example**.

本脚本演示 **multivariate lstm example**。

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
## Step 1 — multivariate lstm example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
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
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 13 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=200, verbose=0)
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

- **概念**: multivariate lstm example 是机器学习中的常用技术。  
  *multivariate lstm example is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
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
# Vanilla Lstm Multivariate Dependent Series / LSTM 网络
# Complete Code / 完整代码
# ===============================

# multivariate lstm example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense

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
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=200, verbose=0)
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

➡️ **Next / 下一步**: File 10 of 18

---

### Multivariate Parallel Series Dataset

# 10 — Multivariate Parallel Series Dataset / 多变量

**Chapter 09 — File 10 of 18 / 第09章 — 第10个文件（共18个）**

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
# Multivariate Parallel Series Dataset / 多变量
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

➡️ **Next / 下一步**: File 11 of 18

---

### Stacked Lstm Multivariate Parallel Series

# 11 — Stacked Lstm Multivariate Parallel Series / LSTM 网络

**Chapter 09 — File 11 of 18 / 第09章 — 第11个文件（共18个）**

---

## Summary / 总结

This script demonstrates **multivariate output stacked lstm example**.

本脚本演示 **multivariate output stacked lstm example**。

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
## Step 1 — multivariate output stacked lstm example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
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
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_features))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 13 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=400, verbose=0)
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

- **概念**: multivariate output stacked lstm example 是机器学习中的常用技术。  
  *multivariate output stacked lstm example is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
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
# Stacked Lstm Multivariate Parallel Series / LSTM 网络
# Complete Code / 完整代码
# ===============================

# multivariate output stacked lstm example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense

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
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_features))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=400, verbose=0)
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

➡️ **Next / 下一步**: File 12 of 18

---

### Multi Step Series Dataset

# 12 — Multi Step Series Dataset / 12 Multi Step Series Dataset

**Chapter 09 — File 12 of 18 / 第09章 — 第12个文件（共18个）**

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
# Multi Step Series Dataset / 12 Multi Step Series Dataset
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

➡️ **Next / 下一步**: File 13 of 18

---

### Stacked Lstm Multi Step

# 13 — Stacked Lstm Multi Step / LSTM 网络

**Chapter 09 — File 13 of 18 / 第09章 — 第13个文件（共18个）**

---

## Summary / 总结

This script demonstrates **univariate multi-step vector-output stacked lstm example**.

本脚本演示 **univariate multi-step vector-output stacked lstm example**。

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
## Step 1 — univariate multi-step vector-output stacked lstm example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
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
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_steps_out))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 11 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=50, verbose=0)
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

- **概念**: univariate multi-step vector-output stacked lstm example 是机器学习中的常用技术。  
  *univariate multi-step vector-output stacked lstm example is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
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
# Stacked Lstm Multi Step / LSTM 网络
# Complete Code / 完整代码
# ===============================

# univariate multi-step vector-output stacked lstm example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense

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
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_steps_out))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=50, verbose=0)
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

➡️ **Next / 下一步**: File 14 of 18

---

### Encoder Decoder Lstm Multi Step

# 14 — Encoder Decoder Lstm Multi Step / LSTM 网络

**Chapter 09 — File 14 of 18 / 第09章 — 第14个文件（共18个）**

---

## Summary / 总结

This script demonstrates **univariate multi-step encoder-decoder lstm example**.

本脚本演示 **univariate multi-step encoder-decoder lstm example**。

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
## Step 1 — univariate multi-step encoder-decoder lstm example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import RepeatVector
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import TimeDistributed
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
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y = y.reshape((y.shape[0], y.shape[1], n_features))
```

---
## Step 10 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(RepeatVector(n_steps_out))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100, activation='relu', return_sequences=True))
# 向模型添加一层 / Add a layer to the model
model.add(TimeDistributed(Dense(1)))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 11 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=100, verbose=0)
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

- **概念**: univariate multi-step encoder-decoder lstm example 是机器学习中的常用技术。  
  *univariate multi-step encoder-decoder lstm example is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
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
# Encoder Decoder Lstm Multi Step / LSTM 网络
# Complete Code / 完整代码
# ===============================

# univariate multi-step encoder-decoder lstm example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import RepeatVector
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import TimeDistributed

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
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y = y.reshape((y.shape[0], y.shape[1], n_features))
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(RepeatVector(n_steps_out))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100, activation='relu', return_sequences=True))
# 向模型添加一层 / Add a layer to the model
model.add(TimeDistributed(Dense(1)))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=100, verbose=0)
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

➡️ **Next / 下一步**: File 15 of 18

---

### Multivariate Dependent Series Multi Step Dataset



---

### Stacked Lstm Dependent Multi Step

# 16 — Stacked Lstm Dependent Multi Step / LSTM 网络

**Chapter 09 — File 16 of 18 / 第09章 — 第16个文件（共18个）**

---

## Summary / 总结

This script demonstrates **multivariate multi-step stacked lstm example**.

本脚本演示 **multivariate multi-step stacked lstm example**。

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
## Step 1 — multivariate multi-step stacked lstm example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
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
## Step 10 — covert into input/output

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
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_steps_out))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 13 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=200, verbose=0)
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

- **概念**: multivariate multi-step stacked lstm example 是机器学习中的常用技术。  
  *multivariate multi-step stacked lstm example is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
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
# Stacked Lstm Dependent Multi Step / LSTM 网络
# Complete Code / 完整代码
# ===============================

# multivariate multi-step stacked lstm example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense

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
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# the dataset knows the number of features, e.g. 2
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_features = X.shape[2]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_steps_out))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=200, verbose=0)
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

➡️ **Next / 下一步**: File 17 of 18

---

### Multivariate Parallel Series Multi Step Dataset



---

### Encoder Decoder Lstm Parallel Multi Step

# 18 — Encoder Decoder Lstm Parallel Multi Step / LSTM 网络

**Chapter 09 — File 18 of 18 / 第09章 — 第18个文件（共18个）**

---

## Summary / 总结

This script demonstrates **multivariate multi-step encoder-decoder lstm example**.

本脚本演示 **multivariate multi-step encoder-decoder lstm example**。

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
## Step 1 — multivariate multi-step encoder-decoder lstm example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import RepeatVector
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import TimeDistributed
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
## Step 10 — covert into input/output

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
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(RepeatVector(n_steps_out))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(200, activation='relu', return_sequences=True))
# 向模型添加一层 / Add a layer to the model
model.add(TimeDistributed(Dense(n_features)))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 13 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=300, verbose=0)
```

---
## Step 14 — demonstrate prediction

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

- **概念**: multivariate multi-step encoder-decoder lstm example 是机器学习中的常用技术。  
  *multivariate multi-step encoder-decoder lstm example is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
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
# Encoder Decoder Lstm Parallel Multi Step / LSTM 网络
# Complete Code / 完整代码
# ===============================

# multivariate multi-step encoder-decoder lstm example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import RepeatVector
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import TimeDistributed

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
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# the dataset knows the number of features, e.g. 2
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_features = X.shape[2]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
# 向模型添加一层 / Add a layer to the model
model.add(RepeatVector(n_steps_out))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(200, activation='relu', return_sequences=True))
# 向模型添加一层 / Add a layer to the model
model.add(TimeDistributed(Dense(n_features)))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=300, verbose=0)
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

# Chapter 09 Summary / 第09章总结

## Theme / 主题: Chapter 09 / Chapter 09

This chapter contains **18 code files** demonstrating chapter 09.

本章包含 **18 个代码文件**，演示Chapter 09。

---
## Evolution / 演化路线

  1. `01_univariate_dataset.ipynb` — Univariate Dataset
  2. `02_vanilla_lstm_univariate.ipynb` — Vanilla Lstm Univariate
  3. `03_stacked_lstm_univariate.ipynb` — Stacked Lstm Univariate
  4. `04_bidirectional_lstm_univariate.ipynb` — Bidirectional Lstm Univariate
  5. `05_cnn_lstm_univariate.ipynb` — Cnn Lstm Univariate
  6. `06_convlstm_univariate.ipynb` — Convlstm Univariate
  7. `07_dependent_series_dataset.ipynb` — Dependent Series Dataset
  8. `08_dependent_series_to_samples.ipynb` — Dependent Series To Samples
  9. `09_vanilla_lstm_multivariate_dependent_series.ipynb` — Vanilla Lstm Multivariate Dependent Series
  10. `10_multivariate_parallel_series_dataset.ipynb` — Multivariate Parallel Series Dataset
  11. `11_stacked_lstm_multivariate_parallel_series.ipynb` — Stacked Lstm Multivariate Parallel Series
  12. `12_multi_step_series_dataset.ipynb` — Multi Step Series Dataset
  13. `13_stacked_lstm_multi_step.ipynb` — Stacked Lstm Multi Step
  14. `14_encoder_decoder_lstm_multi_step.ipynb` — Encoder Decoder Lstm Multi Step
  15. `15_multivariate_dependent_series_multi_step_dataset.ipynb` — Multivariate Dependent Series Multi Step Dataset
  16. `16_stacked_lstm_dependent_multi_step.ipynb` — Stacked Lstm Dependent Multi Step
  17. `17_multivariate_parallel_series_multi_step_dataset.ipynb` — Multivariate Parallel Series Multi Step Dataset
  18. `18_encoder_decoder_lstm_parallel_multi_step.ipynb` — Encoder Decoder Lstm Parallel Multi Step

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 09) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 09）是机器学习流水线中的基础构建块。

---
