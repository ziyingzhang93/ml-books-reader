# 深度学习时间序列预测 / DL Time Series Forecasting
## Chapter 07

---

### Univariate Dataset



---

### Mlp Univariate



---

### Dependent Time Series



---

### Transform Dependent Time Series



---

### Mlp Dependent Time Series

# 05 — Mlp Dependent Time Series / 05 Mlp Dependent Time Series

**Chapter 07 — File 5 of 15 / 第07章 — 第5个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate mlp example**.

本脚本演示 **multivariate mlp example**。

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
## Step 1 — multivariate mlp example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
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
## Step 11 — flatten input

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_input = X.shape[1] * X.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], n_input))
```

---
## Step 12 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(100, activation='relu', input_dim=n_input))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
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
x_input = array([[80, 85], [90, 95], [100, 105]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_input))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate mlp example 是机器学习中的常用技术。  
  *multivariate mlp example is a common technique in machine learning.*

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
# Mlp Dependent Time Series / 05 Mlp Dependent Time Series
# Complete Code / 完整代码
# ===============================

# multivariate mlp example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
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
# flatten input
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_input = X.shape[1] * X.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], n_input))
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(100, activation='relu', input_dim=n_input))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_input))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 6 of 15

---

### Multiheaded Mlp Dependent Time Series

# 06 — Multiheaded Mlp Dependent Time Series / 06 Multiheaded Mlp Dependent Time Series

**Chapter 07 — File 6 of 15 / 第07章 — 第6个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate mlp example**.

本脚本演示 **multivariate mlp example**。

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
## Step 1 — multivariate mlp example

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
## Step 11 — separate input data

```python
X1 = X[:, :, 0]
X2 = X[:, :, 1]
```

---
## Step 12 — first input model

```python
visible1 = Input(shape=(n_steps,))
# 全连接层（Keras） / Fully connected layer (Keras)
dense1 = Dense(100, activation='relu')(visible1)
```

---
## Step 13 — second input model

```python
visible2 = Input(shape=(n_steps,))
# 全连接层（Keras） / Fully connected layer (Keras)
dense2 = Dense(100, activation='relu')(visible2)
```

---
## Step 14 — merge input models

```python
merge = concatenate([dense1, dense2])
# 全连接层（Keras） / Fully connected layer (Keras)
output = Dense(1)(merge)
model = Model(inputs=[visible1, visible2], outputs=output)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 15 — fit model

```python
# 训练模型 / Train the model
model.fit([X1, X2], y, epochs=2000, verbose=0)
```

---
## Step 16 — demonstrate prediction

```python
x_input = array([[80, 85], [90, 95], [100, 105]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x1 = x_input[:, 0].reshape((1, n_steps))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x2 = x_input[:, 1].reshape((1, n_steps))
# 用模型做预测 / Make predictions with model
yhat = model.predict([x1, x2], verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate mlp example 是机器学习中的常用技术。  
  *multivariate mlp example is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
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
# Multiheaded Mlp Dependent Time Series / 06 Multiheaded Mlp Dependent Time Series
# Complete Code / 完整代码
# ===============================

# multivariate mlp example
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
# separate input data
X1 = X[:, :, 0]
X2 = X[:, :, 1]
# first input model
visible1 = Input(shape=(n_steps,))
# 全连接层（Keras） / Fully connected layer (Keras)
dense1 = Dense(100, activation='relu')(visible1)
# second input model
visible2 = Input(shape=(n_steps,))
# 全连接层（Keras） / Fully connected layer (Keras)
dense2 = Dense(100, activation='relu')(visible2)
# merge input models
merge = concatenate([dense1, dense2])
# 全连接层（Keras） / Fully connected layer (Keras)
output = Dense(1)(merge)
model = Model(inputs=[visible1, visible2], outputs=output)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit([X1, X2], y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x1 = x_input[:, 0].reshape((1, n_steps))
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x2 = x_input[:, 1].reshape((1, n_steps))
# 用模型做预测 / Make predictions with model
yhat = model.predict([x1, x2], verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 7 of 15

---

### Split Multivariate Time Series



---

### Mlp Multivariate Time Series

# 08 — Mlp Multivariate Time Series / 多变量

**Chapter 07 — File 8 of 15 / 第07章 — 第8个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate output mlp example**.

本脚本演示 **multivariate output mlp example**。

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
## Step 1 — multivariate output mlp example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
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
## Step 11 — flatten input

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_input = X.shape[1] * X.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], n_input))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_output = y.shape[1]
```

---
## Step 12 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(100, activation='relu', input_dim=n_input))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_output))
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
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_input))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate output mlp example 是机器学习中的常用技术。  
  *multivariate output mlp example is a common technique in machine learning.*

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
# Mlp Multivariate Time Series / 多变量
# Complete Code / 完整代码
# ===============================

# multivariate output mlp example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
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
# flatten input
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_input = X.shape[1] * X.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], n_input))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_output = y.shape[1]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(100, activation='relu', input_dim=n_input))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_output))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([[70,75,145], [80,85,165], [90,95,185]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_input))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 9 of 15

---

### Multi-Output Mlp Multivariate Time Series

# 09 — Multi-Output Mlp Multivariate Time Series / 多变量

**Chapter 07 — File 9 of 15 / 第07章 — 第9个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate output mlp example**.

本脚本演示 **multivariate output mlp example**。

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
## Step 1 — multivariate output mlp example

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
## Step 11 — flatten input

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_input = X.shape[1] * X.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], n_input))
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
visible = Input(shape=(n_input,))
# 全连接层（Keras） / Fully connected layer (Keras)
dense = Dense(100, activation='relu')(visible)
```

---
## Step 14 — define output 1

```python
# 全连接层（Keras） / Fully connected layer (Keras)
output1 = Dense(1)(dense)
```

---
## Step 15 — define output 2

```python
# 全连接层（Keras） / Fully connected layer (Keras)
output2 = Dense(1)(dense)
```

---
## Step 16 — define output 2

```python
# 全连接层（Keras） / Fully connected layer (Keras)
output3 = Dense(1)(dense)
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
x_input = x_input.reshape((1, n_input))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate output mlp example 是机器学习中的常用技术。  
  *multivariate output mlp example is a common technique in machine learning.*

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
# Multi-Output Mlp Multivariate Time Series / 多变量
# Complete Code / 完整代码
# ===============================

# multivariate output mlp example
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
# flatten input
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_input = X.shape[1] * X.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], n_input))
# separate output
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y1 = y[:, 0].reshape((y.shape[0], 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y2 = y[:, 1].reshape((y.shape[0], 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y3 = y[:, 2].reshape((y.shape[0], 1))
# define model
visible = Input(shape=(n_input,))
# 全连接层（Keras） / Fully connected layer (Keras)
dense = Dense(100, activation='relu')(visible)
# define output 1
# 全连接层（Keras） / Fully connected layer (Keras)
output1 = Dense(1)(dense)
# define output 2
# 全连接层（Keras） / Fully connected layer (Keras)
output2 = Dense(1)(dense)
# define output 2
# 全连接层（Keras） / Fully connected layer (Keras)
output3 = Dense(1)(dense)
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
x_input = x_input.reshape((1, n_input))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 10 of 15

---

### Data Prep Multi Step Forecasting

# 10 — Data Prep Multi Step Forecasting / 预测

**Chapter 07 — File 10 of 15 / 第07章 — 第10个文件（共15个）**

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
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Data Prep Multi Step Forecasting / 预测
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

### Mlp Multi Step Forecast



---

### Prepare Data Multi Step Dependent Series



---

### Mlp Multi Step Dependent Series

# 13 — Mlp Multi Step Dependent Series / 13 Mlp Multi Step Dependent Series

**Chapter 07 — File 13 of 15 / 第07章 — 第13个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate multi-step mlp example**.

本脚本演示 **multivariate multi-step mlp example**。

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
## Step 1 — multivariate multi-step mlp example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
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
## Step 10 — convert into input/output

```python
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
```

---
## Step 11 — flatten input

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_input = X.shape[1] * X.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], n_input))
```

---
## Step 12 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(100, activation='relu', input_dim=n_input))
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
x_input = x_input.reshape((1, n_input))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate multi-step mlp example 是机器学习中的常用技术。  
  *multivariate multi-step mlp example is a common technique in machine learning.*

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
# Mlp Multi Step Dependent Series / 13 Mlp Multi Step Dependent Series
# Complete Code / 完整代码
# ===============================

# multivariate multi-step mlp example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
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
# convert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# flatten input
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_input = X.shape[1] * X.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], n_input))
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(100, activation='relu', input_dim=n_input))
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
x_input = x_input.reshape((1, n_input))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 14 of 15

---

### Prepare Data Multi Step Multivariate Series



---

### Mlp Multi Step Multivariate Series

# 15 — Mlp Multi Step Multivariate Series / 多变量

**Chapter 07 — File 15 of 15 / 第07章 — 第15个文件（共15个）**

---

## Summary / 总结

This script demonstrates **multivariate multi-step mlp example**.

本脚本演示 **multivariate multi-step mlp example**。

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
## Step 1 — multivariate multi-step mlp example

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
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
## Step 11 — flatten input

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_input = X.shape[1] * X.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], n_input))
```

---
## Step 12 — flatten output

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_output = y.shape[1] * y.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y = y.reshape((y.shape[0], n_output))
```

---
## Step 13 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(100, activation='relu', input_dim=n_input))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_output))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
```

---
## Step 14 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=2000, verbose=0)
```

---
## Step 15 — demonstrate prediction

```python
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_input))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: multivariate multi-step mlp example 是机器学习中的常用技术。  
  *multivariate multi-step mlp example is a common technique in machine learning.*

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
# Mlp Multi Step Multivariate Series / 多变量
# Complete Code / 完整代码
# ===============================

# multivariate multi-step mlp example
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense

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
# flatten input
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_input = X.shape[1] * X.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = X.reshape((X.shape[0], n_input))
# flatten output
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_output = y.shape[1] * y.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
y = y.reshape((y.shape[0], n_output))
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(100, activation='relu', input_dim=n_input))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(n_output))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='mse')
# fit model
# 训练模型 / Train the model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((1, n_input))
# 用模型做预测 / Make predictions with model
yhat = model.predict(x_input, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

### Chapter Summary / 章节总结

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **15 code files** demonstrating chapter 07.

本章包含 **15 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `01_univariate_dataset.ipynb` — Univariate Dataset
  2. `02_mlp_univariate.ipynb` — Mlp Univariate
  3. `03_dependent_time_series.ipynb` — Dependent Time Series
  4. `04_transform_dependent_time_series.ipynb` — Transform Dependent Time Series
  5. `05_mlp_dependent_time_series.ipynb` — Mlp Dependent Time Series
  6. `06_multiheaded_mlp_dependent_time_series.ipynb` — Multiheaded Mlp Dependent Time Series
  7. `07_split_multivariate_time_series.ipynb` — Split Multivariate Time Series
  8. `08_mlp_multivariate_time_series.ipynb` — Mlp Multivariate Time Series
  9. `09_multi-output_mlp_multivariate_time_series.ipynb` — Multi-Output Mlp Multivariate Time Series
  10. `10_data_prep_multi_step_forecasting.ipynb` — Data Prep Multi Step Forecasting
  11. `11_mlp_multi_step_forecast.ipynb` — Mlp Multi Step Forecast
  12. `12_prepare_data_multi_step_dependent_series.ipynb` — Prepare Data Multi Step Dependent Series
  13. `13_mlp_multi_step_dependent_series.ipynb` — Mlp Multi Step Dependent Series
  14. `14_prepare_data_multi_step_multivariate_series.ipynb` — Prepare Data Multi Step Multivariate Series
  15. `15_mlp_multi_step_multivariate_series.ipynb` — Mlp Multi Step Multivariate Series

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---
