# LSTM网络
## Lesson 10

---

### Bidirectional Lstm

# 01 — Bidirectional Lstm / LSTM 网络

**Chapter 10 — File 1 of 2 / 第10章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **create a cumulative sum sequence**.

本脚本演示 **create a cumulative sum sequence**。

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
## Step 1 — Step 1

```python
from random import random
from numpy import array
from numpy import cumsum
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
```

---
## Step 2 — create a cumulative sum sequence

```python
def get_sequence(n_timesteps):
```

---
## Step 3 — create a sequence of random numbers in [0,1]

```python
X = array([random() for _ in range(n_timesteps)])
```

---
## Step 4 — calculate cut-off value to change class values

```python
limit = n_timesteps/4.0
```

---
## Step 5 — determine the class outcome for each item in cumulative sequence

```python
y = array([0 if x < limit else 1 for x in cumsum(X)])
	return X, y
```

---
## Step 6 — create multiple samples of cumulative sum sequences

```python
def get_sequences(n_sequences, n_timesteps):
	seqX, seqY = list(), list()
```

---
## Step 7 — create and store sequences

```python
for _ in range(n_sequences):
		X, y = get_sequence(n_timesteps)
		seqX.append(X)
		seqY.append(y)
```

---
## Step 8 — reshape input and output for lstm

```python
seqX = array(seqX).reshape(n_sequences, n_timesteps, 1)
	seqY = array(seqY).reshape(n_sequences, n_timesteps, 1)
	return seqX, seqY
```

---
## Step 9 — define problem

```python
n_timesteps = 10
```

---
## Step 10 — define LSTM

```python
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(n_timesteps, 1)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

---
## Step 11 — train LSTM

```python
X, y = get_sequences(50000, n_timesteps)
model.fit(X, y, epochs=1, batch_size=10)
```

---
## Step 12 — evaluate LSTM

```python
X, y = get_sequences(100, n_timesteps)
loss, acc = model.evaluate(X, y, verbose=0)
print('Loss: %f, Accuracy: %f' % (loss, acc*100))
```

---
## Step 13 — make predictions

```python
for _ in range(10):
	X, y = get_sequences(1, n_timesteps)
	yhat = model.predict_classes(X, verbose=0)
	exp, pred = y.reshape(n_timesteps), yhat.reshape(n_timesteps)
	print('y=%s, yhat=%s, correct=%s' % (exp, pred, array_equal(exp,pred)))
```

---
## Learning Notes / 学习笔记

- **概念**: create a cumulative sum sequence 是机器学习中的常用技术。  
  *create a cumulative sum sequence is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
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
# Bidirectional Lstm / LSTM 网络
# Complete Code / 完整代码
# ===============================

from random import random
from numpy import array
from numpy import cumsum
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

# create a cumulative sum sequence
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	return X, y

# create multiple samples of cumulative sum sequences
def get_sequences(n_sequences, n_timesteps):
	seqX, seqY = list(), list()
	# create and store sequences
	for _ in range(n_sequences):
		X, y = get_sequence(n_timesteps)
		seqX.append(X)
		seqY.append(y)
	# reshape input and output for lstm
	seqX = array(seqX).reshape(n_sequences, n_timesteps, 1)
	seqY = array(seqY).reshape(n_sequences, n_timesteps, 1)
	return seqX, seqY

# define problem
n_timesteps = 10

# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(n_timesteps, 1)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# train LSTM
X, y = get_sequences(50000, n_timesteps)
model.fit(X, y, epochs=1, batch_size=10)

# evaluate LSTM
X, y = get_sequences(100, n_timesteps)
loss, acc = model.evaluate(X, y, verbose=0)
print('Loss: %f, Accuracy: %f' % (loss, acc*100))

# make predictions
for _ in range(10):
	X, y = get_sequences(1, n_timesteps)
	yhat = model.predict_classes(X, verbose=0)
	exp, pred = y.reshape(n_timesteps), yhat.reshape(n_timesteps)
	print('y=%s, yhat=%s, correct=%s' % (exp, pred, array_equal(exp,pred)))
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Problem Sequence

# 01 — Problem Sequence / Problem Sequence

**Chapter 10 — File 2 of 2 / 第10章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **create a cumulative sum sequence**.

本脚本演示 **create a cumulative sum sequence**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
from random import random
from numpy import array
from numpy import cumsum
```

---
## Step 2 — create a cumulative sum sequence

```python
def get_sequence(n_timesteps):
```

---
## Step 3 — create a sequence of random numbers in [0,1]

```python
X = array([random() for _ in range(n_timesteps)])
```

---
## Step 4 — calculate cut-off value to change class values

```python
limit = n_timesteps/4.0
```

---
## Step 5 — determine the class outcome for each item in cumulative sequence

```python
y = array([0 if x < limit else 1 for x in cumsum(X)])
	return X, y

X, y = get_sequence(10)
print(X)
print(y)
```

---
## Learning Notes / 学习笔记

- **概念**: create a cumulative sum sequence 是机器学习中的常用技术。  
  *create a cumulative sum sequence is a common technique in machine learning.*

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
# Problem Sequence / Problem Sequence
# Complete Code / 完整代码
# ===============================

from random import random
from numpy import array
from numpy import cumsum

# create a cumulative sum sequence
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	return X, y

X, y = get_sequence(10)
print(X)
print(y)
```

---
