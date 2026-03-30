# LSTM网络
## Lesson 06

---

### Problem Example

# 01 — Problem Example / Problem Example

**Chapter 06 — File 1 of 3 / 第06章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **generate a sequence of random numbers in [0, n_features)**.

本脚本演示 **generate a sequence of random numbers in [0, n_features)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from random import randint
from numpy import array
from numpy import argmax
```

---
## Step 2 — generate a sequence of random numbers in [0, n_features)

```python
def generate_sequence(length, n_features):
	return [randint(0, n_features-1) for _ in range(length)]
```

---
## Step 3 — one hot encode sequence

```python
def one_hot_encode(sequence, n_features):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_features)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)
```

---
## Step 4 — decode a one hot encoded string

```python
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]
```

---
## Step 5 — generate random sequence

```python
sequence = generate_sequence(25, 100)
print(sequence)
```

---
## Step 6 — one hot encode

```python
encoded = one_hot_encode(sequence, 100)
print(encoded)
```

---
## Step 7 — one hot decode

```python
decoded = one_hot_decode(encoded)
print(decoded)
```

---
## Learning Notes / 学习笔记

- **概念**: generate a sequence of random numbers in [0, n_features) 是机器学习中的常用技术。  
  *generate a sequence of random numbers in [0, n_features) is a common technique in machine learning.*

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
# Problem Example / Problem Example
# Complete Code / 完整代码
# ===============================

from random import randint
from numpy import array
from numpy import argmax

# generate a sequence of random numbers in [0, n_features)
def generate_sequence(length, n_features):
	return [randint(0, n_features-1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_features):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_features)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# generate random sequence
sequence = generate_sequence(25, 100)
print(sequence)
# one hot encode
encoded = one_hot_encode(sequence, 100)
print(encoded)
# one hot decode
decoded = one_hot_decode(encoded)
print(decoded)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Problem Example Reshape

# 01 — Problem Example Reshape / Problem Example Reshape

**Chapter 06 — File 2 of 3 / 第06章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **generate a sequence of random numbers in [0, n_features)**.

本脚本演示 **generate a sequence of random numbers in [0, n_features)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from random import randint
from numpy import array
from numpy import argmax
```

---
## Step 2 — generate a sequence of random numbers in [0, n_features)

```python
def generate_sequence(length, n_features):
	return [randint(0, n_features-1) for _ in range(length)]
```

---
## Step 3 — one hot encode sequence

```python
def one_hot_encode(sequence, n_features):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_features)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)
```

---
## Step 4 — decode a one hot encoded string

```python
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]
```

---
## Step 5 — generate one example for an lstm

```python
def generate_example(length, n_features, out_index):
```

---
## Step 6 — generate sequence

```python
sequence = generate_sequence(length, n_features)
```

---
## Step 7 — one hot encode

```python
encoded = one_hot_encode(sequence, n_features)
```

---
## Step 8 — reshape sequence to be 3D

```python
X = encoded.reshape((1, length, n_features))
```

---
## Step 9 — select output

```python
y = encoded[out_index].reshape(1, n_features)
	return X, y

X, y = generate_example(25, 100, 2)
print(X.shape)
print(y.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: generate a sequence of random numbers in [0, n_features) 是机器学习中的常用技术。  
  *generate a sequence of random numbers in [0, n_features) is a common technique in machine learning.*

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
# Problem Example Reshape / Problem Example Reshape
# Complete Code / 完整代码
# ===============================

from random import randint
from numpy import array
from numpy import argmax

# generate a sequence of random numbers in [0, n_features)
def generate_sequence(length, n_features):
	return [randint(0, n_features-1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_features):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_features)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# generate one example for an lstm
def generate_example(length, n_features, out_index):
	# generate sequence
	sequence = generate_sequence(length, n_features)
	# one hot encode
	encoded = one_hot_encode(sequence, n_features)
	# reshape sequence to be 3D
	X = encoded.reshape((1, length, n_features))
	# select output
	y = encoded[out_index].reshape(1, n_features)
	return X, y

X, y = generate_example(25, 100, 2)
print(X.shape)
print(y.shape)
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Vanilla Lstm

# 01 — Vanilla Lstm / LSTM 网络

**Chapter 06 — File 3 of 3 / 第06章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **generate a sequence of random numbers in [0, n_features)**.

本脚本演示 **generate a sequence of random numbers in [0, n_features)**。

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
from random import randint
from numpy import array
from numpy import argmax
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
```

---
## Step 2 — generate a sequence of random numbers in [0, n_features)

```python
def generate_sequence(length, n_features):
	return [randint(0, n_features-1) for _ in range(length)]
```

---
## Step 3 — one hot encode sequence

```python
def one_hot_encode(sequence, n_features):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_features)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)
```

---
## Step 4 — decode a one hot encoded string

```python
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]
```

---
## Step 5 — generate one example for an lstm

```python
def generate_example(length, n_features, out_index):
```

---
## Step 6 — generate sequence

```python
sequence = generate_sequence(length, n_features)
```

---
## Step 7 — one hot encode

```python
encoded = one_hot_encode(sequence, n_features)
```

---
## Step 8 — reshape sequence to be 3D

```python
X = encoded.reshape((1, length, n_features))
```

---
## Step 9 — select output

```python
y = encoded[out_index].reshape(1, n_features)
	return X, y
```

---
## Step 10 — define model

```python
length = 5
n_features = 10
out_index = 2
model = Sequential()
model.add(LSTM(25, input_shape=(length, n_features)))
model.add(Dense(n_features, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

---
## Step 11 — fit model

```python
for i in range(10000):
	X, y = generate_example(length, n_features, out_index)
	model.fit(X, y, epochs=1, verbose=2)
```

---
## Step 12 — evaluate model

```python
correct = 0
for i in range(100):
	X, y = generate_example(length, n_features, out_index)
	yhat = model.predict(X)
	if one_hot_decode(yhat) == one_hot_decode(y):
		correct += 1
print('Accuracy: %f' % ((correct/100.0)*100.0))
```

---
## Step 13 — prediction on new data

```python
X, y = generate_example(length, n_features, out_index)
yhat = model.predict(X)
print('Sequence:  %s' % [one_hot_decode(x) for x in X])
print('Expected:  %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))
```

---
## Learning Notes / 学习笔记

- **概念**: generate a sequence of random numbers in [0, n_features) 是机器学习中的常用技术。  
  *generate a sequence of random numbers in [0, n_features) is a common technique in machine learning.*

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
# Vanilla Lstm / LSTM 网络
# Complete Code / 完整代码
# ===============================

from random import randint
from numpy import array
from numpy import argmax
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# generate a sequence of random numbers in [0, n_features)
def generate_sequence(length, n_features):
	return [randint(0, n_features-1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_features):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_features)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# generate one example for an lstm
def generate_example(length, n_features, out_index):
	# generate sequence
	sequence = generate_sequence(length, n_features)
	# one hot encode
	encoded = one_hot_encode(sequence, n_features)
	# reshape sequence to be 3D
	X = encoded.reshape((1, length, n_features))
	# select output
	y = encoded[out_index].reshape(1, n_features)
	return X, y

# define model
length = 5
n_features = 10
out_index = 2
model = Sequential()
model.add(LSTM(25, input_shape=(length, n_features)))
model.add(Dense(n_features, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# fit model
for i in range(10000):
	X, y = generate_example(length, n_features, out_index)
	model.fit(X, y, epochs=1, verbose=2)

# evaluate model
correct = 0
for i in range(100):
	X, y = generate_example(length, n_features, out_index)
	yhat = model.predict(X)
	if one_hot_decode(yhat) == one_hot_decode(y):
		correct += 1
print('Accuracy: %f' % ((correct/100.0)*100.0))

# prediction on new data
X, y = generate_example(length, n_features, out_index)
yhat = model.predict(X)
print('Sequence:  %s' % [one_hot_decode(x) for x in X])
print('Expected:  %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))
```

---
