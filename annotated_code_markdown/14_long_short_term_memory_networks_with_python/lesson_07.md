# LSTM网络
## Lesson 07

---

### Damped Sine Wave

# 01 — Damped Sine Wave / Damped Sine Wave

**Chapter 07 — File 1 of 6 / 第07章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **create sequence**.

本脚本演示 **create sequence**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
from math import sin
from math import pi
from math import exp
from matplotlib import pyplot
```

---
## Step 2 — create sequence

```python
length = 100
period = 10
decay = 0.05
sequence = [0.5 + 0.5 * sin(2 * pi * i / period) * exp(-decay * i) for i in range(length)]
```

---
## Step 3 — plot sequence

```python
pyplot.plot(sequence)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create sequence 是机器学习中的常用技术。  
  *create sequence is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Damped Sine Wave / Damped Sine Wave
# Complete Code / 完整代码
# ===============================

from math import sin
from math import pi
from math import exp
from matplotlib import pyplot
# create sequence
length = 100
period = 10
decay = 0.05
sequence = [0.5 + 0.5 * sin(2 * pi * i / period) * exp(-decay * i) for i in range(length)]
# plot sequence
pyplot.plot(sequence)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Damped Sine Wave Sequences

# 01 — Damped Sine Wave Sequences / Damped Sine Wave Sequences

**Chapter 07 — File 2 of 6 / 第07章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **generate damped sine wave in [0,1]**.

本脚本演示 **generate damped sine wave in [0,1]**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
from math import sin
from math import pi
from math import exp
from random import randint
from random import uniform
from numpy import array
from matplotlib import pyplot
```

---
## Step 2 — generate damped sine wave in [0,1]

```python
def generate_sequence(length, period, decay):
	return [0.5 + 0.5 * sin(2 * pi * i / period) * exp(-decay * i) for i in range(length)]
```

---
## Step 3 — generate input and output pairs of damped sine waves

```python
def generate_examples(length, n_patterns, output):
	X, y = list(), list()
	for _ in range(n_patterns):
		p = randint(10, 20)
		d = uniform(0.01, 0.1)
		sequence = generate_sequence(length + output, p, d)
		X.append(sequence[:-output])
		y.append(sequence[-output:])
	X = array(X).reshape(n_patterns, length, 1)
	y = array(y).reshape(n_patterns, output)
	return X, y
```

---
## Step 4 — test problem generation

```python
X, y = generate_examples(20, 5, 5)
for i in range(len(X)):
	pyplot.plot([x for x in X[i, :, 0]] + [x for x in y[i]], '-o')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: generate damped sine wave in [0,1] 是机器学习中的常用技术。  
  *generate damped sine wave in [0,1] is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Damped Sine Wave Sequences / Damped Sine Wave Sequences
# Complete Code / 完整代码
# ===============================

from math import sin
from math import pi
from math import exp
from random import randint
from random import uniform
from numpy import array
from matplotlib import pyplot

# generate damped sine wave in [0,1]
def generate_sequence(length, period, decay):
	return [0.5 + 0.5 * sin(2 * pi * i / period) * exp(-decay * i) for i in range(length)]

# generate input and output pairs of damped sine waves
def generate_examples(length, n_patterns, output):
	X, y = list(), list()
	for _ in range(n_patterns):
		p = randint(10, 20)
		d = uniform(0.01, 0.1)
		sequence = generate_sequence(length + output, p, d)
		X.append(sequence[:-output])
		y.append(sequence[-output:])
	X = array(X).reshape(n_patterns, length, 1)
	y = array(y).reshape(n_patterns, output)
	return X, y

# test problem generation
X, y = generate_examples(20, 5, 5)
for i in range(len(X)):
	pyplot.plot([x for x in X[i, :, 0]] + [x for x in y[i]], '-o')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Lstm Return Sequences

# 01 — Lstm Return Sequences / LSTM 网络

**Chapter 07 — File 3 of 6 / 第07章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Example of one output for each input time step**.

本脚本演示 **Example of one output for each input time step**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — Example of one output for each input time step

```python
from keras.models import Sequential
from keras.layers import LSTM
from numpy import array
```

---
## Step 2 — define model where LSTM is also output layer

```python
model = Sequential()
model.add(LSTM(1, return_sequences=True, input_shape=(3,1)))
model.compile(optimizer='adam', loss='mse')
```

---
## Step 3 — input time steps

```python
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
```

---
## Step 4 — make and show prediction

```python
print(model.predict(data))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of one output for each input time step 是机器学习中的常用技术。  
  *Example of one output for each input time step is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lstm Return Sequences / LSTM 网络
# Complete Code / 完整代码
# ===============================

# Example of one output for each input time step
from keras.models import Sequential
from keras.layers import LSTM
from numpy import array
# define model where LSTM is also output layer
model = Sequential()
model.add(LSTM(1, return_sequences=True, input_shape=(3,1)))
model.compile(optimizer='adam', loss='mse')
# input time steps
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Stacked Lstm

# 01 — Stacked Lstm / LSTM 网络

**Chapter 07 — File 6 of 6 / 第07章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **generate damped sine wave in [0,1]**.

本脚本演示 **generate damped sine wave in [0,1]**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
from math import sin
from math import pi
from math import exp
from random import randint
from random import uniform
from numpy import array
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
```

---
## Step 2 — generate damped sine wave in [0,1]

```python
def generate_sequence(length, period, decay):
	return [0.5 + 0.5 * sin(2 * pi * i / period) * exp(-decay * i) for i in range(length)]
```

---
## Step 3 — generate input and output pairs of damped sine waves

```python
def generate_examples(length, n_patterns, output):
	X, y = list(), list()
	for _ in range(n_patterns):
		p = randint(10, 20)
		d = uniform(0.01, 0.1)
		sequence = generate_sequence(length + output, p, d)
		X.append(sequence[:-output])
		y.append(sequence[-output:])
	X = array(X).reshape(n_patterns, length, 1)
	y = array(y).reshape(n_patterns, output)
	return X, y
```

---
## Step 4 — configure problem

```python
length = 50
output = 5
```

---
## Step 5 — define model

```python
model = Sequential()
model.add(LSTM(20, return_sequences=True, input_shape=(length, 1)))
model.add(LSTM(20))
model.add(Dense(output))
model.compile(loss='mae', optimizer='adam')
model.summary()
```

---
## Step 6 — fit model

```python
X, y = generate_examples(length, 10000, output)
history = model.fit(X, y, batch_size=10, epochs=1)
```

---
## Step 7 — evaluate model

```python
X, y = generate_examples(length, 1000, output)
loss = model.evaluate(X, y, verbose=0)
print('MAE: %f' % loss)
```

---
## Step 8 — prediction on new data

```python
X, y = generate_examples(length, 1, output)
yhat = model.predict(X, verbose=0)
pyplot.plot(y[0], label='y')
pyplot.plot(yhat[0], label='yhat')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: generate damped sine wave in [0,1] 是机器学习中的常用技术。  
  *generate damped sine wave in [0,1] is a common technique in machine learning.*

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
| `matplotlib` | 绑图库 | Plotting library |
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
# Stacked Lstm / LSTM 网络
# Complete Code / 完整代码
# ===============================

from math import sin
from math import pi
from math import exp
from random import randint
from random import uniform
from numpy import array
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# generate damped sine wave in [0,1]
def generate_sequence(length, period, decay):
	return [0.5 + 0.5 * sin(2 * pi * i / period) * exp(-decay * i) for i in range(length)]

# generate input and output pairs of damped sine waves
def generate_examples(length, n_patterns, output):
	X, y = list(), list()
	for _ in range(n_patterns):
		p = randint(10, 20)
		d = uniform(0.01, 0.1)
		sequence = generate_sequence(length + output, p, d)
		X.append(sequence[:-output])
		y.append(sequence[-output:])
	X = array(X).reshape(n_patterns, length, 1)
	y = array(y).reshape(n_patterns, output)
	return X, y

# configure problem
length = 50
output = 5

# define model
model = Sequential()
model.add(LSTM(20, return_sequences=True, input_shape=(length, 1)))
model.add(LSTM(20))
model.add(Dense(output))
model.compile(loss='mae', optimizer='adam')
model.summary()

# fit model
X, y = generate_examples(length, 10000, output)
history = model.fit(X, y, batch_size=10, epochs=1)

# evaluate model
X, y = generate_examples(length, 1000, output)
loss = model.evaluate(X, y, verbose=0)
print('MAE: %f' % loss)

# prediction on new data
X, y = generate_examples(length, 1, output)
yhat = model.predict(X, verbose=0)
pyplot.plot(y[0], label='y')
pyplot.plot(yhat[0], label='yhat')
pyplot.legend()
pyplot.show()
```

---
