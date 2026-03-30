# LSTM网络
## Lesson 08

---

### Cnn Lstm

# 01 — Cnn Lstm / 卷积神经网络

**Chapter 08 — File 1 of 2 / 第08章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **generate the next frame in the sequence**.

本脚本演示 **generate the next frame in the sequence**。

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
from random import randint
from numpy import array
from numpy import zeros
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
```

---
## Step 2 — generate the next frame in the sequence

```python
def next_frame(last_step, last_frame, column):
```

---
## Step 3 — define the scope of the next step

```python
lower = max(0, last_step-1)
	upper = min(last_frame.shape[0]-1, last_step+1)
```

---
## Step 4 — choose the row index for the next step

```python
step = randint(lower, upper)
```

---
## Step 5 — copy the prior frame

```python
frame = last_frame.copy()
```

---
## Step 6 — add the new step

```python
frame[step, column] = 1
	return frame, step
```

---
## Step 7 — generate a sequence of frames of a dot moving across an image

```python
def build_frames(size):
	frames = list()
```

---
## Step 8 — create the first frame

```python
frame = zeros((size,size))
	step = randint(0, size-1)
```

---
## Step 9 — decide if we are heading left or right

```python
right = 1 if random() < 0.5 else 0
	col = 0 if right else size-1
	frame[step, col] = 1
	frames.append(frame)
```

---
## Step 10 — create all remaining frames

```python
for i in range(1, size):
		col = i if right else size-1-i
		frame, step = next_frame(step, frame, col)
		frames.append(frame)
	return frames, right
```

---
## Step 11 — generate multiple sequences of frames and reshape for network input

```python
def generate_examples(size, n_patterns):
	X, y = list(), list()
	for _ in range(n_patterns):
		frames, right = build_frames(size)
		X.append(frames)
		y.append(right)
```

---
## Step 12 — resize as [samples, timesteps, width, height, channels]

```python
X = array(X).reshape(n_patterns, size, size, size, 1)
	y = array(y).reshape(n_patterns, 1)
	return X, y
```

---
## Step 13 — configure problem

```python
size = 50
```

---
## Step 14 — define the model

```python
model = Sequential()
model.add(TimeDistributed(Conv2D(2, (2,2), activation='relu'), input_shape=(None,size,size,1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

---
## Step 15 — fit model

```python
X, y = generate_examples(size, 5000)
model.fit(X, y, batch_size=32, epochs=1)
```

---
## Step 16 — evaluate model

```python
X, y = generate_examples(size, 100)
loss, acc = model.evaluate(X, y, verbose=0)
print('loss: %f, acc: %f' % (loss, acc*100))
```

---
## Step 17 — prediction on new data

```python
X, y = generate_examples(size, 1)
yhat = model.predict_classes(X, verbose=0)
expected = "Right" if y[0]==1 else "Left"
predicted = "Right" if yhat[0]==1 else "Left"
print('Expected: %s, Predicted: %s' % (expected, predicted))
```

---
## Learning Notes / 学习笔记

- **概念**: generate the next frame in the sequence 是机器学习中的常用技术。  
  *generate the next frame in the sequence is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |
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
# Cnn Lstm / 卷积神经网络
# Complete Code / 完整代码
# ===============================

from random import random
from random import randint
from numpy import array
from numpy import zeros
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed

# generate the next frame in the sequence
def next_frame(last_step, last_frame, column):
	# define the scope of the next step
	lower = max(0, last_step-1)
	upper = min(last_frame.shape[0]-1, last_step+1)
	# choose the row index for the next step
	step = randint(lower, upper)
	# copy the prior frame
	frame = last_frame.copy()
	# add the new step
	frame[step, column] = 1
	return frame, step

# generate a sequence of frames of a dot moving across an image
def build_frames(size):
	frames = list()
	# create the first frame
	frame = zeros((size,size))
	step = randint(0, size-1)
	# decide if we are heading left or right
	right = 1 if random() < 0.5 else 0
	col = 0 if right else size-1
	frame[step, col] = 1
	frames.append(frame)
	# create all remaining frames
	for i in range(1, size):
		col = i if right else size-1-i
		frame, step = next_frame(step, frame, col)
		frames.append(frame)
	return frames, right

# generate multiple sequences of frames and reshape for network input
def generate_examples(size, n_patterns):
	X, y = list(), list()
	for _ in range(n_patterns):
		frames, right = build_frames(size)
		X.append(frames)
		y.append(right)
	# resize as [samples, timesteps, width, height, channels]
	X = array(X).reshape(n_patterns, size, size, size, 1)
	y = array(y).reshape(n_patterns, 1)
	return X, y

# configure problem
size = 50

# define the model
model = Sequential()
model.add(TimeDistributed(Conv2D(2, (2,2), activation='relu'), input_shape=(None,size,size,1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# fit model
X, y = generate_examples(size, 5000)
model.fit(X, y, batch_size=32, epochs=1)

# evaluate model
X, y = generate_examples(size, 100)
loss, acc = model.evaluate(X, y, verbose=0)
print('loss: %f, acc: %f' % (loss, acc*100))

# prediction on new data
X, y = generate_examples(size, 1)
yhat = model.predict_classes(X, verbose=0)
expected = "Right" if y[0]==1 else "Left"
predicted = "Right" if yhat[0]==1 else "Left"
print('Expected: %s, Predicted: %s' % (expected, predicted))
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Problem

# 01 — Problem / Problem

**Chapter 08 — File 2 of 2 / 第08章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **generate the next frame in the sequence**.

本脚本演示 **generate the next frame in the sequence**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
from numpy import zeros
from random import randint
from random import random
from matplotlib import pyplot
```

---
## Step 2 — generate the next frame in the sequence

```python
def next_frame(last_step, last_frame, column):
```

---
## Step 3 — define the scope of the next step

```python
lower = max(0, last_step-1)
	upper = min(last_frame.shape[0]-1, last_step+1)
```

---
## Step 4 — choose the row index for the next step

```python
step = randint(lower, upper)
```

---
## Step 5 — copy the prior frame

```python
frame = last_frame.copy()
```

---
## Step 6 — add the new step

```python
frame[step, column] = 1
	return frame, step
```

---
## Step 7 — generate a sequence of frames of a dot moving across an image

```python
def build_frames(size):
	frames = list()
```

---
## Step 8 — create the first frame

```python
frame = zeros((size,size))
	step = randint(0, size-1)
```

---
## Step 9 — decide if we are heading left or right

```python
right = 1 if random() < 0.5 else 0
	col = 0 if right else size-1
	frame[step, col] = 1
	frames.append(frame)
```

---
## Step 10 — create all remaining frames

```python
for i in range(1, size):
		col = i if right else size-1-i
		frame, step = next_frame(step, frame, col)
		frames.append(frame)
	return frames, right
```

---
## Step 11 — generate sequence of frames

```python
size = 5
frames, right = build_frames(size)
```

---
## Step 12 — plot all frames

```python
pyplot.figure()
for i in range(size):
```

---
## Step 13 — create a gray scale subplot for each frame

```python
pyplot.subplot(1, size, i+1)
	pyplot.imshow(frames[i], cmap='Greys')
```

---
## Step 14 — turn of the scale to make it clearer

```python
ax = pyplot.gca()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
```

---
## Step 15 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: generate the next frame in the sequence 是机器学习中的常用技术。  
  *generate the next frame in the sequence is a common technique in machine learning.*

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
# Problem / Problem
# Complete Code / 完整代码
# ===============================

from numpy import zeros
from random import randint
from random import random
from matplotlib import pyplot

# generate the next frame in the sequence
def next_frame(last_step, last_frame, column):
	# define the scope of the next step
	lower = max(0, last_step-1)
	upper = min(last_frame.shape[0]-1, last_step+1)
	# choose the row index for the next step
	step = randint(lower, upper)
	# copy the prior frame
	frame = last_frame.copy()
	# add the new step
	frame[step, column] = 1
	return frame, step

# generate a sequence of frames of a dot moving across an image
def build_frames(size):
	frames = list()
	# create the first frame
	frame = zeros((size,size))
	step = randint(0, size-1)
	# decide if we are heading left or right
	right = 1 if random() < 0.5 else 0
	col = 0 if right else size-1
	frame[step, col] = 1
	frames.append(frame)
	# create all remaining frames
	for i in range(1, size):
		col = i if right else size-1-i
		frame, step = next_frame(step, frame, col)
		frames.append(frame)
	return frames, right

# generate sequence of frames
size = 5
frames, right = build_frames(size)
# plot all frames
pyplot.figure()
for i in range(size):
	# create a gray scale subplot for each frame
	pyplot.subplot(1, size, i+1)
	pyplot.imshow(frames[i], cmap='Greys')
	# turn of the scale to make it clearer
	ax = pyplot.gca()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
# show the plot
pyplot.show()
```

---
