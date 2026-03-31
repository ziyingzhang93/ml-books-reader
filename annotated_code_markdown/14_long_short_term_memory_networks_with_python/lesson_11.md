# LSTM 网络实战 / LSTM Networks with Python
## Lesson 11

---

### Chapter Summary / 章节总结

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Lesson 11 / Lesson 11

This chapter contains **4 code files** demonstrating lesson 11.

本章包含 **4 个代码文件**，演示Lesson 11。

---
## Evolution / 演化路线

  1. `generative_lstm.ipynb` — Generative Lstm
  2. `problem_plot.ipynb` — Problem Plot
  3. `problem_random_rect.ipynb` — Problem Random Rect
  4. `problem_sequence.ipynb` — Problem Sequence

---
## ML Relevance / ML 关联

The techniques in this chapter (Lesson 11) are fundamental building blocks in machine learning pipelines.

本章技术（Lesson 11）是机器学习流水线中的基础构建块。

---

### Generative Lstm

# 01 — Generative Lstm / LSTM 网络

**Chapter 11 — File 1 of 4 / 第11章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **generate a rectangle with random width and height**.

本脚本演示 **generate a rectangle with random width and height**。

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
from random import random
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.patches import PathPatch
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.path import Path
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
```

---
## Step 2 — generate a rectangle with random width and height

```python
def random_rectangle():
	width, height = random(), random()
	points = list()
```

---
## Step 3 — bottom left

```python
# 添加元素到列表末尾 / Append element to list end
points.append([0.0, 0.0])
```

---
## Step 4 — bottom right

```python
# 添加元素到列表末尾 / Append element to list end
points.append([width, 0.0])
```

---
## Step 5 — top right

```python
# 添加元素到列表末尾 / Append element to list end
points.append([width, height])
```

---
## Step 6 — top left

```python
# 添加元素到列表末尾 / Append element to list end
points.append([0.0, height])
	return points
```

---
## Step 7 — plot a rectangle

```python
def plot_rectangle(rect):
```

---
## Step 8 — close the rectangle path

```python
# 添加元素到列表末尾 / Append element to list end
rect.append(rect[0])
```

---
## Step 9 — define path

```python
codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
	path = Path(rect, codes)
	axis = pyplot.gca()
	patch = PathPatch(path)
```

---
## Step 10 — add shape to plot

```python
axis.add_patch(patch)
	axis.set_xlim(-0.1,1.1)
	axis.set_ylim(-0.1,1.1)
	pyplot.show()
```

---
## Step 11 — generate input and output sequences for one random rectangle

```python
def get_samples():
```

---
## Step 12 — generate rectangle

```python
rect = random_rectangle()
	X, y = list(), list()
```

---
## Step 13 — create input output pairs for each coordinate

```python
# 获取长度 / Get length
for i in range(1, len(rect)):
  # 添加元素到列表末尾 / Append element to list end
		X.append(rect[i-1])
  # 添加元素到列表末尾 / Append element to list end
		y.append(rect[i])
```

---
## Step 14 — convert input sequence shape to have 1 time step and 2 features

```python
X, y = array(X), array(y)
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	X = X.reshape((X.shape[0], 1, 2))
	return X, y
```

---
## Step 15 — use a fit LSTM model to generate a new rectangle from scratch

```python
def generate_rectangle(model):
	rect = list()
```

---
## Step 16 — use [0,0] to seed the generation process

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
last = array([0.0,0.0]).reshape((1, 1, 2))
 # 添加元素到列表末尾 / Append element to list end
	rect.append([[y for y in x] for x in last[0]][0])
```

---
## Step 17 — generate the remaining 3 coordinates

```python
# 生成整数序列 / Generate integer sequence
for _ in range(3):
```

---
## Step 18 — predict the next coordinate

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(last, verbose=0)
```

---
## Step 19 — use this output as input for the next prediction

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
last = yhat.reshape((1, 1, 2))
```

---
## Step 20 — store coordinate

```python
# 添加元素到列表末尾 / Append element to list end
rect.append([[y for y in x] for x in last[0]][0])
	return rect
```

---
## Step 21 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(10, input_shape=(1, 2)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(2, activation='linear'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mae', optimizer='adam')
model.summary()
```

---
## Step 22 — fit model

```python
# 生成整数序列 / Generate integer sequence
for i in range(25000):
	X, y = get_samples()
 # 训练模型 / Train the model
	model.fit(X, y, epochs=1, verbose=2, shuffle=False)
```

---
## Step 23 — generate new shapes from scratch

```python
rect = generate_rectangle(model)
plot_rectangle(rect)
```

---
## Learning Notes / 学习笔记

- **概念**: generate a rectangle with random width and height 是机器学习中的常用技术。  
  *generate a rectangle with random width and height is a common technique in machine learning.*

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
| `matplotlib` | 绑图库 | Plotting library |
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
# Generative Lstm / LSTM 网络
# Complete Code / 完整代码
# ===============================

from random import random
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.patches import PathPatch
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.path import Path
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense

# generate a rectangle with random width and height
def random_rectangle():
	width, height = random(), random()
	points = list()
	# bottom left
 # 添加元素到列表末尾 / Append element to list end
	points.append([0.0, 0.0])
	# bottom right
 # 添加元素到列表末尾 / Append element to list end
	points.append([width, 0.0])
	# top right
 # 添加元素到列表末尾 / Append element to list end
	points.append([width, height])
	# top left
 # 添加元素到列表末尾 / Append element to list end
	points.append([0.0, height])
	return points

# plot a rectangle
def plot_rectangle(rect):
	# close the rectangle path
 # 添加元素到列表末尾 / Append element to list end
	rect.append(rect[0])
	# define path
	codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
	path = Path(rect, codes)
	axis = pyplot.gca()
	patch = PathPatch(path)
	# add shape to plot
	axis.add_patch(patch)
	axis.set_xlim(-0.1,1.1)
	axis.set_ylim(-0.1,1.1)
	pyplot.show()

# generate input and output sequences for one random rectangle
def get_samples():
	# generate rectangle
	rect = random_rectangle()
	X, y = list(), list()
	# create input output pairs for each coordinate
 # 获取长度 / Get length
	for i in range(1, len(rect)):
  # 添加元素到列表末尾 / Append element to list end
		X.append(rect[i-1])
  # 添加元素到列表末尾 / Append element to list end
		y.append(rect[i])
	# convert input sequence shape to have 1 time step and 2 features
	X, y = array(X), array(y)
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	X = X.reshape((X.shape[0], 1, 2))
	return X, y

# use a fit LSTM model to generate a new rectangle from scratch
def generate_rectangle(model):
	rect = list()
	# use [0,0] to seed the generation process
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	last = array([0.0,0.0]).reshape((1, 1, 2))
 # 添加元素到列表末尾 / Append element to list end
	rect.append([[y for y in x] for x in last[0]][0])
	# generate the remaining 3 coordinates
 # 生成整数序列 / Generate integer sequence
	for _ in range(3):
		# predict the next coordinate
  # 用模型做预测 / Make predictions with model
		yhat = model.predict(last, verbose=0)
		# use this output as input for the next prediction
  # 改变数组形状（不改变数据） / Reshape array (data unchanged)
		last = yhat.reshape((1, 1, 2))
		# store coordinate
  # 添加元素到列表末尾 / Append element to list end
		rect.append([[y for y in x] for x in last[0]][0])
	return rect

# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(10, input_shape=(1, 2)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(2, activation='linear'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mae', optimizer='adam')
model.summary()

# fit model
# 生成整数序列 / Generate integer sequence
for i in range(25000):
	X, y = get_samples()
 # 训练模型 / Train the model
	model.fit(X, y, epochs=1, verbose=2, shuffle=False)

# generate new shapes from scratch
rect = generate_rectangle(model)
plot_rectangle(rect)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Problem Plot

# 01 — Problem Plot / Problem Plot

**Chapter 11 — File 2 of 4 / 第11章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **generate a rectangle with random width and height**.

本脚本演示 **generate a rectangle with random width and height**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
from random import random
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.patches import PathPatch
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.path import Path
```

---
## Step 2 — generate a rectangle with random width and height

```python
def random_rectangle():
	width, height = random(), random()
	points = list()
```

---
## Step 3 — bottom left

```python
# 添加元素到列表末尾 / Append element to list end
points.append([0.0, 0.0])
```

---
## Step 4 — bottom right

```python
# 添加元素到列表末尾 / Append element to list end
points.append([width, 0.0])
```

---
## Step 5 — top right

```python
# 添加元素到列表末尾 / Append element to list end
points.append([width, height])
```

---
## Step 6 — top left

```python
# 添加元素到列表末尾 / Append element to list end
points.append([0.0, height])
	return points
```

---
## Step 7 — plot a rectangle

```python
def plot_rectangle(rect):
```

---
## Step 8 — close the rectangle path

```python
# 添加元素到列表末尾 / Append element to list end
rect.append(rect[0])
```

---
## Step 9 — define path

```python
codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
	path = Path(rect, codes)
	axis = pyplot.gca()
	patch = PathPatch(path)
```

---
## Step 10 — add shape to plot

```python
axis.add_patch(patch)
	axis.set_xlim(-0.1,1.1)
	axis.set_ylim(-0.1,1.1)
	pyplot.show()

rect = random_rectangle()
plot_rectangle(rect)
```

---
## Learning Notes / 学习笔记

- **概念**: generate a rectangle with random width and height 是机器学习中的常用技术。  
  *generate a rectangle with random width and height is a common technique in machine learning.*

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
# Problem Plot / Problem Plot
# Complete Code / 完整代码
# ===============================

from random import random
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.patches import PathPatch
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.path import Path

# generate a rectangle with random width and height
def random_rectangle():
	width, height = random(), random()
	points = list()
	# bottom left
 # 添加元素到列表末尾 / Append element to list end
	points.append([0.0, 0.0])
	# bottom right
 # 添加元素到列表末尾 / Append element to list end
	points.append([width, 0.0])
	# top right
 # 添加元素到列表末尾 / Append element to list end
	points.append([width, height])
	# top left
 # 添加元素到列表末尾 / Append element to list end
	points.append([0.0, height])
	return points

# plot a rectangle
def plot_rectangle(rect):
	# close the rectangle path
 # 添加元素到列表末尾 / Append element to list end
	rect.append(rect[0])
	# define path
	codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
	path = Path(rect, codes)
	axis = pyplot.gca()
	patch = PathPatch(path)
	# add shape to plot
	axis.add_patch(patch)
	axis.set_xlim(-0.1,1.1)
	axis.set_ylim(-0.1,1.1)
	pyplot.show()

rect = random_rectangle()
plot_rectangle(rect)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Problem Random Rect

# 01 — Problem Random Rect / Problem Random Rect

**Chapter 11 — File 3 of 4 / 第11章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **generate a rectangle with random width and height**.

本脚本演示 **generate a rectangle with random width and height**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from random import random
```

---
## Step 2 — generate a rectangle with random width and height

```python
def random_rectangle():
	width, height = random(), random()
	points = list()
```

---
## Step 3 — bottom left

```python
# 添加元素到列表末尾 / Append element to list end
points.append([0.0, 0.0])
```

---
## Step 4 — bottom right

```python
# 添加元素到列表末尾 / Append element to list end
points.append([width, 0.0])
```

---
## Step 5 — top right

```python
# 添加元素到列表末尾 / Append element to list end
points.append([width, height])
```

---
## Step 6 — top left

```python
# 添加元素到列表末尾 / Append element to list end
points.append([0.0, height])
	return points

rect = random_rectangle()
# 打印输出 / Print output
print(rect)
```

---
## Learning Notes / 学习笔记

- **概念**: generate a rectangle with random width and height 是机器学习中的常用技术。  
  *generate a rectangle with random width and height is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Problem Random Rect / Problem Random Rect
# Complete Code / 完整代码
# ===============================

from random import random

# generate a rectangle with random width and height
def random_rectangle():
	width, height = random(), random()
	points = list()
	# bottom left
 # 添加元素到列表末尾 / Append element to list end
	points.append([0.0, 0.0])
	# bottom right
 # 添加元素到列表末尾 / Append element to list end
	points.append([width, 0.0])
	# top right
 # 添加元素到列表末尾 / Append element to list end
	points.append([width, height])
	# top left
 # 添加元素到列表末尾 / Append element to list end
	points.append([0.0, height])
	return points

rect = random_rectangle()
# 打印输出 / Print output
print(rect)
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Problem Sequence

# 01 — Problem Sequence / Problem Sequence

**Chapter 11 — File 4 of 4 / 第11章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **generate a rectangle with random width and height**.

本脚本演示 **generate a rectangle with random width and height**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from random import random
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
```

---
## Step 2 — generate a rectangle with random width and height

```python
def random_rectangle():
	width, height = random(), random()
	points = list()
```

---
## Step 3 — bottom left

```python
# 添加元素到列表末尾 / Append element to list end
points.append([0.0, 0.0])
```

---
## Step 4 — bottom right

```python
# 添加元素到列表末尾 / Append element to list end
points.append([width, 0.0])
```

---
## Step 5 — top right

```python
# 添加元素到列表末尾 / Append element to list end
points.append([width, height])
```

---
## Step 6 — top left

```python
# 添加元素到列表末尾 / Append element to list end
points.append([0.0, height])
	return points
```

---
## Step 7 — generate input and output sequences for one random rectangle

```python
def get_samples():
```

---
## Step 8 — generate rectangle

```python
rect = random_rectangle()
	X, y = list(), list()
```

---
## Step 9 — create input output pairs for each coordinate

```python
# 获取长度 / Get length
for i in range(1, len(rect)):
  # 添加元素到列表末尾 / Append element to list end
		X.append(rect[i-1])
  # 添加元素到列表末尾 / Append element to list end
		y.append(rect[i])
```

---
## Step 10 — convert input sequence shape to have 1 time step and 2 features

```python
X, y = array(X), array(y)
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	X = X.reshape((X.shape[0], 1, 2))
	return X, y

X, y = get_samples()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for i in range(X.shape[0]):
 # 打印输出 / Print output
	print(X[i][0], '=>', y[i])
```

---
## Learning Notes / 学习笔记

- **概念**: generate a rectangle with random width and height 是机器学习中的常用技术。  
  *generate a rectangle with random width and height is a common technique in machine learning.*

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# generate a rectangle with random width and height
def random_rectangle():
	width, height = random(), random()
	points = list()
	# bottom left
 # 添加元素到列表末尾 / Append element to list end
	points.append([0.0, 0.0])
	# bottom right
 # 添加元素到列表末尾 / Append element to list end
	points.append([width, 0.0])
	# top right
 # 添加元素到列表末尾 / Append element to list end
	points.append([width, height])
	# top left
 # 添加元素到列表末尾 / Append element to list end
	points.append([0.0, height])
	return points

# generate input and output sequences for one random rectangle
def get_samples():
	# generate rectangle
	rect = random_rectangle()
	X, y = list(), list()
	# create input output pairs for each coordinate
 # 获取长度 / Get length
	for i in range(1, len(rect)):
  # 添加元素到列表末尾 / Append element to list end
		X.append(rect[i-1])
  # 添加元素到列表末尾 / Append element to list end
		y.append(rect[i])
	# convert input sequence shape to have 1 time step and 2 features
	X, y = array(X), array(y)
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	X = X.reshape((X.shape[0], 1, 2))
	return X, y

X, y = get_samples()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for i in range(X.shape[0]):
 # 打印输出 / Print output
	print(X[i][0], '=>', y[i])
```

---
