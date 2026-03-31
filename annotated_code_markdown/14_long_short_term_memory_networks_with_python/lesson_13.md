# LSTM 网络实战 / LSTM Networks with Python
## Lesson 13

---

### Chapter Summary / 章节总结

# Chapter 13 Summary / 第13章总结

## Theme / 主题: Lesson 13 / Lesson 13

This chapter contains **2 code files** demonstrating lesson 13.

本章包含 **2 个代码文件**，演示Lesson 13。

---
## Evolution / 演化路线

  1. `save_separate_files.ipynb` — Save Separate Files
  2. `save_single_file.ipynb` — Save Single File

---
## ML Relevance / ML 关联

The techniques in this chapter (Lesson 13) are fundamental building blocks in machine learning pipelines.

本章技术（Lesson 13）是机器学习流水线中的基础构建块。

---

### Save Separate Files

# 01 — Save Separate Files / 保存/加载模型

**Chapter 13 — File 1 of 2 / 第13章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **return training data**.

本脚本演示 **return training data**。

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
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import model_from_json
```

---
## Step 2 — return training data

```python
def get_train():
	seq = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]
	seq = array(seq)
	X, y = seq[:, 0], seq[:, 1]
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X = X.reshape((len(X), 1, 1))
	return X, y
```

---
## Step 3 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(10, input_shape=(1,1)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='linear'))
```

---
## Step 4 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mse', optimizer='adam')
```

---
## Step 5 — fit model

```python
X,y = get_train()
# 训练模型 / Train the model
model.fit(X, y, epochs=300, shuffle=False, verbose=0)
```

---
## Step 6 — convert model architecture to JSON format

```python
architecture = model.to_json()
```

---
## Step 7 — save architecture to JSON file

```python
# 打开文件（自动关闭） / Open file (auto-close)
with open('architecture.json', 'wt') as json_file:
    json_file.write(architecture)
```

---
## Step 8 — save weights to hdf5 file

```python
model.save_weights('weights.h5')
```

---
## Step 9 — snip...
later, perhaps run from another script
load architecture from JSON File

```python
json_file = open('architecture.json', 'rt')
architecture = json_file.read()
json_file.close()
```

---
## Step 10 — create model from architecture

```python
model = model_from_json(architecture)
```

---
## Step 11 — load weights from hdf5 file

```python
model.load_weights('weights.h5')
```

---
## Step 12 — make predictions

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: return training data 是机器学习中的常用技术。  
  *return training data is a common technique in machine learning.*

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
# Save Separate Files / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import model_from_json

# return training data
def get_train():
	seq = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]
	seq = array(seq)
	X, y = seq[:, 0], seq[:, 1]
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X = X.reshape((len(X), 1, 1))
	return X, y

# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(10, input_shape=(1,1)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='linear'))
# compile model
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mse', optimizer='adam')
# fit model
X,y = get_train()
# 训练模型 / Train the model
model.fit(X, y, epochs=300, shuffle=False, verbose=0)
# convert model architecture to JSON format
architecture = model.to_json()
# save architecture to JSON file
# 打开文件（自动关闭） / Open file (auto-close)
with open('architecture.json', 'wt') as json_file:
    json_file.write(architecture)
# save weights to hdf5 file
model.save_weights('weights.h5')

# snip...
# later, perhaps run from another script

# load architecture from JSON File
json_file = open('architecture.json', 'rt')
architecture = json_file.read()
json_file.close()
# create model from architecture
model = model_from_json(architecture)
# load weights from hdf5 file
model.load_weights('weights.h5')
# make predictions
# 用模型做预测 / Make predictions with model
yhat = model.predict(X, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Save Single File

# 01 — Save Single File / 保存/加载模型

**Chapter 13 — File 2 of 2 / 第13章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **return training data**.

本脚本演示 **return training data**。

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
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
```

---
## Step 2 — return training data

```python
def get_train():
	seq = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]
	seq = array(seq)
	X, y = seq[:, 0], seq[:, 1]
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X = X.reshape((len(X), 1, 1))
	return X, y
```

---
## Step 3 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(10, input_shape=(1,1)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='linear'))
```

---
## Step 4 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mse', optimizer='adam')
```

---
## Step 5 — fit model

```python
X,y = get_train()
# 训练模型 / Train the model
model.fit(X, y, epochs=300, shuffle=False, verbose=0)
```

---
## Step 6 — save model to single file

```python
# 保存模型到文件 / Save model to file
model.save('lstm_model.h5')
```

---
## Step 7 — snip...
later, perhaps run from another script
load model from single file

```python
# 从文件加载模型 / Load model from file
model = load_model('lstm_model.h5')
```

---
## Step 8 — make predictions

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: return training data 是机器学习中的常用技术。  
  *return training data is a common technique in machine learning.*

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
# Save Single File / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model

# return training data
def get_train():
	seq = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]
	seq = array(seq)
	X, y = seq[:, 0], seq[:, 1]
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X = X.reshape((len(X), 1, 1))
	return X, y

# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(10, input_shape=(1,1)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='linear'))
# compile model
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mse', optimizer='adam')
# fit model
X,y = get_train()
# 训练模型 / Train the model
model.fit(X, y, epochs=300, shuffle=False, verbose=0)
# save model to single file
# 保存模型到文件 / Save model to file
model.save('lstm_model.h5')

# snip...
# later, perhaps run from another script

# load model from single file
# 从文件加载模型 / Load model from file
model = load_model('lstm_model.h5')
# make predictions
# 用模型做预测 / Make predictions with model
yhat = model.predict(X, verbose=0)
# 打印输出 / Print output
print(yhat)
```

---
