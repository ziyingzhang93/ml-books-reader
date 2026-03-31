# Python 深度学习 / Deep Learning with Python
## Chapter 07

---

### Keras First Network



---

### Make Predict

# 11 — Make Predict / 11 Make Predict

**Chapter 07 — File 2 of 2 / 第07章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **first neural network with keras make predictions**.

本脚本演示 **first neural network with keras make predictions**。

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
## Step 1 — first neural network with keras make predictions

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
```

---
## Step 2 — load the dataset

```python
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
```

---
## Step 3 — split into input (X) and output (y) variables

```python
X = dataset[:,0:8]
y = dataset[:,8]
```

---
## Step 4 — define the keras model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(8,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
```

---
## Step 5 — compile the keras model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 6 — fit the keras model on the dataset

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=150, batch_size=10, verbose=0)
```

---
## Step 7 — make class predictions with the model

```python
# 转换数据类型 / Convert data type
predictions = (model.predict(X) > 0.5).astype(int)
```

---
## Step 8 — summarize the first 5 cases

```python
# 生成整数序列 / Generate integer sequence
for i in range(5):
    # 打印输出 / Print output
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
```

---
## Learning Notes / 学习笔记

- **概念**: first neural network with keras make predictions 是机器学习中的常用技术。  
  *first neural network with keras make predictions is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
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
# Make Predict / 11 Make Predict
# Complete Code / 完整代码
# ===============================

# first neural network with keras make predictions
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
# define the keras model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(8,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
# 训练模型 / Train the model
model.fit(X, y, epochs=150, batch_size=10, verbose=0)
# make class predictions with the model
# 转换数据类型 / Convert data type
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
# 生成整数序列 / Generate integer sequence
for i in range(5):
    # 打印输出 / Print output
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
```

---

### Chapter Summary / 章节总结

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **2 code files** demonstrating chapter 07.

本章包含 **2 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `07_keras_first_network.ipynb` — Keras First Network
  2. `11_make_predict.ipynb` — Make Predict

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---
