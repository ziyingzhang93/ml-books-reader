# Python 深度学习 / Deep Learning with Python
## Chapter 17

---

### Plot History

# 02 — Plot History / 02 Plot History

**Chapter 17 — File 1 of 1 / 第17章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Visualize training history**.

本脚本演示 **Visualize training history**。

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
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Visualize training history

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
```

---
## Step 2 — load pima indians dataset

```python
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
```

---
## Step 3 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 4 — create model

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
## Step 5 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 6 — Fit the model

```python
# 训练模型 / Train the model
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
```

---
## Step 7 — list all data in history

```python
# 打印输出 / Print output
print(history.history.keys())
```

---
## Step 8 — summarize history for accuracy

```python
# 绘制折线图 / Draw line plot
plt.plot(history.history['accuracy'])
# 绘制折线图 / Draw line plot
plt.plot(history.history['val_accuracy'])
# 设置图表标题 / Set chart title
plt.title('model accuracy')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('accuracy')
# 设置X轴标签 / Set X-axis label
plt.xlabel('epoch')
# 显示图例 / Show legend
plt.legend(['train', 'test'], loc='upper left')
# 显示图表 / Display the plot
plt.show()
```

---
## Step 9 — summarize history for loss

```python
# 绘制折线图 / Draw line plot
plt.plot(history.history['loss'])
# 绘制折线图 / Draw line plot
plt.plot(history.history['val_loss'])
# 设置图表标题 / Set chart title
plt.title('model loss')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('loss')
# 设置X轴标签 / Set X-axis label
plt.xlabel('epoch')
# 显示图例 / Show legend
plt.legend(['train', 'test'], loc='upper left')
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Visualize training history 是机器学习中的常用技术。  
  *Visualize training history is a common technique in machine learning.*

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
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot History / 02 Plot History
# Complete Code / 完整代码
# ===============================

# Visualize training history
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(8,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# Compile model
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
# 训练模型 / Train the model
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
# 打印输出 / Print output
print(history.history.keys())
# summarize history for accuracy
# 绘制折线图 / Draw line plot
plt.plot(history.history['accuracy'])
# 绘制折线图 / Draw line plot
plt.plot(history.history['val_accuracy'])
# 设置图表标题 / Set chart title
plt.title('model accuracy')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('accuracy')
# 设置X轴标签 / Set X-axis label
plt.xlabel('epoch')
# 显示图例 / Show legend
plt.legend(['train', 'test'], loc='upper left')
# 显示图表 / Display the plot
plt.show()
# summarize history for loss
# 绘制折线图 / Draw line plot
plt.plot(history.history['loss'])
# 绘制折线图 / Draw line plot
plt.plot(history.history['val_loss'])
# 设置图表标题 / Set chart title
plt.title('model loss')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('loss')
# 设置X轴标签 / Set X-axis label
plt.xlabel('epoch')
# 显示图例 / Show legend
plt.legend(['train', 'test'], loc='upper left')
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 17 Summary / 第17章总结

## Theme / 主题: Chapter 17 / Chapter 17

This chapter contains **1 code files** demonstrating chapter 17.

本章包含 **1 个代码文件**，演示Chapter 17。

---
## Evolution / 演化路线

  1. `02_plot_history.ipynb` — Plot History

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 17) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 17）是机器学习流水线中的基础构建块。

---
