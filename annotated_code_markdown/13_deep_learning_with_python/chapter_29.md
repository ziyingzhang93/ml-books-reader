# Python 深度学习 / Deep Learning with Python
## Chapter 29

---

### Mlp

# 14 — Mlp / 14 Mlp

**Chapter 29 — File 1 of 2 / 第29章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **MLP for the IMDB problem**.

本脚本演示 **MLP for the IMDB problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
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
## Step 1 — MLP for the IMDB problem

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import imdb
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Embedding
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing import sequence
```

---
## Step 2 — load the dataset but only keep the top n words, zero the rest

```python
top_words = 5000
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
```

---
## Step 3 — create the model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Embedding(top_words, 32, input_length=max_words))
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(250, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

---
## Step 4 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=2, batch_size=128, verbose=1)
```

---
## Step 5 — Final evaluation of the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("Accuracy: %.2f%%" % (scores[1]*100))
```

---
## Learning Notes / 学习笔记

- **概念**: MLP for the IMDB problem 是机器学习中的常用技术。  
  *MLP for the IMDB problem is a common technique in machine learning.*

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
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mlp / 14 Mlp
# Complete Code / 完整代码
# ===============================

# MLP for the IMDB problem
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import imdb
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Embedding
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing import sequence
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
# create the model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Embedding(top_words, 32, input_length=max_words))
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(250, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Fit the model
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=2, batch_size=128, verbose=1)
# Final evaluation of the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("Accuracy: %.2f%%" % (scores[1]*100))
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Cnn



---

### Chapter Summary / 章节总结

# Chapter 29 Summary / 第29章总结

## Theme / 主题: Chapter 29 / Chapter 29

This chapter contains **2 code files** demonstrating chapter 29.

本章包含 **2 个代码文件**，演示Chapter 29。

---
## Evolution / 演化路线

  1. `14_mlp.ipynb` — Mlp
  2. `19_cnn.ipynb` — Cnn

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 29) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 29）是机器学习流水线中的基础构建块。

---
