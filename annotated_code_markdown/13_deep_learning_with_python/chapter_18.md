# Python深度学习
## Chapter 18

---

### Sigmoid

# 01 — Sigmoid / 01 Sigmoid

**Chapter 18 — File 1 of 4 / 第18章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Sigmoid**.

本脚本演示 **01 Sigmoid**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import tensorflow as tf
from tensorflow.keras.activations import sigmoid

input_array = tf.constant([-1, 0, 1], dtype=tf.float32)
print(sigmoid(input_array))
```

---
## Learning Notes / 学习笔记

- **概念**: Sigmoid 是机器学习中的常用技术。  
  *Sigmoid is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sigmoid / 01 Sigmoid
# Complete Code / 完整代码
# ===============================

import tensorflow as tf
from tensorflow.keras.activations import sigmoid

input_array = tf.constant([-1, 0, 1], dtype=tf.float32)
print(sigmoid(input_array))
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Tanh

# 02 — Tanh / 02 Tanh

**Chapter 18 — File 2 of 4 / 第18章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Tanh**.

本脚本演示 **02 Tanh**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import tensorflow as tf
from tensorflow.keras.activations import tanh

input_array = tf.constant([-1, 0, 1], dtype=tf.float32)
print(tanh(input_array))
```

---
## Learning Notes / 学习笔记

- **概念**: Tanh 是机器学习中的常用技术。  
  *Tanh is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Tanh / 02 Tanh
# Complete Code / 完整代码
# ===============================

import tensorflow as tf
from tensorflow.keras.activations import tanh

input_array = tf.constant([-1, 0, 1], dtype=tf.float32)
print(tanh(input_array))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Relu

# 03 — Relu / 03 Relu

**Chapter 18 — File 3 of 4 / 第18章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Relu**.

本脚本演示 **03 Relu**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import tensorflow as tf
from tensorflow.keras.activations import relu

input_array = tf.constant([-1, 0, 1], dtype=tf.float32)
print(relu(input_array))
```

---
## Learning Notes / 学习笔记

- **概念**: Relu 是机器学习中的常用技术。  
  *Relu is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Relu / 03 Relu
# Complete Code / 完整代码
# ===============================

import tensorflow as tf
from tensorflow.keras.activations import relu

input_array = tf.constant([-1, 0, 1], dtype=tf.float32)
print(relu(input_array))
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Lenet5

# 06 — Lenet5 / 06 Lenet5

**Chapter 18 — File 4 of 4 / 第18章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Lenet5**.

本脚本演示 **06 Lenet5**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model

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
```

---
## Step 1 — Step 1

```python
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Flatten, \
                                    Conv2D, BatchNormalization, MaxPool2D
from tensorflow.keras.models import Model

(trainX, trainY), (testX, testY) = keras.datasets.cifar10.load_data()

input_layer = Input(shape=(32,32,3,))
x = Conv2D(6, (5,5), padding="same", activation="relu")(input_layer)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(16, (5,5), padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Conv2D(120, (5,5), padding="same", activation="relu")(x)
x = Flatten()(x)
x = Dense(units=84, activation="relu")(x)
x = Dense(units=10, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=x)
model.summary()

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics="acc")

history = model.fit(x=trainX, y=trainY, batch_size=256, epochs=10,
                    validation_data=(testX, testY))
```

---
## Learning Notes / 学习笔记

- **概念**: Lenet5 是机器学习中的常用技术。  
  *Lenet5 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lenet5 / 06 Lenet5
# Complete Code / 完整代码
# ===============================

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Flatten, \
                                    Conv2D, BatchNormalization, MaxPool2D
from tensorflow.keras.models import Model

(trainX, trainY), (testX, testY) = keras.datasets.cifar10.load_data()

input_layer = Input(shape=(32,32,3,))
x = Conv2D(6, (5,5), padding="same", activation="relu")(input_layer)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(16, (5,5), padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Conv2D(120, (5,5), padding="same", activation="relu")(x)
x = Flatten()(x)
x = Dense(units=84, activation="relu")(x)
x = Dense(units=10, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=x)
model.summary()

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics="acc")

history = model.fit(x=trainX, y=trainY, batch_size=256, epochs=10,
                    validation_data=(testX, testY))
```

---

### Chapter Summary

# Chapter 18 Summary / 第18章总结

## Theme / 主题: Chapter 18 / Chapter 18

This chapter contains **4 code files** demonstrating chapter 18.

本章包含 **4 个代码文件**，演示Chapter 18。

---
## Evolution / 演化路线

  1. `01_sigmoid.ipynb` — Sigmoid
  2. `02_tanh.ipynb` — Tanh
  3. `03_relu.ipynb` — Relu
  4. `06_lenet5.ipynb` — Lenet5

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 18) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 18）是机器学习流水线中的基础构建块。

---
