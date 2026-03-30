# GAN
## Chapter 02

---

### Functional Mlp

# 1 — Functional Mlp / 1 Functional Mlp

**Chapter 02 — File 1 of 3 / 第02章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **example of a multilayer perceptron**.

本脚本演示 **example of a multilayer perceptron**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of a multilayer perceptron

```python
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
visible = Input(shape=(10,))
hidden1 = Dense(10, activation='relu')(visible)
hidden2 = Dense(20, activation='relu')(hidden1)
hidden3 = Dense(10, activation='relu')(hidden2)
output = Dense(1, activation='sigmoid')(hidden3)
model = Model(inputs=visible, outputs=output)
```

---
## Step 2 — summarize layers

```python
model.summary()
```

---
## Step 3 — plot graph

```python
plot_model(model, to_file='multilayer_perceptron_graph.png')
```

---
## Learning Notes / 学习笔记

- **概念**: example of a multilayer perceptron 是机器学习中的常用技术。  
  *example of a multilayer perceptron is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Functional Mlp / 1 Functional Mlp
# Complete Code / 完整代码
# ===============================

# example of a multilayer perceptron
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
visible = Input(shape=(10,))
hidden1 = Dense(10, activation='relu')(visible)
hidden2 = Dense(20, activation='relu')(hidden1)
hidden3 = Dense(10, activation='relu')(hidden2)
output = Dense(1, activation='sigmoid')(hidden3)
model = Model(inputs=visible, outputs=output)
# summarize layers
model.summary()
# plot graph
plot_model(model, to_file='multilayer_perceptron_graph.png')
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Functional Cnn

# 2 — Functional Cnn / 卷积神经网络

**Chapter 02 — File 2 of 3 / 第02章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **example of a convolutional neural network**.

本脚本演示 **example of a convolutional neural network**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of a convolutional neural network

```python
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
visible = Input(shape=(64,64,1))
conv1 = Conv2D(32, (4,4), activation='relu')(visible)
pool1 = MaxPooling2D()(conv1)
conv2 = Conv2D(16, (4,4), activation='relu')(pool1)
pool2 = MaxPooling2D()(conv2)
flat1 = Flatten()(pool2)
hidden1 = Dense(10, activation='relu')(flat1)
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=visible, outputs=output)
```

---
## Step 2 — summarize layers

```python
model.summary()
```

---
## Step 3 — plot graph

```python
plot_model(model, to_file='convolutional_neural_network.png')
```

---
## Learning Notes / 学习笔记

- **概念**: example of a convolutional neural network 是机器学习中的常用技术。  
  *example of a convolutional neural network is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Functional Cnn / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# example of a convolutional neural network
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
visible = Input(shape=(64,64,1))
conv1 = Conv2D(32, (4,4), activation='relu')(visible)
pool1 = MaxPooling2D()(conv1)
conv2 = Conv2D(16, (4,4), activation='relu')(pool1)
pool2 = MaxPooling2D()(conv2)
flat1 = Flatten()(pool2)
hidden1 = Dense(10, activation='relu')(flat1)
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=visible, outputs=output)
# summarize layers
model.summary()
# plot graph
plot_model(model, to_file='convolutional_neural_network.png')
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Functional Rnn

# 3 — Functional Rnn / 循环神经网络

**Chapter 02 — File 3 of 3 / 第02章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **example of a recurrent neural network**.

本脚本演示 **example of a recurrent neural network**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of a recurrent neural network

```python
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
visible = Input(shape=(100,1))
hidden1 = LSTM(10)(visible)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model(inputs=visible, outputs=output)
```

---
## Step 2 — summarize layers

```python
model.summary()
```

---
## Step 3 — plot graph

```python
plot_model(model, to_file='recurrent_neural_network.png')
```

---
## Learning Notes / 学习笔记

- **概念**: example of a recurrent neural network 是机器学习中的常用技术。  
  *example of a recurrent neural network is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Functional Rnn / 循环神经网络
# Complete Code / 完整代码
# ===============================

# example of a recurrent neural network
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
visible = Input(shape=(100,1))
hidden1 = LSTM(10)(visible)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model(inputs=visible, outputs=output)
# summarize layers
model.summary()
# plot graph
plot_model(model, to_file='recurrent_neural_network.png')
```

---

### Chapter Summary

# Chapter 02 Summary / 第02章总结

## Theme / 主题: Chapter 02 / Chapter 02

This chapter contains **3 code files** demonstrating chapter 02.

本章包含 **3 个代码文件**，演示Chapter 02。

---
## Evolution / 演化路线

  1. `1_functional_mlp.ipynb` — Functional Mlp
  2. `2_functional_cnn.ipynb` — Functional Cnn
  3. `3_functional_rnn.ipynb` — Functional Rnn

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 02) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 02）是机器学习流水线中的基础构建块。

---
