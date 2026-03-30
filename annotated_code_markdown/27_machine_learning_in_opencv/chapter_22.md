# OpenCV ML
## Chapter 22

---

### Train Lenet5

# 05 — Train Lenet5 / 05 Train Lenet5

**Chapter 22 — File 3 of 4 / 第22章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **!/usr/bin/env python**.

本脚本演示 **!/usr/bin/env python**。

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
```

---
## Step 1 — !/usr/bin/env python

```python
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
```

---
## Step 2 — Load MNIST data

```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)
```

---
## Step 3 — LeNet5 model

```python
model = Sequential([
    Conv2D(6, (5,5), input_shape=(28,28,1), padding="same", activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    Conv2D(16, (5,5), activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    Conv2D(120, (5,5), activation="tanh"),
    Flatten(),
    Dense(84, activation="tanh"),
    Dense(10, activation="softmax")
])
```

---
## Step 4 — Reshape data to shape of (n_sample, height, width, n_channel)

```python
X_train = np.expand_dims(X_train, axis=3).astype('float32')
X_test = np.expand_dims(X_test, axis=3).astype('float32')
```

---
## Step 5 — One-hot encode the output

```python
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

---
## Step 6 — Training

```python
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
earlystopping = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32,
          callbacks=[earlystopping])

model.save("lenet5.h5")
```

---
## Learning Notes / 学习笔记

- **概念**: !/usr/bin/env python 是机器学习中的常用技术。  
  *!/usr/bin/env python is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `EarlyStopping` | 早停：验证集不再提升时停止训练 | Stop when validation stops improving |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Lenet5 / 05 Train Lenet5
# Complete Code / 完整代码
# ===============================

#!/usr/bin/env python

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)

# LeNet5 model
model = Sequential([
    Conv2D(6, (5,5), input_shape=(28,28,1), padding="same", activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    Conv2D(16, (5,5), activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    Conv2D(120, (5,5), activation="tanh"),
    Flatten(),
    Dense(84, activation="tanh"),
    Dense(10, activation="softmax")
])

# Reshape data to shape of (n_sample, height, width, n_channel)
X_train = np.expand_dims(X_train, axis=3).astype('float32')
X_test = np.expand_dims(X_test, axis=3).astype('float32')

# One-hot encode the output
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Training
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
earlystopping = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32,
          callbacks=[earlystopping])

model.save("lenet5.h5")
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Test Lenet5

# 09 — Test Lenet5 / 09 Test Lenet5

**Chapter 22 — File 4 of 4 / 第22章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load the frozen model in OpenCV**.

本脚本演示 **Load the frozen model in OpenCV**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 评估模型效果 / Evaluate model performance


---
## Step 1 — Step 1

```python
import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
```

---
## Step 2 — Load the frozen model in OpenCV

```python
net = cv2.dnn.readNetFromONNX('lenet5.onnx')
```

---
## Step 3 — Prepare input image

```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
correct = 0
wrong = 0
for i in range(len(X_test)):
    img = X_test[i]
    label = y_test[i]

    blob = cv2.dnn.blobFromImage(img, 1.0, (28, 28))
```

---
## Step 4 — Run inference

```python
net.setInput(blob)
    output = net.forward()
    prediction = np.argmax(output)
    if prediction == label:
        correct += 1
    else:
        wrong += 1

print("count of test samples:", len(X_test))
print("accuracy:", (correct/(correct+wrong)))
```

---
## Learning Notes / 学习笔记

- **概念**: Load the frozen model in OpenCV 是机器学习中的常用技术。  
  *Load the frozen model in OpenCV is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Test Lenet5 / 09 Test Lenet5
# Complete Code / 完整代码
# ===============================

import numpy as np
import cv2
from tensorflow.keras.datasets import mnist

# Load the frozen model in OpenCV
net = cv2.dnn.readNetFromONNX('lenet5.onnx')

# Prepare input image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
correct = 0
wrong = 0
for i in range(len(X_test)):
    img = X_test[i]
    label = y_test[i]

    blob = cv2.dnn.blobFromImage(img, 1.0, (28, 28))

    # Run inference
    net.setInput(blob)
    output = net.forward()
    prediction = np.argmax(output)
    if prediction == label:
        correct += 1
    else:
        wrong += 1

print("count of test samples:", len(X_test))
print("accuracy:", (correct/(correct+wrong)))
```

---

### Chapter Summary

# Chapter 22 Summary / 第22章总结

## Theme / 主题: Chapter 22 / Chapter 22

This chapter contains **4 code files** demonstrating chapter 22.

本章包含 **4 个代码文件**，演示Chapter 22。

---
## Evolution / 演化路线

  1. `01_load_data.ipynb` — Load Data
  2. `02_model.ipynb` — Model
  3. `05_train_lenet5.ipynb` — Train Lenet5
  4. `09_test_lenet5.ipynb` — Test Lenet5

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 22) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 22）是机器学习流水线中的基础构建块。

---
