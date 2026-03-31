# OpenCV 机器学习 / Machine Learning in OpenCV
## Chapter 22

---

### Load Data



---

### Model



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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import EarlyStopping
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
```

---
## Step 2 — Load MNIST data

```python
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X_train.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(y_train.shape)
```

---
## Step 3 — LeNet5 model

```python
model = Sequential([
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(6, (5,5), input_shape=(28,28,1), padding="same", activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(16, (5,5), activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(120, (5,5), activation="tanh"),
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    Flatten(),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(84, activation="tanh"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(10, activation="softmax")
])
```

---
## Step 4 — Reshape data to shape of (n_sample, height, width, n_channel)

```python
# 转换数据类型 / Convert data type
X_train = np.expand_dims(X_train, axis=3).astype('float32')
# 转换数据类型 / Convert data type
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
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# 早停：验证集不再提升时自动停止训练 / EarlyStopping: stop when validation stops improving
earlystopping = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32,
          callbacks=[earlystopping])

# 保存模型到文件 / Save model to file
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

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import EarlyStopping
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical

# Load MNIST data
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X_train.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(y_train.shape)

# LeNet5 model
model = Sequential([
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(6, (5,5), input_shape=(28,28,1), padding="same", activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(16, (5,5), activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(120, (5,5), activation="tanh"),
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    Flatten(),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(84, activation="tanh"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(10, activation="softmax")
])

# Reshape data to shape of (n_sample, height, width, n_channel)
# 转换数据类型 / Convert data type
X_train = np.expand_dims(X_train, axis=3).astype('float32')
# 转换数据类型 / Convert data type
X_test = np.expand_dims(X_test, axis=3).astype('float32')

# One-hot encode the output
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Training
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# 早停：验证集不再提升时自动停止训练 / EarlyStopping: stop when validation stops improving
earlystopping = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32,
          callbacks=[earlystopping])

# 保存模型到文件 / Save model to file
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
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
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
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
correct = 0
wrong = 0
# 获取长度 / Get length
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
    # 找最大值的索引位置 / Find index of maximum value
    prediction = np.argmax(output)
    if prediction == label:
        correct += 1
    else:
        wrong += 1

# 打印输出 / Print output
print("count of test samples:", len(X_test))
# 打印输出 / Print output
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

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist

# Load the frozen model in OpenCV
net = cv2.dnn.readNetFromONNX('lenet5.onnx')

# Prepare input image
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
correct = 0
wrong = 0
# 获取长度 / Get length
for i in range(len(X_test)):
    img = X_test[i]
    label = y_test[i]

    blob = cv2.dnn.blobFromImage(img, 1.0, (28, 28))

    # Run inference
    net.setInput(blob)
    output = net.forward()
    # 找最大值的索引位置 / Find index of maximum value
    prediction = np.argmax(output)
    if prediction == label:
        correct += 1
    else:
        wrong += 1

# 打印输出 / Print output
print("count of test samples:", len(X_test))
# 打印输出 / Print output
print("accuracy:", (correct/(correct+wrong)))
```

---

### Chapter Summary / 章节总结

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
