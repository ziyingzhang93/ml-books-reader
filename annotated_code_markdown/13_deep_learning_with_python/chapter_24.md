# Python 深度学习 / Deep Learning with Python
## Chapter 24

---

### Classification



---

### Visualize

# 07 — Visualize / 07 Visualize

**Chapter 24 — File 2 of 2 / 第24章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **rescale image**.

本脚本演示 **rescale image**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.constraints import MaxNorm
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets.cifar10 import load_data


# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = load_data()
```

---
## Step 2 — rescale image

```python
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

model = Sequential([
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(32, (3,3), input_shape=(32, 32, 3), padding="same",
           activation="relu", kernel_constraint=MaxNorm(3)),
    Dropout(0.3),
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(32, (3,3), padding="same",
           activation="relu", kernel_constraint=MaxNorm(3)),
    # 最大池化层（Keras） / Max pooling layer (Keras)
    MaxPooling2D(),
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    Flatten(),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(512, activation="relu", kernel_constraint=MaxNorm(3)),
    Dropout(0.5),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(10, activation="sigmoid")
])
```

---
## Step 3 — Train the network with CIFAR10 dataset

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics="sparse_categorical_accuracy")

# 训练模型 / Train the model
model.fit(X_train_scaled, y_train, epochs=25, batch_size=32,
          validation_data=(X_test_scaled, y_test))
```

---
## Step 4 — Visualize the input image

```python
# 显示图像 / Display image
plt.imshow(X_train_scaled[7])
# 显示图表 / Display the plot
plt.show()
```

---
## Step 5 — Extract output from each layer

```python
extractor = tf.keras.Model(inputs=model.inputs,
                           outputs=[layer.output for layer in model.layers])
# 增加一个维度 / Add a dimension
features = extractor(np.expand_dims(X_train_scaled[7], 0))
```

---
## Step 6 — Show the 32 feature maps from the first layer

```python
l0_features = features[0].numpy()[0]

fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
# 生成整数序列 / Generate integer sequence
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(l0_features[..., i])

# 显示图表 / Display the plot
plt.show()
```

---
## Step 7 — Show the 32 feature maps from the third layer

```python
l2_features = features[2].numpy()[0]

fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
# 生成整数序列 / Generate integer sequence
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(l2_features[..., i])

# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: rescale image 是机器学习中的常用技术。  
  *rescale image is a common technique in machine learning.*

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
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Visualize / 07 Visualize
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.constraints import MaxNorm
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets.cifar10 import load_data


# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = load_data()

# rescale image
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

model = Sequential([
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(32, (3,3), input_shape=(32, 32, 3), padding="same",
           activation="relu", kernel_constraint=MaxNorm(3)),
    Dropout(0.3),
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(32, (3,3), padding="same",
           activation="relu", kernel_constraint=MaxNorm(3)),
    # 最大池化层（Keras） / Max pooling layer (Keras)
    MaxPooling2D(),
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    Flatten(),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(512, activation="relu", kernel_constraint=MaxNorm(3)),
    Dropout(0.5),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(10, activation="sigmoid")
])

# Train the network with CIFAR10 dataset
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics="sparse_categorical_accuracy")

# 训练模型 / Train the model
model.fit(X_train_scaled, y_train, epochs=25, batch_size=32,
          validation_data=(X_test_scaled, y_test))

# Visualize the input image
# 显示图像 / Display image
plt.imshow(X_train_scaled[7])
# 显示图表 / Display the plot
plt.show()

# Extract output from each layer
extractor = tf.keras.Model(inputs=model.inputs,
                           outputs=[layer.output for layer in model.layers])
# 增加一个维度 / Add a dimension
features = extractor(np.expand_dims(X_train_scaled[7], 0))

# Show the 32 feature maps from the first layer
l0_features = features[0].numpy()[0]

fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
# 生成整数序列 / Generate integer sequence
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(l0_features[..., i])

# 显示图表 / Display the plot
plt.show()

# Show the 32 feature maps from the third layer
l2_features = features[2].numpy()[0]

fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
# 生成整数序列 / Generate integer sequence
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(l2_features[..., i])

# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结



---
