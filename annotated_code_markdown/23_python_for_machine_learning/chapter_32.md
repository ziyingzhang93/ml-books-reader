# Python 机器学习 / Python for Machine Learning
## Chapter 32

---

### Lenet5

# 11 — Lenet5 / 11 Lenet5

**Chapter 32 — File 1 of 2 / 第32章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **creating layers in initializer**.

本脚本演示 **creating layers in initializer**。

---
## Step 1 — Step 1

```python
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D
from keras.models import Model

class LeNet5(tf.keras.Model):
  def __init__(self):
    super(LeNet5, self).__init__()
```

---
## Step 2 — creating layers in initializer

```python
self.conv1 = Conv2D(filters=6, kernel_size=(5,5), padding="same", activation="relu")
    self.max_pool2x2 = MaxPool2D(pool_size=(2,2))
    self.conv2 = Conv2D(filters=16, kernel_size=(5,5), padding="same", activation="relu")
    self.flatten = Flatten()
    self.fc1 = Dense(units=120, activation="relu")
    self.fc2 = Dense(units=84, activation="relu")
    self.fc3 = Dense(units=10, activation="softmax")
  def call(self, input_tensor):
    conv1 = self.conv1(input_tensor)
    maxpool1 = self.max_pool2x2(conv1)
    conv2 = self.conv2(maxpool1)
    maxpool2 = self.max_pool2x2(conv2)
    flatten = self.flatten(maxpool2)
    fc1 = self.fc1(flatten)
    fc2 = self.fc2(fc1)
    fc3 = self.fc3(fc2)
    return fc3
```

---
## Learning Notes / 学习笔记

- **概念**: creating layers in initializer 是机器学习中的常用技术。  
  *creating layers in initializer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lenet5 / 11 Lenet5
# Complete Code / 完整代码
# ===============================

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D
from keras.models import Model

class LeNet5(tf.keras.Model):
  def __init__(self):
    super(LeNet5, self).__init__()
    #creating layers in initializer
    self.conv1 = Conv2D(filters=6, kernel_size=(5,5), padding="same", activation="relu")
    self.max_pool2x2 = MaxPool2D(pool_size=(2,2))
    self.conv2 = Conv2D(filters=16, kernel_size=(5,5), padding="same", activation="relu")
    self.flatten = Flatten()
    self.fc1 = Dense(units=120, activation="relu")
    self.fc2 = Dense(units=84, activation="relu")
    self.fc3 = Dense(units=10, activation="softmax")
  def call(self, input_tensor):
    conv1 = self.conv1(input_tensor)
    maxpool1 = self.max_pool2x2(conv1)
    conv2 = self.conv2(maxpool1)
    maxpool2 = self.max_pool2x2(conv2)
    flatten = self.flatten(maxpool2)
    fc1 = self.fc1(flatten)
    fc2 = self.fc2(fc1)
    fc3 = self.fc3(fc2)
    return fc3
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Checkpoint

# 15 — Checkpoint / 15 Checkpoint

**Chapter 32 — File 2 of 2 / 第32章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **saving checkpoints**.

本脚本演示 **saving checkpoints**。

---
## Step 1 — Step 1

```python
import os
from google.colab import drive
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D
from keras.models import Model

MOUNTPOINT = "/content/gdrive"
DATADIR = os.path.join(MOUNTPOINT, "MyDrive")
drive.mount(MOUNTPOINT)

class LeNet5(tf.keras.Model):
  def __init__(self):
    super(LeNet5, self).__init__()
    self.conv1 = Conv2D(filters=6, kernel_size=(5,5), padding="same", activation="relu")
    self.max_pool2x2 = MaxPool2D(pool_size=(2,2))
    self.conv2 = Conv2D(filters=16, kernel_size=(5,5), padding="same", activation="relu")
    self.flatten = Flatten()
    self.fc1 = Dense(units=120, activation="relu")
    self.fc2 = Dense(units=84, activation="relu")
    self.fc3=Dense(units=10, activation="softmax")
  def call(self, input_tensor):
    conv1 = self.conv1(input_tensor)
    maxpool1 = self.max_pool2x2(conv1)
    conv2 = self.conv2(maxpool1)
    maxpool2 = self.max_pool2x2(conv2)
    flatten = self.flatten(maxpool2)
    fc1 = self.fc1(flatten)
    fc2 = self.fc2(fc1)
    fc3 = self.fc3(fc2)
    return fc3

mnist_digits = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_digits.load_data()
```

---
## Step 2 — saving checkpoints

```python
checkpoint_path = DATADIR + "/checkpoints/cp-epoch-{epoch}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
input_layer = Input(shape=(28,28,1))
model = LeNet5()(input_layer)
model = Model(inputs=input_layer, outputs=model)
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics="acc")
model.fit(x=train_images, y=train_labels, validation_data=[test_images, test_labels],
          batch_size=256, epochs=5, callbacks=[cp_callback])
```

---
## Learning Notes / 学习笔记

- **概念**: saving checkpoints 是机器学习中的常用技术。  
  *saving checkpoints is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Checkpoint / 15 Checkpoint
# Complete Code / 完整代码
# ===============================

import os
from google.colab import drive
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D
from keras.models import Model

MOUNTPOINT = "/content/gdrive"
DATADIR = os.path.join(MOUNTPOINT, "MyDrive")
drive.mount(MOUNTPOINT)

class LeNet5(tf.keras.Model):
  def __init__(self):
    super(LeNet5, self).__init__()
    self.conv1 = Conv2D(filters=6, kernel_size=(5,5), padding="same", activation="relu")
    self.max_pool2x2 = MaxPool2D(pool_size=(2,2))
    self.conv2 = Conv2D(filters=16, kernel_size=(5,5), padding="same", activation="relu")
    self.flatten = Flatten()
    self.fc1 = Dense(units=120, activation="relu")
    self.fc2 = Dense(units=84, activation="relu")
    self.fc3=Dense(units=10, activation="softmax")
  def call(self, input_tensor):
    conv1 = self.conv1(input_tensor)
    maxpool1 = self.max_pool2x2(conv1)
    conv2 = self.conv2(maxpool1)
    maxpool2 = self.max_pool2x2(conv2)
    flatten = self.flatten(maxpool2)
    fc1 = self.fc1(flatten)
    fc2 = self.fc2(fc1)
    fc3 = self.fc3(fc2)
    return fc3

mnist_digits = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_digits.load_data()

# saving checkpoints
checkpoint_path = DATADIR + "/checkpoints/cp-epoch-{epoch}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
input_layer = Input(shape=(28,28,1))
model = LeNet5()(input_layer)
model = Model(inputs=input_layer, outputs=model)
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics="acc")
model.fit(x=train_images, y=train_labels, validation_data=[test_images, test_labels],
          batch_size=256, epochs=5, callbacks=[cp_callback])
```

---

### Chapter Summary / 章节总结

# Chapter 32 Summary / 第32章总结

## Theme / 主题: Chapter 32 / Chapter 32

This chapter contains **2 code files** demonstrating chapter 32.

本章包含 **2 个代码文件**，演示Chapter 32。

---
## Evolution / 演化路线

  1. `11_lenet5.ipynb` — Lenet5
  2. `15_checkpoint.ipynb` — Checkpoint

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 32) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 32）是机器学习流水线中的基础构建块。

---
