# Python 机器学习 / Python for Machine Learning
## Chapter 05

---

### Class



---

### Methods



---

### Customcallback

# 12 — Customcallback / 12 Customcallback

**Chapter 05 — File 3 of 5 / 第05章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Loading the MNIST training and testing data splits**.

本脚本演示 **Loading the MNIST training and testing data splits**。

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
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow.keras as keras
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical

class EpochCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # 打印输出 / Print output
        print("Starting epoch {}".format(epoch + 1))

    def on_epoch_end(self, epoch, logs=None):
        # 打印输出 / Print output
        print("Finished epoch {}".format(epoch + 1))


def simple_model():
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Flatten(input_shape=(28, 28)))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(128, activation="relu"))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(10, activation="softmax"))

    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss="categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    return model
```

---
## Step 2 — Loading the MNIST training and testing data splits

```python
# 加载数据集 / Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

---
## Step 3 — Pre-processing the training data

```python
x_train = x_train / 255.0
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_train = x_train.reshape(60000, 28, 28, 1)
y_train_cat = to_categorical(y_train, 10)

model = simple_model()
# 训练模型 / Train the model
model.fit(x_train,
          y_train_cat,
          batch_size=32,
          epochs=5,
          callbacks=[EpochCallback()],
          verbose=0)
```

---
## Learning Notes / 学习笔记

- **概念**: Loading the MNIST training and testing data splits 是机器学习中的常用技术。  
  *Loading the MNIST training and testing data splits is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
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
# Customcallback / 12 Customcallback
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow.keras as keras
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical

class EpochCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # 打印输出 / Print output
        print("Starting epoch {}".format(epoch + 1))

    def on_epoch_end(self, epoch, logs=None):
        # 打印输出 / Print output
        print("Finished epoch {}".format(epoch + 1))


def simple_model():
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Flatten(input_shape=(28, 28)))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(128, activation="relu"))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(10, activation="softmax"))

    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss="categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    return model


# Loading the MNIST training and testing data splits
# 加载数据集 / Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pre-processing the training data
x_train = x_train / 255.0
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_train = x_train.reshape(60000, 28, 28, 1)
y_train_cat = to_categorical(y_train, 10)

model = simple_model()
# 训练模型 / Train the model
model.fit(x_train,
          y_train_cat,
          batch_size=32,
          epochs=5,
          callbacks=[EpochCallback()],
          verbose=0)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Customcallback

# 15 — Customcallback / 15 Customcallback

**Chapter 05 — File 4 of 5 / 第05章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Loading the MNIST training and testing data splits**.

本脚本演示 **Loading the MNIST training and testing data splits**。

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
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow.keras as keras
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical

class EpochCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # 打印输出 / Print output
        print("Starting epoch {}".format(epoch + 1))

    def on_epoch_end(self, epoch, logs=None):
        # 打印输出 / Print output
        print("Finished epoch {}".format(epoch + 1))


class CheckpointCallback(keras.callbacks.Callback):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        super(CheckpointCallback, self).__init__()
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("loss")
        # 打印输出 / Print output
        print("Current loss is {}".format(current_loss))
        if np.less(current_loss, self.best_loss):
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()
            # 打印输出 / Print output
            print("Storing the model weights at epoch {} \n".format(epoch + 1))


def simple_model():
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Flatten(input_shape=(28, 28)))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(128, activation="relu"))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(10, activation="softmax"))

    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss="categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    return model
```

---
## Step 2 — Loading the MNIST training and testing data splits

```python
# 加载数据集 / Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

---
## Step 3 — Pre-processing the training data

```python
x_train = x_train / 255.0
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_train = x_train.reshape(60000, 28, 28, 1)
y_train_cat = to_categorical(y_train, 10)

model = simple_model()
# 训练模型 / Train the model
model.fit(x_train,
          y_train_cat,
          batch_size=32,
          epochs=5,
          callbacks=[EpochCallback(), CheckpointCallback()],
          verbose=0)
```

---
## Learning Notes / 学习笔记

- **概念**: Loading the MNIST training and testing data splits 是机器学习中的常用技术。  
  *Loading the MNIST training and testing data splits is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
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
# Customcallback / 15 Customcallback
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow.keras as keras
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical

class EpochCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # 打印输出 / Print output
        print("Starting epoch {}".format(epoch + 1))

    def on_epoch_end(self, epoch, logs=None):
        # 打印输出 / Print output
        print("Finished epoch {}".format(epoch + 1))


class CheckpointCallback(keras.callbacks.Callback):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        super(CheckpointCallback, self).__init__()
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("loss")
        # 打印输出 / Print output
        print("Current loss is {}".format(current_loss))
        if np.less(current_loss, self.best_loss):
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()
            # 打印输出 / Print output
            print("Storing the model weights at epoch {} \n".format(epoch + 1))


def simple_model():
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Flatten(input_shape=(28, 28)))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(128, activation="relu"))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(10, activation="softmax"))

    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss="categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    return model


# Loading the MNIST training and testing data splits
# 加载数据集 / Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pre-processing the training data
x_train = x_train / 255.0
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_train = x_train.reshape(60000, 28, 28, 1)
y_train_cat = to_categorical(y_train, 10)

model = simple_model()
# 训练模型 / Train the model
model.fit(x_train,
          y_train_cat,
          batch_size=32,
          epochs=5,
          callbacks=[EpochCallback(), CheckpointCallback()],
          verbose=0)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Custommetrics

# 16 — Custommetrics / 16 Custommetrics

**Chapter 05 — File 5 of 5 / 第05章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Custommetrics**.

本脚本演示 **16 Custommetrics**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf

class BinaryTruePositives(tf.keras.metrics.Metric):

  # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
  def __init__(self, name='binary_true_positives', **kwargs):
    super(BinaryTruePositives, self).__init__(name=name, **kwargs)
    self.true_positives = self.add_weight(name='tp', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
    values = tf.cast(values, self.dtype)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype)
      values = tf.multiply(values, sample_weight)
    self.true_positives.assign_add(tf.reduce_sum(values))

  def result(self):
    return self.true_positives

  def reset_states(self):
    self.true_positives.assign(0)

m = BinaryTruePositives()
m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
# 打印输出 / Print output
print('Intermediate result:', float(m.result()))

m.update_state([1, 1, 1, 1], [0, 1, 1, 0])
# 打印输出 / Print output
print('Final result:', float(m.result()))
```

---
## Learning Notes / 学习笔记

- **概念**: Custommetrics 是机器学习中的常用技术。  
  *Custommetrics is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Custommetrics / 16 Custommetrics
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf

class BinaryTruePositives(tf.keras.metrics.Metric):

  # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
  def __init__(self, name='binary_true_positives', **kwargs):
    super(BinaryTruePositives, self).__init__(name=name, **kwargs)
    self.true_positives = self.add_weight(name='tp', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
    values = tf.cast(values, self.dtype)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype)
      values = tf.multiply(values, sample_weight)
    self.true_positives.assign_add(tf.reduce_sum(values))

  def result(self):
    return self.true_positives

  def reset_states(self):
    self.true_positives.assign(0)

m = BinaryTruePositives()
m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
# 打印输出 / Print output
print('Intermediate result:', float(m.result()))

m.update_state([1, 1, 1, 1], [0, 1, 1, 0])
# 打印输出 / Print output
print('Final result:', float(m.result()))
```

---

### Chapter Summary / 章节总结



---
