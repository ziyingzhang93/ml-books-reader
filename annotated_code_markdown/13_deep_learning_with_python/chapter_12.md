# Python 深度学习 / Deep Learning with Python
## Chapter 12

---

### Sequential

# 02 — Sequential / 02 Sequential

**Chapter 12 — File 1 of 5 / 第12章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Sequential**.

本脚本演示 **02 Sequential**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPool2D

model = Sequential([
          Input(shape=(32,32,3,)),
          # 二维卷积层（Keras） / 2D convolution layer (Keras)
          Conv2D(6, (5,5), padding="same", activation="relu"),
          MaxPool2D(pool_size=(2,2)),
          # 二维卷积层（Keras） / 2D convolution layer (Keras)
          Conv2D(16, (5,5), padding="same", activation="relu"),
          MaxPool2D(pool_size=(2, 2)),
          # 二维卷积层（Keras） / 2D convolution layer (Keras)
          Conv2D(120, (5,5), padding="same", activation="relu"),
          # 展平层：多维→一维 / Flatten: multi-dim → 1D
          Flatten(),
          # 全连接层（Keras） / Fully connected layer (Keras)
          Dense(units=84, activation="relu"),
          # 全连接层（Keras） / Fully connected layer (Keras)
          Dense(units=10, activation="softmax"),
      ])

model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: Sequential 是机器学习中的常用技术。  
  *Sequential is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sequential / 02 Sequential
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPool2D

model = Sequential([
          Input(shape=(32,32,3,)),
          # 二维卷积层（Keras） / 2D convolution layer (Keras)
          Conv2D(6, (5,5), padding="same", activation="relu"),
          MaxPool2D(pool_size=(2,2)),
          # 二维卷积层（Keras） / 2D convolution layer (Keras)
          Conv2D(16, (5,5), padding="same", activation="relu"),
          MaxPool2D(pool_size=(2, 2)),
          # 二维卷积层（Keras） / 2D convolution layer (Keras)
          Conv2D(120, (5,5), padding="same", activation="relu"),
          # 展平层：多维→一维 / Flatten: multi-dim → 1D
          Flatten(),
          # 全连接层（Keras） / Fully connected layer (Keras)
          Dense(units=84, activation="relu"),
          # 全连接层（Keras） / Fully connected layer (Keras)
          Dense(units=10, activation="softmax"),
      ])

model.summary()
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Train Sequential

# 04 — Train Sequential / 04 Train Sequential

**Chapter 12 — File 2 of 5 / 第12章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Train Sequential**.

本脚本演示 **04 Train Sequential**。

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
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPool2D

# 加载数据集 / Load dataset
(trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()

model = Sequential([
          Input(shape=(32,32,3,)),
          # 二维卷积层（Keras） / 2D convolution layer (Keras)
          Conv2D(6, (5,5), padding="same", activation="relu"),
          MaxPool2D(pool_size=(2,2)),
          # 二维卷积层（Keras） / 2D convolution layer (Keras)
          Conv2D(16, (5,5), padding="same", activation="relu"),
          MaxPool2D(pool_size=(2, 2)),
          # 二维卷积层（Keras） / 2D convolution layer (Keras)
          Conv2D(120, (5,5), padding="same", activation="relu"),
          # 展平层：多维→一维 / Flatten: multi-dim → 1D
          Flatten(),
          # 全连接层（Keras） / Fully connected layer (Keras)
          Dense(units=84, activation="relu"),
          # 全连接层（Keras） / Fully connected layer (Keras)
          Dense(units=10, activation="softmax"),
      ])

model.summary()

# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics="acc")
# 训练模型 / Train the model
history = model.fit(x=trainX, y=trainY, batch_size=256, epochs=10,
                    validation_data=(testX, testY))
```

---
## Learning Notes / 学习笔记

- **概念**: Train Sequential 是机器学习中的常用技术。  
  *Train Sequential is a common technique in machine learning.*

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
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
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
# Train Sequential / 04 Train Sequential
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPool2D

# 加载数据集 / Load dataset
(trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()

model = Sequential([
          Input(shape=(32,32,3,)),
          # 二维卷积层（Keras） / 2D convolution layer (Keras)
          Conv2D(6, (5,5), padding="same", activation="relu"),
          MaxPool2D(pool_size=(2,2)),
          # 二维卷积层（Keras） / 2D convolution layer (Keras)
          Conv2D(16, (5,5), padding="same", activation="relu"),
          MaxPool2D(pool_size=(2, 2)),
          # 二维卷积层（Keras） / 2D convolution layer (Keras)
          Conv2D(120, (5,5), padding="same", activation="relu"),
          # 展平层：多维→一维 / Flatten: multi-dim → 1D
          Flatten(),
          # 全连接层（Keras） / Fully connected layer (Keras)
          Dense(units=84, activation="relu"),
          # 全连接层（Keras） / Fully connected layer (Keras)
          Dense(units=10, activation="softmax"),
      ])

model.summary()

# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics="acc")
# 训练模型 / Train the model
history = model.fit(x=trainX, y=trainY, batch_size=256, epochs=10,
                    validation_data=(testX, testY))
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Functional

# 06 — Functional / 06 Functional

**Chapter 12 — File 3 of 5 / 第12章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Functional**.

本脚本演示 **06 Functional**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPool2D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Model

input_layer = Input(shape=(32,32,3,))
# 二维卷积层（Keras） / 2D convolution layer (Keras)
x = Conv2D(6, (5,5), padding="same", activation="relu")(input_layer)
x = MaxPool2D(pool_size=(2,2))(x)
# 二维卷积层（Keras） / 2D convolution layer (Keras)
x = Conv2D(16, (5,5), padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2, 2))(x)
# 二维卷积层（Keras） / 2D convolution layer (Keras)
x = Conv2D(120, (5,5), padding="same", activation="relu")(x)
# 展平层：多维→一维 / Flatten: multi-dim → 1D
x = Flatten()(x)
# 全连接层（Keras） / Fully connected layer (Keras)
x = Dense(units=84, activation="relu")(x)
# 全连接层（Keras） / Fully connected layer (Keras)
x = Dense(units=10, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=x)

model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: Functional 是机器学习中的常用技术。  
  *Functional is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Functional / 06 Functional
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPool2D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Model

input_layer = Input(shape=(32,32,3,))
# 二维卷积层（Keras） / 2D convolution layer (Keras)
x = Conv2D(6, (5,5), padding="same", activation="relu")(input_layer)
x = MaxPool2D(pool_size=(2,2))(x)
# 二维卷积层（Keras） / 2D convolution layer (Keras)
x = Conv2D(16, (5,5), padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2, 2))(x)
# 二维卷积层（Keras） / 2D convolution layer (Keras)
x = Conv2D(120, (5,5), padding="same", activation="relu")(x)
# 展平层：多维→一维 / Flatten: multi-dim → 1D
x = Flatten()(x)
# 全连接层（Keras） / Fully connected layer (Keras)
x = Dense(units=84, activation="relu")(x)
# 全连接层（Keras） / Fully connected layer (Keras)
x = Dense(units=10, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=x)

model.summary()
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Residual

# 09 — Residual / 09 Residual

**Chapter 12 — File 4 of 5 / 第12章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **store the input tensor to be added later as the identity**.

本脚本演示 **store the input tensor to be added later as the identity**。

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
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, \
                                    MaxPool2D, Flatten, Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.activations import relu
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Model

def residual_block(x, filters):
```

---
## Step 2 — store the input tensor to be added later as the identity

```python
identity = x
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    x = Conv2D(filters = filters, kernel_size=(3, 3), strides = (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = relu(x)
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    x = Conv2D(filters = filters, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([identity, x])
    x = relu(x)

    return x

# 加载数据集 / Load dataset
(trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()

input_layer = Input(shape=(32,32,3,))
# 二维卷积层（Keras） / 2D convolution layer (Keras)
x = Conv2D(32, (3, 3), padding="same", activation="relu")(input_layer)
x = residual_block(x, 32)
# 二维卷积层（Keras） / 2D convolution layer (Keras)
x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
x = residual_block(x, 64)
# 二维卷积层（Keras） / 2D convolution layer (Keras)
x = Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
x = residual_block(x, 128)
# 展平层：多维→一维 / Flatten: multi-dim → 1D
x = Flatten()(x)
# 全连接层（Keras） / Fully connected layer (Keras)
x = Dense(units=84, activation="relu")(x)
# 全连接层（Keras） / Fully connected layer (Keras)
x = Dense(units=10, activation="softmax")(x)

model = Model(inputs=input_layer, outputs = x)
model.summary()

# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics="acc")

# 训练模型 / Train the model
history = model.fit(x=trainX, y=trainY, batch_size=256, epochs=10,
                    validation_data=(testX, testY))
```

---
## Learning Notes / 学习笔记

- **概念**: store the input tensor to be added later as the identity 是机器学习中的常用技术。  
  *store the input tensor to be added later as the identity is a common technique in machine learning.*

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
# Residual / 09 Residual
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, \
                                    MaxPool2D, Flatten, Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.activations import relu
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Model

def residual_block(x, filters):
# store the input tensor to be added later as the identity
    identity = x
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    x = Conv2D(filters = filters, kernel_size=(3, 3), strides = (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = relu(x)
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    x = Conv2D(filters = filters, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([identity, x])
    x = relu(x)

    return x

# 加载数据集 / Load dataset
(trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()

input_layer = Input(shape=(32,32,3,))
# 二维卷积层（Keras） / 2D convolution layer (Keras)
x = Conv2D(32, (3, 3), padding="same", activation="relu")(input_layer)
x = residual_block(x, 32)
# 二维卷积层（Keras） / 2D convolution layer (Keras)
x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
x = residual_block(x, 64)
# 二维卷积层（Keras） / 2D convolution layer (Keras)
x = Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
x = residual_block(x, 128)
# 展平层：多维→一维 / Flatten: multi-dim → 1D
x = Flatten()(x)
# 全连接层（Keras） / Fully connected layer (Keras)
x = Dense(units=84, activation="relu")(x)
# 全连接层（Keras） / Fully connected layer (Keras)
x = Dense(units=10, activation="softmax")(x)

model = Model(inputs=input_layer, outputs = x)
model.summary()

# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics="acc")

# 训练模型 / Train the model
history = model.fit(x=trainX, y=trainY, batch_size=256, epochs=10,
                    validation_data=(testX, testY))
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Subclass

# 14 — Subclass / 14 Subclass

**Chapter 12 — File 5 of 5 / 第12章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **creating layers in initializer**.

本脚本演示 **creating layers in initializer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPool2D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Model

class LeNet5(tf.keras.Model):
  # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
  def __init__(self):
    super(LeNet5, self).__init__()
```

---
## Step 2 — creating layers in initializer

```python
# 二维卷积层（Keras） / 2D convolution layer (Keras)
self.conv1 = Conv2D(6, (5,5), padding="same", activation="relu")
    self.max_pool2x2 = MaxPool2D(pool_size=(2,2))
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    self.conv2 = Conv2D(16, (5,5), padding="same", activation="relu")
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    self.conv3 = Conv2D(120, (5,5), padding="same", activation="relu")
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    self.flatten = Flatten()
    # 全连接层（Keras） / Fully connected layer (Keras)
    self.fc2 = Dense(units=84, activation="relu")
    # 全连接层（Keras） / Fully connected layer (Keras)
    self.fc3=Dense(units=10, activation="softmax")

  def call(self, input_tensor):
```

---
## Step 3 — don't add layers here, need to create the layers in initializer,
otherwise you will get the tf.Variable can only be created once error

```python
x = self.conv1(input_tensor)
    x = self.max_pool2x2(x)
    x = self.conv2(x)
    x = self.max_pool2x2(x)
    x = self.conv3(x)
    # 展平为一维数组 / Flatten to 1D array
    x = self.flatten(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x

input_layer = Input(shape=(32,32,3,))
x = LeNet5()(input_layer)
model = Model(inputs=input_layer, outputs=x)
model.summary(expand_nested=True)
```

---
## Learning Notes / 学习笔记

- **概念**: creating layers in initializer 是机器学习中的常用技术。  
  *creating layers in initializer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Subclass / 14 Subclass
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPool2D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Model

class LeNet5(tf.keras.Model):
  # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
  def __init__(self):
    super(LeNet5, self).__init__()
    #creating layers in initializer
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    self.conv1 = Conv2D(6, (5,5), padding="same", activation="relu")
    self.max_pool2x2 = MaxPool2D(pool_size=(2,2))
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    self.conv2 = Conv2D(16, (5,5), padding="same", activation="relu")
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    self.conv3 = Conv2D(120, (5,5), padding="same", activation="relu")
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    self.flatten = Flatten()
    # 全连接层（Keras） / Fully connected layer (Keras)
    self.fc2 = Dense(units=84, activation="relu")
    # 全连接层（Keras） / Fully connected layer (Keras)
    self.fc3=Dense(units=10, activation="softmax")

  def call(self, input_tensor):
    # don't add layers here, need to create the layers in initializer,
    # otherwise you will get the tf.Variable can only be created once error
    x = self.conv1(input_tensor)
    x = self.max_pool2x2(x)
    x = self.conv2(x)
    x = self.max_pool2x2(x)
    x = self.conv3(x)
    # 展平为一维数组 / Flatten to 1D array
    x = self.flatten(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x

input_layer = Input(shape=(32,32,3,))
x = LeNet5()(input_layer)
model = Model(inputs=input_layer, outputs=x)
model.summary(expand_nested=True)
```

---

### Chapter Summary / 章节总结

# Chapter 12 Summary / 第12章总结

## Theme / 主题: Chapter 12 / Chapter 12

This chapter contains **5 code files** demonstrating chapter 12.

本章包含 **5 个代码文件**，演示Chapter 12。

---
## Evolution / 演化路线

  1. `02_sequential.ipynb` — Sequential
  2. `04_train_sequential.ipynb` — Train Sequential
  3. `06_functional.ipynb` — Functional
  4. `09_residual.ipynb` — Residual
  5. `14_subclass.ipynb` — Subclass

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 12) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 12）是机器学习流水线中的基础构建块。

---
