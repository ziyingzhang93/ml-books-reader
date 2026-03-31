# Python 深度学习 / Deep Learning with Python
## Chapter 22

---

### Numpy Dataset

# 01 — Numpy Dataset / 01 Numpy Dataset

**Chapter 22 — File 1 of 5 / 第22章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Numpy Dataset**.

本脚本演示 **01 Numpy Dataset**。

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
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
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
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets.fashion_mnist import load_data
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential

# 加载数据集 / Load dataset
(train_image, train_label), (test_image, test_label) = load_data()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(train_image.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(train_label.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(test_image.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(test_label.shape)

model = Sequential([
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    Flatten(input_shape=(28,28)),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(10, activation="sigmoid")
])
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics="sparse_categorical_accuracy")
# 训练模型 / Train the model
history = model.fit(train_image, train_label,
                    batch_size=32, epochs=50,
                    validation_data=(test_image, test_label), verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
print(model.evaluate(test_image, test_label))

# 绘制折线图 / Draw line plot
plt.plot(history.history['val_sparse_categorical_accuracy'])
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Numpy Dataset 是机器学习中的常用技术。  
  *Numpy Dataset is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
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
# Numpy Dataset / 01 Numpy Dataset
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets.fashion_mnist import load_data
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential

# 加载数据集 / Load dataset
(train_image, train_label), (test_image, test_label) = load_data()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(train_image.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(train_label.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(test_image.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(test_label.shape)

model = Sequential([
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    Flatten(input_shape=(28,28)),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(10, activation="sigmoid")
])
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics="sparse_categorical_accuracy")
# 训练模型 / Train the model
history = model.fit(train_image, train_label,
                    batch_size=32, epochs=50,
                    validation_data=(test_image, test_label), verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
print(model.evaluate(test_image, test_label))

# 绘制折线图 / Draw line plot
plt.plot(history.history['val_sparse_categorical_accuracy'])
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Generator

# 04 — Generator / 04 Generator

**Chapter 22 — File 2 of 5 / 第22章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Generator**.

本脚本演示 **04 Generator**。

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
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
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
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets.fashion_mnist import load_data
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential

model = Sequential([
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    Flatten(input_shape=(28,28)),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(10, activation="sigmoid")
])

def batch_generator(image, label, batchsize):
    # 获取长度 / Get length
    N = len(image)
    i = 0
    while True:
        yield image[i:i+batchsize], label[i:i+batchsize]
        i = i + batchsize
        if i + batchsize > N:
            i = 0

# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics="sparse_categorical_accuracy")
# 训练模型 / Train the model
history = model.fit(batch_generator(train_image, train_label, 32),
                    # 获取长度 / Get length
                    steps_per_epoch=len(train_image)//32,
                    epochs=50, validation_data=(test_image, test_label), verbose=2)
# 评估模型在测试集上的表现 / Evaluate model on test set
print(model.evaluate(test_image, test_label))

# 绘制折线图 / Draw line plot
plt.plot(history.history['val_sparse_categorical_accuracy'])
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Generator 是机器学习中的常用技术。  
  *Generator is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Generator / 04 Generator
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets.fashion_mnist import load_data
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential

model = Sequential([
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    Flatten(input_shape=(28,28)),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(10, activation="sigmoid")
])

def batch_generator(image, label, batchsize):
    # 获取长度 / Get length
    N = len(image)
    i = 0
    while True:
        yield image[i:i+batchsize], label[i:i+batchsize]
        i = i + batchsize
        if i + batchsize > N:
            i = 0

# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics="sparse_categorical_accuracy")
# 训练模型 / Train the model
history = model.fit(batch_generator(train_image, train_label, 32),
                    # 获取长度 / Get length
                    steps_per_epoch=len(train_image)//32,
                    epochs=50, validation_data=(test_image, test_label), verbose=2)
# 评估模型在测试集上的表现 / Evaluate model on test set
print(model.evaluate(test_image, test_label))

# 绘制折线图 / Draw line plot
plt.plot(history.history['val_sparse_categorical_accuracy'])
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Tfdata

# 07 — Tfdata / 07 Tfdata

**Chapter 22 — File 3 of 5 / 第22章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Tfdata**.

本脚本演示 **07 Tfdata**。

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
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
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
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets.fashion_mnist import load_data
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential

# 加载数据集 / Load dataset
(train_image, train_label), (test_image, test_label) = load_data()
dataset = tf.data.Dataset.from_tensor_slices((train_image, train_label))

model = Sequential([
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    Flatten(input_shape=(28,28)),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(10, activation="sigmoid")
])
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics="sparse_categorical_accuracy")
# 训练模型 / Train the model
history = model.fit(dataset.batch(32),
                    epochs=50, validation_data=(test_image, test_label), verbose=2)
# 评估模型在测试集上的表现 / Evaluate model on test set
print(model.evaluate(test_image, test_label))

# 绘制折线图 / Draw line plot
plt.plot(history.history['val_sparse_categorical_accuracy'])
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Tfdata 是机器学习中的常用技术。  
  *Tfdata is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Tfdata / 07 Tfdata
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets.fashion_mnist import load_data
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential

# 加载数据集 / Load dataset
(train_image, train_label), (test_image, test_label) = load_data()
dataset = tf.data.Dataset.from_tensor_slices((train_image, train_label))

model = Sequential([
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    Flatten(input_shape=(28,28)),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(10, activation="sigmoid")
])
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics="sparse_categorical_accuracy")
# 训练模型 / Train the model
history = model.fit(dataset.batch(32),
                    epochs=50, validation_data=(test_image, test_label), verbose=2)
# 评估模型在测试集上的表现 / Evaluate model on test set
print(model.evaluate(test_image, test_label))

# 绘制折线图 / Draw line plot
plt.plot(history.history['val_sparse_categorical_accuracy'])
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Generator Dataset

# 11 — Generator Dataset / 11 Generator Dataset

**Chapter 22 — File 4 of 5 / 第22章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Generator Dataset**.

本脚本演示 **11 Generator Dataset**。

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
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
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
from tensorflow.keras.datasets.fashion_mnist import load_data
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential

# 加载数据集 / Load dataset
(train_image, train_label), (test_image, test_label) = load_data()

def shuffle_generator(image, label, seed):
    # 生成等差数组 / Generate array with step
    idx = np.arange(len(image))
    # 生成随机数 / Generate random numbers
    np.random.default_rng(seed).shuffle(idx)
    for i in idx:
        yield image[i], label[i]

dataset = tf.data.Dataset.from_generator(
    shuffle_generator,
    args=[train_image, train_label, 42],
    output_signature=(
        tf.TensorSpec(shape=(28,28), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.uint8)))

model = Sequential([
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    Flatten(input_shape=(28,28)),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(10, activation="sigmoid")
])
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics="sparse_categorical_accuracy")
# 训练模型 / Train the model
history = model.fit(dataset.batch(32),
                    epochs=50, validation_data=(test_image, test_label), verbose=2)
# 评估模型在测试集上的表现 / Evaluate model on test set
print(model.evaluate(test_image, test_label))

# 绘制折线图 / Draw line plot
plt.plot(history.history['val_sparse_categorical_accuracy'])
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Generator Dataset 是机器学习中的常用技术。  
  *Generator Dataset is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Generator Dataset / 11 Generator Dataset
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets.fashion_mnist import load_data
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential

# 加载数据集 / Load dataset
(train_image, train_label), (test_image, test_label) = load_data()

def shuffle_generator(image, label, seed):
    # 生成等差数组 / Generate array with step
    idx = np.arange(len(image))
    # 生成随机数 / Generate random numbers
    np.random.default_rng(seed).shuffle(idx)
    for i in idx:
        yield image[i], label[i]

dataset = tf.data.Dataset.from_generator(
    shuffle_generator,
    args=[train_image, train_label, 42],
    output_signature=(
        tf.TensorSpec(shape=(28,28), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.uint8)))

model = Sequential([
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    Flatten(input_shape=(28,28)),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(100, activation="relu"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(10, activation="sigmoid")
])
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics="sparse_categorical_accuracy")
# 训练模型 / Train the model
history = model.fit(dataset.batch(32),
                    epochs=50, validation_data=(test_image, test_label), verbose=2)
# 评估模型在测试集上的表现 / Evaluate model on test set
print(model.evaluate(test_image, test_label))

# 绘制折线图 / Draw line plot
plt.plot(history.history['val_sparse_categorical_accuracy'])
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Shuffle



---

### Chapter Summary / 章节总结



---
