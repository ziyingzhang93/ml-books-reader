# Python 深度学习 / Deep Learning with Python
## Chapter 25

---

### Load Mnist

# 01 — Load Mnist / 01 Load Mnist

**Chapter 25 — File 1 of 4 / 第25章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Plot ad hoc mnist instances**.

本脚本演示 **Plot ad hoc mnist instances**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Plot ad hoc mnist instances

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
```

---
## Step 2 — load (downloaded if needed) the MNIST dataset

```python
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

---
## Step 3 — plot 4 images as gray scale

```python
# 创建子图 / Create subplot
plt.subplot(221)
# 显示图像 / Display image
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
# 创建子图 / Create subplot
plt.subplot(222)
# 显示图像 / Display image
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
# 创建子图 / Create subplot
plt.subplot(223)
# 显示图像 / Display image
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
# 创建子图 / Create subplot
plt.subplot(224)
# 显示图像 / Display image
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
```

---
## Step 4 — show the plot

```python
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Plot ad hoc mnist instances 是机器学习中的常用技术。  
  *Plot ad hoc mnist instances is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Mnist / 01 Load Mnist
# Complete Code / 完整代码
# ===============================

# Plot ad hoc mnist instances
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
# 创建子图 / Create subplot
plt.subplot(221)
# 显示图像 / Display image
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
# 创建子图 / Create subplot
plt.subplot(222)
# 显示图像 / Display image
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
# 创建子图 / Create subplot
plt.subplot(223)
# 显示图像 / Display image
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
# 创建子图 / Create subplot
plt.subplot(224)
# 显示图像 / Display image
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Baseline

# 09 — Baseline / 09 Baseline

**Chapter 25 — File 2 of 4 / 第25章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Baseline MLP for MNIST dataset**.

本脚本演示 **Baseline MLP for MNIST dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — Baseline MLP for MNIST dataset

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
```

---
## Step 2 — load data

```python
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

---
## Step 3 — flatten 28*28 images to a 784 vector for each image

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
num_pixels = X_train.shape[1] * X_train.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
```

---
## Step 4 — normalize inputs from 0-255 to 0-1

```python
X_train = X_train / 255
X_test = X_test / 255
```

---
## Step 5 — one-hot encode outputs

```python
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
num_classes = y_test.shape[1]
```

---
## Step 6 — define baseline model

```python
def baseline_model():
```

---
## Step 7 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(num_pixels, input_shape=(num_pixels,),
                    kernel_initializer='normal', activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
```

---
## Step 8 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

---
## Step 9 — build the model

```python
model = baseline_model()
```

---
## Step 10 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X_train, y_train, epochs=10, batch_size=200,
          validation_data=(X_test, y_test), verbose=2)
```

---
## Step 11 — Final evaluation of the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
```

---
## Learning Notes / 学习笔记

- **概念**: Baseline MLP for MNIST dataset 是机器学习中的常用技术。  
  *Baseline MLP for MNIST dataset is a common technique in machine learning.*

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
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Baseline / 09 Baseline
# Complete Code / 完整代码
# ===============================

# Baseline MLP for MNIST dataset
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# load data
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# flatten 28*28 images to a 784 vector for each image
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
num_pixels = X_train.shape[1] * X_train.shape[2]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one-hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
num_classes = y_test.shape[1]
# define baseline model
def baseline_model():
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(num_pixels, input_shape=(num_pixels,),
                    kernel_initializer='normal', activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build the model
model = baseline_model()
# Fit the model
# 训练模型 / Train the model
model.fit(X_train, y_train, epochs=10, batch_size=200,
          validation_data=(X_test, y_test), verbose=2)
# Final evaluation of the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Cnn

# 15 — Cnn / 卷积神经网络

**Chapter 25 — File 3 of 4 / 第25章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Simple CNN for the MNIST Dataset**.

本脚本演示 **Simple CNN for the MNIST Dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — Simple CNN for the MNIST Dataset

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv2D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import MaxPooling2D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
```

---
## Step 2 — load data

```python
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

---
## Step 3 — reshape to be [samples][width][height][channels]

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
```

---
## Step 4 — normalize inputs from 0-255 to 0-1

```python
X_train = X_train / 255
X_test = X_test / 255
```

---
## Step 5 — one-hot encode outputs

```python
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
num_classes = y_test.shape[1]
```

---
## Step 6 — define a simple CNN model

```python
def baseline_model():
```

---
## Step 7 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(MaxPooling2D())
    # 向模型添加一层 / Add a layer to the model
    model.add(Dropout(0.2))
    # 向模型添加一层 / Add a layer to the model
    model.add(Flatten())
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(128, activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(num_classes, activation='softmax'))
```

---
## Step 8 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

---
## Step 9 — build the model

```python
model = baseline_model()
```

---
## Step 10 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
```

---
## Step 11 — Final evaluation of the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("CNN Error: %.2f%%" % (100-scores[1]*100))
```

---
## Learning Notes / 学习笔记

- **概念**: Simple CNN for the MNIST Dataset 是机器学习中的常用技术。  
  *Simple CNN for the MNIST Dataset is a common technique in machine learning.*

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
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cnn / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# Simple CNN for the MNIST Dataset
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv2D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import MaxPooling2D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# load data
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][width][height][channels]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one-hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
num_classes = y_test.shape[1]
# define a simple CNN model
def baseline_model():
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(MaxPooling2D())
    # 向模型添加一层 / Add a layer to the model
    model.add(Dropout(0.2))
    # 向模型添加一层 / Add a layer to the model
    model.add(Flatten())
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(128, activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build the model
model = baseline_model()
# Fit the model
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("CNN Error: %.2f%%" % (100-scores[1]*100))
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Larger

# 19 — Larger / 19 Larger

**Chapter 25 — File 4 of 4 / 第25章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Larger CNN for the MNIST Dataset**.

本脚本演示 **Larger CNN for the MNIST Dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — Larger CNN for the MNIST Dataset

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv2D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import MaxPooling2D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
```

---
## Step 2 — load data

```python
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

---
## Step 3 — reshape to be [samples][width][height][channels]

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
```

---
## Step 4 — normalize inputs from 0-255 to 0-1

```python
X_train = X_train / 255
X_test = X_test / 255
```

---
## Step 5 — one-hot encode outputs

```python
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
num_classes = y_test.shape[1]
```

---
## Step 6 — define the larger model

```python
def larger_model():
```

---
## Step 7 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(MaxPooling2D())
    # 向模型添加一层 / Add a layer to the model
    model.add(Conv2D(15, (3, 3), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(MaxPooling2D())
    # 向模型添加一层 / Add a layer to the model
    model.add(Dropout(0.2))
    # 向模型添加一层 / Add a layer to the model
    model.add(Flatten())
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(128, activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(50, activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(num_classes, activation='softmax'))
```

---
## Step 8 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

---
## Step 9 — build the model

```python
model = larger_model()
```

---
## Step 10 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
```

---
## Step 11 — Final evaluation of the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
```

---
## Learning Notes / 学习笔记

- **概念**: Larger CNN for the MNIST Dataset 是机器学习中的常用技术。  
  *Larger CNN for the MNIST Dataset is a common technique in machine learning.*

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
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Larger / 19 Larger
# Complete Code / 完整代码
# ===============================

# Larger CNN for the MNIST Dataset
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv2D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import MaxPooling2D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# load data
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][width][height][channels]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one-hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
num_classes = y_test.shape[1]
# define the larger model
def larger_model():
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(MaxPooling2D())
    # 向模型添加一层 / Add a layer to the model
    model.add(Conv2D(15, (3, 3), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(MaxPooling2D())
    # 向模型添加一层 / Add a layer to the model
    model.add(Dropout(0.2))
    # 向模型添加一层 / Add a layer to the model
    model.add(Flatten())
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(128, activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(50, activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build the model
model = larger_model()
# Fit the model
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
```

---

### Chapter Summary / 章节总结

# Chapter 25 Summary / 第25章总结

## Theme / 主题: Chapter 25 / Chapter 25

This chapter contains **4 code files** demonstrating chapter 25.

本章包含 **4 个代码文件**，演示Chapter 25。

---
## Evolution / 演化路线

  1. `01_load_mnist.ipynb` — Load Mnist
  2. `09_baseline.ipynb` — Baseline
  3. `15_cnn.ipynb` — Cnn
  4. `19_larger.ipynb` — Larger

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 25) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 25）是机器学习流水线中的基础构建块。

---
