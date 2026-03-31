# Python 机器学习 / Python for Machine Learning
## Chapter 14

---

### Dump

# 01 — Dump / 01 Dump

**Chapter 14 — File 1 of 13 / 第14章 — 第1个文件（共13个）**

---

## Summary / 总结

This script demonstrates **"wb" argument opens the file in binary mode**.

本脚本演示 **"wb" argument opens the file in binary mode**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
# 导入对象序列化模块 / Import object serialization module
import pickle

test_dict = {"Hello": "World!"}
# 打开文件（自动关闭） / Open file (auto-close)
with open("test.pickle", "wb") as outfile:
```

---
## Step 2 — "wb" argument opens the file in binary mode

```python
pickle.dump(test_dict, outfile)
```

---
## Learning Notes / 学习笔记

- **概念**: "wb" argument opens the file in binary mode 是机器学习中的常用技术。  
  *"wb" argument opens the file in binary mode is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dump / 01 Dump
# Complete Code / 完整代码
# ===============================

# 导入对象序列化模块 / Import object serialization module
import pickle

test_dict = {"Hello": "World!"}
# 打开文件（自动关闭） / Open file (auto-close)
with open("test.pickle", "wb") as outfile:
 	# "wb" argument opens the file in binary mode
	pickle.dump(test_dict, outfile)
```

---

➡️ **Next / 下一步**: File 2 of 13

---

### Load

# 02 — Load / 02 Load

**Chapter 14 — File 2 of 13 / 第14章 — 第2个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Load**.

本脚本演示 **02 Load**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
# 导入对象序列化模块 / Import object serialization module
import pickle

# 打开文件（自动关闭） / Open file (auto-close)
with open("test.pickle", "rb") as infile:
 	test_dict_reconstructed = pickle.load(infile)
```

---
## Learning Notes / 学习笔记

- **概念**: Load 是机器学习中的常用技术。  
  *Load is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load / 02 Load
# Complete Code / 完整代码
# ===============================

# 导入对象序列化模块 / Import object serialization module
import pickle

# 打开文件（自动关闭） / Open file (auto-close)
with open("test.pickle", "rb") as infile:
 	test_dict_reconstructed = pickle.load(infile)
```

---

➡️ **Next / 下一步**: File 3 of 13

---

### Serialization



---

### Userdefined

# 06 — Userdefined / 06 Userdefined

**Chapter 14 — File 4 of 13 / 第14章 — 第4个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Create an object of NewClass**.

本脚本演示 **Create an object of NewClass**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
# 导入对象序列化模块 / Import object serialization module
import pickle

class NewClass:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, data):
        # 打印输出 / Print output
        print(data)
        self.data = data
```

---
## Step 2 — Create an object of NewClass

```python
new_class = NewClass(1)
```

---
## Step 3 — Serialize and deserialize

```python
pickled_data = pickle.dumps(new_class)
reconstructed = pickle.loads(pickled_data)
```

---
## Step 4 — Verify

```python
# 打印输出 / Print output
print("Data from reconstructed object:", reconstructed.data)
```

---
## Learning Notes / 学习笔记

- **概念**: Create an object of NewClass 是机器学习中的常用技术。  
  *Create an object of NewClass is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Userdefined / 06 Userdefined
# Complete Code / 完整代码
# ===============================

# 导入对象序列化模块 / Import object serialization module
import pickle

class NewClass:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, data):
        # 打印输出 / Print output
        print(data)
        self.data = data

# Create an object of NewClass
new_class = NewClass(1)

# Serialize and deserialize
pickled_data = pickle.dumps(new_class)
reconstructed = pickle.loads(pickled_data)

# Verify
# 打印输出 / Print output
print("Data from reconstructed object:", reconstructed.data)
```

---

➡️ **Next / 下一步**: File 5 of 13

---

### Functions

# 07 — Functions / 07 Functions

**Chapter 14 — File 5 of 13 / 第14章 — 第5个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Serialize and deserialize**.

本脚本演示 **Serialize and deserialize**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入对象序列化模块 / Import object serialization module
import pickle

def test():
    return "Hello world!"
```

---
## Step 2 — Serialize and deserialize

```python
pickled_function = pickle.dumps(test)
reconstructed_function = pickle.loads(pickled_function)
```

---
## Step 3 — Verify

```python
print (reconstructed_function()) #prints "Hello, world!"
```

---
## Learning Notes / 学习笔记

- **概念**: Serialize and deserialize 是机器学习中的常用技术。  
  *Serialize and deserialize is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Functions / 07 Functions
# Complete Code / 完整代码
# ===============================

# 导入对象序列化模块 / Import object serialization module
import pickle

def test():
    return "Hello world!"

# Serialize and deserialize
pickled_function = pickle.dumps(test)
reconstructed_function = pickle.loads(pickled_function)

# Verify
print (reconstructed_function()) #prints "Hello, world!"
```

---

➡️ **Next / 下一步**: File 6 of 13

---

### Keras

# 08 — Keras / Keras

**Chapter 14 — File 6 of 13 / 第14章 — 第6个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Load MNIST digits**.

本脚本演示 **Load MNIST digits**。

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
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — Step 1

```python
# 导入对象序列化模块 / Import object serialization module
import pickle

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Dropout, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import EarlyStopping
```

---
## Step 2 — Load MNIST digits

```python
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

---
## Step 3 — Reshape data to (n_samples, height, wiedth, n_channel)

```python
# 转换数据类型 / Convert data type
X_train = np.expand_dims(X_train, axis=3).astype("float32")
# 转换数据类型 / Convert data type
X_test = np.expand_dims(X_test, axis=3).astype("float32")
```

---
## Step 4 — One-hot encode the output

```python
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

---
## Step 5 — LeNet5 model

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
## Step 6 — Train the model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# 早停：验证集不再提升时自动停止训练 / EarlyStopping: stop when validation stops improving
earlystopping = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=100, batch_size=32, callbacks=[earlystopping])
```

---
## Step 7 — Evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
print(model.evaluate(X_test, y_test, verbose=0))
```

---
## Step 8 — Pickle to serialize and deserialize

```python
pickled_model = pickle.dumps(model)
reconstructed = pickle.loads(pickled_model)
```

---
## Step 9 — Evaluate again

```python
# 打印输出 / Print output
print(reconstructed.evaluate(X_test, y_test, verbose=0))
```

---
## Learning Notes / 学习笔记

- **概念**: Load MNIST digits 是机器学习中的常用技术。  
  *Load MNIST digits is a common technique in machine learning.*

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
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Keras / Keras
# Complete Code / 完整代码
# ===============================

# 导入对象序列化模块 / Import object serialization module
import pickle

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Dropout, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import EarlyStopping

# Load MNIST digits
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data to (n_samples, height, wiedth, n_channel)
# 转换数据类型 / Convert data type
X_train = np.expand_dims(X_train, axis=3).astype("float32")
# 转换数据类型 / Convert data type
X_test = np.expand_dims(X_test, axis=3).astype("float32")

# One-hot encode the output
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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

# Train the model
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# 早停：验证集不再提升时自动停止训练 / EarlyStopping: stop when validation stops improving
earlystopping = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=100, batch_size=32, callbacks=[earlystopping])

# Evaluate the model
# 评估模型在测试集上的表现 / Evaluate model on test set
print(model.evaluate(X_test, y_test, verbose=0))

# Pickle to serialize and deserialize
pickled_model = pickle.dumps(model)
reconstructed = pickle.loads(pickled_model)

# Evaluate again
# 打印输出 / Print output
print(reconstructed.evaluate(X_test, y_test, verbose=0))
```

---

➡️ **Next / 下一步**: File 7 of 13

---

### Hdf5

# 09 — Hdf5 / 09 Hdf5

**Chapter 14 — File 7 of 13 / 第14章 — 第7个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Hdf5**.

本脚本演示 **09 Hdf5**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import h5py

with h5py.File("test.hdf5", "w") as file:
    dataset = file.create_dataset("test_dataset", (100,), dtype="i4")
```

---
## Learning Notes / 学习笔记

- **概念**: Hdf5 是机器学习中的常用技术。  
  *Hdf5 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hdf5 / 09 Hdf5
# Complete Code / 完整代码
# ===============================

import h5py

with h5py.File("test.hdf5", "w") as file:
    dataset = file.create_dataset("test_dataset", (100,), dtype="i4")
```

---

➡️ **Next / 下一步**: File 8 of 13

---

### Hdf5

# 12 — Hdf5 / 12 Hdf5

**Chapter 14 — File 8 of 13 / 第14章 — 第8个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Hdf5**.

本脚本演示 **12 Hdf5**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import h5py

with h5py.File("test.hdf5", "r") as file:
    # 获取字典的所有键 / Get all dict keys
    print (file.keys()) #gets names of datasets that are in the file
    dataset = file["test_dataset"]
```

---
## Learning Notes / 学习笔记

- **概念**: Hdf5 是机器学习中的常用技术。  
  *Hdf5 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hdf5 / 12 Hdf5
# Complete Code / 完整代码
# ===============================

import h5py

with h5py.File("test.hdf5", "r") as file:
    # 获取字典的所有键 / Get all dict keys
    print (file.keys()) #gets names of datasets that are in the file
    dataset = file["test_dataset"]
```

---

➡️ **Next / 下一步**: File 9 of 13

---

### Groups



---

### Create With Groups



---

### Keras Save

# 15 — Keras Save / 保存/加载模型

**Chapter 14 — File 11 of 13 / 第14章 — 第11个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Create model**.

本脚本演示 **Create model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import keras
```

---
## Step 2 — Create model

```python
model = keras.models.Sequential([
 	keras.layers.Input(shape=(10,)),
  # 全连接层（Keras） / Fully connected layer (Keras)
 	keras.layers.Dense(1)
])

# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam", loss="mse")
```

---
## Step 3 — using the .h5 extension in the file name specifies that the model
should be saved in HDF5 format

```python
# 保存模型到文件 / Save model to file
model.save("my_model.h5")
```

---
## Learning Notes / 学习笔记

- **概念**: Create model 是机器学习中的常用技术。  
  *Create model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Keras Save / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow import keras

# Create model
model = keras.models.Sequential([
 	keras.layers.Input(shape=(10,)),
  # 全连接层（Keras） / Fully connected layer (Keras)
 	keras.layers.Dense(1)
])

# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer="adam", loss="mse")

# using the .h5 extension in the file name specifies that the model
# should be saved in HDF5 format
# 保存模型到文件 / Save model to file
model.save("my_model.h5")
```

---

➡️ **Next / 下一步**: File 12 of 13

---

### Keras Layer

# 17 — Keras Layer / Keras

**Chapter 14 — File 12 of 13 / 第14章 — 第12个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Keras Layer**.

本脚本演示 **Keras**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import h5py

with h5py.File("my_model.h5", "r") as infile:
    # 打印输出 / Print output
    print(infile["/model_weights/dense/dense/kernel:0"][:])
```

---
## Learning Notes / 学习笔记

- **概念**: Keras Layer 是机器学习中的常用技术。  
  *Keras Layer is a common technique in machine learning.*

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
# Keras Layer / Keras
# Complete Code / 完整代码
# ===============================

import h5py

with h5py.File("my_model.h5", "r") as infile:
    # 打印输出 / Print output
    print(infile["/model_weights/dense/dense/kernel:0"][:])
```

---

➡️ **Next / 下一步**: File 13 of 13

---

### Print Metadata

# 18 — Print Metadata / 18 Print Metadata

**Chapter 14 — File 13 of 13 / 第14章 — 第13个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Print Metadata**.

本脚本演示 **18 Print Metadata**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入JSON处理模块 / Import JSON processing module
import json
import h5py

with h5py.File("my_model.h5", "r") as infile:
    # 获取字典的所有键 / Get all dict keys
    for key in infile.attrs.keys():
        formatted = infile.attrs[key]
        if key.endswith("_config"):
            # 读取JSON文件 / Read JSON file
            formatted = json.dumps(json.loads(formatted), indent=4)
        # 打印输出 / Print output
        print(f"{key}: {formatted}")
```

---
## Learning Notes / 学习笔记

- **概念**: Print Metadata 是机器学习中的常用技术。  
  *Print Metadata is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Print Metadata / 18 Print Metadata
# Complete Code / 完整代码
# ===============================

# 导入JSON处理模块 / Import JSON processing module
import json
import h5py

with h5py.File("my_model.h5", "r") as infile:
    # 获取字典的所有键 / Get all dict keys
    for key in infile.attrs.keys():
        formatted = infile.attrs[key]
        if key.endswith("_config"):
            # 读取JSON文件 / Read JSON file
            formatted = json.dumps(json.loads(formatted), indent=4)
        # 打印输出 / Print output
        print(f"{key}: {formatted}")
```

---

### Chapter Summary / 章节总结

# Chapter 14 Summary / 第14章总结

## Theme / 主题: Chapter 14 / Chapter 14

This chapter contains **13 code files** demonstrating chapter 14.

本章包含 **13 个代码文件**，演示Chapter 14。

---
## Evolution / 演化路线

  1. `01_dump.ipynb` — Dump
  2. `02_load.ipynb` — Load
  3. `03_serialization.ipynb` — Serialization
  4. `06_userdefined.ipynb` — Userdefined
  5. `07_functions.ipynb` — Functions
  6. `08_keras.ipynb` — Keras
  7. `09_hdf5.ipynb` — Hdf5
  8. `12_hdf5.ipynb` — Hdf5
  9. `13_groups.ipynb` — Groups
  10. `14_create_with_groups.ipynb` — Create With Groups
  11. `15_keras_save.ipynb` — Keras Save
  12. `17_keras_layer.ipynb` — Keras Layer
  13. `18_print_metadata.ipynb` — Print Metadata

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 14) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 14）是机器学习流水线中的基础构建块。

---
