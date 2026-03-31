# Python 机器学习 / Python for Machine Learning
## Chapter 30

---

### Flask

# 01 — Flask / 01 Flask

**Chapter 30 — File 1 of 3 / 第30章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Flask**.

本脚本演示 **01 Flask**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from datetime import datetime
import pytz
from flask import Flask

app = Flask("time now")

@app.route("/now/<path:timezone>")
def timenow(timezone):
    try:
        zone = pytz.timezone(timezone)
        now = datetime.now(zone)
        return now.strftime("%Y-%m-%d %H:%M:%S %z %Z\n")
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Unknown time zone: {timezone}\n"

app.run()
```

---
## Learning Notes / 学习笔记

- **概念**: Flask 是机器学习中的常用技术。  
  *Flask is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Flask / 01 Flask
# Complete Code / 完整代码
# ===============================

from datetime import datetime
import pytz
from flask import Flask

app = Flask("time now")

@app.route("/now/<path:timezone>")
def timenow(timezone):
    try:
        zone = pytz.timezone(timezone)
        now = datetime.now(zone)
        return now.strftime("%Y-%m-%d %H:%M:%S %z %Z\n")
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Unknown time zone: {timezone}\n"

app.run()
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Utc

# 03 — Utc / 03 Utc

**Chapter 30 — File 2 of 3 / 第30章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Utc**.

本脚本演示 **03 Utc**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from datetime import datetime
import pytz
from flask import Flask

app = Flask("time now")

@app.route('/now', defaults={'timezone': ''})
@app.route("/now/<path:timezone>")
def timenow(timezone):
    try:
        if not timezone:
            zone = pytz.utc
        else:
            zone = pytz.timezone(timezone)
        now = datetime.now(zone)
        return now.strftime("%Y-%m-%d %H:%M:%S %z %Z\n")
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Unknown timezone: {timezone}\n"

app.run()
```

---
## Learning Notes / 学习笔记

- **概念**: Utc 是机器学习中的常用技术。  
  *Utc is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Utc / 03 Utc
# Complete Code / 完整代码
# ===============================

from datetime import datetime
import pytz
from flask import Flask

app = Flask("time now")

@app.route('/now', defaults={'timezone': ''})
@app.route("/now/<path:timezone>")
def timenow(timezone):
    try:
        if not timezone:
            zone = pytz.utc
        else:
            zone = pytz.timezone(timezone)
        now = datetime.now(zone)
        return now.strftime("%Y-%m-%d %H:%M:%S %z %Z\n")
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Unknown timezone: {timezone}\n"

app.run()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Lenet5

# 05 — Lenet5 / 05 Lenet5

**Chapter 30 — File 3 of 3 / 第30章 — 第3个文件（共3个）**

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
```

---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
```

---
## Step 2 — Load MNIST digits

```python
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

---
## Step 3 — Reshape data to (n_samples, height, width, n_channel)

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
    Conv2D(6, (5,5), activation="tanh",
           input_shape=(28,28,1), padding="same"),
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
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)
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
# Lenet5 / 05 Lenet5
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical

# Load MNIST digits
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data to (n_samples, height, width, n_channel)
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
    Conv2D(6, (5,5), activation="tanh",
           input_shape=(28,28,1), padding="same"),
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
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)
```

---

### Chapter Summary / 章节总结



---
