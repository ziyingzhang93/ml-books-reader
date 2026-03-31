# Python 深度学习 / Deep Learning with Python
## Chapter 21

---

### Time Based Decay

# 01 — Time Based Decay / 01 Time Based Decay

**Chapter 21 — File 1 of 2 / 第21章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Time Based Learning Rate Decay**.

本脚本演示 **Time Based Learning Rate Decay**。

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
```

---
## Step 1 — Time Based Learning Rate Decay

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.optimizers import SGD
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("ionosphere.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
```

---
## Step 3 — split into input (X) and output (Y) variables

```python
# 转换数据类型 / Convert data type
X = dataset[:,0:34].astype(float)
Y = dataset[:,34]
```

---
## Step 4 — encode class values as integers

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
Y = encoder.transform(Y)
```

---
## Step 5 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(34, input_shape=(34,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
```

---
## Step 6 — Compile model

```python
epochs = 50
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate,
          nesterov=False)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
```

---
## Step 7 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=28, verbose=2)
```

---
## Learning Notes / 学习笔记

- **概念**: Time Based Learning Rate Decay 是机器学习中的常用技术。  
  *Time Based Learning Rate Decay is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Time Based Decay / 01 Time Based Decay
# Complete Code / 完整代码
# ===============================

# Time Based Learning Rate Decay
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.optimizers import SGD
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("ionosphere.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# split into input (X) and output (Y) variables
# 转换数据类型 / Convert data type
X = dataset[:,0:34].astype(float)
Y = dataset[:,34]
# encode class values as integers
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
Y = encoder.transform(Y)
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(34, input_shape=(34,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# Compile model
epochs = 50
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate,
          nesterov=False)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Fit the model
# 训练模型 / Train the model
model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=28, verbose=2)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Drop Based Decay

# 02 — Drop Based Decay / 02 Drop Based Decay

**Chapter 21 — File 2 of 2 / 第21章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Drop-Based Learning Rate Decay**.

本脚本演示 **Drop-Based Learning Rate Decay**。

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
```

---
## Step 1 — Drop-Based Learning Rate Decay

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入数学函数库 / Import math functions library
import math
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.optimizers import SGD
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import LearningRateScheduler
```

---
## Step 2 — learning rate schedule

```python
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
```

---
## Step 3 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("ionosphere.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
```

---
## Step 4 — split into input (X) and output (Y) variables

```python
# 转换数据类型 / Convert data type
X = dataset[:,0:34].astype(float)
Y = dataset[:,34]
```

---
## Step 5 — encode class values as integers

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
Y = encoder.transform(Y)
```

---
## Step 6 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(34, input_shape=(34,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
```

---
## Step 7 — Compile model

```python
sgd = SGD(learning_rate=0.0, momentum=0.9)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
```

---
## Step 8 — learning schedule callback

```python
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
```

---
## Step 9 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X, Y, validation_split=0.33, epochs=50, batch_size=28,
          callbacks=callbacks_list, verbose=2)
```

---
## Learning Notes / 学习笔记

- **概念**: Drop-Based Learning Rate Decay 是机器学习中的常用技术。  
  *Drop-Based Learning Rate Decay is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Drop Based Decay / 02 Drop Based Decay
# Complete Code / 完整代码
# ===============================

# Drop-Based Learning Rate Decay
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入数学函数库 / Import math functions library
import math
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.optimizers import SGD
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import LearningRateScheduler

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv("ionosphere.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# split into input (X) and output (Y) variables
# 转换数据类型 / Convert data type
X = dataset[:,0:34].astype(float)
Y = dataset[:,34]
# encode class values as integers
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(Y)
# 用已拟合的模型转换数据 / Transform data with fitted model
Y = encoder.transform(Y)
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(34, input_shape=(34,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# Compile model
sgd = SGD(learning_rate=0.0, momentum=0.9)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
# Fit the model
# 训练模型 / Train the model
model.fit(X, Y, validation_split=0.33, epochs=50, batch_size=28,
          callbacks=callbacks_list, verbose=2)
```

---

### Chapter Summary / 章节总结



---
