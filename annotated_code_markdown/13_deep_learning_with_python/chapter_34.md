# Python 深度学习 / Deep Learning with Python
## Chapter 34

---

### Lstm Mapping

# 12 — Lstm Mapping / LSTM 网络

**Chapter 34 — File 1 of 6 / 第34章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Naive LSTM to learn one-char to one-char mapping**.

本脚本演示 **Naive LSTM to learn one-char to one-char mapping**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — Naive LSTM to learn one-char to one-char mapping

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
```

---
## Step 2 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 3 — define the raw dataset

```python
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

---
## Step 4 — create mapping of characters to integers (0-25) and the reverse

```python
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
```

---
## Step 5 — prepare the dataset of input to output pairs encoded as integers

```python
seq_length = 1
dataX = []
dataY = []
# 获取长度 / Get length
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
    # 打印输出 / Print output
    print(seq_in, '->', seq_out)
```

---
## Step 6 — reshape X to be [samples, time steps, features]

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (len(dataX), seq_length, 1))
```

---
## Step 7 — normalize

```python
# 获取长度 / Get length
X = X / float(len(alphabet))
```

---
## Step 8 — one-hot encode the output variable

```python
y = to_categorical(dataY)
```

---
## Step 9 — create and fit the model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型 / Train the model
model.fit(X, y, epochs=500, batch_size=1, verbose=2)
```

---
## Step 10 — summarize performance of the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, y, verbose=0)
# 打印输出 / Print output
print("Model Accuracy: %.2f%%" % (scores[1]*100))
```

---
## Step 11 — demonstrate some model predictions

```python
for pattern in dataX:
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(pattern, (1, len(pattern), 1))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # 打印输出 / Print output
    print(seq_in, "->", result)
```

---
## Learning Notes / 学习笔记

- **概念**: Naive LSTM to learn one-char to one-char mapping 是机器学习中的常用技术。  
  *Naive LSTM to learn one-char to one-char mapping is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lstm Mapping / LSTM 网络
# Complete Code / 完整代码
# ===============================

# Naive LSTM to learn one-char to one-char mapping
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []
# 获取长度 / Get length
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
    # 打印输出 / Print output
    print(seq_in, '->', seq_out)
# reshape X to be [samples, time steps, features]
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (len(dataX), seq_length, 1))
# normalize
# 获取长度 / Get length
X = X / float(len(alphabet))
# one-hot encode the output variable
y = to_categorical(dataY)
# create and fit the model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型 / Train the model
model.fit(X, y, epochs=500, batch_size=1, verbose=2)
# summarize performance of the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, y, verbose=0)
# 打印输出 / Print output
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
for pattern in dataX:
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(pattern, (1, len(pattern), 1))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # 打印输出 / Print output
    print(seq_in, "->", result)
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Three To One

# 16 — Three To One / 16 Three To One

**Chapter 34 — File 2 of 6 / 第34章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Naive LSTM to learn three-char window to one-char mapping**.

本脚本演示 **Naive LSTM to learn three-char window to one-char mapping**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — Naive LSTM to learn three-char window to one-char mapping

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
```

---
## Step 2 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 3 — define the raw dataset

```python
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

---
## Step 4 — create mapping of characters to integers (0-25) and the reverse

```python
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
```

---
## Step 5 — prepare the dataset of input to output pairs encoded as integers

```python
seq_length = 3
dataX = []
dataY = []
# 获取长度 / Get length
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
    # 打印输出 / Print output
    print(seq_in, '->', seq_out)
```

---
## Step 6 — reshape X to be [samples, time steps, features]

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (len(dataX), 1, seq_length))
```

---
## Step 7 — normalize

```python
# 获取长度 / Get length
X = X / float(len(alphabet))
```

---
## Step 8 — one-hot encode the output variable

```python
y = to_categorical(dataY)
```

---
## Step 9 — create and fit the model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型 / Train the model
model.fit(X, y, epochs=500, batch_size=1, verbose=2)
```

---
## Step 10 — summarize performance of the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, y, verbose=0)
# 打印输出 / Print output
print("Model Accuracy: %.2f%%" % (scores[1]*100))
```

---
## Step 11 — demonstrate some model predictions

```python
for pattern in dataX:
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(pattern, (1, 1, len(pattern)))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # 打印输出 / Print output
    print(seq_in, "->", result)
```

---
## Learning Notes / 学习笔记

- **概念**: Naive LSTM to learn three-char window to one-char mapping 是机器学习中的常用技术。  
  *Naive LSTM to learn three-char window to one-char mapping is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Three To One / 16 Three To One
# Complete Code / 完整代码
# ===============================

# Naive LSTM to learn three-char window to one-char mapping
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 3
dataX = []
dataY = []
# 获取长度 / Get length
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
    # 打印输出 / Print output
    print(seq_in, '->', seq_out)
# reshape X to be [samples, time steps, features]
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (len(dataX), 1, seq_length))
# normalize
# 获取长度 / Get length
X = X / float(len(alphabet))
# one-hot encode the output variable
y = to_categorical(dataY)
# create and fit the model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型 / Train the model
model.fit(X, y, epochs=500, batch_size=1, verbose=2)
# summarize performance of the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, y, verbose=0)
# 打印输出 / Print output
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
for pattern in dataX:
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(pattern, (1, 1, len(pattern)))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # 打印输出 / Print output
    print(seq_in, "->", result)
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Three To One

# 19 — Three To One / 19 Three To One

**Chapter 34 — File 3 of 6 / 第34章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Naive LSTM to learn three-char time steps to one-char mapping**.

本脚本演示 **Naive LSTM to learn three-char time steps to one-char mapping**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — Naive LSTM to learn three-char time steps to one-char mapping

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
```

---
## Step 2 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 3 — define the raw dataset

```python
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

---
## Step 4 — create mapping of characters to integers (0-25) and the reverse

```python
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
```

---
## Step 5 — prepare the dataset of input to output pairs encoded as integers

```python
seq_length = 3
dataX = []
dataY = []
# 获取长度 / Get length
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
    # 打印输出 / Print output
    print(seq_in, '->', seq_out)
```

---
## Step 6 — reshape X to be [samples, time steps, features]

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (len(dataX), seq_length, 1))
```

---
## Step 7 — normalize

```python
# 获取长度 / Get length
X = X / float(len(alphabet))
```

---
## Step 8 — one-hot encode the output variable

```python
y = to_categorical(dataY)
```

---
## Step 9 — create and fit the model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型 / Train the model
model.fit(X, y, epochs=500, batch_size=1, verbose=2)
```

---
## Step 10 — summarize performance of the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, y, verbose=0)
# 打印输出 / Print output
print("Model Accuracy: %.2f%%" % (scores[1]*100))
```

---
## Step 11 — demonstrate some model predictions

```python
for pattern in dataX:
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(pattern, (1, len(pattern), 1))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # 打印输出 / Print output
    print(seq_in, "->", result)
```

---
## Learning Notes / 学习笔记

- **概念**: Naive LSTM to learn three-char time steps to one-char mapping 是机器学习中的常用技术。  
  *Naive LSTM to learn three-char time steps to one-char mapping is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Three To One / 19 Three To One
# Complete Code / 完整代码
# ===============================

# Naive LSTM to learn three-char time steps to one-char mapping
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 3
dataX = []
dataY = []
# 获取长度 / Get length
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
    # 打印输出 / Print output
    print(seq_in, '->', seq_out)
# reshape X to be [samples, time steps, features]
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (len(dataX), seq_length, 1))
# normalize
# 获取长度 / Get length
X = X / float(len(alphabet))
# one-hot encode the output variable
y = to_categorical(dataY)
# create and fit the model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型 / Train the model
model.fit(X, y, epochs=500, batch_size=1, verbose=2)
# summarize performance of the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, y, verbose=0)
# 打印输出 / Print output
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
for pattern in dataX:
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(pattern, (1, len(pattern), 1))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # 打印输出 / Print output
    print(seq_in, "->", result)
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Batch

# 21 — Batch / 21 Batch

**Chapter 34 — File 4 of 6 / 第34章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Naive LSTM to learn one-char to one-char mapping with all data in each batch**.

本脚本演示 **Naive LSTM to learn one-char to one-char mapping with all data in each batch**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — Naive LSTM to learn one-char to one-char mapping with all data in each batch

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

---
## Step 2 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 3 — define the raw dataset

```python
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

---
## Step 4 — create mapping of characters to integers (0-25) and the reverse

```python
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
```

---
## Step 5 — prepare the dataset of input to output pairs encoded as integers

```python
seq_length = 1
dataX = []
dataY = []
# 获取长度 / Get length
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
    # 打印输出 / Print output
    print(seq_in, '->', seq_out)
```

---
## Step 6 — convert list of lists to array and pad sequences if needed

```python
X = pad_sequences(dataX, maxlen=seq_length, dtype='float32')
```

---
## Step 7 — reshape X to be [samples, time steps, features]

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = np.reshape(dataX, (X.shape[0], seq_length, 1))
```

---
## Step 8 — normalize

```python
# 获取长度 / Get length
X = X / float(len(alphabet))
```

---
## Step 9 — one-hot encode the output variable

```python
y = to_categorical(dataY)
```

---
## Step 10 — create and fit the model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(16, input_shape=(X.shape[1], X.shape[2])))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型 / Train the model
model.fit(X, y, epochs=5000, batch_size=len(dataX), verbose=2, shuffle=False)
```

---
## Step 11 — summarize performance of the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, y, verbose=0)
# 打印输出 / Print output
print("Model Accuracy: %.2f%%" % (scores[1]*100))
```

---
## Step 12 — demonstrate some model predictions

```python
for pattern in dataX:
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(pattern, (1, len(pattern), 1))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # 打印输出 / Print output
    print(seq_in, "->", result)
```

---
## Step 13 — demonstrate predicting random patterns

```python
# 打印输出 / Print output
print("Test a Random Pattern:")
# 生成整数序列 / Generate integer sequence
for i in range(0,20):
    # 生成随机数 / Generate random numbers
    pattern_index = np.random.randint(len(dataX))
    pattern = dataX[pattern_index]
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(pattern, (1, len(pattern), 1))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # 打印输出 / Print output
    print(seq_in, "->", result)
```

---
## Learning Notes / 学习笔记

- **概念**: Naive LSTM to learn one-char to one-char mapping with all data in each batch 是机器学习中的常用技术。  
  *Naive LSTM to learn one-char to one-char mapping with all data in each batch is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.random` | 随机数生成 | Random number generation |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Batch / 21 Batch
# Complete Code / 完整代码
# ===============================

# Naive LSTM to learn one-char to one-char mapping with all data in each batch
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.sequence import pad_sequences
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []
# 获取长度 / Get length
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
    # 打印输出 / Print output
    print(seq_in, '->', seq_out)
# convert list of lists to array and pad sequences if needed
X = pad_sequences(dataX, maxlen=seq_length, dtype='float32')
# reshape X to be [samples, time steps, features]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = np.reshape(dataX, (X.shape[0], seq_length, 1))
# normalize
# 获取长度 / Get length
X = X / float(len(alphabet))
# one-hot encode the output variable
y = to_categorical(dataY)
# create and fit the model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(16, input_shape=(X.shape[1], X.shape[2])))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型 / Train the model
model.fit(X, y, epochs=5000, batch_size=len(dataX), verbose=2, shuffle=False)
# summarize performance of the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, y, verbose=0)
# 打印输出 / Print output
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
for pattern in dataX:
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(pattern, (1, len(pattern), 1))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # 打印输出 / Print output
    print(seq_in, "->", result)
# demonstrate predicting random patterns
# 打印输出 / Print output
print("Test a Random Pattern:")
# 生成整数序列 / Generate integer sequence
for i in range(0,20):
    # 生成随机数 / Generate random numbers
    pattern_index = np.random.randint(len(dataX))
    pattern = dataX[pattern_index]
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(pattern, (1, len(pattern), 1))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # 打印输出 / Print output
    print(seq_in, "->", result)
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Stateful

# 27 — Stateful / 27 Stateful

**Chapter 34 — File 5 of 6 / 第34章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Stateful LSTM to learn one-char to one-char mapping**.

本脚本演示 **Stateful LSTM to learn one-char to one-char mapping**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — Stateful LSTM to learn one-char to one-char mapping

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
```

---
## Step 2 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 3 — define the raw dataset

```python
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

---
## Step 4 — create mapping of characters to integers (0-25) and the reverse

```python
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
```

---
## Step 5 — prepare the dataset of input to output pairs encoded as integers

```python
seq_length = 1
dataX = []
dataY = []
# 获取长度 / Get length
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
    # 打印输出 / Print output
    print(seq_in, '->', seq_out)
```

---
## Step 6 — reshape X to be [samples, time steps, features]

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (len(dataX), seq_length, 1))
```

---
## Step 7 — normalize

```python
# 获取长度 / Get length
X = X / float(len(alphabet))
```

---
## Step 8 — one-hot encode the output variable

```python
y = to_categorical(dataY)
```

---
## Step 9 — create and fit the model

```python
batch_size = 1
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(50, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 生成整数序列 / Generate integer sequence
for i in range(300):
    # 训练模型 / Train the model
    model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()
```

---
## Step 10 — summarize performance of the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, y, batch_size=batch_size, verbose=0)
model.reset_states()
# 打印输出 / Print output
print("Model Accuracy: %.2f%%" % (scores[1]*100))
```

---
## Step 11 — demonstrate some model predictions

```python
seed = [char_to_int[alphabet[0]]]
# 获取长度 / Get length
for i in range(0, len(alphabet)-1):
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(seed, (1, len(seed), 1))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    # 打印输出 / Print output
    print(int_to_char[seed[0]], "->", int_to_char[index])
    seed = [index]
model.reset_states()
```

---
## Step 12 — demonstrate a random starting point

```python
letter = "K"
seed = [char_to_int[letter]]
# 打印输出 / Print output
print("New start: ", letter)
# 生成整数序列 / Generate integer sequence
for i in range(0, 5):
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(seed, (1, len(seed), 1))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    # 打印输出 / Print output
    print(int_to_char[seed[0]], "->", int_to_char[index])
    seed = [index]
model.reset_states()
```

---
## Learning Notes / 学习笔记

- **概念**: Stateful LSTM to learn one-char to one-char mapping 是机器学习中的常用技术。  
  *Stateful LSTM to learn one-char to one-char mapping is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Stateful / 27 Stateful
# Complete Code / 完整代码
# ===============================

# Stateful LSTM to learn one-char to one-char mapping
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []
# 获取长度 / Get length
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
    # 打印输出 / Print output
    print(seq_in, '->', seq_out)
# reshape X to be [samples, time steps, features]
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (len(dataX), seq_length, 1))
# normalize
# 获取长度 / Get length
X = X / float(len(alphabet))
# one-hot encode the output variable
y = to_categorical(dataY)
# create and fit the model
batch_size = 1
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(50, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 生成整数序列 / Generate integer sequence
for i in range(300):
    # 训练模型 / Train the model
    model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()
# summarize performance of the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, y, batch_size=batch_size, verbose=0)
model.reset_states()
# 打印输出 / Print output
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
seed = [char_to_int[alphabet[0]]]
# 获取长度 / Get length
for i in range(0, len(alphabet)-1):
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(seed, (1, len(seed), 1))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    # 打印输出 / Print output
    print(int_to_char[seed[0]], "->", int_to_char[index])
    seed = [index]
model.reset_states()
# demonstrate a random starting point
letter = "K"
seed = [char_to_int[letter]]
# 打印输出 / Print output
print("New start: ", letter)
# 生成整数序列 / Generate integer sequence
for i in range(0, 5):
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(seed, (1, len(seed), 1))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    # 打印输出 / Print output
    print(int_to_char[seed[0]], "->", int_to_char[index])
    seed = [index]
model.reset_states()
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Variable

# 30 — Variable / 30 Variable

**Chapter 34 — File 6 of 6 / 第34章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **LSTM with Variable Length Input Sequences to One Character Output**.

本脚本演示 **LSTM with Variable Length Input Sequences to One Character Output**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — LSTM with Variable Length Input Sequences to One Character Output

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

---
## Step 2 — fix random seed for reproducibility

```python
# 生成随机数 / Generate random numbers
np.random.seed(7)
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 3 — define the raw dataset

```python
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

---
## Step 4 — create mapping of characters to integers (0-25) and the reverse

```python
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
```

---
## Step 5 — prepare the dataset of input to output pairs encoded as integers

```python
num_inputs = 1000
max_len = 5
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(num_inputs):
    # 生成随机数 / Generate random numbers
    start = np.random.randint(len(alphabet)-2)
    # 生成随机数 / Generate random numbers
    end = np.random.randint(start, min(start+max_len,len(alphabet)-1))
    sequence_in = alphabet[start:end+1]
    sequence_out = alphabet[end + 1]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in sequence_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[sequence_out])
    # 打印输出 / Print output
    print(sequence_in, '->', sequence_out)
```

---
## Step 6 — convert list of lists to array and pad sequences if needed

```python
X = pad_sequences(dataX, maxlen=max_len, dtype='float32')
```

---
## Step 7 — reshape X to be [samples, time steps, features]

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = np.reshape(X, (X.shape[0], max_len, 1))
```

---
## Step 8 — normalize

```python
# 获取长度 / Get length
X = X / float(len(alphabet))
```

---
## Step 9 — one-hot encode the output variable

```python
y = to_categorical(dataY)
```

---
## Step 10 — create and fit the model

```python
batch_size = 1
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(32, input_shape=(X.shape[1], 1)))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型 / Train the model
model.fit(X, y, epochs=500, batch_size=batch_size, verbose=2)
```

---
## Step 11 — summarize performance of the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, y, verbose=0)
# 打印输出 / Print output
print("Model Accuracy: %.2f%%" % (scores[1]*100))
```

---
## Step 12 — demonstrate some model predictions

```python
# 生成整数序列 / Generate integer sequence
for i in range(20):
    # 生成随机数 / Generate random numbers
    pattern_index = np.random.randint(len(dataX))
    pattern = dataX[pattern_index]
    x = pad_sequences([pattern], maxlen=max_len, dtype='float32')
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(x, (1, max_len, 1))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # 打印输出 / Print output
    print(seq_in, "->", result)
```

---
## Learning Notes / 学习笔记

- **概念**: LSTM with Variable Length Input Sequences to One Character Output 是机器学习中的常用技术。  
  *LSTM with Variable Length Input Sequences to One Character Output is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.random` | 随机数生成 | Random number generation |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Variable / 30 Variable
# Complete Code / 完整代码
# ===============================

# LSTM with Variable Length Input Sequences to One Character Output
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.sequence import pad_sequences
# fix random seed for reproducibility
# 生成随机数 / Generate random numbers
np.random.seed(7)
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
num_inputs = 1000
max_len = 5
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(num_inputs):
    # 生成随机数 / Generate random numbers
    start = np.random.randint(len(alphabet)-2)
    # 生成随机数 / Generate random numbers
    end = np.random.randint(start, min(start+max_len,len(alphabet)-1))
    sequence_in = alphabet[start:end+1]
    sequence_out = alphabet[end + 1]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in sequence_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[sequence_out])
    # 打印输出 / Print output
    print(sequence_in, '->', sequence_out)
# convert list of lists to array and pad sequences if needed
X = pad_sequences(dataX, maxlen=max_len, dtype='float32')
# reshape X to be [samples, time steps, features]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X = np.reshape(X, (X.shape[0], max_len, 1))
# normalize
# 获取长度 / Get length
X = X / float(len(alphabet))
# one-hot encode the output variable
y = to_categorical(dataY)
# create and fit the model
batch_size = 1
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(32, input_shape=(X.shape[1], 1)))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型 / Train the model
model.fit(X, y, epochs=500, batch_size=batch_size, verbose=2)
# summarize performance of the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X, y, verbose=0)
# 打印输出 / Print output
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
# 生成整数序列 / Generate integer sequence
for i in range(20):
    # 生成随机数 / Generate random numbers
    pattern_index = np.random.randint(len(dataX))
    pattern = dataX[pattern_index]
    x = pad_sequences([pattern], maxlen=max_len, dtype='float32')
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(x, (1, max_len, 1))
    # 获取长度 / Get length
    x = x / float(len(alphabet))
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    # 打印输出 / Print output
    print(seq_in, "->", result)
```

---

### Chapter Summary / 章节总结



---
