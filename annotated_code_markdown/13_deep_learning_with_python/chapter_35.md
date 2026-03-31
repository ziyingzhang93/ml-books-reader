# Python 深度学习 / Deep Learning with Python
## Chapter 35

---

### Lstm

# 10 — Lstm / LSTM 网络

**Chapter 35 — File 1 of 4 / 第35章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Small LSTM Network to Generate Text for Alice in Wonderland**.

本脚本演示 **Small LSTM Network to Generate Text for Alice in Wonderland**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  🏋️ 训练模型 / Train Model
```

---
## Step 1 — Small LSTM Network to Generate Text for Alice in Wonderland

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
```

---
## Step 2 — load ascii text and covert to lowercase

```python
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
```

---
## Step 3 — create mapping of unique chars to integers

```python
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))
```

---
## Step 4 — summarize the loaded data

```python
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)
```

---
## Step 5 — prepare the dataset of input to output pairs encoded as integers

```python
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
# 打印输出 / Print output
print("Total Patterns: ", n_patterns)
```

---
## Step 6 — reshape X to be [samples, time steps, features]

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (n_patterns, seq_length, 1))
```

---
## Step 7 — normalize

```python
X = X / float(n_vocab)
```

---
## Step 8 — one-hot encode the output variable

```python
y = to_categorical(dataY)
```

---
## Step 9 — define the LSTM model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

---
## Step 10 — define the checkpoint

```python
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# 模型检查点：训练中保存最佳模型 / ModelCheckpoint: save best model during training
checkpoint = ModelCheckpoint(filepath, monitor='loss',
                             verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
```

---
## Step 11 — fit the model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
```

---
## Learning Notes / 学习笔记

- **概念**: Small LSTM Network to Generate Text for Alice in Wonderland 是机器学习中的常用技术。  
  *Small LSTM Network to Generate Text for Alice in Wonderland is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `ModelCheckpoint` | 保存最佳模型 | Save best model during training |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lstm / LSTM 网络
# Complete Code / 完整代码
# ===============================

# Small LSTM Network to Generate Text for Alice in Wonderland
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
# 打印输出 / Print output
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one-hot encode the output variable
y = to_categorical(dataY)
# define the LSTM model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# 模型检查点：训练中保存最佳模型 / ModelCheckpoint: save best model during training
checkpoint = ModelCheckpoint(filepath, monitor='loss',
                             verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
# 训练模型 / Train the model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Generate

# 14 — Generate / 14 Generate

**Chapter 35 — File 2 of 4 / 第35章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load LSTM network and generate text**.

本脚本演示 **Load LSTM network and generate text**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Load LSTM network and generate text

```python
# 导入系统相关功能 / Import system utilities
import sys
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
```

---
## Step 2 — load ascii text and covert to lowercase

```python
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
```

---
## Step 3 — create mapping of unique chars to integers, and a reverse mapping

```python
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(chars))
```

---
## Step 4 — summarize the loaded data

```python
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)
```

---
## Step 5 — prepare the dataset of input to output pairs encoded as integers

```python
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
# 打印输出 / Print output
print("Total Patterns: ", n_patterns)
```

---
## Step 6 — reshape X to be [samples, time steps, features]

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (n_patterns, seq_length, 1))
```

---
## Step 7 — normalize

```python
X = X / float(n_vocab)
```

---
## Step 8 — one-hot encode the output variable

```python
y = to_categorical(dataY)
```

---
## Step 9 — define the LSTM model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
```

---
## Step 10 — load the network weights (modify to your filename)

```python
filename = "weights-improvement-19-1.9435.hdf5"
model.load_weights(filename)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

---
## Step 11 — pick a random seed

```python
# 生成随机数 / Generate random numbers
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
# 打印输出 / Print output
print("Seed:")
# 打印输出 / Print output
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
```

---
## Step 12 — generate characters

```python
# 生成整数序列 / Generate integer sequence
for i in range(1000):
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    # 添加元素到列表末尾 / Append element to list end
    pattern.append(index)
    # 获取长度 / Get length
    pattern = pattern[1:len(pattern)]
# 打印输出 / Print output
print("\nDone.")
```

---
## Learning Notes / 学习笔记

- **概念**: Load LSTM network and generate text 是机器学习中的常用技术。  
  *Load LSTM network and generate text is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `ModelCheckpoint` | 保存最佳模型 | Save best model during training |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
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
# Generate / 14 Generate
# Complete Code / 完整代码
# ===============================

# Load LSTM network and generate text
# 导入系统相关功能 / Import system utilities
import sys
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
# 打印输出 / Print output
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one-hot encode the output variable
y = to_categorical(dataY)
# define the LSTM model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights (modify to your filename)
filename = "weights-improvement-19-1.9435.hdf5"
model.load_weights(filename)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
# 生成随机数 / Generate random numbers
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
# 打印输出 / Print output
print("Seed:")
# 打印输出 / Print output
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
# 生成整数序列 / Generate integer sequence
for i in range(1000):
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    # 添加元素到列表末尾 / Append element to list end
    pattern.append(index)
    # 获取长度 / Get length
    pattern = pattern[1:len(pattern)]
# 打印输出 / Print output
print("\nDone.")
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Larger

# 17 — Larger / 17 Larger

**Chapter 35 — File 3 of 4 / 第35章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Larger LSTM Network to Generate Text for Alice in Wonderland**.

本脚本演示 **Larger LSTM Network to Generate Text for Alice in Wonderland**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  🏋️ 训练模型 / Train Model
```

---
## Step 1 — Larger LSTM Network to Generate Text for Alice in Wonderland

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
```

---
## Step 2 — load ascii text and covert to lowercase

```python
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
```

---
## Step 3 — create mapping of unique chars to integers

```python
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))
```

---
## Step 4 — summarize the loaded data

```python
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)
```

---
## Step 5 — prepare the dataset of input to output pairs encoded as integers

```python
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
# 打印输出 / Print output
print("Total Patterns: ", n_patterns)
```

---
## Step 6 — reshape X to be [samples, time steps, features]

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (n_patterns, seq_length, 1))
```

---
## Step 7 — normalize

```python
X = X / float(n_vocab)
```

---
## Step 8 — one-hot encode the output variable

```python
y = to_categorical(dataY)
```

---
## Step 9 — define the LSTM model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(256))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

---
## Step 10 — define the checkpoint

```python
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
# 模型检查点：训练中保存最佳模型 / ModelCheckpoint: save best model during training
checkpoint = ModelCheckpoint(filepath, monitor='loss',
                             verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
```

---
## Step 11 — fit the model

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)
```

---
## Learning Notes / 学习笔记

- **概念**: Larger LSTM Network to Generate Text for Alice in Wonderland 是机器学习中的常用技术。  
  *Larger LSTM Network to Generate Text for Alice in Wonderland is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `ModelCheckpoint` | 保存最佳模型 | Save best model during training |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Larger / 17 Larger
# Complete Code / 完整代码
# ===============================

# Larger LSTM Network to Generate Text for Alice in Wonderland
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
# 打印输出 / Print output
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one-hot encode the output variable
y = to_categorical(dataY)
# define the LSTM model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(256))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
# 模型检查点：训练中保存最佳模型 / ModelCheckpoint: save best model during training
checkpoint = ModelCheckpoint(filepath, monitor='loss',
                             verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
# 训练模型 / Train the model
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Generate Larger

# 18 — Generate Larger / 18 Generate Larger

**Chapter 35 — File 4 of 4 / 第35章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load Larger LSTM network and generate text**.

本脚本演示 **Load Larger LSTM network and generate text**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Load Larger LSTM network and generate text

```python
# 导入系统相关功能 / Import system utilities
import sys
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
```

---
## Step 2 — load ascii text and covert to lowercase

```python
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
```

---
## Step 3 — create mapping of unique chars to integers, and a reverse mapping

```python
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(chars))
```

---
## Step 4 — summarize the loaded data

```python
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)
```

---
## Step 5 — prepare the dataset of input to output pairs encoded as integers

```python
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
# 打印输出 / Print output
print("Total Patterns: ", n_patterns)
```

---
## Step 6 — reshape X to be [samples, time steps, features]

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (n_patterns, seq_length, 1))
```

---
## Step 7 — normalize

```python
X = X / float(n_vocab)
```

---
## Step 8 — one-hot encode the output variable

```python
y = to_categorical(dataY)
```

---
## Step 9 — define the LSTM model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(256))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
```

---
## Step 10 — load the network weights (modify to your filename)

```python
filename = "weights-improvement-47-1.2219-bigger.hdf5"
model.load_weights(filename)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

---
## Step 11 — pick a random seed

```python
# 生成随机数 / Generate random numbers
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
# 打印输出 / Print output
print("Seed:")
# 打印输出 / Print output
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
```

---
## Step 12 — generate characters

```python
# 生成整数序列 / Generate integer sequence
for i in range(1000):
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    # 添加元素到列表末尾 / Append element to list end
    pattern.append(index)
    # 获取长度 / Get length
    pattern = pattern[1:len(pattern)]
# 打印输出 / Print output
print("\nDone.")
```

---
## Learning Notes / 学习笔记

- **概念**: Load Larger LSTM network and generate text 是机器学习中的常用技术。  
  *Load Larger LSTM network and generate text is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `ModelCheckpoint` | 保存最佳模型 | Save best model during training |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
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
# Generate Larger / 18 Generate Larger
# Complete Code / 完整代码
# ===============================

# Load Larger LSTM network and generate text
# 导入系统相关功能 / Import system utilities
import sys
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))
# 同时获取索引和值 / Get both index and value
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
# 打印输出 / Print output
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one-hot encode the output variable
y = to_categorical(dataY)
# define the LSTM model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(256))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights (modify to your filename)
filename = "weights-improvement-47-1.2219-bigger.hdf5"
model.load_weights(filename)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
# 生成随机数 / Generate random numbers
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
# 打印输出 / Print output
print("Seed:")
# 打印输出 / Print output
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
# 生成整数序列 / Generate integer sequence
for i in range(1000):
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    # 用模型做预测 / Make predictions with model
    prediction = model.predict(x, verbose=0)
    # 找最大值的索引位置 / Find index of maximum value
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    # 添加元素到列表末尾 / Append element to list end
    pattern.append(index)
    # 获取长度 / Get length
    pattern = pattern[1:len(pattern)]
# 打印输出 / Print output
print("\nDone.")
```

---

### Chapter Summary / 章节总结



---
