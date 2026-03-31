# 注意力与Transformer / Transformer Models with Attention
## Chapter 09

---

### Fibonacci

# 02 — Fibonacci / 02 Fibonacci

**Chapter 09 — File 1 of 7 / 第09章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Get the Fibonacci sequence**.

本脚本演示 **Get the Fibonacci sequence**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler

def get_fib_seq(n, scale_data=True):
```

---
## Step 2 — Get the Fibonacci sequence

```python
# 创建全零数组 / Create array of zeros
seq = np.zeros(n)
    fib_n1 = 0.0
    fib_n = 1.0
    # 生成整数序列 / Generate integer sequence
    for i in range(n):
            seq[i] = fib_n1 + fib_n
            fib_n1 = fib_n
            fib_n = seq[i]
    scaler = []
    if scale_data:
        # 归一化到[0,1]范围 / Normalize to [0,1] range
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
        seq = np.reshape(seq, (n, 1))
        # 展平为一维数组 / Flatten to 1D array
        seq = scaler.fit_transform(seq).flatten()
    return seq, scaler

fib_seq, _ = get_fib_seq(10, False)
# 打印输出 / Print output
print(fib_seq)
```

---
## Learning Notes / 学习笔记

- **概念**: Get the Fibonacci sequence 是机器学习中的常用技术。  
  *Get the Fibonacci sequence is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `np.zeros` | 全零数组 | Array filled with zeros |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Fibonacci / 02 Fibonacci
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler

def get_fib_seq(n, scale_data=True):
    # Get the Fibonacci sequence
    # 创建全零数组 / Create array of zeros
    seq = np.zeros(n)
    fib_n1 = 0.0
    fib_n = 1.0
    # 生成整数序列 / Generate integer sequence
    for i in range(n):
            seq[i] = fib_n1 + fib_n
            fib_n1 = fib_n
            fib_n = seq[i]
    scaler = []
    if scale_data:
        # 归一化到[0,1]范围 / Normalize to [0,1] range
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
        seq = np.reshape(seq, (n, 1))
        # 展平为一维数组 / Flatten to 1D array
        seq = scaler.fit_transform(seq).flatten()
    return seq, scaler

fib_seq, _ = get_fib_seq(10, False)
# 打印输出 / Print output
print(fib_seq)
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Split



---

### Rnn

# 04 — Rnn / 循环神经网络

**Chapter 09 — File 3 of 7 / 第09章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Set up parameters**.

本脚本演示 **Set up parameters**。

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
```

---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, SimpleRNN
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
```

---
## Step 2 — Set up parameters

```python
time_steps = 20
hidden_units = 2
epochs = 30
```

---
## Step 3 — Create a traditional RNN network

```python
def create_RNN(hidden_units, dense_units, input_shape, activation):
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(units=dense_units, activation=activation[1]))
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mse', optimizer='adam')
    return model

model_RNN = create_RNN(hidden_units=hidden_units, dense_units=1,
                       input_shape=(time_steps,1), activation=['tanh', 'tanh'])
model_RNN.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: Set up parameters 是机器学习中的常用技术。  
  *Set up parameters is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Rnn / 循环神经网络
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, SimpleRNN
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential

# Set up parameters
time_steps = 20
hidden_units = 2
epochs = 30

# Create a traditional RNN network
def create_RNN(hidden_units, dense_units, input_shape, activation):
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(units=dense_units, activation=activation[1]))
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mse', optimizer='adam')
    return model

model_RNN = create_RNN(hidden_units=hidden_units, dense_units=1,
                       input_shape=(time_steps,1), activation=['tanh', 'tanh'])
model_RNN.summary()
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Training

# 05 — Training / 05 Training

**Chapter 09 — File 4 of 7 / 第09章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Set up parameters**.

本脚本演示 **Set up parameters**。

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
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, SimpleRNN
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
```

---
## Step 2 — Set up parameters

```python
time_steps = 20
hidden_units = 2
epochs = 30

def get_fib_seq(n, scale_data=True):
```

---
## Step 3 — Get the Fibonacci sequence

```python
# 创建全零数组 / Create array of zeros
seq = np.zeros(n)
    fib_n1 = 0.0
    fib_n = 1.0
    # 生成整数序列 / Generate integer sequence
    for i in range(n):
            seq[i] = fib_n1 + fib_n
            fib_n1 = fib_n
            fib_n = seq[i]
    scaler = []
    if scale_data:
        # 归一化到[0,1]范围 / Normalize to [0,1] range
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
        seq = np.reshape(seq, (n, 1))
        # 展平为一维数组 / Flatten to 1D array
        seq = scaler.fit_transform(seq).flatten()
    return seq, scaler

def get_fib_XY(total_fib_numbers, time_steps, train_percent, scale_data=True):
    dat, scaler = get_fib_seq(total_fib_numbers, scale_data)
    # 生成等差数组 / Generate array with step
    Y_ind = np.arange(time_steps, len(dat), 1)
    Y = dat[Y_ind]
    # 获取长度 / Get length
    rows_x = len(Y)
    X = dat[0:rows_x]
    # 生成整数序列 / Generate integer sequence
    for i in range(time_steps-1):
        temp = dat[i+1:rows_x+i+1]
        X = np.column_stack((X, temp))
```

---
## Step 4 — random permutation with fixed seed

```python
# 生成随机数 / Generate random numbers
rand = np.random.RandomState(seed=13)
    idx = rand.permutation(rows_x)
    split = int(train_percent*rows_x)
    train_ind = idx[0:split]
    test_ind = idx[split:]
    trainX = X[train_ind]
    trainY = Y[train_ind]
    testX = X[test_ind]
    testY = Y[test_ind]
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    trainX = np.reshape(trainX, (len(trainX), time_steps, 1))
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    testX = np.reshape(testX, (len(testX), time_steps, 1))
    return trainX, trainY, testX, testY, scaler
```

---
## Step 5 — Create a traditional RNN network

```python
def create_RNN(hidden_units, dense_units, input_shape, activation):
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(units=dense_units, activation=activation[1]))
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mse', optimizer='adam')
    return model

model_RNN = create_RNN(hidden_units=hidden_units, dense_units=1,
                       input_shape=(time_steps,1), activation=['tanh', 'tanh'])
```

---
## Step 6 — Generate the dataset

```python
trainX, trainY, testX, testY, scaler  = get_fib_XY(1200, time_steps, 0.7)

model_RNN.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)
```

---
## Step 7 — Evalute model

```python
train_mse = model_RNN.evaluate(trainX, trainY)
test_mse = model_RNN.evaluate(testX, testY)
```

---
## Step 8 — Print error

```python
# 打印输出 / Print output
print("Train set MSE = ", train_mse)
# 打印输出 / Print output
print("Test set MSE = ", test_mse)
```

---
## Learning Notes / 学习笔记

- **概念**: Set up parameters 是机器学习中的常用技术。  
  *Set up parameters is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `np.random` | 随机数生成 | Random number generation |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `np.zeros` | 全零数组 | Array filled with zeros |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Training / 05 Training
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense, SimpleRNN
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler

# Set up parameters
time_steps = 20
hidden_units = 2
epochs = 30

def get_fib_seq(n, scale_data=True):
    # Get the Fibonacci sequence
    # 创建全零数组 / Create array of zeros
    seq = np.zeros(n)
    fib_n1 = 0.0
    fib_n = 1.0
    # 生成整数序列 / Generate integer sequence
    for i in range(n):
            seq[i] = fib_n1 + fib_n
            fib_n1 = fib_n
            fib_n = seq[i]
    scaler = []
    if scale_data:
        # 归一化到[0,1]范围 / Normalize to [0,1] range
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
        seq = np.reshape(seq, (n, 1))
        # 展平为一维数组 / Flatten to 1D array
        seq = scaler.fit_transform(seq).flatten()
    return seq, scaler

def get_fib_XY(total_fib_numbers, time_steps, train_percent, scale_data=True):
    dat, scaler = get_fib_seq(total_fib_numbers, scale_data)
    # 生成等差数组 / Generate array with step
    Y_ind = np.arange(time_steps, len(dat), 1)
    Y = dat[Y_ind]
    # 获取长度 / Get length
    rows_x = len(Y)
    X = dat[0:rows_x]
    # 生成整数序列 / Generate integer sequence
    for i in range(time_steps-1):
        temp = dat[i+1:rows_x+i+1]
        X = np.column_stack((X, temp))
    # random permutation with fixed seed
    # 生成随机数 / Generate random numbers
    rand = np.random.RandomState(seed=13)
    idx = rand.permutation(rows_x)
    split = int(train_percent*rows_x)
    train_ind = idx[0:split]
    test_ind = idx[split:]
    trainX = X[train_ind]
    trainY = Y[train_ind]
    testX = X[test_ind]
    testY = Y[test_ind]
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    trainX = np.reshape(trainX, (len(trainX), time_steps, 1))
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    testX = np.reshape(testX, (len(testX), time_steps, 1))
    return trainX, trainY, testX, testY, scaler

# Create a traditional RNN network
def create_RNN(hidden_units, dense_units, input_shape, activation):
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(units=dense_units, activation=activation[1]))
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mse', optimizer='adam')
    return model

model_RNN = create_RNN(hidden_units=hidden_units, dense_units=1,
                       input_shape=(time_steps,1), activation=['tanh', 'tanh'])

# Generate the dataset
trainX, trainY, testX, testY, scaler  = get_fib_XY(1200, time_steps, 0.7)

model_RNN.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)

# Evalute model
train_mse = model_RNN.evaluate(trainX, trainY)
test_mse = model_RNN.evaluate(testX, testY)

# Print error
# 打印输出 / Print output
print("Train set MSE = ", train_mse)
# 打印输出 / Print output
print("Test set MSE = ", test_mse)
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Attention



---

### Training

# 08 — Training / 08 Training

**Chapter 09 — File 6 of 7 / 第09章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Get the Fibonacci sequence**.

本脚本演示 **Get the Fibonacci sequence**。

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
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras import Model
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Input, Dense, SimpleRNN
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Layer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow.keras.backend as K

def get_fib_seq(n, scale_data=True):
```

---
## Step 2 — Get the Fibonacci sequence

```python
# 创建全零数组 / Create array of zeros
seq = np.zeros(n)
    fib_n1 = 0.0
    fib_n = 1.0
    # 生成整数序列 / Generate integer sequence
    for i in range(n):
            seq[i] = fib_n1 + fib_n
            fib_n1 = fib_n
            fib_n = seq[i]
    scaler = []
    if scale_data:
        # 归一化到[0,1]范围 / Normalize to [0,1] range
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
        seq = np.reshape(seq, (n, 1))
        # 展平为一维数组 / Flatten to 1D array
        seq = scaler.fit_transform(seq).flatten()
    return seq, scaler

def get_fib_XY(total_fib_numbers, time_steps, train_percent, scale_data=True):
    dat, scaler = get_fib_seq(total_fib_numbers, scale_data)
    # 生成等差数组 / Generate array with step
    Y_ind = np.arange(time_steps, len(dat), 1)
    Y = dat[Y_ind]
    # 获取长度 / Get length
    rows_x = len(Y)
    X = dat[0:rows_x]
    # 生成整数序列 / Generate integer sequence
    for i in range(time_steps-1):
        temp = dat[i+1:rows_x+i+1]
        X = np.column_stack((X, temp))
```

---
## Step 3 — random permutation with fixed seed

```python
# 生成随机数 / Generate random numbers
rand = np.random.RandomState(seed=13)
    idx = rand.permutation(rows_x)
    split = int(train_percent*rows_x)
    train_ind = idx[0:split]
    test_ind = idx[split:]
    trainX = X[train_ind]
    trainY = Y[train_ind]
    testX = X[test_ind]
    testY = Y[test_ind]
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    trainX = np.reshape(trainX, (len(trainX), time_steps, 1))
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    testX = np.reshape(testX, (len(testX), time_steps, 1))
    return trainX, trainY, testX, testY, scaler
```

---
## Step 4 — Set up parameters

```python
time_steps = 20
hidden_units = 2
epochs = 30
```

---
## Step 5 — Generate the dataset

```python
trainX, trainY, testX, testY, scaler  = get_fib_XY(1200, time_steps, 0.7)

class attention(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1),
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1),
                               initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self,x):
```

---
## Step 6 — Alignment scores. Pass them through tanh function

```python
e = K.tanh(K.dot(x,self.W)+self.b)
```

---
## Step 7 — Remove dimension of size 1

```python
e = K.squeeze(e, axis=-1)
```

---
## Step 8 — Compute the weights

```python
alpha = K.softmax(e)
```

---
## Step 9 — Reshape to tensorFlow format

```python
alpha = K.expand_dims(alpha, axis=-1)
```

---
## Step 10 — Compute the context vector

```python
context = x * alpha
        context = K.sum(context, axis=1)
        return context

def create_RNN_with_attention(hidden_units, dense_units, input_shape, activation):
    x = Input(shape=input_shape)
    RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
    attention_layer = attention()(RNN_layer)
    # 全连接层（Keras） / Fully connected layer (Keras)
    outputs = Dense(dense_units, trainable=True, activation=activation)(attention_layer)
    model = Model(x,outputs)
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mse', optimizer='adam')
    return model

model_attention = create_RNN_with_attention(hidden_units=hidden_units, dense_units=1,
                                            input_shape=(time_steps,1), activation='tanh')
model_attention.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)
```

---
## Step 11 — Evalute model

```python
train_mse_attn = model_attention.evaluate(trainX, trainY)
test_mse_attn = model_attention.evaluate(testX, testY)
```

---
## Step 12 — Print error

```python
# 打印输出 / Print output
print("Train set MSE with attention = ", train_mse_attn)
# 打印输出 / Print output
print("Test set MSE with attention = ", test_mse_attn)
```

---
## Learning Notes / 学习笔记

- **概念**: Get the Fibonacci sequence 是机器学习中的常用技术。  
  *Get the Fibonacci sequence is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `np.random` | 随机数生成 | Random number generation |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `np.zeros` | 全零数组 | Array filled with zeros |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Training / 08 Training
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras import Model
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Input, Dense, SimpleRNN
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Layer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow.keras.backend as K

def get_fib_seq(n, scale_data=True):
    # Get the Fibonacci sequence
    # 创建全零数组 / Create array of zeros
    seq = np.zeros(n)
    fib_n1 = 0.0
    fib_n = 1.0
    # 生成整数序列 / Generate integer sequence
    for i in range(n):
            seq[i] = fib_n1 + fib_n
            fib_n1 = fib_n
            fib_n = seq[i]
    scaler = []
    if scale_data:
        # 归一化到[0,1]范围 / Normalize to [0,1] range
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
        seq = np.reshape(seq, (n, 1))
        # 展平为一维数组 / Flatten to 1D array
        seq = scaler.fit_transform(seq).flatten()
    return seq, scaler

def get_fib_XY(total_fib_numbers, time_steps, train_percent, scale_data=True):
    dat, scaler = get_fib_seq(total_fib_numbers, scale_data)
    # 生成等差数组 / Generate array with step
    Y_ind = np.arange(time_steps, len(dat), 1)
    Y = dat[Y_ind]
    # 获取长度 / Get length
    rows_x = len(Y)
    X = dat[0:rows_x]
    # 生成整数序列 / Generate integer sequence
    for i in range(time_steps-1):
        temp = dat[i+1:rows_x+i+1]
        X = np.column_stack((X, temp))
    # random permutation with fixed seed
    # 生成随机数 / Generate random numbers
    rand = np.random.RandomState(seed=13)
    idx = rand.permutation(rows_x)
    split = int(train_percent*rows_x)
    train_ind = idx[0:split]
    test_ind = idx[split:]
    trainX = X[train_ind]
    trainY = Y[train_ind]
    testX = X[test_ind]
    testY = Y[test_ind]
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    trainX = np.reshape(trainX, (len(trainX), time_steps, 1))
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    testX = np.reshape(testX, (len(testX), time_steps, 1))
    return trainX, trainY, testX, testY, scaler

# Set up parameters
time_steps = 20
hidden_units = 2
epochs = 30

# Generate the dataset
trainX, trainY, testX, testY, scaler  = get_fib_XY(1200, time_steps, 0.7)

class attention(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1),
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1),
                               initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def create_RNN_with_attention(hidden_units, dense_units, input_shape, activation):
    x = Input(shape=input_shape)
    RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
    attention_layer = attention()(RNN_layer)
    # 全连接层（Keras） / Fully connected layer (Keras)
    outputs = Dense(dense_units, trainable=True, activation=activation)(attention_layer)
    model = Model(x,outputs)
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mse', optimizer='adam')
    return model

model_attention = create_RNN_with_attention(hidden_units=hidden_units, dense_units=1,
                                            input_shape=(time_steps,1), activation='tanh')
model_attention.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)

# Evalute model
train_mse_attn = model_attention.evaluate(trainX, trainY)
test_mse_attn = model_attention.evaluate(testX, testY)

# Print error
# 打印输出 / Print output
print("Train set MSE with attention = ", train_mse_attn)
# 打印输出 / Print output
print("Test set MSE with attention = ", test_mse_attn)
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Complete

# 09 — Complete / 09 Complete

**Chapter 09 — File 7 of 7 / 第09章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Prepare data**.

本脚本演示 **Prepare data**。

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
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras import Model
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Input, Dense, SimpleRNN
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Layer
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.metrics import mean_squared_error
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow.keras.backend as K
```

---
## Step 2 — Prepare data

```python
def get_fib_seq(n, scale_data=True):
```

---
## Step 3 — Get the Fibonacci sequence

```python
# 创建全零数组 / Create array of zeros
seq = np.zeros(n)
    fib_n1 = 0.0
    fib_n = 1.0
    # 生成整数序列 / Generate integer sequence
    for i in range(n):
            seq[i] = fib_n1 + fib_n
            fib_n1 = fib_n
            fib_n = seq[i]
    scaler = []
    if scale_data:
        # 归一化到[0,1]范围 / Normalize to [0,1] range
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
        seq = np.reshape(seq, (n, 1))
        # 展平为一维数组 / Flatten to 1D array
        seq = scaler.fit_transform(seq).flatten()
    return seq, scaler

def get_fib_XY(total_fib_numbers, time_steps, train_percent, scale_data=True):
    dat, scaler = get_fib_seq(total_fib_numbers, scale_data)
    # 生成等差数组 / Generate array with step
    Y_ind = np.arange(time_steps, len(dat), 1)
    Y = dat[Y_ind]
    # 获取长度 / Get length
    rows_x = len(Y)
    X = dat[0:rows_x]
    # 生成整数序列 / Generate integer sequence
    for i in range(time_steps-1):
        temp = dat[i+1:rows_x+i+1]
        X = np.column_stack((X, temp))
```

---
## Step 4 — random permutation with fixed seed

```python
# 生成随机数 / Generate random numbers
rand = np.random.RandomState(seed=13)
    idx = rand.permutation(rows_x)
    split = int(train_percent*rows_x)
    train_ind = idx[0:split]
    test_ind = idx[split:]
    trainX = X[train_ind]
    trainY = Y[train_ind]
    testX = X[test_ind]
    testY = Y[test_ind]
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    trainX = np.reshape(trainX, (len(trainX), time_steps, 1))
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    testX = np.reshape(testX, (len(testX), time_steps, 1))
    return trainX, trainY, testX, testY, scaler
```

---
## Step 5 — Set up parameters

```python
time_steps = 20
hidden_units = 2
epochs = 30
```

---
## Step 6 — Create a traditional RNN network

```python
def create_RNN(hidden_units, dense_units, input_shape, activation):
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(units=dense_units, activation=activation[1]))
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mse', optimizer='adam')
    return model

model_RNN = create_RNN(hidden_units=hidden_units, dense_units=1,
                       input_shape=(time_steps,1), activation=['tanh', 'tanh'])
```

---
## Step 7 — Generate the dataset for the network

```python
trainX, trainY, testX, testY, scaler  = get_fib_XY(1200, time_steps, 0.7)
```

---
## Step 8 — Train the network

```python
model_RNN.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)
```

---
## Step 9 — Evalute model

```python
train_mse = model_RNN.evaluate(trainX, trainY)
test_mse = model_RNN.evaluate(testX, testY)
```

---
## Step 10 — Print error

```python
# 打印输出 / Print output
print("Train set MSE = ", train_mse)
# 打印输出 / Print output
print("Test set MSE = ", test_mse)
```

---
## Step 11 — Add attention layer to the deep learning network

```python
class attention(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1),
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1),
                               initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self,x):
```

---
## Step 12 — Alignment scores. Pass them through tanh function

```python
e = K.tanh(K.dot(x,self.W)+self.b)
```

---
## Step 13 — Remove dimension of size 1

```python
e = K.squeeze(e, axis=-1)
```

---
## Step 14 — Compute the weights

```python
alpha = K.softmax(e)
```

---
## Step 15 — Reshape to tensorFlow format

```python
alpha = K.expand_dims(alpha, axis=-1)
```

---
## Step 16 — Compute the context vector

```python
context = x * alpha
        context = K.sum(context, axis=1)
        return context

def create_RNN_with_attention(hidden_units, dense_units, input_shape, activation):
    x = Input(shape=input_shape)
    RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
    attention_layer = attention()(RNN_layer)
    # 全连接层（Keras） / Fully connected layer (Keras)
    outputs = Dense(dense_units, trainable=True, activation=activation)(attention_layer)
    model = Model(x,outputs)
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mse', optimizer='adam')
    return model
```

---
## Step 17 — Create the model with attention, train and evaluate

```python
model_attention = create_RNN_with_attention(hidden_units=hidden_units, dense_units=1,
                                  input_shape=(time_steps,1), activation='tanh')
model_attention.summary()
model_attention.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)
```

---
## Step 18 — Evalute model

```python
train_mse_attn = model_attention.evaluate(trainX, trainY)
test_mse_attn = model_attention.evaluate(testX, testY)
```

---
## Step 19 — Print error

```python
# 打印输出 / Print output
print("Train set MSE with attention = ", train_mse_attn)
# 打印输出 / Print output
print("Test set MSE with attention = ", test_mse_attn)
```

---
## Learning Notes / 学习笔记

- **概念**: Prepare data 是机器学习中的常用技术。  
  *Prepare data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `np.random` | 随机数生成 | Random number generation |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `np.zeros` | 全零数组 | Array filled with zeros |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Complete / 09 Complete
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras import Model
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Input, Dense, SimpleRNN
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Layer
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.metrics import mean_squared_error
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow.keras.backend as K

# Prepare data
def get_fib_seq(n, scale_data=True):
    # Get the Fibonacci sequence
    # 创建全零数组 / Create array of zeros
    seq = np.zeros(n)
    fib_n1 = 0.0
    fib_n = 1.0
    # 生成整数序列 / Generate integer sequence
    for i in range(n):
            seq[i] = fib_n1 + fib_n
            fib_n1 = fib_n
            fib_n = seq[i]
    scaler = []
    if scale_data:
        # 归一化到[0,1]范围 / Normalize to [0,1] range
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
        seq = np.reshape(seq, (n, 1))
        # 展平为一维数组 / Flatten to 1D array
        seq = scaler.fit_transform(seq).flatten()
    return seq, scaler

def get_fib_XY(total_fib_numbers, time_steps, train_percent, scale_data=True):
    dat, scaler = get_fib_seq(total_fib_numbers, scale_data)
    # 生成等差数组 / Generate array with step
    Y_ind = np.arange(time_steps, len(dat), 1)
    Y = dat[Y_ind]
    # 获取长度 / Get length
    rows_x = len(Y)
    X = dat[0:rows_x]
    # 生成整数序列 / Generate integer sequence
    for i in range(time_steps-1):
        temp = dat[i+1:rows_x+i+1]
        X = np.column_stack((X, temp))
    # random permutation with fixed seed
    # 生成随机数 / Generate random numbers
    rand = np.random.RandomState(seed=13)
    idx = rand.permutation(rows_x)
    split = int(train_percent*rows_x)
    train_ind = idx[0:split]
    test_ind = idx[split:]
    trainX = X[train_ind]
    trainY = Y[train_ind]
    testX = X[test_ind]
    testY = Y[test_ind]
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    trainX = np.reshape(trainX, (len(trainX), time_steps, 1))
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    testX = np.reshape(testX, (len(testX), time_steps, 1))
    return trainX, trainY, testX, testY, scaler

# Set up parameters
time_steps = 20
hidden_units = 2
epochs = 30

# Create a traditional RNN network
def create_RNN(hidden_units, dense_units, input_shape, activation):
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(units=dense_units, activation=activation[1]))
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mse', optimizer='adam')
    return model

model_RNN = create_RNN(hidden_units=hidden_units, dense_units=1,
                       input_shape=(time_steps,1), activation=['tanh', 'tanh'])

# Generate the dataset for the network
trainX, trainY, testX, testY, scaler  = get_fib_XY(1200, time_steps, 0.7)
# Train the network
model_RNN.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)

# Evalute model
train_mse = model_RNN.evaluate(trainX, trainY)
test_mse = model_RNN.evaluate(testX, testY)

# Print error
# 打印输出 / Print output
print("Train set MSE = ", train_mse)
# 打印输出 / Print output
print("Test set MSE = ", test_mse)

# Add attention layer to the deep learning network
class attention(Layer):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1),
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1),
                               initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def create_RNN_with_attention(hidden_units, dense_units, input_shape, activation):
    x = Input(shape=input_shape)
    RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
    attention_layer = attention()(RNN_layer)
    # 全连接层（Keras） / Fully connected layer (Keras)
    outputs = Dense(dense_units, trainable=True, activation=activation)(attention_layer)
    model = Model(x,outputs)
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='mse', optimizer='adam')
    return model

# Create the model with attention, train and evaluate
model_attention = create_RNN_with_attention(hidden_units=hidden_units, dense_units=1,
                                  input_shape=(time_steps,1), activation='tanh')
model_attention.summary()
model_attention.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)

# Evalute model
train_mse_attn = model_attention.evaluate(trainX, trainY)
test_mse_attn = model_attention.evaluate(testX, testY)

# Print error
# 打印输出 / Print output
print("Train set MSE with attention = ", train_mse_attn)
# 打印输出 / Print output
print("Test set MSE with attention = ", test_mse_attn)
```

---

### Chapter Summary / 章节总结

# Chapter 09 Summary / 第09章总结

## Theme / 主题: Chapter 09 / Chapter 09

This chapter contains **7 code files** demonstrating chapter 09.

本章包含 **7 个代码文件**，演示Chapter 09。

---
## Evolution / 演化路线

  1. `02_fibonacci.ipynb` — Fibonacci
  2. `03_split.ipynb` — Split
  3. `04_rnn.ipynb` — Rnn
  4. `05_training.ipynb` — Training
  5. `07_attention.ipynb` — Attention
  6. `08_training.ipynb` — Training
  7. `09_complete.ipynb` — Complete

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 09) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 09）是机器学习流水线中的基础构建块。

---
