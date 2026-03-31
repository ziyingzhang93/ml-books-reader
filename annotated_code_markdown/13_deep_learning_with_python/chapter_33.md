# Python 深度学习 / Deep Learning with Python
## Chapter 33

---

### Simple



---

### Dropout

# 08 — Dropout / 随机失活

**Chapter 33 — File 2 of 5 / 第33章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **LSTM with Dropout for sequence classification in the IMDB dataset**.

本脚本演示 **LSTM with Dropout for sequence classification in the IMDB dataset**。

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
## Step 1 — LSTM with Dropout for sequence classification in the IMDB dataset

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import imdb
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Embedding
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing import sequence
```

---
## Step 2 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 3 — load the dataset but only keep the top n words, zero the rest

```python
top_words = 5000
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
```

---
## Step 4 — truncate and pad input sequences

```python
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
```

---
## Step 5 — create the model

```python
embedding_vecor_length = 32
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# 训练模型 / Train the model
model.fit(X_train, y_train, epochs=3, batch_size=64)
```

---
## Step 6 — Final evaluation of the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("Accuracy: %.2f%%" % (scores[1]*100))
```

---
## Learning Notes / 学习笔记

- **概念**: LSTM with Dropout for sequence classification in the IMDB dataset 是机器学习中的常用技术。  
  *LSTM with Dropout for sequence classification in the IMDB dataset is a common technique in machine learning.*

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
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
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
# Dropout / 随机失活
# Complete Code / 完整代码
# ===============================

# LSTM with Dropout for sequence classification in the IMDB dataset
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import imdb
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dropout
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Embedding
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing import sequence
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100))
# 向模型添加一层 / Add a layer to the model
model.add(Dropout(0.2))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# 训练模型 / Train the model
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("Accuracy: %.2f%%" % (scores[1]*100))
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Dropout Gates

# 10 — Dropout Gates / 随机失活

**Chapter 33 — File 3 of 5 / 第33章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **LSTM with dropout for sequence classification in the IMDB dataset**.

本脚本演示 **LSTM with dropout for sequence classification in the IMDB dataset**。

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
## Step 1 — LSTM with dropout for sequence classification in the IMDB dataset

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import imdb
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Embedding
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing import sequence
```

---
## Step 2 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 3 — load the dataset but only keep the top n words, zero the rest

```python
top_words = 5000
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
```

---
## Step 4 — truncate and pad input sequences

```python
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
```

---
## Step 5 — create the model

```python
embedding_vecor_length = 32
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# 训练模型 / Train the model
model.fit(X_train, y_train, epochs=3, batch_size=64)
```

---
## Step 6 — Final evaluation of the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("Accuracy: %.2f%%" % (scores[1]*100))
```

---
## Learning Notes / 学习笔记

- **概念**: LSTM with dropout for sequence classification in the IMDB dataset 是机器学习中的常用技术。  
  *LSTM with dropout for sequence classification in the IMDB dataset is a common technique in machine learning.*

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
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
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
# Dropout Gates / 随机失活
# Complete Code / 完整代码
# ===============================

# LSTM with dropout for sequence classification in the IMDB dataset
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import imdb
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Embedding
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing import sequence
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# 训练模型 / Train the model
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("Accuracy: %.2f%%" % (scores[1]*100))
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Bidirectional

# 12 — Bidirectional / 12 Bidirectional

**Chapter 33 — File 4 of 5 / 第33章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **LSTM with dropout for sequence classification in the IMDB dataset**.

本脚本演示 **LSTM with dropout for sequence classification in the IMDB dataset**。

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
## Step 1 — LSTM with dropout for sequence classification in the IMDB dataset

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import imdb
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Bidirectional
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Embedding
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing import sequence
```

---
## Step 2 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 3 — load the dataset but only keep the top n words, zero the rest

```python
top_words = 5000
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
```

---
## Step 4 — truncate and pad input sequences

```python
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
```

---
## Step 5 — create the model

```python
embedding_vecor_length = 32
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# 向模型添加一层 / Add a layer to the model
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# 训练模型 / Train the model
model.fit(X_train, y_train, epochs=3, batch_size=64)
```

---
## Step 6 — Final evaluation of the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("Accuracy: %.2f%%" % (scores[1]*100))
```

---
## Learning Notes / 学习笔记

- **概念**: LSTM with dropout for sequence classification in the IMDB dataset 是机器学习中的常用技术。  
  *LSTM with dropout for sequence classification in the IMDB dataset is a common technique in machine learning.*

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
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
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
# Bidirectional / 12 Bidirectional
# Complete Code / 完整代码
# ===============================

# LSTM with dropout for sequence classification in the IMDB dataset
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import imdb
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Bidirectional
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Embedding
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing import sequence
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# 向模型添加一层 / Add a layer to the model
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# 训练模型 / Train the model
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("Accuracy: %.2f%%" % (scores[1]*100))
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Lstm Cnn

# 14 — Lstm Cnn / 卷积神经网络

**Chapter 33 — File 5 of 5 / 第33章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **LSTM and CNN for sequence classification in the IMDB dataset**.

本脚本演示 **LSTM and CNN for sequence classification in the IMDB dataset**。

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
## Step 1 — LSTM and CNN for sequence classification in the IMDB dataset

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import imdb
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv1D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import MaxPooling1D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Embedding
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing import sequence
```

---
## Step 2 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 3 — load the dataset but only keep the top n words, zero the rest

```python
top_words = 5000
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
```

---
## Step 4 — truncate and pad input sequences

```python
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
```

---
## Step 5 — create the model

```python
embedding_vecor_length = 32
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(32, 3, padding='same', activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling1D(pool_size=2))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# 训练模型 / Train the model
model.fit(X_train, y_train, epochs=3, batch_size=64)
```

---
## Step 6 — Final evaluation of the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("Accuracy: %.2f%%" % (scores[1]*100))
```

---
## Learning Notes / 学习笔记

- **概念**: LSTM and CNN for sequence classification in the IMDB dataset 是机器学习中的常用技术。  
  *LSTM and CNN for sequence classification in the IMDB dataset is a common technique in machine learning.*

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
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
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
# Lstm Cnn / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# LSTM and CNN for sequence classification in the IMDB dataset
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import imdb
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv1D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import MaxPooling1D
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Embedding
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing import sequence
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(32, 3, padding='same', activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling1D(pool_size=2))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(100))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# 训练模型 / Train the model
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X_test, y_test, verbose=0)
# 打印输出 / Print output
print("Accuracy: %.2f%%" % (scores[1]*100))
```

---

### Chapter Summary / 章节总结



---
