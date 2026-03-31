# NLP 深度学习 / Deep Learning for NLP
## Chapter 13

---

### Embedding Example

# 1 — Embedding Example / 词嵌入

**Chapter 13 — File 1 of 2 / 第13章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **define documents**.

本脚本演示 **define documents**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import one_hot
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.embeddings import Embedding
```

---
## Step 2 — define documents

```python
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
```

---
## Step 3 — define class labels

```python
labels = array([1,1,1,1,1,0,0,0,0,0])
```

---
## Step 4 — integer encode the documents

```python
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
# 打印输出 / Print output
print(encoded_docs)
```

---
## Step 5 — pad documents to a max length of 4 words

```python
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# 打印输出 / Print output
print(padded_docs)
```

---
## Step 6 — define the model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Embedding(vocab_size, 8, input_length=max_length))
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
```

---
## Step 7 — compile the model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---
## Step 8 — summarize the model

```python
model.summary()
```

---
## Step 9 — fit the model

```python
# 训练模型 / Train the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
```

---
## Step 10 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
# 打印输出 / Print output
print('Accuracy: %f' % (accuracy*100))
```

---
## Learning Notes / 学习笔记

- **概念**: define documents 是机器学习中的常用技术。  
  *define documents is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
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
# Embedding Example / 词嵌入
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import one_hot
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.embeddings import Embedding
# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
# 打印输出 / Print output
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# 打印输出 / Print output
print(padded_docs)
# define the model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Embedding(vocab_size, 8, input_length=max_length))
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# compile the model
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
model.summary()
# fit the model
# 训练模型 / Train the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
# 评估模型在测试集上的表现 / Evaluate model on test set
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
# 打印输出 / Print output
print('Accuracy: %f' % (accuracy*100))
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Pretrained Embedding

# 2 — Pretrained Embedding / 词嵌入

**Chapter 13 — File 2 of 2 / 第13章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **define documents**.

本脚本演示 **define documents**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import Tokenizer
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Embedding
```

---
## Step 2 — define documents

```python
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
```

---
## Step 3 — define class labels

```python
labels = array([1,1,1,1,1,0,0,0,0,0])
```

---
## Step 4 — prepare tokenizer

```python
t = Tokenizer()
t.fit_on_texts(docs)
# 获取长度 / Get length
vocab_size = len(t.word_index) + 1
```

---
## Step 5 — integer encode the documents

```python
encoded_docs = t.texts_to_sequences(docs)
# 打印输出 / Print output
print(encoded_docs)
```

---
## Step 6 — pad documents to a max length of 4 words

```python
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# 打印输出 / Print output
print(padded_docs)
```

---
## Step 7 — load the whole embedding into memory

```python
embeddings_index = dict()
f = open('glove.6B.100d.txt', mode='rt', encoding='utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
# 打印输出 / Print output
print('Loaded %s word vectors.' % len(embeddings_index))
```

---
## Step 8 — create a weight matrix for words in training docs

```python
embedding_matrix = zeros((vocab_size, 100))
# 获取字典的键值对 / Get dict key-value pairs
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
```

---
## Step 9 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
# 向模型添加一层 / Add a layer to the model
model.add(e)
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
```

---
## Step 10 — compile the model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---
## Step 11 — summarize the model

```python
model.summary()
```

---
## Step 12 — fit the model

```python
# 训练模型 / Train the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
```

---
## Step 13 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
# 打印输出 / Print output
print('Accuracy: %f' % (accuracy*100))
```

---
## Learning Notes / 学习笔记

- **概念**: define documents 是机器学习中的常用技术。  
  *define documents is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pretrained Embedding / 词嵌入
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import Tokenizer
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Embedding
# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
# 获取长度 / Get length
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
# 打印输出 / Print output
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# 打印输出 / Print output
print(padded_docs)
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt', mode='rt', encoding='utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
# 打印输出 / Print output
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
# 获取字典的键值对 / Get dict key-value pairs
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
# 向模型添加一层 / Add a layer to the model
model.add(e)
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# compile the model
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
model.summary()
# fit the model
# 训练模型 / Train the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
# 评估模型在测试集上的表现 / Evaluate model on test set
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
# 打印输出 / Print output
print('Accuracy: %f' % (accuracy*100))
```

---

### Chapter Summary / 章节总结



---
