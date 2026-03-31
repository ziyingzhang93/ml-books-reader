# NLP 深度学习 / Deep Learning for NLP
## Chapter 19

---

### Model1

# 1 — Model1 / 1 Model1

**Chapter 19 — File 1 of 3 / 第19章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **generate a sequence from the model**.

本脚本演示 **generate a sequence from the model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
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
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import Tokenizer
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils.vis_utils import plot_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Embedding
```

---
## Step 2 — generate a sequence from the model

```python
def generate_seq(model, tokenizer, seed_text, n_words):
	in_text, result = seed_text, seed_text
```

---
## Step 3 — generate a fixed number of words

```python
# 生成整数序列 / Generate integer sequence
for _ in range(n_words):
```

---
## Step 4 — encode the text as integer

```python
encoded = tokenizer.texts_to_sequences([in_text])[0]
		encoded = array(encoded)
```

---
## Step 5 — predict a word in the vocabulary

```python
yhat = model.predict_classes(encoded, verbose=0)
```

---
## Step 6 — map predicted word index to word

```python
out_word = ''
  # 获取字典的键值对 / Get dict key-value pairs
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
```

---
## Step 7 — append to input

```python
in_text, result = out_word, result + ' ' + out_word
	return result
```

---
## Step 8 — define the model

```python
def define_model(vocab_size):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Embedding(vocab_size, 10, input_length=1))
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(50))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(vocab_size, activation='softmax'))
```

---
## Step 9 — compile network

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 10 — summarize defined model

```python
model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)
	return model
```

---
## Step 11 — source text

```python
data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """
```

---
## Step 12 — integer encode text

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
```

---
## Step 13 — determine the vocabulary size

```python
# 获取长度 / Get length
vocab_size = len(tokenizer.word_index) + 1
# 打印输出 / Print output
print('Vocabulary Size: %d' % vocab_size)
```

---
## Step 14 — create word -> word sequences

```python
sequences = list()
# 获取长度 / Get length
for i in range(1, len(encoded)):
	sequence = encoded[i-1:i+1]
 # 添加元素到列表末尾 / Append element to list end
	sequences.append(sequence)
# 打印输出 / Print output
print('Total Sequences: %d' % len(sequences))
```

---
## Step 15 — split into X and y elements

```python
sequences = array(sequences)
X, y = sequences[:,0],sequences[:,1]
```

---
## Step 16 — one hot encode outputs

```python
y = to_categorical(y, num_classes=vocab_size)
```

---
## Step 17 — define model

```python
model = define_model(vocab_size)
```

---
## Step 18 — fit network

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=500, verbose=2)
```

---
## Step 19 — evaluate

```python
# 打印输出 / Print output
print(generate_seq(model, tokenizer, 'Jack', 6))
```

---
## Learning Notes / 学习笔记

- **概念**: generate a sequence from the model 是机器学习中的常用技术。  
  *generate a sequence from the model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model1 / 1 Model1
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import Tokenizer
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils.vis_utils import plot_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Embedding

# generate a sequence from the model
def generate_seq(model, tokenizer, seed_text, n_words):
	in_text, result = seed_text, seed_text
	# generate a fixed number of words
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		encoded = array(encoded)
		# predict a word in the vocabulary
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
  # 获取字典的键值对 / Get dict key-value pairs
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text, result = out_word, result + ' ' + out_word
	return result

# define the model
def define_model(vocab_size):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Embedding(vocab_size, 10, input_length=1))
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(50))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(vocab_size, activation='softmax'))
	# compile network
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize defined model
	model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

# source text
data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """
# integer encode text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
# determine the vocabulary size
# 获取长度 / Get length
vocab_size = len(tokenizer.word_index) + 1
# 打印输出 / Print output
print('Vocabulary Size: %d' % vocab_size)
# create word -> word sequences
sequences = list()
# 获取长度 / Get length
for i in range(1, len(encoded)):
	sequence = encoded[i-1:i+1]
 # 添加元素到列表末尾 / Append element to list end
	sequences.append(sequence)
# 打印输出 / Print output
print('Total Sequences: %d' % len(sequences))
# split into X and y elements
sequences = array(sequences)
X, y = sequences[:,0],sequences[:,1]
# one hot encode outputs
y = to_categorical(y, num_classes=vocab_size)
# define model
model = define_model(vocab_size)
# fit network
# 训练模型 / Train the model
model.fit(X, y, epochs=500, verbose=2)
# evaluate
# 打印输出 / Print output
print(generate_seq(model, tokenizer, 'Jack', 6))
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Model2

# 2 — Model2 / 2 Model2

**Chapter 19 — File 2 of 3 / 第19章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **generate a sequence from a language model**.

本脚本演示 **generate a sequence from a language model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
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
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import Tokenizer
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils.vis_utils import plot_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Embedding
```

---
## Step 2 — generate a sequence from a language model

```python
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
```

---
## Step 3 — generate a fixed number of words

```python
# 生成整数序列 / Generate integer sequence
for _ in range(n_words):
```

---
## Step 4 — encode the text as integer

```python
encoded = tokenizer.texts_to_sequences([in_text])[0]
```

---
## Step 5 — pre-pad sequences to a fixed length

```python
encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
```

---
## Step 6 — predict probabilities for each word

```python
yhat = model.predict_classes(encoded, verbose=0)
```

---
## Step 7 — map predicted word index to word

```python
out_word = ''
  # 获取字典的键值对 / Get dict key-value pairs
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
```

---
## Step 8 — append to input

```python
in_text += ' ' + out_word
	return in_text
```

---
## Step 9 — define the model

```python
def define_model(vocab_size, max_length):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Embedding(vocab_size, 10, input_length=max_length-1))
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(50))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(vocab_size, activation='softmax'))
```

---
## Step 10 — compile network

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 11 — summarize defined model

```python
model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)
	return model
```

---
## Step 12 — source text

```python
data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """
```

---
## Step 13 — prepare the tokenizer on the source text

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
```

---
## Step 14 — determine the vocabulary size

```python
# 获取长度 / Get length
vocab_size = len(tokenizer.word_index) + 1
# 打印输出 / Print output
print('Vocabulary Size: %d' % vocab_size)
```

---
## Step 15 — create line-based sequences

```python
sequences = list()
for line in data.split('\n'):
	encoded = tokenizer.texts_to_sequences([line])[0]
 # 获取长度 / Get length
	for i in range(1, len(encoded)):
		sequence = encoded[:i+1]
  # 添加元素到列表末尾 / Append element to list end
		sequences.append(sequence)
# 打印输出 / Print output
print('Total Sequences: %d' % len(sequences))
```

---
## Step 16 — pad input sequences

```python
# 获取长度 / Get length
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
# 打印输出 / Print output
print('Max Sequence Length: %d' % max_length)
```

---
## Step 17 — split into input and output elements

```python
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
```

---
## Step 18 — define model

```python
model = define_model(vocab_size, max_length)
```

---
## Step 19 — fit network

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=500, verbose=2)
```

---
## Step 20 — evaluate model

```python
# 打印输出 / Print output
print(generate_seq(model, tokenizer, max_length-1, 'Jack', 4))
# 打印输出 / Print output
print(generate_seq(model, tokenizer, max_length-1, 'Jill', 4))
```

---
## Learning Notes / 学习笔记

- **概念**: generate a sequence from a language model 是机器学习中的常用技术。  
  *generate a sequence from a language model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model2 / 2 Model2
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import Tokenizer
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils.vis_utils import plot_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Embedding

# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	# generate a fixed number of words
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# pre-pad sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
  # 获取字典的键值对 / Get dict key-value pairs
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
	return in_text

# define the model
def define_model(vocab_size, max_length):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Embedding(vocab_size, 10, input_length=max_length-1))
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(50))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(vocab_size, activation='softmax'))
	# compile network
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize defined model
	model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

# source text
data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """
# prepare the tokenizer on the source text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
# determine the vocabulary size
# 获取长度 / Get length
vocab_size = len(tokenizer.word_index) + 1
# 打印输出 / Print output
print('Vocabulary Size: %d' % vocab_size)
# create line-based sequences
sequences = list()
for line in data.split('\n'):
	encoded = tokenizer.texts_to_sequences([line])[0]
 # 获取长度 / Get length
	for i in range(1, len(encoded)):
		sequence = encoded[:i+1]
  # 添加元素到列表末尾 / Append element to list end
		sequences.append(sequence)
# 打印输出 / Print output
print('Total Sequences: %d' % len(sequences))
# pad input sequences
# 获取长度 / Get length
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
# 打印输出 / Print output
print('Max Sequence Length: %d' % max_length)
# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
# define model
model = define_model(vocab_size, max_length)
# fit network
# 训练模型 / Train the model
model.fit(X, y, epochs=500, verbose=2)
# evaluate model
# 打印输出 / Print output
print(generate_seq(model, tokenizer, max_length-1, 'Jack', 4))
# 打印输出 / Print output
print(generate_seq(model, tokenizer, max_length-1, 'Jill', 4))
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Model3

# 3 — Model3 / 3 Model3

**Chapter 19 — File 3 of 3 / 第19章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **generate a sequence from a language model**.

本脚本演示 **generate a sequence from a language model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
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
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import Tokenizer
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils.vis_utils import plot_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Embedding
```

---
## Step 2 — generate a sequence from a language model

```python
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
```

---
## Step 3 — generate a fixed number of words

```python
# 生成整数序列 / Generate integer sequence
for _ in range(n_words):
```

---
## Step 4 — encode the text as integer

```python
encoded = tokenizer.texts_to_sequences([in_text])[0]
```

---
## Step 5 — pre-pad sequences to a fixed length

```python
encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
```

---
## Step 6 — predict probabilities for each word

```python
yhat = model.predict_classes(encoded, verbose=0)
```

---
## Step 7 — map predicted word index to word

```python
out_word = ''
  # 获取字典的键值对 / Get dict key-value pairs
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
```

---
## Step 8 — append to input

```python
in_text += ' ' + out_word
	return in_text
```

---
## Step 9 — define the model

```python
def define_model(vocab_size, max_length):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Embedding(vocab_size, 10, input_length=max_length-1))
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(50))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(vocab_size, activation='softmax'))
```

---
## Step 10 — compile network

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 11 — summarize defined model

```python
model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)
	return model
```

---
## Step 12 — source text

```python
data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """
```

---
## Step 13 — integer encode sequences of words

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
```

---
## Step 14 — retrieve vocabulary size

```python
# 获取长度 / Get length
vocab_size = len(tokenizer.word_index) + 1
# 打印输出 / Print output
print('Vocabulary Size: %d' % vocab_size)
```

---
## Step 15 — encode 2 words -> 1 word

```python
sequences = list()
# 获取长度 / Get length
for i in range(2, len(encoded)):
	sequence = encoded[i-2:i+1]
 # 添加元素到列表末尾 / Append element to list end
	sequences.append(sequence)
# 打印输出 / Print output
print('Total Sequences: %d' % len(sequences))
```

---
## Step 16 — pad sequences

```python
# 获取长度 / Get length
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
# 打印输出 / Print output
print('Max Sequence Length: %d' % max_length)
```

---
## Step 17 — split into input and output elements

```python
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
```

---
## Step 18 — define model

```python
model = define_model(vocab_size, max_length)
```

---
## Step 19 — fit network

```python
# 训练模型 / Train the model
model.fit(X, y, epochs=500, verbose=2)
```

---
## Step 20 — evaluate model

```python
# 打印输出 / Print output
print(generate_seq(model, tokenizer, max_length-1, 'Jack and', 5))
# 打印输出 / Print output
print(generate_seq(model, tokenizer, max_length-1, 'And Jill', 3))
# 打印输出 / Print output
print(generate_seq(model, tokenizer, max_length-1, 'fell down', 5))
# 打印输出 / Print output
print(generate_seq(model, tokenizer, max_length-1, 'pail of', 5))
```

---
## Learning Notes / 学习笔记

- **概念**: generate a sequence from a language model 是机器学习中的常用技术。  
  *generate a sequence from a language model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model3 / 3 Model3
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import Tokenizer
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils.vis_utils import plot_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Embedding

# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	# generate a fixed number of words
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# pre-pad sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
  # 获取字典的键值对 / Get dict key-value pairs
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
	return in_text

# define the model
def define_model(vocab_size, max_length):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Embedding(vocab_size, 10, input_length=max_length-1))
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(50))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(vocab_size, activation='softmax'))
	# compile network
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize defined model
	model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

# source text
data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
# retrieve vocabulary size
# 获取长度 / Get length
vocab_size = len(tokenizer.word_index) + 1
# 打印输出 / Print output
print('Vocabulary Size: %d' % vocab_size)
# encode 2 words -> 1 word
sequences = list()
# 获取长度 / Get length
for i in range(2, len(encoded)):
	sequence = encoded[i-2:i+1]
 # 添加元素到列表末尾 / Append element to list end
	sequences.append(sequence)
# 打印输出 / Print output
print('Total Sequences: %d' % len(sequences))
# pad sequences
# 获取长度 / Get length
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
# 打印输出 / Print output
print('Max Sequence Length: %d' % max_length)
# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
# define model
model = define_model(vocab_size, max_length)
# fit network
# 训练模型 / Train the model
model.fit(X, y, epochs=500, verbose=2)
# evaluate model
# 打印输出 / Print output
print(generate_seq(model, tokenizer, max_length-1, 'Jack and', 5))
# 打印输出 / Print output
print(generate_seq(model, tokenizer, max_length-1, 'And Jill', 3))
# 打印输出 / Print output
print(generate_seq(model, tokenizer, max_length-1, 'fell down', 5))
# 打印输出 / Print output
print(generate_seq(model, tokenizer, max_length-1, 'pail of', 5))
```

---

### Chapter Summary / 章节总结



---
