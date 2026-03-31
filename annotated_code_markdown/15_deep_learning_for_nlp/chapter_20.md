# NLP 深度学习 / Deep Learning for NLP
## Chapter 20

---

### Prepare Text

# 1 — Prepare Text / 数据准备

**Chapter 20 — File 1 of 3 / 第20章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **load doc into memory**.

本脚本演示 **load doc into memory**。

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
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  ⚙️ 配置训练 / Configure Training
```

---
## Step 1 — Step 1

```python
import string
# 导入正则表达式模块 / Import regex module
import re
```

---
## Step 2 — load doc into memory

```python
def load_doc(filename):
```

---
## Step 3 — open the file as read only

```python
file = open(filename, 'r')
```

---
## Step 4 — read all text

```python
text = file.read()
```

---
## Step 5 — close the file

```python
file.close()
	return text
```

---
## Step 6 — turn a doc into clean tokens

```python
def clean_doc(doc):
```

---
## Step 7 — replace '--' with a space ' '

```python
doc = doc.replace('--', ' ')
```

---
## Step 8 — split into tokens by white space

```python
tokens = doc.split()
```

---
## Step 9 — prepare regex for char filtering

```python
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
```

---
## Step 10 — remove punctuation from each word

```python
tokens = [re_punc.sub('', w) for w in tokens]
```

---
## Step 11 — remove remaining tokens that are not alphabetic

```python
tokens = [word for word in tokens if word.isalpha()]
```

---
## Step 12 — make lower case

```python
tokens = [word.lower() for word in tokens]
	return tokens
```

---
## Step 13 — save tokens to file, one dialog per line

```python
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
```

---
## Step 14 — load document

```python
in_filename = 'republic_clean.txt'
doc = load_doc(in_filename)
# 打印输出 / Print output
print(doc[:200])
```

---
## Step 15 — clean document

```python
tokens = clean_doc(doc)
# 打印输出 / Print output
print(tokens[:200])
# 打印输出 / Print output
print('Total Tokens: %d' % len(tokens))
# 打印输出 / Print output
print('Unique Tokens: %d' % len(set(tokens)))
```

---
## Step 16 — organize into sequences of tokens

```python
length = 50 + 1
sequences = list()
# 获取长度 / Get length
for i in range(length, len(tokens)):
```

---
## Step 17 — select sequence of tokens

```python
seq = tokens[i-length:i]
```

---
## Step 18 — convert into a line

```python
line = ' '.join(seq)
```

---
## Step 19 — store

```python
# 添加元素到列表末尾 / Append element to list end
sequences.append(line)
# 打印输出 / Print output
print('Total Sequences: %d' % len(sequences))
```

---
## Step 20 — save sequences to file

```python
out_filename = 'republic_sequences.txt'
save_doc(sequences, out_filename)
```

---
## Learning Notes / 学习笔记

- **概念**: load doc into memory 是机器学习中的常用技术。  
  *load doc into memory is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Prepare Text / 数据准备
# Complete Code / 完整代码
# ===============================

import string
# 导入正则表达式模块 / Import regex module
import re

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# replace '--' with a space ' '
	doc = doc.replace('--', ' ')
	# split into tokens by white space
	tokens = doc.split()
	# prepare regex for char filtering
	re_punc = re.compile('[%s]' % re.escape(string.punctuation))
	# remove punctuation from each word
	tokens = [re_punc.sub('', w) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# load document
in_filename = 'republic_clean.txt'
doc = load_doc(in_filename)
# 打印输出 / Print output
print(doc[:200])
# clean document
tokens = clean_doc(doc)
# 打印输出 / Print output
print(tokens[:200])
# 打印输出 / Print output
print('Total Tokens: %d' % len(tokens))
# 打印输出 / Print output
print('Unique Tokens: %d' % len(set(tokens)))
# organize into sequences of tokens
length = 50 + 1
sequences = list()
# 获取长度 / Get length
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
 # 添加元素到列表末尾 / Append element to list end
	sequences.append(line)
# 打印输出 / Print output
print('Total Sequences: %d' % len(sequences))
# save sequences to file
out_filename = 'republic_sequences.txt'
save_doc(sequences, out_filename)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Train Model

# 2 — Train Model / 2 Train Model

**Chapter 20 — File 2 of 3 / 第20章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **load doc into memory**.

本脚本演示 **load doc into memory**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
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
  🏋️ 训练模型 / Train Model
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
from pickle import dump
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import Tokenizer
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils.vis_utils import plot_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
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
## Step 2 — load doc into memory

```python
def load_doc(filename):
```

---
## Step 3 — open the file as read only

```python
file = open(filename, 'r')
```

---
## Step 4 — read all text

```python
text = file.read()
```

---
## Step 5 — close the file

```python
file.close()
	return text
```

---
## Step 6 — define the model

```python
def define_model(vocab_size, seq_length):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Embedding(vocab_size, 50, input_length=seq_length))
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(100, return_sequences=True))
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(100))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(100, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(vocab_size, activation='softmax'))
```

---
## Step 7 — compile network

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 8 — summarize defined model

```python
model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)
	return model
```

---
## Step 9 — load

```python
in_filename = 'republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
```

---
## Step 10 — integer encode sequences of words

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
```

---
## Step 11 — vocabulary size

```python
# 获取长度 / Get length
vocab_size = len(tokenizer.word_index) + 1
```

---
## Step 12 — separate into input and output

```python
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
seq_length = X.shape[1]
```

---
## Step 13 — define model

```python
model = define_model(vocab_size, seq_length)
```

---
## Step 14 — fit model

```python
# 训练模型 / Train the model
model.fit(X, y, batch_size=128, epochs=100)
```

---
## Step 15 — save the model to file

```python
# 保存模型到文件 / Save model to file
model.save('model.h5')
```

---
## Step 16 — save the tokenizer

```python
dump(tokenizer, open('tokenizer.pkl', 'wb'))
```

---
## Learning Notes / 学习笔记

- **概念**: load doc into memory 是机器学习中的常用技术。  
  *load doc into memory is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Model / 2 Train Model
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
from pickle import dump
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import Tokenizer
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils.vis_utils import plot_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Embedding

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# define the model
def define_model(vocab_size, seq_length):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Embedding(vocab_size, 50, input_length=seq_length))
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(100, return_sequences=True))
 # 向模型添加一层 / Add a layer to the model
	model.add(LSTM(100))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(100, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(vocab_size, activation='softmax'))
	# compile network
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize defined model
	model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

# load
in_filename = 'republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
# vocabulary size
# 获取长度 / Get length
vocab_size = len(tokenizer.word_index) + 1
# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
seq_length = X.shape[1]
# define model
model = define_model(vocab_size, seq_length)
# fit model
# 训练模型 / Train the model
model.fit(X, y, batch_size=128, epochs=100)
# save the model to file
# 保存模型到文件 / Save model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Generate

# 3 — Generate / 3 Generate

**Chapter 20 — File 3 of 3 / 第20章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **load doc into memory**.

本脚本演示 **load doc into memory**。

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
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
from random import randint
from pickle import load
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
```

---
## Step 2 — load doc into memory

```python
def load_doc(filename):
```

---
## Step 3 — open the file as read only

```python
file = open(filename, 'r')
```

---
## Step 4 — read all text

```python
text = file.read()
```

---
## Step 5 — close the file

```python
file.close()
	return text
```

---
## Step 6 — generate a sequence from a language model

```python
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
```

---
## Step 7 — generate a fixed number of words

```python
# 生成整数序列 / Generate integer sequence
for _ in range(n_words):
```

---
## Step 8 — encode the text as integer

```python
encoded = tokenizer.texts_to_sequences([in_text])[0]
```

---
## Step 9 — truncate sequences to a fixed length

```python
encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
```

---
## Step 10 — predict probabilities for each word

```python
yhat = model.predict_classes(encoded, verbose=0)
```

---
## Step 11 — map predicted word index to word

```python
out_word = ''
  # 获取字典的键值对 / Get dict key-value pairs
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
```

---
## Step 12 — append to input

```python
in_text += ' ' + out_word
  # 添加元素到列表末尾 / Append element to list end
		result.append(out_word)
	return ' '.join(result)
```

---
## Step 13 — load cleaned text sequences

```python
in_filename = 'republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
# 获取长度 / Get length
seq_length = len(lines[0].split()) - 1
```

---
## Step 14 — load the model

```python
# 从文件加载模型 / Load model from file
model = load_model('model.h5')
```

---
## Step 15 — load the tokenizer

```python
tokenizer = load(open('tokenizer.pkl', 'rb'))
```

---
## Step 16 — select a seed text

```python
# 获取长度 / Get length
seed_text = lines[randint(0,len(lines))]
# 打印输出 / Print output
print(seed_text + '\n')
```

---
## Step 17 — generate new text

```python
generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
# 打印输出 / Print output
print(generated)
```

---
## Learning Notes / 学习笔记

- **概念**: load doc into memory 是机器学习中的常用技术。  
  *load doc into memory is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Generate / 3 Generate
# Complete Code / 完整代码
# ===============================

from random import randint
from pickle import load
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
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
  # 添加元素到列表末尾 / Append element to list end
		result.append(out_word)
	return ' '.join(result)

# load cleaned text sequences
in_filename = 'republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
# 获取长度 / Get length
seq_length = len(lines[0].split()) - 1
# load the model
# 从文件加载模型 / Load model from file
model = load_model('model.h5')
# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# select a seed text
# 获取长度 / Get length
seed_text = lines[randint(0,len(lines))]
# 打印输出 / Print output
print(seed_text + '\n')
# generate new text
generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
# 打印输出 / Print output
print(generated)
```

---

### Chapter Summary / 章节总结

# Chapter 20 Summary / 第20章总结

## Theme / 主题: Chapter 20 / Chapter 20

This chapter contains **3 code files** demonstrating chapter 20.

本章包含 **3 个代码文件**，演示Chapter 20。

---
## Evolution / 演化路线

  1. `1_prepare_text.ipynb` — Prepare Text
  2. `2_train_model.ipynb` — Train Model
  3. `3_generate.ipynb` — Generate

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 20) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 20）是机器学习流水线中的基础构建块。

---
