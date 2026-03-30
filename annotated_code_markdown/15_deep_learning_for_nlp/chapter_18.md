# NLP深度学习
## Chapter 18

---

### Prepare Data

# 1 — Prepare Data / 数据准备

**Chapter 18 — File 1 of 3 / 第18章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **load doc into memory**.

本脚本演示 **load doc into memory**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — load doc into memory

```python
def load_doc(filename):
```

---
## Step 2 — open the file as read only

```python
file = open(filename, 'r')
```

---
## Step 3 — read all text

```python
text = file.read()
```

---
## Step 4 — close the file

```python
file.close()
	return text
```

---
## Step 5 — save tokens to file, one dialog per line

```python
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
```

---
## Step 6 — load text

```python
raw_text = load_doc('rhyme.txt')
print(raw_text)
```

---
## Step 7 — clean

```python
tokens = raw_text.split()
raw_text = ' '.join(tokens)
```

---
## Step 8 — organize into sequences of characters

```python
length = 10
sequences = list()
for i in range(length, len(raw_text)):
```

---
## Step 9 — select sequence of tokens

```python
seq = raw_text[i-length:i+1]
```

---
## Step 10 — store

```python
sequences.append(seq)
print('Total Sequences: %d' % len(sequences))
```

---
## Step 11 — save sequences to file

```python
out_filename = 'char_sequences.txt'
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
# Prepare Data / 数据准备
# Complete Code / 完整代码
# ===============================

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# load text
raw_text = load_doc('rhyme.txt')
print(raw_text)
# clean
tokens = raw_text.split()
raw_text = ' '.join(tokens)
# organize into sequences of characters
length = 10
sequences = list()
for i in range(length, len(raw_text)):
	# select sequence of tokens
	seq = raw_text[i-length:i+1]
	# store
	sequences.append(seq)
print('Total Sequences: %d' % len(sequences))
# save sequences to file
out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Train Model

# 2 — Train Model / 2 Train Model

**Chapter 18 — File 2 of 3 / 第18章 — 第2个文件（共3个）**

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
## Step 1 — Step 1

```python
from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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
def define_model(X):
	model = Sequential()
	model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dense(vocab_size, activation='softmax'))
```

---
## Step 7 — compile model

```python
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
in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')
```

---
## Step 10 — integer encode sequences of characters

```python
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in lines:
```

---
## Step 11 — integer encode line

```python
encoded_seq = [mapping[char] for char in line]
```

---
## Step 12 — store

```python
sequences.append(encoded_seq)
```

---
## Step 13 — vocabulary size

```python
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)
```

---
## Step 14 — separate into input and output

```python
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)
```

---
## Step 15 — define model

```python
model = define_model(X)
```

---
## Step 16 — fit model

```python
model.fit(X, y, epochs=100, verbose=2)
```

---
## Step 17 — save the model to file

```python
model.save('model.h5')
```

---
## Step 18 — save the mapping

```python
dump(mapping, open('mapping.pkl', 'wb'))
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
# Train Model / 2 Train Model
# Complete Code / 完整代码
# ===============================

from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

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
def define_model(X):
	model = Sequential()
	model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dense(vocab_size, activation='softmax'))
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize defined model
	model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

# load
in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')
# integer encode sequences of characters
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in lines:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences.append(encoded_seq)
# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)
# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)
# define model
model = define_model(X)
# fit model
model.fit(X, y, epochs=100, verbose=2)
# save the model to file
model.save('model.h5')
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Generate

# 3 — Generate / 3 Generate

**Chapter 18 — File 3 of 3 / 第18章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **generate a sequence of characters with a language model**.

本脚本演示 **generate a sequence of characters with a language model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from pickle import load
from numpy import array
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
```

---
## Step 2 — generate a sequence of characters with a language model

```python
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
```

---
## Step 3 — generate a fixed number of characters

```python
for _ in range(n_chars):
```

---
## Step 4 — encode the characters as integers

```python
encoded = [mapping[char] for char in in_text]
```

---
## Step 5 — truncate sequences to a fixed length

```python
encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
```

---
## Step 6 — one hot encode

```python
encoded = to_categorical(encoded, num_classes=len(mapping))
```

---
## Step 7 — predict character

```python
yhat = model.predict_classes(encoded, verbose=0)
```

---
## Step 8 — reverse map integer to character

```python
out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
```

---
## Step 9 — append to input

```python
in_text += out_char
	return in_text
```

---
## Step 10 — load the model

```python
model = load_model('model.h5')
```

---
## Step 11 — load the mapping

```python
mapping = load(open('mapping.pkl', 'rb'))
```

---
## Step 12 — test start of rhyme

```python
print(generate_seq(model, mapping, 10, 'Sing a son', 20))
```

---
## Step 13 — test mid-line

```python
print(generate_seq(model, mapping, 10, 'king was i', 20))
```

---
## Step 14 — test not in original

```python
print(generate_seq(model, mapping, 10, 'hello worl', 20))
```

---
## Learning Notes / 学习笔记

- **概念**: generate a sequence of characters with a language model 是机器学习中的常用技术。  
  *generate a sequence of characters with a language model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Generate / 3 Generate
# Complete Code / 完整代码
# ===============================

from pickle import load
from numpy import array
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
		encoded = to_categorical(encoded, num_classes=len(mapping))
		# predict character
		yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += out_char
	return in_text

# load the model
model = load_model('model.h5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))
# test start of rhyme
print(generate_seq(model, mapping, 10, 'Sing a son', 20))
# test mid-line
print(generate_seq(model, mapping, 10, 'king was i', 20))
# test not in original
print(generate_seq(model, mapping, 10, 'hello worl', 20))
```

---

### Chapter Summary

# Chapter 18 Summary / 第18章总结

## Theme / 主题: Chapter 18 / Chapter 18

This chapter contains **3 code files** demonstrating chapter 18.

本章包含 **3 个代码文件**，演示Chapter 18。

---
## Evolution / 演化路线

  1. `1_prepare_data.ipynb` — Prepare Data
  2. `2_train_model.ipynb` — Train Model
  3. `3_generate.ipynb` — Generate

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 18) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 18）是机器学习流水线中的基础构建块。

---
