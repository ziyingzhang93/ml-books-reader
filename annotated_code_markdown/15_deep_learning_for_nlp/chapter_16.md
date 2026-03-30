# NLP深度学习
## Chapter 16

---

### Clean Review

# 1 — Clean Review / 数据清洗

**Chapter 16 — File 1 of 4 / 第16章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **load doc into memory**.

本脚本演示 **load doc into memory**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from nltk.corpus import stopwords
import string
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
## Step 7 — split into tokens by white space

```python
tokens = doc.split()
```

---
## Step 8 — prepare regex for char filtering

```python
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
```

---
## Step 9 — remove punctuation from each word

```python
tokens = [re_punc.sub('', w) for w in tokens]
```

---
## Step 10 — remove remaining tokens that are not alphabetic

```python
tokens = [word for word in tokens if word.isalpha()]
```

---
## Step 11 — filter out stop words

```python
stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
```

---
## Step 12 — filter out short tokens

```python
tokens = [word for word in tokens if len(word) > 1]
	return tokens
```

---
## Step 13 — load the document

```python
filename = 'txt_sentoken/pos/cv000_29590.txt'
text = load_doc(filename)
tokens = clean_doc(text)
print(tokens)
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
# Clean Review / 数据清洗
# Complete Code / 完整代码
# ===============================

from nltk.corpus import stopwords
import string
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
	# split into tokens by white space
	tokens = doc.split()
	# prepare regex for char filtering
	re_punc = re.compile('[%s]' % re.escape(string.punctuation))
	# remove punctuation from each word
	tokens = [re_punc.sub('', w) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load the document
filename = 'txt_sentoken/pos/cv000_29590.txt'
text = load_doc(filename)
tokens = clean_doc(text)
print(tokens)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Clean All Reviews

# 2 — Clean All Reviews / 数据清洗

**Chapter 16 — File 2 of 4 / 第16章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **load doc into memory**.

本脚本演示 **load doc into memory**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import string
import re
from os import listdir
from nltk.corpus import stopwords
from pickle import dump
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
## Step 7 — split into tokens by white space

```python
tokens = doc.split()
```

---
## Step 8 — prepare regex for char filtering

```python
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
```

---
## Step 9 — remove punctuation from each word

```python
tokens = [re_punc.sub('', w) for w in tokens]
```

---
## Step 10 — remove remaining tokens that are not alphabetic

```python
tokens = [word for word in tokens if word.isalpha()]
```

---
## Step 11 — filter out stop words

```python
stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
```

---
## Step 12 — filter out short tokens

```python
tokens = [word for word in tokens if len(word) > 1]
	tokens = ' '.join(tokens)
	return tokens
```

---
## Step 13 — load all docs in a directory

```python
def process_docs(directory, is_train):
	documents = list()
```

---
## Step 14 — walk through all files in the folder

```python
for filename in listdir(directory):
```

---
## Step 15 — skip any reviews in the test set

```python
if is_train and filename.startswith('cv9'):
			continue
		if not is_train and not filename.startswith('cv9'):
			continue
```

---
## Step 16 — create the full path of the file to open

```python
path = directory + '/' + filename
```

---
## Step 17 — load the doc

```python
doc = load_doc(path)
```

---
## Step 18 — clean doc

```python
tokens = clean_doc(doc)
```

---
## Step 19 — add to list

```python
documents.append(tokens)
	return documents
```

---
## Step 20 — load and clean a dataset

```python
def load_clean_dataset(is_train):
```

---
## Step 21 — load documents

```python
neg = process_docs('txt_sentoken/neg', is_train)
	pos = process_docs('txt_sentoken/pos', is_train)
	docs = neg + pos
```

---
## Step 22 — prepare labels

```python
labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
	return docs, labels
```

---
## Step 23 — save a dataset to file

```python
def save_dataset(dataset, filename):
	dump(dataset, open(filename, 'wb'))
	print('Saved: %s' % filename)
```

---
## Step 24 — load and clean all reviews

```python
train_docs, ytrain = load_clean_dataset(True)
test_docs, ytest = load_clean_dataset(False)
```

---
## Step 25 — save training datasets

```python
save_dataset([train_docs, ytrain], 'train.pkl')
save_dataset([test_docs, ytest], 'test.pkl')
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
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Clean All Reviews / 数据清洗
# Complete Code / 完整代码
# ===============================

import string
import re
from os import listdir
from nltk.corpus import stopwords
from pickle import dump

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
	# split into tokens by white space
	tokens = doc.split()
	# prepare regex for char filtering
	re_punc = re.compile('[%s]' % re.escape(string.punctuation))
	# remove punctuation from each word
	tokens = [re_punc.sub('', w) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	tokens = ' '.join(tokens)
	return tokens

# load all docs in a directory
def process_docs(directory, is_train):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_train and filename.startswith('cv9'):
			continue
		if not is_train and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc)
		# add to list
		documents.append(tokens)
	return documents

# load and clean a dataset
def load_clean_dataset(is_train):
	# load documents
	neg = process_docs('txt_sentoken/neg', is_train)
	pos = process_docs('txt_sentoken/pos', is_train)
	docs = neg + pos
	# prepare labels
	labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
	return docs, labels

# save a dataset to file
def save_dataset(dataset, filename):
	dump(dataset, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load and clean all reviews
train_docs, ytrain = load_clean_dataset(True)
test_docs, ytest = load_clean_dataset(False)
# save training datasets
save_dataset([train_docs, ytrain], 'train.pkl')
save_dataset([test_docs, ytest], 'test.pkl')
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Model

# 3 — Model / 3 Model

**Chapter 16 — File 3 of 4 / 第16章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **load a clean dataset**.

本脚本演示 **load a clean dataset**。

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
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
```

---
## Step 2 — load a clean dataset

```python
def load_dataset(filename):
	return load(open(filename, 'rb'))
```

---
## Step 3 — fit a tokenizer

```python
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
```

---
## Step 4 — calculate the maximum document length

```python
def max_length(lines):
	return max([len(s.split()) for s in lines])
```

---
## Step 5 — encode a list of lines

```python
def encode_text(tokenizer, lines, length):
```

---
## Step 6 — integer encode

```python
encoded = tokenizer.texts_to_sequences(lines)
```

---
## Step 7 — pad encoded sequences

```python
padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded
```

---
## Step 8 — define the model

```python
def define_model(length, vocab_size):
```

---
## Step 9 — channel 1

```python
inputs1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_size, 100)(inputs1)
	conv1 = Conv1D(32, 4, activation='relu')(embedding1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D()(drop1)
	flat1 = Flatten()(pool1)
```

---
## Step 10 — channel 2

```python
inputs2 = Input(shape=(length,))
	embedding2 = Embedding(vocab_size, 100)(inputs2)
	conv2 = Conv1D(32, 6, activation='relu')(embedding2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D()(drop2)
	flat2 = Flatten()(pool2)
```

---
## Step 11 — channel 3

```python
inputs3 = Input(shape=(length,))
	embedding3 = Embedding(vocab_size, 100)(inputs3)
	conv3 = Conv1D(32, 8, activation='relu')(embedding3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D()(drop3)
	flat3 = Flatten()(pool3)
```

---
## Step 12 — merge

```python
merged = concatenate([flat1, flat2, flat3])
```

---
## Step 13 — interpretation

```python
dense1 = Dense(10, activation='relu')(merged)
	outputs = Dense(1, activation='sigmoid')(dense1)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
```

---
## Step 14 — compile

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 15 — summarize

```python
model.summary()
	plot_model(model, show_shapes=True, to_file='model.png')
	return model
```

---
## Step 16 — load training dataset

```python
trainLines, trainLabels = load_dataset('train.pkl')
```

---
## Step 17 — create tokenizer

```python
tokenizer = create_tokenizer(trainLines)
```

---
## Step 18 — calculate max document length

```python
length = max_length(trainLines)
print('Max document length: %d' % length)
```

---
## Step 19 — calculate vocabulary size

```python
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)
```

---
## Step 20 — encode data

```python
trainX = encode_text(tokenizer, trainLines, length)
```

---
## Step 21 — define model

```python
model = define_model(length, vocab_size)
```

---
## Step 22 — fit model

```python
model.fit([trainX,trainX,trainX], array(trainLabels), epochs=7, batch_size=16)
```

---
## Step 23 — save the model

```python
model.save('model.h5')
```

---
## Learning Notes / 学习笔记

- **概念**: load a clean dataset 是机器学习中的常用技术。  
  *load a clean dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
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
# Model / 3 Model
# Complete Code / 完整代码
# ===============================

from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

# load a clean dataset
def load_dataset(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the maximum document length
def max_length(lines):
	return max([len(s.split()) for s in lines])

# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded

# define the model
def define_model(length, vocab_size):
	# channel 1
	inputs1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_size, 100)(inputs1)
	conv1 = Conv1D(32, 4, activation='relu')(embedding1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D()(drop1)
	flat1 = Flatten()(pool1)
	# channel 2
	inputs2 = Input(shape=(length,))
	embedding2 = Embedding(vocab_size, 100)(inputs2)
	conv2 = Conv1D(32, 6, activation='relu')(embedding2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D()(drop2)
	flat2 = Flatten()(pool2)
	# channel 3
	inputs3 = Input(shape=(length,))
	embedding3 = Embedding(vocab_size, 100)(inputs3)
	conv3 = Conv1D(32, 8, activation='relu')(embedding3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D()(drop3)
	flat3 = Flatten()(pool3)
	# merge
	merged = concatenate([flat1, flat2, flat3])
	# interpretation
	dense1 = Dense(10, activation='relu')(merged)
	outputs = Dense(1, activation='sigmoid')(dense1)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
	# compile
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize
	model.summary()
	plot_model(model, show_shapes=True, to_file='model.png')
	return model

# load training dataset
trainLines, trainLabels = load_dataset('train.pkl')
# create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
print('Max document length: %d' % length)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
# define model
model = define_model(length, vocab_size)
# fit model
model.fit([trainX,trainX,trainX], array(trainLabels), epochs=7, batch_size=16)
# save the model
model.save('model.h5')
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Evaluate

# 4 — Evaluate / 模型评估

**Chapter 16 — File 4 of 4 / 第16章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **load a clean dataset**.

本脚本演示 **load a clean dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — Step 1

```python
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
```

---
## Step 2 — load a clean dataset

```python
def load_dataset(filename):
	return load(open(filename, 'rb'))
```

---
## Step 3 — fit a tokenizer

```python
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
```

---
## Step 4 — calculate the maximum document length

```python
def max_length(lines):
	return max([len(s.split()) for s in lines])
```

---
## Step 5 — encode a list of lines

```python
def encode_text(tokenizer, lines, length):
```

---
## Step 6 — integer encode

```python
encoded = tokenizer.texts_to_sequences(lines)
```

---
## Step 7 — pad encoded sequences

```python
padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded
```

---
## Step 8 — load datasets

```python
trainLines, trainLabels = load_dataset('train.pkl')
testLines, testLabels = load_dataset('test.pkl')
```

---
## Step 9 — create tokenizer

```python
tokenizer = create_tokenizer(trainLines)
```

---
## Step 10 — calculate max document length

```python
length = max_length(trainLines)
print('Max document length: %d' % length)
```

---
## Step 11 — calculate vocabulary size

```python
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)
```

---
## Step 12 — encode data

```python
trainX = encode_text(tokenizer, trainLines, length)
testX = encode_text(tokenizer, testLines, length)
```

---
## Step 13 — load the model

```python
model = load_model('model.h5')
```

---
## Step 14 — evaluate model on training dataset

```python
_, acc = model.evaluate([trainX,trainX,trainX], array(trainLabels), verbose=0)
print('Train Accuracy: %.2f' % (acc*100))
```

---
## Step 15 — evaluate model on test dataset dataset

```python
_, acc = model.evaluate([testX,testX,testX], array(testLabels), verbose=0)
print('Test Accuracy: %.2f' % (acc*100))
```

---
## Learning Notes / 学习笔记

- **概念**: load a clean dataset 是机器学习中的常用技术。  
  *load a clean dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate / 模型评估
# Complete Code / 完整代码
# ===============================

from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# load a clean dataset
def load_dataset(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the maximum document length
def max_length(lines):
	return max([len(s.split()) for s in lines])

# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded

# load datasets
trainLines, trainLabels = load_dataset('train.pkl')
testLines, testLabels = load_dataset('test.pkl')
# create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
print('Max document length: %d' % length)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
testX = encode_text(tokenizer, testLines, length)
# load the model
model = load_model('model.h5')
# evaluate model on training dataset
_, acc = model.evaluate([trainX,trainX,trainX], array(trainLabels), verbose=0)
print('Train Accuracy: %.2f' % (acc*100))
# evaluate model on test dataset dataset
_, acc = model.evaluate([testX,testX,testX], array(testLabels), verbose=0)
print('Test Accuracy: %.2f' % (acc*100))
```

---
