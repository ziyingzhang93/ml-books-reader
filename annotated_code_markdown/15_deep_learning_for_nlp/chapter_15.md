# NLP 深度学习 / Deep Learning for NLP
## Chapter 15

---

### Clean Review



---

### Select Vocab

# 2 — Select Vocab / 特征选择

**Chapter 15 — File 2 of 5 / 第15章 — 第2个文件（共5个）**

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
from os import listdir
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
from nltk.corpus import stopwords
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
# 获取长度 / Get length
tokens = [word for word in tokens if len(word) > 1]
	return tokens
```

---
## Step 13 — load doc and add to vocab

```python
def add_doc_to_vocab(filename, vocab):
```

---
## Step 14 — load doc

```python
doc = load_doc(filename)
```

---
## Step 15 — clean doc

```python
tokens = clean_doc(doc)
```

---
## Step 16 — update counts

```python
vocab.update(tokens)
```

---
## Step 17 — load all docs in a directory

```python
def process_docs(directory, vocab):
```

---
## Step 18 — walk through all files in the folder

```python
for filename in listdir(directory):
```

---
## Step 19 — skip any reviews in the test set

```python
if filename.startswith('cv9'):
			continue
```

---
## Step 20 — create the full path of the file to open

```python
path = directory + '/' + filename
```

---
## Step 21 — add doc to vocab

```python
add_doc_to_vocab(path, vocab)
```

---
## Step 22 — define vocab

```python
vocab = Counter()
```

---
## Step 23 — add all docs to vocab

```python
process_docs('txt_sentoken/pos', vocab)
process_docs('txt_sentoken/neg', vocab)
```

---
## Step 24 — print the size of the vocab

```python
# 打印输出 / Print output
print(len(vocab))
```

---
## Step 25 — print the top words in the vocab

```python
# 打印输出 / Print output
print(vocab.most_common(50))
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
# Select Vocab / 特征选择
# Complete Code / 完整代码
# ===============================

import string
# 导入正则表达式模块 / Import regex module
import re
from os import listdir
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
from nltk.corpus import stopwords

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
 # 获取长度 / Get length
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('txt_sentoken/pos', vocab)
process_docs('txt_sentoken/neg', vocab)
# print the size of the vocab
# 打印输出 / Print output
print(len(vocab))
# print the top words in the vocab
# 打印输出 / Print output
print(vocab.most_common(50))
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Filter Vocab

# 3 — Filter Vocab / 3 Filter Vocab

**Chapter 15 — File 3 of 5 / 第15章 — 第3个文件（共5个）**

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
from os import listdir
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
from nltk.corpus import stopwords
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
# 获取长度 / Get length
tokens = [word for word in tokens if len(word) > 1]
	return tokens
```

---
## Step 13 — load doc and add to vocab

```python
def add_doc_to_vocab(filename, vocab):
```

---
## Step 14 — load doc

```python
doc = load_doc(filename)
```

---
## Step 15 — clean doc

```python
tokens = clean_doc(doc)
```

---
## Step 16 — update counts

```python
vocab.update(tokens)
```

---
## Step 17 — load all docs in a directory

```python
def process_docs(directory, vocab):
```

---
## Step 18 — walk through all files in the folder

```python
for filename in listdir(directory):
```

---
## Step 19 — skip any reviews in the test set

```python
if filename.startswith('cv9'):
			continue
```

---
## Step 20 — create the full path of the file to open

```python
path = directory + '/' + filename
```

---
## Step 21 — add doc to vocab

```python
add_doc_to_vocab(path, vocab)
```

---
## Step 22 — save list to file

```python
def save_list(lines, filename):
```

---
## Step 23 — convert lines to a single blob of text

```python
data = '\n'.join(lines)
```

---
## Step 24 — open file

```python
file = open(filename, 'w')
```

---
## Step 25 — write text

```python
file.write(data)
```

---
## Step 26 — close file

```python
file.close()
```

---
## Step 27 — define vocab

```python
vocab = Counter()
```

---
## Step 28 — add all docs to vocab

```python
process_docs('txt_sentoken/pos', vocab)
process_docs('txt_sentoken/neg', vocab)
```

---
## Step 29 — print the size of the vocab

```python
# 打印输出 / Print output
print(len(vocab))
```

---
## Step 30 — keep tokens with a min occurrence

```python
min_occurrence = 2
# 获取字典的键值对 / Get dict key-value pairs
tokens = [k for k,c in vocab.items() if c >= min_occurrence]
# 打印输出 / Print output
print(len(tokens))
```

---
## Step 31 — save tokens to a vocabulary file

```python
save_list(tokens, 'vocab.txt')
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
# Filter Vocab / 3 Filter Vocab
# Complete Code / 完整代码
# ===============================

import string
# 导入正则表达式模块 / Import regex module
import re
from os import listdir
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
from nltk.corpus import stopwords

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
 # 获取长度 / Get length
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# save list to file
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('txt_sentoken/pos', vocab)
process_docs('txt_sentoken/neg', vocab)
# print the size of the vocab
# 打印输出 / Print output
print(len(vocab))
# keep tokens with a min occurrence
min_occurrence = 2
# 获取字典的键值对 / Get dict key-value pairs
tokens = [k for k,c in vocab.items() if c >= min_occurrence]
# 打印输出 / Print output
print(len(tokens))
# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Cnn Model



---

### Evaluate

# 5 — Evaluate / 模型评估

**Chapter 15 — File 5 of 5 / 第15章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **load doc into memory**.

本脚本演示 **load doc into memory**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
import string
# 导入正则表达式模块 / Import regex module
import re
from os import listdir
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import Tokenizer
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
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
def clean_doc(doc, vocab):
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
## Step 10 — filter out tokens not in vocab

```python
tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens
```

---
## Step 11 — load all docs in a directory

```python
def process_docs(directory, vocab, is_train):
	documents = list()
```

---
## Step 12 — walk through all files in the folder

```python
for filename in listdir(directory):
```

---
## Step 13 — skip any reviews in the test set

```python
if is_train and filename.startswith('cv9'):
			continue
		if not is_train and not filename.startswith('cv9'):
			continue
```

---
## Step 14 — create the full path of the file to open

```python
path = directory + '/' + filename
```

---
## Step 15 — load the doc

```python
doc = load_doc(path)
```

---
## Step 16 — clean doc

```python
tokens = clean_doc(doc, vocab)
```

---
## Step 17 — add to list

```python
# 添加元素到列表末尾 / Append element to list end
documents.append(tokens)
	return documents
```

---
## Step 18 — load and clean a dataset

```python
def load_clean_dataset(vocab, is_train):
```

---
## Step 19 — load documents

```python
neg = process_docs('txt_sentoken/neg', vocab, is_train)
	pos = process_docs('txt_sentoken/pos', vocab, is_train)
	docs = neg + pos
```

---
## Step 20 — prepare labels

```python
# 获取长度 / Get length
labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
	return docs, labels
```

---
## Step 21 — fit a tokenizer

```python
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
```

---
## Step 22 — integer encode and pad documents

```python
def encode_docs(tokenizer, max_length, docs):
```

---
## Step 23 — integer encode

```python
encoded = tokenizer.texts_to_sequences(docs)
```

---
## Step 24 — pad sequences

```python
padded = pad_sequences(encoded, maxlen=max_length, padding='post')
	return padded
```

---
## Step 25 — classify a review as negative or positive

```python
def predict_sentiment(review, vocab, tokenizer, max_length, model):
```

---
## Step 26 — clean review

```python
line = clean_doc(review, vocab)
```

---
## Step 27 — encode and pad review

```python
padded = encode_docs(tokenizer, max_length, [line])
```

---
## Step 28 — predict sentiment

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(padded, verbose=0)
```

---
## Step 29 — retrieve predicted percentage and label

```python
percent_pos = yhat[0,0]
	if round(percent_pos) == 0:
		return (1-percent_pos), 'NEGATIVE'
	return percent_pos, 'POSITIVE'
```

---
## Step 30 — load the vocabulary

```python
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
```

---
## Step 31 — load all reviews

```python
train_docs, ytrain = load_clean_dataset(vocab, True)
test_docs, ytest = load_clean_dataset(vocab, False)
```

---
## Step 32 — create the tokenizer

```python
tokenizer = create_tokenizer(train_docs)
```

---
## Step 33 — define vocabulary size

```python
# 获取长度 / Get length
vocab_size = len(tokenizer.word_index) + 1
# 打印输出 / Print output
print('Vocabulary size: %d' % vocab_size)
```

---
## Step 34 — calculate the maximum sequence length

```python
# 获取长度 / Get length
max_length = max([len(s.split()) for s in train_docs])
# 打印输出 / Print output
print('Maximum length: %d' % max_length)
```

---
## Step 35 — encode data

```python
Xtrain = encode_docs(tokenizer, max_length, train_docs)
Xtest = encode_docs(tokenizer, max_length, test_docs)
```

---
## Step 36 — load the model

```python
# 从文件加载模型 / Load model from file
model = load_model('model.h5')
```

---
## Step 37 — evaluate model on training dataset

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, acc = model.evaluate(Xtrain, ytrain, verbose=0)
# 打印输出 / Print output
print('Train Accuracy: %.2f' % (acc*100))
```

---
## Step 38 — evaluate model on test dataset

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, acc = model.evaluate(Xtest, ytest, verbose=0)
# 打印输出 / Print output
print('Test Accuracy: %.2f' % (acc*100))
```

---
## Step 39 — test positive text

```python
text = 'Everyone will enjoy this film. I love it, recommended!'
percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
# 打印输出 / Print output
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
```

---
## Step 40 — test negative text

```python
text = 'This is a bad movie. Do not watch it. It sucks.'
percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
# 打印输出 / Print output
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
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
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate / 模型评估
# Complete Code / 完整代码
# ===============================

import string
# 导入正则表达式模块 / Import regex module
import re
from os import listdir
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import Tokenizer
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model

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
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# prepare regex for char filtering
	re_punc = re.compile('[%s]' % re.escape(string.punctuation))
	# remove punctuation from each word
	tokens = [re_punc.sub('', w) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_train):
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
		tokens = clean_doc(doc, vocab)
		# add to list
  # 添加元素到列表末尾 / Append element to list end
		documents.append(tokens)
	return documents

# load and clean a dataset
def load_clean_dataset(vocab, is_train):
	# load documents
	neg = process_docs('txt_sentoken/neg', vocab, is_train)
	pos = process_docs('txt_sentoken/pos', vocab, is_train)
	docs = neg + pos
	# prepare labels
 # 获取长度 / Get length
	labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
	return docs, labels

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
	# integer encode
	encoded = tokenizer.texts_to_sequences(docs)
	# pad sequences
	padded = pad_sequences(encoded, maxlen=max_length, padding='post')
	return padded

# classify a review as negative or positive
def predict_sentiment(review, vocab, tokenizer, max_length, model):
	# clean review
	line = clean_doc(review, vocab)
	# encode and pad review
	padded = encode_docs(tokenizer, max_length, [line])
	# predict sentiment
 # 用模型做预测 / Make predictions with model
	yhat = model.predict(padded, verbose=0)
	# retrieve predicted percentage and label
	percent_pos = yhat[0,0]
	if round(percent_pos) == 0:
		return (1-percent_pos), 'NEGATIVE'
	return percent_pos, 'POSITIVE'

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load all reviews
train_docs, ytrain = load_clean_dataset(vocab, True)
test_docs, ytest = load_clean_dataset(vocab, False)
# create the tokenizer
tokenizer = create_tokenizer(train_docs)
# define vocabulary size
# 获取长度 / Get length
vocab_size = len(tokenizer.word_index) + 1
# 打印输出 / Print output
print('Vocabulary size: %d' % vocab_size)
# calculate the maximum sequence length
# 获取长度 / Get length
max_length = max([len(s.split()) for s in train_docs])
# 打印输出 / Print output
print('Maximum length: %d' % max_length)
# encode data
Xtrain = encode_docs(tokenizer, max_length, train_docs)
Xtest = encode_docs(tokenizer, max_length, test_docs)
# load the model
# 从文件加载模型 / Load model from file
model = load_model('model.h5')
# evaluate model on training dataset
# 评估模型在测试集上的表现 / Evaluate model on test set
_, acc = model.evaluate(Xtrain, ytrain, verbose=0)
# 打印输出 / Print output
print('Train Accuracy: %.2f' % (acc*100))
# evaluate model on test dataset
# 评估模型在测试集上的表现 / Evaluate model on test set
_, acc = model.evaluate(Xtest, ytest, verbose=0)
# 打印输出 / Print output
print('Test Accuracy: %.2f' % (acc*100))

# test positive text
text = 'Everyone will enjoy this film. I love it, recommended!'
percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
# 打印输出 / Print output
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
# test negative text
text = 'This is a bad movie. Do not watch it. It sucks.'
percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
# 打印输出 / Print output
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
```

---

### Chapter Summary / 章节总结



---
