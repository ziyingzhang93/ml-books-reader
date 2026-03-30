# NLP深度学习
## Chapter 10

---

### Select Vocab

# 2 — Select Vocab / 特征选择

**Chapter 10 — File 2 of 8 / 第10章 — 第2个文件（共8个）**

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
print(len(vocab))
```

---
## Step 25 — print the top words in the vocab

```python
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
import re
from os import listdir
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
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Filter Vocab

# 3 — Filter Vocab / 3 Filter Vocab

**Chapter 10 — File 3 of 8 / 第10章 — 第3个文件（共8个）**

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
print(len(vocab))
```

---
## Step 30 — keep tokens with a min occurrence

```python
min_occurrence = 2
tokens = [k for k,c in vocab.items() if c >= min_occurrence]
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
import re
from os import listdir
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
print(len(vocab))
# keep tokens with a min occurrence
min_occurrence = 2
tokens = [k for k,c in vocab.items() if c >= min_occurrence]
print(len(tokens))
# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Filter All Reviews

# 4 — Filter All Reviews / 4 Filter All Reviews

**Chapter 10 — File 4 of 8 / 第10章 — 第4个文件（共8个）**

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
## Step 13 — load doc, clean and return line of tokens

```python
def doc_to_line(filename, vocab):
```

---
## Step 14 — load the doc

```python
doc = load_doc(filename)
```

---
## Step 15 — clean doc

```python
tokens = clean_doc(doc)
```

---
## Step 16 — filter by vocab

```python
tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)
```

---
## Step 17 — load all docs in a directory

```python
def process_docs(directory, vocab):
	lines = list()
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
## Step 21 — load and clean the doc

```python
line = doc_to_line(path, vocab)
```

---
## Step 22 — add to list

```python
lines.append(line)
	return lines
```

---
## Step 23 — load and clean a dataset

```python
def load_clean_dataset(vocab):
```

---
## Step 24 — load documents

```python
neg = process_docs('txt_sentoken/neg', vocab)
	pos = process_docs('txt_sentoken/pos', vocab)
	docs = neg + pos
```

---
## Step 25 — prepare labels

```python
labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
	return docs, labels
```

---
## Step 26 — load the vocabulary

```python
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
```

---
## Step 27 — load all training reviews

```python
docs, labels = load_clean_dataset(vocab)
```

---
## Step 28 — summarize what we have

```python
print(len(docs), len(labels))
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
# Filter All Reviews / 4 Filter All Reviews
# Complete Code / 完整代码
# ===============================

import string
import re
from os import listdir
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
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines

# load and clean a dataset
def load_clean_dataset(vocab):
	# load documents
	neg = process_docs('txt_sentoken/neg', vocab)
	pos = process_docs('txt_sentoken/pos', vocab)
	docs = neg + pos
	# prepare labels
	labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
	return docs, labels

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# load all training reviews
docs, labels = load_clean_dataset(vocab)
# summarize what we have
print(len(docs), len(labels))
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Chapter Summary

# Chapter 10 Summary / 第10章总结

## Theme / 主题: Chapter 10 / Chapter 10

This chapter contains **8 code files** demonstrating chapter 10.

本章包含 **8 个代码文件**，演示Chapter 10。

---
## Evolution / 演化路线

  1. `1_clean_review.ipynb` — Clean Review
  2. `2_select_vocab.ipynb` — Select Vocab
  3. `3_filter_vocab.ipynb` — Filter Vocab
  4. `4_filter_all_reviews.ipynb` — Filter All Reviews
  5. `5_prepare_data.ipynb` — Prepare Data
  6. `6_mlp_bow_model.ipynb` — Mlp Bow Model
  7. `7_compare_encodings.ipynb` — Compare Encodings
  8. `8_prediction.ipynb` — Prediction

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 10) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 10）是机器学习流水线中的基础构建块。

---
