# NLP 深度学习 / Deep Learning for NLP
## Chapter 09

---

### Load File



---

### Load All Files

# 02 — Load All Files / 02 Load All Files

**Chapter 09 — File 2 of 10 / 第09章 — 第2个文件（共10个）**

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
from os import listdir
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
## Step 6 — specify directory to load

```python
directory = 'txt_sentoken/neg'
```

---
## Step 7 — walk through all files in the folder

```python
for filename in listdir(directory):
```

---
## Step 8 — skip files that do not have the right extension

```python
if not filename.endswith(".txt"):
		continue
```

---
## Step 9 — create the full path of the file to open

```python
path = directory + '/' + filename
```

---
## Step 10 — load document

```python
doc = load_doc(path)
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
# Load All Files / 02 Load All Files
# Complete Code / 完整代码
# ===============================

from os import listdir

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# specify directory to load
directory = 'txt_sentoken/neg'
# walk through all files in the folder
for filename in listdir(directory):
	# skip files that do not have the right extension
	if not filename.endswith(".txt"):
		continue
	# create the full path of the file to open
	path = directory + '/' + filename
	# load document
	doc = load_doc(path)
	# print('Loaded %s' % filename)
```

---

➡️ **Next / 下一步**: File 3 of 10

---

### Load All Files With Func



---

### Load And Split



---

### Clean Review



---

### Clean Review Func



---

### Clean And Build Vocab

# 07 — Clean And Build Vocab / 数据清洗

**Chapter 09 — File 7 of 10 / 第09章 — 第7个文件（共10个）**

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
## Step 19 — skip files that do not have the right extension

```python
if not filename.endswith(".txt"):
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
process_docs('txt_sentoken/neg', vocab)
process_docs('txt_sentoken/pos', vocab)
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
# Clean And Build Vocab / 数据清洗
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
		# skip files that do not have the right extension
		if not filename.endswith(".txt"):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('txt_sentoken/neg', vocab)
process_docs('txt_sentoken/pos', vocab)
# print the size of the vocab
# 打印输出 / Print output
print(len(vocab))
# print the top words in the vocab
# 打印输出 / Print output
print(vocab.most_common(50))
```

---

➡️ **Next / 下一步**: File 8 of 10

---

### Build Vocabulary

# 08 — Build Vocabulary / 08 Build Vocabulary

**Chapter 09 — File 8 of 10 / 第09章 — 第8个文件（共10个）**

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
## Step 19 — skip files that do not have the right extension

```python
if not filename.endswith(".txt"):
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
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
```

---
## Step 23 — define vocab

```python
vocab = Counter()
```

---
## Step 24 — add all docs to vocab

```python
process_docs('txt_sentoken/neg', vocab)
process_docs('txt_sentoken/pos', vocab)
```

---
## Step 25 — print the size of the vocab

```python
# 打印输出 / Print output
print(len(vocab))
```

---
## Step 26 — print the top words in the vocab

```python
# 打印输出 / Print output
print(vocab.most_common(50))
```

---
## Step 27 — keep tokens with > 5 occurrence

```python
min_occurrence = 5
# 获取字典的键值对 / Get dict key-value pairs
tokens = [k for k,c in vocab.items() if c >= min_occurrence]
# 打印输出 / Print output
print(len(tokens))
```

---
## Step 28 — save tokens to a vocabulary file

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
# Build Vocabulary / 08 Build Vocabulary
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
		# skip files that do not have the right extension
		if not filename.endswith(".txt"):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# save list to file
def save_list(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('txt_sentoken/neg', vocab)
process_docs('txt_sentoken/pos', vocab)
# print the size of the vocab
# 打印输出 / Print output
print(len(vocab))
# print the top words in the vocab
# 打印输出 / Print output
print(vocab.most_common(50))
# keep tokens with > 5 occurrence
min_occurrence = 5
# 获取字典的键值对 / Get dict key-value pairs
tokens = [k for k,c in vocab.items() if c >= min_occurrence]
# 打印输出 / Print output
print(len(tokens))
# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')
```

---

➡️ **Next / 下一步**: File 9 of 10

---

### Load Vocab

# 09 — Load Vocab / 09 Load Vocab

**Chapter 09 — File 9 of 10 / 第09章 — 第9个文件（共10个）**

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
```

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
## Step 5 — load vocabulary

```python
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
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
# Load Vocab / 09 Load Vocab
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

# load vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
```

---

➡️ **Next / 下一步**: File 10 of 10

---

### Save Clean Filtered Reviews

# 10 — Save Clean Filtered Reviews / 保存/加载模型

**Chapter 09 — File 10 of 10 / 第09章 — 第10个文件（共10个）**

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
## Step 13 — save list to file

```python
def save_list(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
```

---
## Step 14 — load doc, clean and return line of tokens

```python
def doc_to_line(filename, vocab):
```

---
## Step 15 — load the doc

```python
doc = load_doc(filename)
```

---
## Step 16 — clean doc

```python
tokens = clean_doc(doc)
```

---
## Step 17 — filter by vocab

```python
tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)
```

---
## Step 18 — load all docs in a directory

```python
def process_docs(directory, vocab):
	lines = list()
```

---
## Step 19 — walk through all files in the folder

```python
for filename in listdir(directory):
```

---
## Step 20 — skip files that do not have the right extension

```python
if not filename.endswith(".txt"):
			continue
```

---
## Step 21 — create the full path of the file to open

```python
path = directory + '/' + filename
```

---
## Step 22 — load and clean the doc

```python
line = doc_to_line(path, vocab)
```

---
## Step 23 — add to list

```python
# 添加元素到列表末尾 / Append element to list end
lines.append(line)
	return lines
```

---
## Step 24 — load vocabulary

```python
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
```

---
## Step 25 — prepare negative reviews

```python
negative_lines = process_docs('txt_sentoken/neg', vocab)
save_list(negative_lines, 'negative.txt')
```

---
## Step 26 — prepare positive reviews

```python
positive_lines = process_docs('txt_sentoken/pos', vocab)
save_list(positive_lines, 'positive.txt')
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
# Save Clean Filtered Reviews / 保存/加载模型
# Complete Code / 完整代码
# ===============================

import string
# 导入正则表达式模块 / Import regex module
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
 # 获取长度 / Get length
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# save list to file
def save_list(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

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
		# skip files that do not have the right extension
		if not filename.endswith(".txt"):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
  # 添加元素到列表末尾 / Append element to list end
		lines.append(line)
	return lines

# load vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# prepare negative reviews
negative_lines = process_docs('txt_sentoken/neg', vocab)
save_list(negative_lines, 'negative.txt')
# prepare positive reviews
positive_lines = process_docs('txt_sentoken/pos', vocab)
save_list(positive_lines, 'positive.txt')
```

---

### Chapter Summary / 章节总结

# Chapter 09 Summary / 第09章总结

## Theme / 主题: Chapter 09 / Chapter 09

This chapter contains **10 code files** demonstrating chapter 09.

本章包含 **10 个代码文件**，演示Chapter 09。

---
## Evolution / 演化路线

  1. `01_load_file.ipynb` — Load File
  2. `02_load_all_files.ipynb` — Load All Files
  3. `03_load_all_files_with_func.ipynb` — Load All Files With Func
  4. `04_load_and_split.ipynb` — Load And Split
  5. `05_clean_review.ipynb` — Clean Review
  6. `06_clean_review_func.ipynb` — Clean Review Func
  7. `07_clean_and_build_vocab.ipynb` — Clean And Build Vocab
  8. `08_build_vocabulary.ipynb` — Build Vocabulary
  9. `09_load_vocab.ipynb` — Load Vocab
  10. `10_save_clean_filtered_reviews.ipynb` — Save Clean Filtered Reviews

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 09) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 09）是机器学习流水线中的基础构建块。

---
