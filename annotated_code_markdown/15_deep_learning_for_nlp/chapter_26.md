# NLP 深度学习 / Deep Learning for NLP
## Chapter 26

---

### Extract Features



---

### Data Prep

# 2 — Data Prep / 2 Data Prep

**Chapter 26 — File 2 of 7 / 第26章 — 第2个文件（共7个）**

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
## Step 6 — extract descriptions for images

```python
def load_descriptions(doc):
	mapping = dict()
```

---
## Step 7 — process lines

```python
for line in doc.split('\n'):
```

---
## Step 8 — split line by white space

```python
tokens = line.split()
  # 获取长度 / Get length
		if len(line) < 2:
			continue
```

---
## Step 9 — take the first token as the image id, the rest as the description

```python
image_id, image_desc = tokens[0], tokens[1:]
```

---
## Step 10 — remove filename from image id

```python
image_id = image_id.split('.')[0]
```

---
## Step 11 — convert description tokens back to string

```python
image_desc = ' '.join(image_desc)
```

---
## Step 12 — create the list if needed

```python
if image_id not in mapping:
			mapping[image_id] = list()
```

---
## Step 13 — store description

```python
# 添加元素到列表末尾 / Append element to list end
mapping[image_id].append(image_desc)
	return mapping

def clean_descriptions(descriptions):
```

---
## Step 14 — prepare regex for char filtering

```python
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
 # 获取字典的键值对 / Get dict key-value pairs
	for _, desc_list in descriptions.items():
  # 获取长度 / Get length
		for i in range(len(desc_list)):
			desc = desc_list[i]
```

---
## Step 15 — tokenize

```python
desc = desc.split()
```

---
## Step 16 — convert to lower case

```python
desc = [word.lower() for word in desc]
```

---
## Step 17 — remove punctuation from each token

```python
desc = [re_punc.sub('', w) for w in desc]
```

---
## Step 18 — remove hanging 's' and 'a'

```python
# 获取长度 / Get length
desc = [word for word in desc if len(word)>1]
```

---
## Step 19 — remove tokens with numbers in them

```python
desc = [word for word in desc if word.isalpha()]
```

---
## Step 20 — store as string

```python
desc_list[i] =  ' '.join(desc)
```

---
## Step 21 — convert the loaded descriptions into a vocabulary of words

```python
def to_vocabulary(descriptions):
```

---
## Step 22 — build a list of all description strings

```python
all_desc = set()
 # 获取字典的所有键 / Get all dict keys
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc
```

---
## Step 23 — save descriptions to file, one per line

```python
def save_descriptions(descriptions, filename):
	lines = list()
 # 获取字典的键值对 / Get dict key-value pairs
	for key, desc_list in descriptions.items():
		for desc in desc_list:
   # 添加元素到列表末尾 / Append element to list end
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

filename = 'Flickr8k_text/Flickr8k.token.txt'
```

---
## Step 24 — load descriptions

```python
doc = load_doc(filename)
```

---
## Step 25 — parse descriptions

```python
descriptions = load_descriptions(doc)
# 打印输出 / Print output
print('Loaded: %d ' % len(descriptions))
```

---
## Step 26 — clean descriptions

```python
clean_descriptions(descriptions)
```

---
## Step 27 — summarize vocabulary

```python
vocabulary = to_vocabulary(descriptions)
# 打印输出 / Print output
print('Vocabulary Size: %d' % len(vocabulary))
```

---
## Step 28 — save to file

```python
save_descriptions(descriptions, 'descriptions.txt')
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
# Data Prep / 2 Data Prep
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

# extract descriptions for images
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
  # 获取长度 / Get length
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
  # 添加元素到列表末尾 / Append element to list end
		mapping[image_id].append(image_desc)
	return mapping

def clean_descriptions(descriptions):
	# prepare regex for char filtering
	re_punc = re.compile('[%s]' % re.escape(string.punctuation))
 # 获取字典的键值对 / Get dict key-value pairs
	for _, desc_list in descriptions.items():
  # 获取长度 / Get length
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [re_punc.sub('', w) for w in desc]
			# remove hanging 's' and 'a'
   # 获取长度 / Get length
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
 # 获取字典的所有键 / Get all dict keys
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
 # 获取字典的键值对 / Get dict key-value pairs
	for key, desc_list in descriptions.items():
		for desc in desc_list:
   # 添加元素到列表末尾 / Append element to list end
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

filename = 'Flickr8k_text/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)
# parse descriptions
descriptions = load_descriptions(doc)
# 打印输出 / Print output
print('Loaded: %d ' % len(descriptions))
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
# 打印输出 / Print output
print('Vocabulary Size: %d' % len(vocabulary))
# save to file
save_descriptions(descriptions, 'descriptions.txt')
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Load Prepared Data

# 3 — Load Prepared Data / 数据准备

**Chapter 26 — File 3 of 7 / 第26章 — 第3个文件（共7个）**

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
from pickle import load
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
## Step 6 — load a pre-defined list of photo identifiers

```python
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
```

---
## Step 7 — process line by line

```python
for line in doc.split('\n'):
```

---
## Step 8 — skip empty lines

```python
# 获取长度 / Get length
if len(line) < 1:
			continue
```

---
## Step 9 — get the image identifier

```python
identifier = line.split('.')[0]
  # 添加元素到列表末尾 / Append element to list end
		dataset.append(identifier)
	return set(dataset)
```

---
## Step 10 — load clean descriptions into memory

```python
def load_clean_descriptions(filename, dataset):
```

---
## Step 11 — load document

```python
doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
```

---
## Step 12 — split line by white space

```python
tokens = line.split()
```

---
## Step 13 — split id from description

```python
image_id, image_desc = tokens[0], tokens[1:]
```

---
## Step 14 — skip images not in the set

```python
if image_id in dataset:
```

---
## Step 15 — create list

```python
if image_id not in descriptions:
				descriptions[image_id] = list()
```

---
## Step 16 — wrap description in tokens

```python
desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
```

---
## Step 17 — store

```python
# 添加元素到列表末尾 / Append element to list end
descriptions[image_id].append(desc)
	return descriptions
```

---
## Step 18 — load photo features

```python
def load_photo_features(filename, dataset):
```

---
## Step 19 — load all features

```python
all_features = load(open(filename, 'rb'))
```

---
## Step 20 — filter features

```python
features = {k: all_features[k] for k in dataset}
	return features
```

---
## Step 21 — load training dataset (6K)

```python
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
# 打印输出 / Print output
print('Dataset: %d' % len(train))
```

---
## Step 22 — descriptions

```python
train_descriptions = load_clean_descriptions('descriptions.txt', train)
# 打印输出 / Print output
print('Descriptions: train=%d' % len(train_descriptions))
```

---
## Step 23 — photo features

```python
train_features = load_photo_features('features.pkl', train)
# 打印输出 / Print output
print('Photos: train=%d' % len(train_features))
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
# Load Prepared Data / 数据准备
# Complete Code / 完整代码
# ===============================

from pickle import load

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
  # 获取长度 / Get length
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
  # 添加元素到列表末尾 / Append element to list end
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
   # 添加元素到列表末尾 / Append element to list end
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
# 打印输出 / Print output
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
# 打印输出 / Print output
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
# 打印输出 / Print output
print('Photos: train=%d' % len(train_features))
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Train Model



---

### Evaluate Model



---

### Save Tokenizer

# 6 — Save Tokenizer / 分词

**Chapter 26 — File 6 of 7 / 第26章 — 第6个文件（共7个）**

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
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import Tokenizer
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
## Step 6 — load a pre-defined list of photo identifiers

```python
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
```

---
## Step 7 — process line by line

```python
for line in doc.split('\n'):
```

---
## Step 8 — skip empty lines

```python
# 获取长度 / Get length
if len(line) < 1:
			continue
```

---
## Step 9 — get the image identifier

```python
identifier = line.split('.')[0]
  # 添加元素到列表末尾 / Append element to list end
		dataset.append(identifier)
	return set(dataset)
```

---
## Step 10 — load clean descriptions into memory

```python
def load_clean_descriptions(filename, dataset):
```

---
## Step 11 — load document

```python
doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
```

---
## Step 12 — split line by white space

```python
tokens = line.split()
```

---
## Step 13 — split id from description

```python
image_id, image_desc = tokens[0], tokens[1:]
```

---
## Step 14 — skip images not in the set

```python
if image_id in dataset:
```

---
## Step 15 — create list

```python
if image_id not in descriptions:
				descriptions[image_id] = list()
```

---
## Step 16 — wrap description in tokens

```python
desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
```

---
## Step 17 — store

```python
# 添加元素到列表末尾 / Append element to list end
descriptions[image_id].append(desc)
	return descriptions
```

---
## Step 18 — covert a dictionary of clean descriptions to a list of descriptions

```python
def to_lines(descriptions):
	all_desc = list()
 # 获取字典的所有键 / Get all dict keys
	for key in descriptions.keys():
  # 添加元素到列表末尾 / Append element to list end
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
```

---
## Step 19 — fit a tokenizer given caption descriptions

```python
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
```

---
## Step 20 — load training dataset

```python
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
# 打印输出 / Print output
print('Dataset: %d' % len(train))
```

---
## Step 21 — descriptions

```python
train_descriptions = load_clean_descriptions('descriptions.txt', train)
# 打印输出 / Print output
print('Descriptions: train=%d' % len(train_descriptions))
```

---
## Step 22 — prepare tokenizer

```python
tokenizer = create_tokenizer(train_descriptions)
```

---
## Step 23 — save the tokenizer

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
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Save Tokenizer / 分词
# Complete Code / 完整代码
# ===============================

# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.text import Tokenizer
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

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
  # 获取长度 / Get length
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
  # 添加元素到列表末尾 / Append element to list end
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
   # 添加元素到列表末尾 / Append element to list end
			descriptions[image_id].append(desc)
	return descriptions

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
 # 获取字典的所有键 / Get all dict keys
	for key in descriptions.keys():
  # 添加元素到列表末尾 / Append element to list end
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# load training dataset
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
# 打印输出 / Print output
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
# 打印输出 / Print output
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Generate Description

# 7 — Generate Description / 7 Generate Description

**Chapter 26 — File 7 of 7 / 第26章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **extract features from each photo in the directory**.

本脚本演示 **extract features from each photo in the directory**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
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
  📊 评估模型 / Evaluate Model
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
from pickle import load
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.applications.vgg16 import VGG16
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.applications.vgg16 import preprocess_input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
```

---
## Step 2 — extract features from each photo in the directory

```python
def extract_features(filename):
```

---
## Step 3 — load the model

```python
model = VGG16()
```

---
## Step 4 — re-structure the model

```python
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
```

---
## Step 5 — load the photo

```python
image = load_img(filename, target_size=(224, 224))
```

---
## Step 6 — convert the image pixels to a numpy array

```python
image = img_to_array(image)
```

---
## Step 7 — reshape data for the model

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
```

---
## Step 8 — prepare the image for the VGG model

```python
image = preprocess_input(image)
```

---
## Step 9 — get features

```python
# 用模型做预测 / Make predictions with model
feature = model.predict(image, verbose=0)
	return feature
```

---
## Step 10 — map an integer to a word

```python
def word_for_id(integer, tokenizer):
 # 获取字典的键值对 / Get dict key-value pairs
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
```

---
## Step 11 — remove start/end sequence tokens from a summary

```python
def cleanup_summary(summary):
```

---
## Step 12 — remove start of sequence token

```python
index = summary.find('startseq ')
	if index > -1:
  # 获取长度 / Get length
		summary = summary[len('startseq '):]
```

---
## Step 13 — remove end of sequence token

```python
index = summary.find(' endseq')
	if index > -1:
		summary = summary[:index]
	return summary
```

---
## Step 14 — generate a description for an image

```python
def generate_desc(model, tokenizer, photo, max_length):
```

---
## Step 15 — seed the generation process

```python
in_text = 'startseq'
```

---
## Step 16 — iterate over the whole length of the sequence

```python
# 生成整数序列 / Generate integer sequence
for _ in range(max_length):
```

---
## Step 17 — integer encode input sequence

```python
sequence = tokenizer.texts_to_sequences([in_text])[0]
```

---
## Step 18 — pad input

```python
sequence = pad_sequences([sequence], maxlen=max_length)
```

---
## Step 19 — predict next word

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict([photo,sequence], verbose=0)
```

---
## Step 20 — convert probability to integer

```python
yhat = argmax(yhat)
```

---
## Step 21 — map integer to word

```python
word = word_for_id(yhat, tokenizer)
```

---
## Step 22 — stop if we cannot map the word

```python
if word is None:
			break
```

---
## Step 23 — append as input for generating the next word

```python
in_text += ' ' + word
```

---
## Step 24 — stop if we predict the end of the sequence

```python
if word == 'endseq':
			break
	return in_text
```

---
## Step 25 — load the tokenizer

```python
tokenizer = load(open('tokenizer.pkl', 'rb'))
```

---
## Step 26 — pre-define the max sequence length (from training)

```python
max_length = 34
```

---
## Step 27 — load the model

```python
# 从文件加载模型 / Load model from file
model = load_model('model.h5')
```

---
## Step 28 — load and prepare the photograph

```python
photo = extract_features('example.jpg')
```

---
## Step 29 — generate description

```python
description = generate_desc(model, tokenizer, photo, max_length)
description = cleanup_summary(description)
# 打印输出 / Print output
print(description)
```

---
## Learning Notes / 学习笔记

- **概念**: extract features from each photo in the directory 是机器学习中的常用技术。  
  *extract features from each photo in the directory is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Generate Description / 7 Generate Description
# Complete Code / 完整代码
# ===============================

from pickle import load
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.sequence import pad_sequences
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.applications.vgg16 import VGG16
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.applications.vgg16 import preprocess_input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model

# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
 # 用模型做预测 / Make predictions with model
	feature = model.predict(image, verbose=0)
	return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
 # 获取字典的键值对 / Get dict key-value pairs
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# remove start/end sequence tokens from a summary
def cleanup_summary(summary):
	# remove start of sequence token
	index = summary.find('startseq ')
	if index > -1:
  # 获取长度 / Get length
		summary = summary[len('startseq '):]
	# remove end of sequence token
	index = summary.find(' endseq')
	if index > -1:
		summary = summary[:index]
	return summary

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
 # 生成整数序列 / Generate integer sequence
	for _ in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
  # 用模型做预测 / Make predictions with model
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
# 从文件加载模型 / Load model from file
model = load_model('model.h5')
# load and prepare the photograph
photo = extract_features('example.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
description = cleanup_summary(description)
# 打印输出 / Print output
print(description)
```

---

### Chapter Summary / 章节总结

# Chapter 26 Summary / 第26章总结

## Theme / 主题: Chapter 26 / Chapter 26

This chapter contains **7 code files** demonstrating chapter 26.

本章包含 **7 个代码文件**，演示Chapter 26。

---
## Evolution / 演化路线

  1. `1_extract_features.ipynb` — Extract Features
  2. `2_data_prep.ipynb` — Data Prep
  3. `3_load_prepared_data.ipynb` — Load Prepared Data
  4. `4_train_model.ipynb` — Train Model
  5. `5_evaluate_model.ipynb` — Evaluate Model
  6. `6_save_tokenizer.ipynb` — Save Tokenizer
  7. `7_generate_description.ipynb` — Generate Description

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 26) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 26）是机器学习流水线中的基础构建块。

---
