# NLP深度学习
## Chapter 25

---

### Load Photos

# 01 — Load Photos / 01 Load Photos

**Chapter 25 — File 1 of 7 / 第25章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **load an image from file**.

本脚本演示 **load an image from file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from os import listdir
from os import path
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

def load_photos(directory):
	images = dict()
	for name in listdir(directory):
```

---
## Step 2 — load an image from file

```python
filename = path.join(directory, name)
		image = load_img(filename, target_size=(224, 224))
```

---
## Step 3 — convert the image pixels to a numpy array

```python
image = img_to_array(image)
```

---
## Step 4 — reshape data for the model

```python
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
```

---
## Step 5 — prepare the image for the VGG model

```python
image = preprocess_input(image)
```

---
## Step 6 — get image id

```python
image_id = name.split('.')[0]
		images[image_id] = image
	return images
```

---
## Step 7 — load images

```python
directory = 'Flicker8k_Dataset'
images = load_photos(directory)
print('Loaded Images: %d' % len(images))
```

---
## Learning Notes / 学习笔记

- **概念**: load an image from file 是机器学习中的常用技术。  
  *load an image from file is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Photos / 01 Load Photos
# Complete Code / 完整代码
# ===============================

from os import listdir
from os import path
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

def load_photos(directory):
	images = dict()
	for name in listdir(directory):
		# load an image from file
		filename = path.join(directory, name)
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get image id
		image_id = name.split('.')[0]
		images[image_id] = image
	return images

# load images
directory = 'Flicker8k_Dataset'
images = load_photos(directory)
print('Loaded Images: %d' % len(images))
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Load Descriptions

# 03 — Load Descriptions / 03 Load Descriptions

**Chapter 25 — File 3 of 7 / 第25章 — 第3个文件（共7个）**

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
## Step 5 — extract descriptions for images

```python
def load_descriptions(doc):
	mapping = dict()
```

---
## Step 6 — process lines

```python
for line in doc.split('\n'):
```

---
## Step 7 — split line by white space

```python
tokens = line.split()
		if len(line) < 2:
			continue
```

---
## Step 8 — take the first token as the image id, the rest as the description

```python
image_id, image_desc = tokens[0], tokens[1:]
```

---
## Step 9 — remove filename from image id

```python
image_id = image_id.split('.')[0]
```

---
## Step 10 — convert description tokens back to string

```python
image_desc = ' '.join(image_desc)
```

---
## Step 11 — store the first description for each image

```python
if image_id not in mapping:
			mapping[image_id] = image_desc
	return mapping

filename = 'Flickr8k_text/Flickr8k.token.txt'
doc = load_doc(filename)
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
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
# Load Descriptions / 03 Load Descriptions
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

# extract descriptions for images
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# store the first description for each image
		if image_id not in mapping:
			mapping[image_id] = image_desc
	return mapping

filename = 'Flickr8k_text/Flickr8k.token.txt'
doc = load_doc(filename)
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Whole Description Model

# 05 — Whole Description Model / 05 Whole Description Model

**Chapter 25 — File 5 of 7 / 第25章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **load doc into memory**.

本脚本演示 **load doc into memory**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
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
## Step 6 — load clean descriptions into memory

```python
def load_clean_descriptions(filename):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
```

---
## Step 7 — split line by white space

```python
tokens = line.split()
```

---
## Step 8 — split id from description

```python
image_id, image_desc = tokens[0], tokens[1:]
```

---
## Step 9 — store

```python
descriptions[image_id] = ' '.join(image_desc)
	return descriptions

descriptions = load_clean_descriptions('descriptions.txt')
print('Loaded %d' % (len(descriptions)))
```

---
## Step 10 — extract all text

```python
desc_text = list(descriptions.values())
```

---
## Step 11 — prepare tokenizer

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_text)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
```

---
## Step 12 — integer encode descriptions

```python
sequences = tokenizer.texts_to_sequences(desc_text)
```

---
## Step 13 — pad all sequences to a fixed length

```python
max_length = max(len(s) for s in sequences)
print('Description Length: %d' % max_length)
padded = pad_sequences(sequences, maxlen=max_length, padding='post')
```

---
## Step 14 — one hot encode

```python
y = to_categorical(padded, num_classes=vocab_size)
y = y.reshape((len(descriptions), max_length, vocab_size))
print(y.shape)
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
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Whole Description Model / 05 Whole Description Model
# Complete Code / 完整代码
# ===============================

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load clean descriptions into memory
def load_clean_descriptions(filename):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# store
		descriptions[image_id] = ' '.join(image_desc)
	return descriptions

descriptions = load_clean_descriptions('descriptions.txt')
print('Loaded %d' % (len(descriptions)))
# extract all text
desc_text = list(descriptions.values())
# prepare tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_text)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# integer encode descriptions
sequences = tokenizer.texts_to_sequences(desc_text)
# pad all sequences to a fixed length
max_length = max(len(s) for s in sequences)
print('Description Length: %d' % max_length)
padded = pad_sequences(sequences, maxlen=max_length, padding='post')
# one hot encode
y = to_categorical(padded, num_classes=vocab_size)
y = y.reshape((len(descriptions), max_length, vocab_size))
print(y.shape)
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Word By Word

# 06 — Word By Word / 06 Word By Word

**Chapter 25 — File 6 of 7 / 第25章 — 第6个文件（共7个）**

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
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
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
## Step 6 — load clean descriptions into memory

```python
def load_clean_descriptions(filename):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
```

---
## Step 7 — split line by white space

```python
tokens = line.split()
```

---
## Step 8 — split id from description

```python
image_id, image_desc = tokens[0], tokens[1:]
```

---
## Step 9 — store

```python
descriptions[image_id] = ' '.join(image_desc)
	return descriptions

descriptions = load_clean_descriptions('descriptions.txt')
print('Loaded %d' % (len(descriptions)))
```

---
## Step 10 — extract all text

```python
desc_text = list(descriptions.values())
```

---
## Step 11 — prepare tokenizer

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_text)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
```

---
## Step 12 — integer encode descriptions

```python
sequences = tokenizer.texts_to_sequences(desc_text)
```

---
## Step 13 — determine the maximum sequence length

```python
max_length = max(len(s) for s in sequences)
print('Description Length: %d' % max_length)

X, y = list(), list()
for img_no, seq in enumerate(sequences):
```

---
## Step 14 — split one sequence into multiple X,y pairs

```python
for i in range(1, len(seq)):
```

---
## Step 15 — split into input and output pair

```python
in_seq, out_seq = seq[:i], seq[i]
```

---
## Step 16 — pad input sequence

```python
in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
```

---
## Step 17 — encode output sequence

```python
out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
```

---
## Step 18 — store

```python
X.append(in_seq)
		y.append(out_seq)
```

---
## Step 19 — convert to numpy arrays

```python
X, y = array(X), array(y)
print(X.shape)
print(y.shape)
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
| `numpy` | 数值计算库 | Numerical computing library |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Word By Word / 06 Word By Word
# Complete Code / 完整代码
# ===============================

from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load clean descriptions into memory
def load_clean_descriptions(filename):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# store
		descriptions[image_id] = ' '.join(image_desc)
	return descriptions

descriptions = load_clean_descriptions('descriptions.txt')
print('Loaded %d' % (len(descriptions)))
# extract all text
desc_text = list(descriptions.values())
# prepare tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_text)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# integer encode descriptions
sequences = tokenizer.texts_to_sequences(desc_text)
# determine the maximum sequence length
max_length = max(len(s) for s in sequences)
print('Description Length: %d' % max_length)

X, y = list(), list()
for img_no, seq in enumerate(sequences):
	# split one sequence into multiple X,y pairs
	for i in range(1, len(seq)):
		# split into input and output pair
		in_seq, out_seq = seq[:i], seq[i]
		# pad input sequence
		in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
		# encode output sequence
		out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
		# store
		X.append(in_seq)
		y.append(out_seq)

# convert to numpy arrays
X, y = array(X), array(y)
print(X.shape)
print(y.shape)
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Progressive Loading

# 07 — Progressive Loading / 07 Progressive Loading

**Chapter 25 — File 7 of 7 / 第25章 — 第7个文件（共7个）**

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
from os import path
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
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
## Step 6 — load clean descriptions into memory

```python
def load_clean_descriptions(filename):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
```

---
## Step 7 — split line by white space

```python
tokens = line.split()
```

---
## Step 8 — split id from description

```python
image_id, image_desc = tokens[0], tokens[1:]
```

---
## Step 9 — store

```python
descriptions[image_id] = ' '.join(image_desc)
	return descriptions
```

---
## Step 10 — fit a tokenizer given caption descriptions

```python
def create_tokenizer(descriptions):
	lines = list(descriptions.values())
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
```

---
## Step 11 — load a single photo intended as input for the VGG feature extractor model

```python
def load_photo(filename):
	image = load_img(filename, target_size=(224, 224))
```

---
## Step 12 — convert the image pixels to a numpy array

```python
image = img_to_array(image)
```

---
## Step 13 — reshape data for the model

```python
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
```

---
## Step 14 — prepare the image for the VGG model

```python
image = preprocess_input(image)[0]
```

---
## Step 15 — get image id

```python
image_id = path.basename(filename).split('.')[0]
	return image, image_id
```

---
## Step 16 — create sequences of images, input sequences and output words for an image

```python
def create_sequences(tokenizer, max_length, desc, image):
	Ximages, XSeq, y = list(), list(),list()
	vocab_size = len(tokenizer.word_index) + 1
```

---
## Step 17 — integer encode the description

```python
seq = tokenizer.texts_to_sequences([desc])[0]
```

---
## Step 18 — split one sequence into multiple X,y pairs

```python
for i in range(1, len(seq)):
```

---
## Step 19 — select

```python
in_seq, out_seq = seq[:i], seq[i]
```

---
## Step 20 — pad input sequence

```python
in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
```

---
## Step 21 — encode output sequence

```python
out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
```

---
## Step 22 — store

```python
Ximages.append(image)
		XSeq.append(in_seq)
		y.append(out_seq)
	Ximages, XSeq, y = array(Ximages), array(XSeq), array(y)
	return [Ximages, XSeq, y]
```

---
## Step 23 — data generator, intended to be used in a call to model.fit_generator()

```python
def data_generator(descriptions, tokenizer, max_length):
```

---
## Step 24 — loop for ever over images

```python
directory = 'Flicker8k_Dataset'
	while 1:
		for name in listdir(directory):
```

---
## Step 25 — load an image from file

```python
filename = path.join(directory, name)
			image, image_id = load_photo(filename)
```

---
## Step 26 — create word sequences

```python
desc = descriptions[image_id]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc, image)
			yield [[in_img, in_seq], out_word]
```

---
## Step 27 — load mapping of ids to descriptions

```python
descriptions = load_clean_descriptions('descriptions.txt')
```

---
## Step 28 — integer encode sequences of words

```python
tokenizer = create_tokenizer(descriptions)
```

---
## Step 29 — pad to fixed length

```python
max_length = max(len(s.split()) for s in list(descriptions.values()))
print('Description Length: %d' % max_length)
```

---
## Step 30 — test the data generator

```python
generator = data_generator(descriptions, tokenizer, max_length)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)
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
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Progressive Loading / 07 Progressive Loading
# Complete Code / 完整代码
# ===============================

from os import listdir
from os import path
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load clean descriptions into memory
def load_clean_descriptions(filename):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# store
		descriptions[image_id] = ' '.join(image_desc)
	return descriptions

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = list(descriptions.values())
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# load a single photo intended as input for the VGG feature extractor model
def load_photo(filename):
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)[0]
	# get image id
	image_id = path.basename(filename).split('.')[0]
	return image, image_id

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc, image):
	Ximages, XSeq, y = list(), list(),list()
	vocab_size = len(tokenizer.word_index) + 1
	# integer encode the description
	seq = tokenizer.texts_to_sequences([desc])[0]
	# split one sequence into multiple X,y pairs
	for i in range(1, len(seq)):
		# select
		in_seq, out_seq = seq[:i], seq[i]
		# pad input sequence
		in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
		# encode output sequence
		out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
		# store
		Ximages.append(image)
		XSeq.append(in_seq)
		y.append(out_seq)
	Ximages, XSeq, y = array(Ximages), array(XSeq), array(y)
	return [Ximages, XSeq, y]

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, tokenizer, max_length):
	# loop for ever over images
	directory = 'Flicker8k_Dataset'
	while 1:
		for name in listdir(directory):
			# load an image from file
			filename = path.join(directory, name)
			image, image_id = load_photo(filename)
			# create word sequences
			desc = descriptions[image_id]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc, image)
			yield [[in_img, in_seq], out_word]

# load mapping of ids to descriptions
descriptions = load_clean_descriptions('descriptions.txt')
# integer encode sequences of words
tokenizer = create_tokenizer(descriptions)
# pad to fixed length
max_length = max(len(s.split()) for s in list(descriptions.values()))
print('Description Length: %d' % max_length)
# test the data generator
generator = data_generator(descriptions, tokenizer, max_length)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)
```

---

### Chapter Summary

# Chapter 25 Summary / 第25章总结

## Theme / 主题: Chapter 25 / Chapter 25

This chapter contains **7 code files** demonstrating chapter 25.

本章包含 **7 个代码文件**，演示Chapter 25。

---
## Evolution / 演化路线

  1. `01_load_photos.ipynb` — Load Photos
  2. `02_pre_calculate_features.ipynb` — Pre Calculate Features
  3. `03_load_descriptions.ipynb` — Load Descriptions
  4. `04_clean_descriptions.ipynb` — Clean Descriptions
  5. `05_whole_description_model.ipynb` — Whole Description Model
  6. `06_word_by_word.ipynb` — Word By Word
  7. `07_progressive_loading.ipynb` — Progressive Loading

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 25) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 25）是机器学习流水线中的基础构建块。

---
