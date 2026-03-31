# NLP 深度学习 / Deep Learning for NLP
## Chapter 05

---

### Manual Load Data

# 01 — Manual Load Data / 01 Manual Load Data

**Chapter 05 — File 1 of 12 / 第05章 — 第1个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load text**.

本脚本演示 **load text**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — load text

```python
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
```

---
## Learning Notes / 学习笔记

- **概念**: load text 是机器学习中的常用技术。  
  *load text is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Manual Load Data / 01 Manual Load Data
# Complete Code / 完整代码
# ===============================

# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
```

---

➡️ **Next / 下一步**: File 2 of 12

---

### Manual Split

# 02 — Manual Split / 02 Manual Split

**Chapter 05 — File 2 of 12 / 第05章 — 第2个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load text**.

本脚本演示 **load text**。

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
## Step 1 — load text

```python
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
```

---
## Step 2 — split into words by white space

```python
words = text.split()
# 打印输出 / Print output
print(words[:100])
```

---
## Learning Notes / 学习笔记

- **概念**: load text 是机器学习中的常用技术。  
  *load text is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Manual Split / 02 Manual Split
# Complete Code / 完整代码
# ===============================

# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()
# 打印输出 / Print output
print(words[:100])
```

---

➡️ **Next / 下一步**: File 3 of 12

---

### Manual Select Words

# 03 — Manual Select Words / 特征选择

**Chapter 05 — File 3 of 12 / 第05章 — 第3个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load text**.

本脚本演示 **load text**。

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
## Step 1 — Step 1

```python
# 导入正则表达式模块 / Import regex module
import re
```

---
## Step 2 — load text

```python
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
```

---
## Step 3 — split based on words only

```python
words = re.split(r'\W+', text)
# 打印输出 / Print output
print(words[:100])
```

---
## Learning Notes / 学习笔记

- **概念**: load text 是机器学习中的常用技术。  
  *load text is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Manual Select Words / 特征选择
# Complete Code / 完整代码
# ===============================

# 导入正则表达式模块 / Import regex module
import re
# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split based on words only
words = re.split(r'\W+', text)
# 打印输出 / Print output
print(words[:100])
```

---

➡️ **Next / 下一步**: File 4 of 12

---

### Manual Remove Punctuation

# 04 — Manual Remove Punctuation / 04 Manual Remove Punctuation

**Chapter 05 — File 4 of 12 / 第05章 — 第4个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load text**.

本脚本演示 **load text**。

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
## Step 2 — load text

```python
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
```

---
## Step 3 — split into words by white space

```python
words = text.split()
```

---
## Step 4 — prepare regex for char filtering

```python
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
```

---
## Step 5 — remove punctuation from each word

```python
stripped = [re_punc.sub('', w) for w in words]
# 打印输出 / Print output
print(stripped[:100])
```

---
## Learning Notes / 学习笔记

- **概念**: load text 是机器学习中的常用技术。  
  *load text is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Manual Remove Punctuation / 04 Manual Remove Punctuation
# Complete Code / 完整代码
# ===============================

import string
# 导入正则表达式模块 / Import regex module
import re
# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()
# prepare regex for char filtering
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
# remove punctuation from each word
stripped = [re_punc.sub('', w) for w in words]
# 打印输出 / Print output
print(stripped[:100])
```

---

➡️ **Next / 下一步**: File 5 of 12

---

### Manual Normalize Case

# 05 — Manual Normalize Case / 05 Manual Normalize Case

**Chapter 05 — File 5 of 12 / 第05章 — 第5个文件（共12个）**

---

## Summary / 总结

This script demonstrates **split into words by white space**.

本脚本演示 **split into words by white space**。

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
## Step 1 — Step 1

```python
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
```

---
## Step 2 — split into words by white space

```python
words = text.split()
```

---
## Step 3 — convert to lower case

```python
words = [word.lower() for word in words]
# 打印输出 / Print output
print(words[:100])
```

---
## Learning Notes / 学习笔记

- **概念**: split into words by white space 是机器学习中的常用技术。  
  *split into words by white space is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Manual Normalize Case / 05 Manual Normalize Case
# Complete Code / 完整代码
# ===============================

filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()
# convert to lower case
words = [word.lower() for word in words]
# 打印输出 / Print output
print(words[:100])
```

---

➡️ **Next / 下一步**: File 6 of 12

---

### Nltk Download

# 06 — Nltk Download / 06 Nltk Download

**Chapter 05 — File 6 of 12 / 第05章 — 第6个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Nltk Download**.

本脚本演示 **06 Nltk Download**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import nltk
nltk.download()
```

---
## Learning Notes / 学习笔记

- **概念**: Nltk Download 是机器学习中的常用技术。  
  *Nltk Download is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nltk Download / 06 Nltk Download
# Complete Code / 完整代码
# ===============================

import nltk
nltk.download()
```

---

➡️ **Next / 下一步**: File 7 of 12

---

### Nltk Split Sentences

# 07 — Nltk Split Sentences / 07 Nltk Split Sentences

**Chapter 05 — File 7 of 12 / 第05章 — 第7个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load data**.

本脚本演示 **load data**。

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
## Step 1 — Step 1

```python
from nltk import sent_tokenize
```

---
## Step 2 — load data

```python
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
```

---
## Step 3 — split into sentences

```python
sentences = sent_tokenize(text)
# 打印输出 / Print output
print(sentences[0])
```

---
## Learning Notes / 学习笔记

- **概念**: load data 是机器学习中的常用技术。  
  *load data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nltk Split Sentences / 07 Nltk Split Sentences
# Complete Code / 完整代码
# ===============================

from nltk import sent_tokenize
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into sentences
sentences = sent_tokenize(text)
# 打印输出 / Print output
print(sentences[0])
```

---

➡️ **Next / 下一步**: File 8 of 12

---

### Nltk Split Words

# 08 — Nltk Split Words / 08 Nltk Split Words

**Chapter 05 — File 8 of 12 / 第05章 — 第8个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load data**.

本脚本演示 **load data**。

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
## Step 1 — Step 1

```python
from nltk.tokenize import word_tokenize
```

---
## Step 2 — load data

```python
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
```

---
## Step 3 — split into words

```python
tokens = word_tokenize(text)
# 打印输出 / Print output
print(tokens[:100])
```

---
## Learning Notes / 学习笔记

- **概念**: load data 是机器学习中的常用技术。  
  *load data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nltk Split Words / 08 Nltk Split Words
# Complete Code / 完整代码
# ===============================

from nltk.tokenize import word_tokenize
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
tokens = word_tokenize(text)
# 打印输出 / Print output
print(tokens[:100])
```

---

➡️ **Next / 下一步**: File 9 of 12

---

### Nltk Remove Punctuation

# 09 — Nltk Remove Punctuation / 09 Nltk Remove Punctuation

**Chapter 05 — File 9 of 12 / 第05章 — 第9个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load data**.

本脚本演示 **load data**。

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
## Step 1 — Step 1

```python
from nltk.tokenize import word_tokenize
```

---
## Step 2 — load data

```python
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
```

---
## Step 3 — split into words

```python
tokens = word_tokenize(text)
```

---
## Step 4 — remove all tokens that are not alphabetic

```python
words = [word for word in tokens if word.isalpha()]
# 打印输出 / Print output
print(words[:100])
```

---
## Learning Notes / 学习笔记

- **概念**: load data 是机器学习中的常用技术。  
  *load data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nltk Remove Punctuation / 09 Nltk Remove Punctuation
# Complete Code / 完整代码
# ===============================

from nltk.tokenize import word_tokenize
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
tokens = word_tokenize(text)
# remove all tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]
# 打印输出 / Print output
print(words[:100])
```

---

➡️ **Next / 下一步**: File 10 of 12

---

### Nltk Stop Words

# 10 — Nltk Stop Words / 10 Nltk Stop Words

**Chapter 05 — File 10 of 12 / 第05章 — 第10个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Nltk Stop Words**.

本脚本演示 **10 Nltk Stop Words**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# 打印输出 / Print output
print(stop_words)
```

---
## Learning Notes / 学习笔记

- **概念**: Nltk Stop Words 是机器学习中的常用技术。  
  *Nltk Stop Words is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nltk Stop Words / 10 Nltk Stop Words
# Complete Code / 完整代码
# ===============================

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# 打印输出 / Print output
print(stop_words)
```

---

➡️ **Next / 下一步**: File 11 of 12

---

### Nltk Filter Stop Words

# 11 — Nltk Filter Stop Words / 11 Nltk Filter Stop Words

**Chapter 05 — File 11 of 12 / 第05章 — 第11个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load data**.

本脚本演示 **load data**。

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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
```

---
## Step 2 — load data

```python
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
```

---
## Step 3 — split into words

```python
tokens = word_tokenize(text)
```

---
## Step 4 — convert to lower case

```python
tokens = [w.lower() for w in tokens]
```

---
## Step 5 — prepare regex for char filtering

```python
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
```

---
## Step 6 — remove punctuation from each word

```python
stripped = [re_punc.sub('', w) for w in tokens]
```

---
## Step 7 — remove remaining tokens that are not alphabetic

```python
words = [word for word in stripped if word.isalpha()]
```

---
## Step 8 — filter out stop words

```python
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
# 打印输出 / Print output
print(words[:100])
```

---
## Learning Notes / 学习笔记

- **概念**: load data 是机器学习中的常用技术。  
  *load data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nltk Filter Stop Words / 11 Nltk Filter Stop Words
# Complete Code / 完整代码
# ===============================

import string
# 导入正则表达式模块 / Import regex module
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
tokens = word_tokenize(text)
# convert to lower case
tokens = [w.lower() for w in tokens]
# prepare regex for char filtering
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
# remove punctuation from each word
stripped = [re_punc.sub('', w) for w in tokens]
# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]
# filter out stop words
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
# 打印输出 / Print output
print(words[:100])
```

---

➡️ **Next / 下一步**: File 12 of 12

---

### Nltk Stemming

# 12 — Nltk Stemming / 12 Nltk Stemming

**Chapter 05 — File 12 of 12 / 第05章 — 第12个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load data**.

本脚本演示 **load data**。

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
## Step 1 — Step 1

```python
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
```

---
## Step 2 — load data

```python
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
```

---
## Step 3 — split into words

```python
tokens = word_tokenize(text)
```

---
## Step 4 — stemming of words

```python
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
# 打印输出 / Print output
print(stemmed[:100])
```

---
## Learning Notes / 学习笔记

- **概念**: load data 是机器学习中的常用技术。  
  *load data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nltk Stemming / 12 Nltk Stemming
# Complete Code / 完整代码
# ===============================

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
tokens = word_tokenize(text)
# stemming of words
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
# 打印输出 / Print output
print(stemmed[:100])
```

---

### Chapter Summary / 章节总结

# Chapter 05 Summary / 第05章总结

## Theme / 主题: Chapter 05 / Chapter 05

This chapter contains **12 code files** demonstrating chapter 05.

本章包含 **12 个代码文件**，演示Chapter 05。

---
## Evolution / 演化路线

  1. `01_manual_load_data.ipynb` — Manual Load Data
  2. `02_manual_split.ipynb` — Manual Split
  3. `03_manual_select_words.ipynb` — Manual Select Words
  4. `04_manual_remove_punctuation.ipynb` — Manual Remove Punctuation
  5. `05_manual_normalize_case.ipynb` — Manual Normalize Case
  6. `06_nltk_download.ipynb` — Nltk Download
  7. `07_nltk_split_sentences.ipynb` — Nltk Split Sentences
  8. `08_nltk_split_words.ipynb` — Nltk Split Words
  9. `09_nltk_remove_punctuation.ipynb` — Nltk Remove Punctuation
  10. `10_nltk_stop_words.ipynb` — Nltk Stop Words
  11. `11_nltk_filter_stop_words.ipynb` — Nltk Filter Stop Words
  12. `12_nltk_stemming.ipynb` — Nltk Stemming

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 05) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 05）是机器学习流水线中的基础构建块。

---
