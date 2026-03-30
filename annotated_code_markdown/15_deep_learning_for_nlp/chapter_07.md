# NLP深度学习
## Chapter 07

---

### Split Words

# 1 — Split Words / 1 Split Words

**Chapter 07 — File 1 of 5 / 第07章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **define the document**.

本脚本演示 **define the document**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from keras.preprocessing.text import text_to_word_sequence
```

---
## Step 2 — define the document

```python
text = 'The quick brown fox jumped over the lazy dog.'
```

---
## Step 3 — tokenize the document

```python
result = text_to_word_sequence(text)
print(result)
```

---
## Learning Notes / 学习笔记

- **概念**: define the document 是机器学习中的常用技术。  
  *define the document is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Split Words / 1 Split Words
# Complete Code / 完整代码
# ===============================

from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# tokenize the document
result = text_to_word_sequence(text)
print(result)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Prepare Vocab

# 2 — Prepare Vocab / 数据准备

**Chapter 07 — File 2 of 5 / 第07章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **define the document**.

本脚本演示 **define the document**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from keras.preprocessing.text import text_to_word_sequence
```

---
## Step 2 — define the document

```python
text = 'The quick brown fox jumped over the lazy dog.'
```

---
## Step 3 — estimate the size of the vocabulary

```python
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
```

---
## Learning Notes / 学习笔记

- **概念**: define the document 是机器学习中的常用技术。  
  *define the document is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Prepare Vocab / 数据准备
# Complete Code / 完整代码
# ===============================

from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# estimate the size of the vocabulary
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### One Hot Encode

# 3 — One Hot Encode / 数据编码

**Chapter 07 — File 3 of 5 / 第07章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **define the document**.

本脚本演示 **define the document**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
```

---
## Step 2 — define the document

```python
text = 'The quick brown fox jumped over the lazy dog.'
```

---
## Step 3 — estimate the size of the vocabulary

```python
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
```

---
## Step 4 — integer encode the document

```python
result = one_hot(text, round(vocab_size*1.3))
print(result)
```

---
## Learning Notes / 学习笔记

- **概念**: define the document 是机器学习中的常用技术。  
  *define the document is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# One Hot Encode / 数据编码
# Complete Code / 完整代码
# ===============================

from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# estimate the size of the vocabulary
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
# integer encode the document
result = one_hot(text, round(vocab_size*1.3))
print(result)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Hash Encode

# 4 — Hash Encode / 数据编码

**Chapter 07 — File 4 of 5 / 第07章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **define the document**.

本脚本演示 **define the document**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
```

---
## Step 2 — define the document

```python
text = 'The quick brown fox jumped over the lazy dog.'
```

---
## Step 3 — estimate the size of the vocabulary

```python
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
```

---
## Step 4 — integer encode the document

```python
result = hashing_trick(text, round(vocab_size*1.3), hash_function='md5')
print(result)
```

---
## Learning Notes / 学习笔记

- **概念**: define the document 是机器学习中的常用技术。  
  *define the document is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hash Encode / 数据编码
# Complete Code / 完整代码
# ===============================

from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# estimate the size of the vocabulary
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
# integer encode the document
result = hashing_trick(text, round(vocab_size*1.3), hash_function='md5')
print(result)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Chapter Summary

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **5 code files** demonstrating chapter 07.

本章包含 **5 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `1_split_words.ipynb` — Split Words
  2. `2_prepare_vocab.ipynb` — Prepare Vocab
  3. `3_one_hot_encode.ipynb` — One Hot Encode
  4. `4_hash_encode.ipynb` — Hash Encode
  5. `5_example_tokenizer.ipynb` — Example Tokenizer

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---
