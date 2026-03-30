# NLP深度学习
## Chapter 06

---

### Example Countvector

# 1 — Example Countvector / 1 Example Countvector

**Chapter 06 — File 1 of 3 / 第06章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **list of text documents**.

本脚本演示 **list of text documents**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 训练模型 / Train the model


---
## Step 1 — Step 1

```python
from sklearn.feature_extraction.text import CountVectorizer
```

---
## Step 2 — list of text documents

```python
text = ["The quick brown fox jumped over the lazy dog."]
```

---
## Step 3 — create the transform

```python
vectorizer = CountVectorizer()
```

---
## Step 4 — tokenize and build vocab

```python
vectorizer.fit(text)
```

---
## Step 5 — summarize

```python
print(vectorizer.vocabulary_)
```

---
## Step 6 — encode document

```python
vector = vectorizer.transform(text)
```

---
## Step 7 — summarize encoded vector

```python
print(vector.shape)
print(type(vector))
print(vector.toarray())
```

---
## Learning Notes / 学习笔记

- **概念**: list of text documents 是机器学习中的常用技术。  
  *list of text documents is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example Countvector / 1 Example Countvector
# Complete Code / 完整代码
# ===============================

from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Example Tfidf

# 2 — Example Tfidf / 2 Example Tfidf

**Chapter 06 — File 2 of 3 / 第06章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **list of text documents**.

本脚本演示 **list of text documents**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 训练模型 / Train the model


---
## Step 1 — Step 1

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

---
## Step 2 — list of text documents

```python
text = ["The quick brown fox jumped over the lazy dog.",
		"The dog.",
		"The fox"]
```

---
## Step 3 — create the transform

```python
vectorizer = TfidfVectorizer()
```

---
## Step 4 — tokenize and build vocab

```python
vectorizer.fit(text)
```

---
## Step 5 — summarize

```python
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
```

---
## Step 6 — encode document

```python
vector = vectorizer.transform([text[0]])
```

---
## Step 7 — summarize encoded vector

```python
print(vector.shape)
print(vector.toarray())
```

---
## Learning Notes / 学习笔记

- **概念**: list of text documents 是机器学习中的常用技术。  
  *list of text documents is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example Tfidf / 2 Example Tfidf
# Complete Code / 完整代码
# ===============================

from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog.",
		"The dog.",
		"The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Example Hash

# 3 — Example Hash / 3 Example Hash

**Chapter 06 — File 3 of 3 / 第06章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **list of text documents**.

本脚本演示 **list of text documents**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
from sklearn.feature_extraction.text import HashingVectorizer
```

---
## Step 2 — list of text documents

```python
text = ["The quick brown fox jumped over the lazy dog."]
```

---
## Step 3 — create the transform

```python
vectorizer = HashingVectorizer(n_features=20)
```

---
## Step 4 — encode document

```python
vector = vectorizer.transform(text)
```

---
## Step 5 — summarize encoded vector

```python
print(vector.shape)
print(vector.toarray())
```

---
## Learning Notes / 学习笔记

- **概念**: list of text documents 是机器学习中的常用技术。  
  *list of text documents is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example Hash / 3 Example Hash
# Complete Code / 完整代码
# ===============================

from sklearn.feature_extraction.text import HashingVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = HashingVectorizer(n_features=20)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())
```

---

### Chapter Summary

# Chapter 06 Summary / 第06章总结

## Theme / 主题: Chapter 06 / Chapter 06

This chapter contains **3 code files** demonstrating chapter 06.

本章包含 **3 个代码文件**，演示Chapter 06。

---
## Evolution / 演化路线

  1. `1_example_countvector.ipynb` — Example Countvector
  2. `2_example_tfidf.ipynb` — Example Tfidf
  3. `3_example_hash.ipynb` — Example Hash

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 06) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 06）是机器学习流水线中的基础构建块。

---
