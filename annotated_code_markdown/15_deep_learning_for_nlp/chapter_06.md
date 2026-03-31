# NLP 深度学习 / Deep Learning for NLP
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
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏋️ 训练模型 / Train Model
```

---
## Step 1 — Step 1

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
# 打印输出 / Print output
print(vectorizer.vocabulary_)
```

---
## Step 6 — encode document

```python
# 用已拟合的模型转换数据 / Transform data with fitted model
vector = vectorizer.transform(text)
```

---
## Step 7 — summarize encoded vector

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(vector.shape)
# 打印输出 / Print output
print(type(vector))
# 打印输出 / Print output
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

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
# 打印输出 / Print output
print(vectorizer.vocabulary_)
# encode document
# 用已拟合的模型转换数据 / Transform data with fitted model
vector = vectorizer.transform(text)
# summarize encoded vector
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(vector.shape)
# 打印输出 / Print output
print(type(vector))
# 打印输出 / Print output
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
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏋️ 训练模型 / Train Model
```

---
## Step 1 — Step 1

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
# 打印输出 / Print output
print(vectorizer.vocabulary_)
# 打印输出 / Print output
print(vectorizer.idf_)
```

---
## Step 6 — encode document

```python
# 用已拟合的模型转换数据 / Transform data with fitted model
vector = vectorizer.transform([text[0]])
```

---
## Step 7 — summarize encoded vector

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(vector.shape)
# 打印输出 / Print output
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

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
# 打印输出 / Print output
print(vectorizer.vocabulary_)
# 打印输出 / Print output
print(vectorizer.idf_)
# encode document
# 用已拟合的模型转换数据 / Transform data with fitted model
vector = vectorizer.transform([text[0]])
# summarize encoded vector
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(vector.shape)
# 打印输出 / Print output
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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
# 用已拟合的模型转换数据 / Transform data with fitted model
vector = vectorizer.transform(text)
```

---
## Step 5 — summarize encoded vector

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(vector.shape)
# 打印输出 / Print output
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

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_extraction.text import HashingVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = HashingVectorizer(n_features=20)
# encode document
# 用已拟合的模型转换数据 / Transform data with fitted model
vector = vectorizer.transform(text)
# summarize encoded vector
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(vector.shape)
# 打印输出 / Print output
print(vector.toarray())
```

---

### Chapter Summary / 章节总结

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
