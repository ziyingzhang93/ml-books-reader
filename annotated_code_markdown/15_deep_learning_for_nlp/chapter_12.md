# NLP 深度学习 / Deep Learning for NLP
## Chapter 12

---

### Example Word2Vec



---

### Plot Model



---

### Word Vector Arithmetic



---

### Example Glove

# 4 — Example Glove / 4 Example Glove

**Chapter 12 — File 4 of 4 / 第12章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **convert glove to word2vec format**.

本脚本演示 **convert glove to word2vec format**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
```

---
## Step 2 — convert glove to word2vec format

```python
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
```

---
## Step 3 — load the converted model

```python
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
```

---
## Step 4 — calculate: (king - man) + woman = ?

```python
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# 打印输出 / Print output
print(result)
```

---
## Learning Notes / 学习笔记

- **概念**: convert glove to word2vec format 是机器学习中的常用技术。  
  *convert glove to word2vec format is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example Glove / 4 Example Glove
# Complete Code / 完整代码
# ===============================

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# convert glove to word2vec format
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

# load the converted model
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# 打印输出 / Print output
print(result)
```

---

### Chapter Summary / 章节总结



---
