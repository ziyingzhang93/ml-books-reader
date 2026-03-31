# NLP 深度学习 / Deep Learning for NLP
## Chapter 24

---

### Sentence Bleu

# 01 — Sentence Bleu / 01 Sentence Bleu

**Chapter 24 — File 1 of 13 / 第24章 — 第1个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Sentence Bleu**.

本脚本演示 **01 Sentence Bleu**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — Step 1

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: Sentence Bleu 是机器学习中的常用技术。  
  *Sentence Bleu is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sentence Bleu / 01 Sentence Bleu
# Complete Code / 完整代码
# ===============================

from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---

➡️ **Next / 下一步**: File 2 of 13

---

### Corpus Bleu

# 02 — Corpus Bleu / 02 Corpus Bleu

**Chapter 24 — File 2 of 13 / 第24章 — 第2个文件（共13个）**

---

## Summary / 总结

This script demonstrates **two references for one document**.

本脚本演示 **two references for one document**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — two references for one document

```python
from nltk.translate.bleu_score import corpus_bleu
references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
candidates = [['this', 'is', 'a', 'test']]
score = corpus_bleu(references, candidates)
# 打印输出 / Print output
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: two references for one document 是机器学习中的常用技术。  
  *two references for one document is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Corpus Bleu / 02 Corpus Bleu
# Complete Code / 完整代码
# ===============================

# two references for one document
from nltk.translate.bleu_score import corpus_bleu
references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
candidates = [['this', 'is', 'a', 'test']]
score = corpus_bleu(references, candidates)
# 打印输出 / Print output
print(score)
```

---

➡️ **Next / 下一步**: File 3 of 13

---

### Individual 1Gram Bleu

# 03 — Individual 1Gram Bleu / 03 Individual 1Gram Bleu

**Chapter 24 — File 3 of 13 / 第24章 — 第3个文件（共13个）**

---

## Summary / 总结

This script demonstrates **1-gram individual BLEU**.

本脚本演示 **1-gram individual BLEU**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — 1-gram individual BLEU

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
# 打印输出 / Print output
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: 1-gram individual BLEU 是机器学习中的常用技术。  
  *1-gram individual BLEU is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Individual 1Gram Bleu / 03 Individual 1Gram Bleu
# Complete Code / 完整代码
# ===============================

# 1-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
# 打印输出 / Print output
print(score)
```

---

➡️ **Next / 下一步**: File 4 of 13

---

### Individual Ngram Bleu

# 04 — Individual Ngram Bleu / 04 Individual Ngram Bleu

**Chapter 24 — File 4 of 13 / 第24章 — 第4个文件（共13个）**

---

## Summary / 总结

This script demonstrates **n-gram individual BLEU**.

本脚本演示 **n-gram individual BLEU**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — n-gram individual BLEU

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'a', 'test']]
candidate = ['this', 'is', 'a', 'test']
# 打印输出 / Print output
print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
# 打印输出 / Print output
print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
# 打印输出 / Print output
print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
# 打印输出 / Print output
print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))
```

---
## Learning Notes / 学习笔记

- **概念**: n-gram individual BLEU 是机器学习中的常用技术。  
  *n-gram individual BLEU is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Individual Ngram Bleu / 04 Individual Ngram Bleu
# Complete Code / 完整代码
# ===============================

# n-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'a', 'test']]
candidate = ['this', 'is', 'a', 'test']
# 打印输出 / Print output
print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
# 打印输出 / Print output
print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
# 打印输出 / Print output
print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
# 打印输出 / Print output
print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))
```

---

➡️ **Next / 下一步**: File 5 of 13

---

### Cumulative 4Gram Bleu

# 05 — Cumulative 4Gram Bleu / 05 Cumulative 4Gram Bleu

**Chapter 24 — File 5 of 13 / 第24章 — 第5个文件（共13个）**

---

## Summary / 总结

This script demonstrates **4-gram cumulative BLEU**.

本脚本演示 **4-gram cumulative BLEU**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — 4-gram cumulative BLEU

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
# 打印输出 / Print output
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: 4-gram cumulative BLEU 是机器学习中的常用技术。  
  *4-gram cumulative BLEU is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cumulative 4Gram Bleu / 05 Cumulative 4Gram Bleu
# Complete Code / 完整代码
# ===============================

# 4-gram cumulative BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
# 打印输出 / Print output
print(score)
```

---

➡️ **Next / 下一步**: File 6 of 13

---

### Cumulative Ngram Bleu

# 06 — Cumulative Ngram Bleu / 06 Cumulative Ngram Bleu

**Chapter 24 — File 6 of 13 / 第24章 — 第6个文件（共13个）**

---

## Summary / 总结

This script demonstrates **cumulative BLEU scores**.

本脚本演示 **cumulative BLEU scores**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — cumulative BLEU scores

```python
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
# 打印输出 / Print output
print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
# 打印输出 / Print output
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
# 打印输出 / Print output
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
# 打印输出 / Print output
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
```

---
## Learning Notes / 学习笔记

- **概念**: cumulative BLEU scores 是机器学习中的常用技术。  
  *cumulative BLEU scores is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cumulative Ngram Bleu / 06 Cumulative Ngram Bleu
# Complete Code / 完整代码
# ===============================

# cumulative BLEU scores
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
# 打印输出 / Print output
print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
# 打印输出 / Print output
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
# 打印输出 / Print output
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
# 打印输出 / Print output
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
```

---

➡️ **Next / 下一步**: File 7 of 13

---

### Example Perfect

# 07 — Example Perfect / 07 Example Perfect

**Chapter 24 — File 7 of 13 / 第24章 — 第7个文件（共13个）**

---

## Summary / 总结

This script demonstrates **prefect match**.

本脚本演示 **prefect match**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — prefect match

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: prefect match 是机器学习中的常用技术。  
  *prefect match is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example Perfect / 07 Example Perfect
# Complete Code / 完整代码
# ===============================

# prefect match
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---

➡️ **Next / 下一步**: File 8 of 13

---

### Example One Word Diff

# 08 — Example One Word Diff / 08 Example One Word Diff

**Chapter 24 — File 8 of 13 / 第24章 — 第8个文件（共13个）**

---

## Summary / 总结

This script demonstrates **one word different**.

本脚本演示 **one word different**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — one word different

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: one word different 是机器学习中的常用技术。  
  *one word different is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example One Word Diff / 08 Example One Word Diff
# Complete Code / 完整代码
# ===============================

# one word different
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---

➡️ **Next / 下一步**: File 9 of 13

---

### Example Two Words Diff

# 09 — Example Two Words Diff / 09 Example Two Words Diff

**Chapter 24 — File 9 of 13 / 第24章 — 第9个文件（共13个）**

---

## Summary / 总结

This script demonstrates **two words different**.

本脚本演示 **two words different**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — two words different

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'sleepy', 'dog']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: two words different 是机器学习中的常用技术。  
  *two words different is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example Two Words Diff / 09 Example Two Words Diff
# Complete Code / 完整代码
# ===============================

# two words different
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'sleepy', 'dog']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---

➡️ **Next / 下一步**: File 10 of 13

---

### Example All Diff

# 10 — Example All Diff / 10 Example All Diff

**Chapter 24 — File 10 of 13 / 第24章 — 第10个文件（共13个）**

---

## Summary / 总结

This script demonstrates **all words different**.

本脚本演示 **all words different**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — all words different

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: all words different 是机器学习中的常用技术。  
  *all words different is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example All Diff / 10 Example All Diff
# Complete Code / 完整代码
# ===============================

# all words different
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---

➡️ **Next / 下一步**: File 11 of 13

---

### Example Shorter

# 11 — Example Shorter / 11 Example Shorter

**Chapter 24 — File 11 of 13 / 第24章 — 第11个文件（共13个）**

---

## Summary / 总结

This script demonstrates **shorter candidate**.

本脚本演示 **shorter candidate**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — shorter candidate

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: shorter candidate 是机器学习中的常用技术。  
  *shorter candidate is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example Shorter / 11 Example Shorter
# Complete Code / 完整代码
# ===============================

# shorter candidate
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---

➡️ **Next / 下一步**: File 12 of 13

---

### Example Longer

# 12 — Example Longer / 12 Example Longer

**Chapter 24 — File 12 of 13 / 第24章 — 第12个文件（共13个）**

---

## Summary / 总结

This script demonstrates **longer candidate**.

本脚本演示 **longer candidate**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — longer candidate

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', 'from', 'space']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: longer candidate 是机器学习中的常用技术。  
  *longer candidate is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example Longer / 12 Example Longer
# Complete Code / 完整代码
# ===============================

# longer candidate
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', 'from', 'space']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---

➡️ **Next / 下一步**: File 13 of 13

---

### Example Too Short

# 13 — Example Too Short / 13 Example Too Short

**Chapter 24 — File 13 of 13 / 第24章 — 第13个文件（共13个）**

---

## Summary / 总结

This script demonstrates **very short**.

本脚本演示 **very short**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — very short

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: very short 是机器学习中的常用技术。  
  *very short is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example Too Short / 13 Example Too Short
# Complete Code / 完整代码
# ===============================

# very short
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick']
score = sentence_bleu(reference, candidate)
# 打印输出 / Print output
print(score)
```

---

### Chapter Summary / 章节总结

# Chapter 24 Summary / 第24章总结

## Theme / 主题: Chapter 24 / Chapter 24

This chapter contains **13 code files** demonstrating chapter 24.

本章包含 **13 个代码文件**，演示Chapter 24。

---
## Evolution / 演化路线

  1. `01_sentence_bleu.ipynb` — Sentence Bleu
  2. `02_corpus_bleu.ipynb` — Corpus Bleu
  3. `03_individual_1gram_bleu.ipynb` — Individual 1Gram Bleu
  4. `04_individual_ngram_bleu.ipynb` — Individual Ngram Bleu
  5. `05_cumulative_4gram_bleu.ipynb` — Cumulative 4Gram Bleu
  6. `06_cumulative_ngram_bleu.ipynb` — Cumulative Ngram Bleu
  7. `07_example_perfect.ipynb` — Example Perfect
  8. `08_example_one_word_diff.ipynb` — Example One Word Diff
  9. `09_example_two_words_diff.ipynb` — Example Two Words Diff
  10. `10_example_all_diff.ipynb` — Example All Diff
  11. `11_example_shorter.ipynb` — Example Shorter
  12. `12_example_longer.ipynb` — Example Longer
  13. `13_example_too_short.ipynb` — Example Too Short

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 24) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 24）是机器学习流水线中的基础构建块。

---
