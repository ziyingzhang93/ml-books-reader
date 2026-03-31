# 注意力与Transformer / Transformer Models with Attention
## Chapter 23

---

### Summary

# 02 — Summary / 02 Summary

**Chapter 23 — File 1 of 2 / 第23章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Summary**.

本脚本演示 **02 Summary**。

---
## Step 1 — Step 1

```python
from summarizer import Summarizer
text = open("article.txt").read()
model = Summarizer('distilbert-base-uncased')
result = model(text, num_sentences=3)
print(result)
```

---
## Learning Notes / 学习笔记

- **概念**: Summary 是机器学习中的常用技术。  
  *Summary is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Summary / 02 Summary
# Complete Code / 完整代码
# ===============================

from summarizer import Summarizer
text = open("article.txt").read()
model = Summarizer('distilbert-base-uncased')
result = model(text, num_sentences=3)
print(result)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Answering

# 03 — Answering / 03 Answering

**Chapter 23 — File 2 of 2 / 第23章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Answering**.

本脚本演示 **03 Answering**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


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
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import pipeline
text = open("article.txt").read()
question = "What is BOE doing?"

answering = pipeline("question-answering",
                     model='distilbert-base-uncased-distilled-squad')
result = answering(question=question, context=text)
# 打印输出 / Print output
print(result)
```

---
## Learning Notes / 学习笔记

- **概念**: Answering 是机器学习中的常用技术。  
  *Answering is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Answering / 03 Answering
# Complete Code / 完整代码
# ===============================

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import pipeline
text = open("article.txt").read()
question = "What is BOE doing?"

answering = pipeline("question-answering",
                     model='distilbert-base-uncased-distilled-squad')
result = answering(question=question, context=text)
# 打印输出 / Print output
print(result)
```

---

### Chapter Summary / 章节总结

# Chapter 23 Summary / 第23章总结

## Theme / 主题: Chapter 23 / Chapter 23

This chapter contains **2 code files** demonstrating chapter 23.

本章包含 **2 个代码文件**，演示Chapter 23。

---
## Evolution / 演化路线

  1. `02_summary.ipynb` — Summary
  2. `03_answering.ipynb` — Answering

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 23) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 23）是机器学习流水线中的基础构建块。

---
