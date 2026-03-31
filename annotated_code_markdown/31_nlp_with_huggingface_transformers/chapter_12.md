# HuggingFace Transformers NLP / NLP with HF Transformers
## Chapter 12

---

### Qna

# 01 — Qna / 01 Qna

**Chapter 12 — File 1 of 4 / 第12章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load pre-trained model and tokenizer**.

本脚本演示 **Load pre-trained model and tokenizer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
```

---
## Step 2 — Load pre-trained model and tokenizer

```python
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
```

---
## Step 3 — Define a context and a question

```python
question = "What is machine learning?"
context = """Machine learning is a field of inquiry devoted to understanding and building
methods that 'learn', that is, methods that leverage data to improve performance on some
set of tasks. It is seen as a part of artificial intelligence.  Machine learning
algorithms build a model based on sample data, known as training data, in order to make
predictions or decisions without being explicitly programmed to do so. Machine learning
algorithms are used in a wide variety of applications, such as in medicine, email
filtering, speech recognition, and computer vision, where it is difficult or unfeasible to
develop conventional algorithms to perform the needed tasks."""
```

---
## Step 4 — Tokenize the input and run the model

```python
inputs = tokenizer(question, context, return_tensors="pt")
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    outputs = model(**inputs)
```

---
## Step 5 — Process the answer

```python
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits)
answer_tokens = inputs.input_ids[0, answer_start: answer_end + 1]
answer = tokenizer.decode(answer_tokens)

# 打印输出 / Print output
print(f"Question: {question}")
# 打印输出 / Print output
print(f"Answer: {answer}")
```

---
## Learning Notes / 学习笔记

- **概念**: Load pre-trained model and tokenizer 是机器学习中的常用技术。  
  *Load pre-trained model and tokenizer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Qna / 01 Qna
# Complete Code / 完整代码
# ===============================

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# Define a context and a question
question = "What is machine learning?"
context = """Machine learning is a field of inquiry devoted to understanding and building
methods that 'learn', that is, methods that leverage data to improve performance on some
set of tasks. It is seen as a part of artificial intelligence.  Machine learning
algorithms build a model based on sample data, known as training data, in order to make
predictions or decisions without being explicitly programmed to do so. Machine learning
algorithms are used in a wide variety of applications, such as in medicine, email
filtering, speech recognition, and computer vision, where it is difficult or unfeasible to
develop conventional algorithms to perform the needed tasks."""

# Tokenize the input and run the model
inputs = tokenizer(question, context, return_tensors="pt")
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    outputs = model(**inputs)

# Process the answer
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits)
answer_tokens = inputs.input_ids[0, answer_start: answer_end + 1]
answer = tokenizer.decode(answer_tokens)

# 打印输出 / Print output
print(f"Question: {question}")
# 打印输出 / Print output
print(f"Answer: {answer}")
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Logit



---

### Sliding



---

### Ensemble



---

### Chapter Summary / 章节总结

# Chapter 12 Summary / 第12章总结

## Theme / 主题: Chapter 12 / Chapter 12

This chapter contains **4 code files** demonstrating chapter 12.

本章包含 **4 个代码文件**，演示Chapter 12。

---
## Evolution / 演化路线

  1. `01_qna.ipynb` — Qna
  2. `02_logit.ipynb` — Logit
  3. `03_sliding.ipynb` — Sliding
  4. `04_ensemble.ipynb` — Ensemble

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 12) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 12）是机器学习流水线中的基础构建块。

---
