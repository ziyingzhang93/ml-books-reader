# HF Transformers
## Chapter 02

---

### Verbose

# 01 — Verbose / 01 Verbose

**Chapter 02 — File 1 of 6 / 第02章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Verbose**.

本脚本演示 **01 Verbose**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model_name = "KernAI/stock-news-distilbert"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

text = "Machine Learning Mastery is a nice website."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
```

---
## Learning Notes / 学习笔记

- **概念**: Verbose 是机器学习中的常用技术。  
  *Verbose is a common technique in machine learning.*

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
# Verbose / 01 Verbose
# Complete Code / 完整代码
# ===============================

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model_name = "KernAI/stock-news-distilbert"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

text = "Machine Learning Mastery is a nice website."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Chapter Summary

# Chapter 02 Summary / 第02章总结

## Theme / 主题: Chapter 02 / Chapter 02

This chapter contains **6 code files** demonstrating chapter 02.

本章包含 **6 个代码文件**，演示Chapter 02。

---
## Evolution / 演化路线

  1. `01_verbose.ipynb` — Verbose
  2. `02_othermodel.ipynb` — Othermodel
  3. `03_auto.ipynb` — Auto
  4. `04_tf.ipynb` — Tf
  5. `05_warning.ipynb` — Warning
  6. `06_pipeline.ipynb` — Pipeline

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 02) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 02）是机器学习流水线中的基础构建块。

---
