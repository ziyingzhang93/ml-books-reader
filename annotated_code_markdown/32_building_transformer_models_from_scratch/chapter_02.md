# 从零构建Transformer
## Chapter 02

---

### Causal Mask

# 01 — Causal Mask / 01 Causal Mask

**Chapter 02 — File 1 of 4 / 第02章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Causal Mask**.

本脚本演示 **01 Causal Mask**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import torch

seq_len = 4
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
print(causal_mask)
```

---
## Learning Notes / 学习笔记

- **概念**: Causal Mask 是机器学习中的常用技术。  
  *Causal Mask is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Causal Mask / 01 Causal Mask
# Complete Code / 完整代码
# ===============================

import torch

seq_len = 4
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
print(causal_mask)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Bert Model

# 02 — Bert Model / 02 Bert Model

**Chapter 02 — File 2 of 4 / 第02章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Bert Model**.

本脚本演示 **02 Bert Model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
from transformers import BertModel, BertConfig

config = BertConfig()
model = BertModel(config=config)
print(model)
```

---
## Learning Notes / 学习笔记

- **概念**: Bert Model 是机器学习中的常用技术。  
  *Bert Model is a common technique in machine learning.*

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
# Bert Model / 02 Bert Model
# Complete Code / 完整代码
# ===============================

from transformers import BertModel, BertConfig

config = BertConfig()
model = BertModel(config=config)
print(model)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Gpt2 Model

# 03 — Gpt2 Model / 03 Gpt2 Model

**Chapter 02 — File 3 of 4 / 第02章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Gpt2 Model**.

本脚本演示 **03 Gpt2 Model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
from transformers import GPT2LMHeadModel, GPT2Config

config = GPT2Config()
model = GPT2LMHeadModel(config=config)
print(model)
```

---
## Learning Notes / 学习笔记

- **概念**: Gpt2 Model 是机器学习中的常用技术。  
  *Gpt2 Model is a common technique in machine learning.*

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
# Gpt2 Model / 03 Gpt2 Model
# Complete Code / 完整代码
# ===============================

from transformers import GPT2LMHeadModel, GPT2Config

config = GPT2Config()
model = GPT2LMHeadModel(config=config)
print(model)
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Load Gpt2

# 04 — Load Gpt2 / 04 Load Gpt2

**Chapter 02 — File 4 of 4 / 第02章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load Gpt2**.

本脚本演示 **04 Load Gpt2**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
print(model)
```

---
## Learning Notes / 学习笔记

- **概念**: Load Gpt2 是机器学习中的常用技术。  
  *Load Gpt2 is a common technique in machine learning.*

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
# Load Gpt2 / 04 Load Gpt2
# Complete Code / 完整代码
# ===============================

from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
print(model)
```

---

### Chapter Summary

# Chapter 02 Summary / 第02章总结

## Theme / 主题: Chapter 02 / Chapter 02

This chapter contains **4 code files** demonstrating chapter 02.

本章包含 **4 个代码文件**，演示Chapter 02。

---
## Evolution / 演化路线

  1. `01_causal_mask.ipynb` — Causal Mask
  2. `02_bert_model.ipynb` — Bert Model
  3. `03_gpt2_model.ipynb` — Gpt2 Model
  4. `04_load_gpt2.ipynb` — Load Gpt2

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 02) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 02）是机器学习流水线中的基础构建块。

---
