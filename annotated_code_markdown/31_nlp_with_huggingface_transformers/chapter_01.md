# HuggingFace Transformers NLP / NLP with HF Transformers
## Chapter 01

---

### Simple

# 02 — Simple / 02 Simple

**Chapter 01 — File 1 of 5 / 第01章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Simple**.

本脚本演示 **02 Simple**。

---
## Step 1 — Step 1

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
input_ids = tokenizer("Hello, world!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**input_ids)
output_tokens = outputs.logits.argmax(dim=-1)
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(output_text)
```

---
## Learning Notes / 学习笔记

- **概念**: Simple 是机器学习中的常用技术。  
  *Simple is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Simple / 02 Simple
# Complete Code / 完整代码
# ===============================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
input_ids = tokenizer("Hello, world!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**input_ids)
output_tokens = outputs.logits.argmax(dim=-1)
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(output_text)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Envvar

# 03 — Envvar / 03 Envvar

**Chapter 01 — File 2 of 5 / 第01章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Envvar**.

本脚本演示 **03 Envvar**。

---
## Step 1 — Step 1

```python
import os
os.environ["HF_TOKEN"] = "hf_YourTokenHere"
os.environ["HF_HOME"] = "~/.cache/huggingface"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
input_ids = tokenizer("Hello, world!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**input_ids)
output_tokens = outputs.logits.argmax(dim=-1)
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(output_text)
```

---
## Learning Notes / 学习笔记

- **概念**: Envvar 是机器学习中的常用技术。  
  *Envvar is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Envvar / 03 Envvar
# Complete Code / 完整代码
# ===============================

import os
os.environ["HF_TOKEN"] = "hf_YourTokenHere"
os.environ["HF_HOME"] = "~/.cache/huggingface"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
input_ids = tokenizer("Hello, world!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**input_ids)
output_tokens = outputs.logits.argmax(dim=-1)
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(output_text)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Pipeline

# 04 — Pipeline / 管道

**Chapter 01 — File 3 of 5 / 第01章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Pipeline**.

本脚本演示 **管道**。

---
## Step 1 — Step 1

```python
from transformers import pipeline

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_id)
result = classifier("Machine Learning Mastery is a great website for machine learning.")
print(result)
```

---
## Learning Notes / 学习笔记

- **概念**: Pipeline 是机器学习中的常用技术。  
  *Pipeline is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pipeline / 管道
# Complete Code / 完整代码
# ===============================

from transformers import pipeline

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_id)
result = classifier("Machine Learning Mastery is a great website for machine learning.")
print(result)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Print

# 05 — Print / 05 Print

**Chapter 01 — File 4 of 5 / 第01章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Print**.

本脚本演示 **05 Print**。

---
## Step 1 — Step 1

```python
from transformers import pipeline

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_id)
print(classifier.model)
print(classifier.tokenizer)
```

---
## Learning Notes / 学习笔记

- **概念**: Print 是机器学习中的常用技术。  
  *Print is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Print / 05 Print
# Complete Code / 完整代码
# ===============================

from transformers import pipeline

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_id)
print(classifier.model)
print(classifier.tokenizer)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Create

# 06 — Create / 06 Create

**Chapter 01 — File 5 of 5 / 第01章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Create**.

本脚本演示 **06 Create**。

---
## Step 1 — Step 1

```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = DistilBertForSequenceClassification.from_pretrained(model_id)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_id)
```

---
## Learning Notes / 学习笔记

- **概念**: Create 是机器学习中的常用技术。  
  *Create is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Create / 06 Create
# Complete Code / 完整代码
# ===============================

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = DistilBertForSequenceClassification.from_pretrained(model_id)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_id)
```

---

### Chapter Summary / 章节总结

# Chapter 01 Summary / 第01章总结

## Theme / 主题: Chapter 01 / Chapter 01

This chapter contains **5 code files** demonstrating chapter 01.

本章包含 **5 个代码文件**，演示Chapter 01。

---
## Evolution / 演化路线

  1. `02_simple.ipynb` — Simple
  2. `03_envvar.ipynb` — Envvar
  3. `04_pipeline.ipynb` — Pipeline
  4. `05_print.ipynb` — Print
  5. `06_create.ipynb` — Create

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 01) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 01）是机器学习流水线中的基础构建块。

---
