# HuggingFace Transformers NLP / NLP with HF Transformers
## Chapter 10

---

### Translate



---

### Multiple



---

### Scores

# 05 — Scores / 05 Scores

**Chapter 10 — File 3 of 4 / 第10章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Scores**.

本脚本演示 **05 Scores**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name="t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
model.to(device)

input_text = "This is an important message that needs accurate translation."
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
inputs = inputs.to(device)

outputs = model.generate(**inputs, max_length=512, num_beams=4*4, num_beam_groups=4,
                         num_return_sequences=4, diversity_penalty=0.8,
                         length_penalty=0.6, early_stopping=True, output_scores=True,
                         return_dict_in_generate=True)
transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
)
# 同时获取索引和值 / Get both index and value
for idx, (out_tok, out_score) in enumerate(zip(outputs.sequences, transition_scores)):
    translation = tokenizer.decode(out_tok, skip_special_tokens=True)
    # 打印输出 / Print output
    print(f"Translation: {translation}")
    # 打印输出 / Print output
    print("token | token string   | logits  | probability")
    # 将多个序列配对 / Pair multiple sequences
    for tok, score in zip(out_tok[1:], out_score.cpu()):
        # 打印输出 / Print output
        print(f"| {tok:5d} | {tokenizer.decode(tok):14s} | {score.numpy():.4f} "
              f"| {np.exp(score.numpy()):.2%}")
```

---
## Learning Notes / 学习笔记

- **概念**: Scores 是机器学习中的常用技术。  
  *Scores is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Scores / 05 Scores
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name="t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
model.to(device)

input_text = "This is an important message that needs accurate translation."
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
inputs = inputs.to(device)

outputs = model.generate(**inputs, max_length=512, num_beams=4*4, num_beam_groups=4,
                         num_return_sequences=4, diversity_penalty=0.8,
                         length_penalty=0.6, early_stopping=True, output_scores=True,
                         return_dict_in_generate=True)
transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
)
# 同时获取索引和值 / Get both index and value
for idx, (out_tok, out_score) in enumerate(zip(outputs.sequences, transition_scores)):
    translation = tokenizer.decode(out_tok, skip_special_tokens=True)
    # 打印输出 / Print output
    print(f"Translation: {translation}")
    # 打印输出 / Print output
    print("token | token string   | logits  | probability")
    # 将多个序列配对 / Pair multiple sequences
    for tok, score in zip(out_tok[1:], out_score.cpu()):
        # 打印输出 / Print output
        print(f"| {tok:5d} | {tokenizer.decode(tok):14s} | {score.numpy():.4f} "
              f"| {np.exp(score.numpy()):.2%}")
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Bleu



---

### Chapter Summary / 章节总结

# Chapter 10 Summary / 第10章总结

## Theme / 主题: Chapter 10 / Chapter 10

This chapter contains **4 code files** demonstrating chapter 10.

本章包含 **4 个代码文件**，演示Chapter 10。

---
## Evolution / 演化路线

  1. `02_translate.ipynb` — Translate
  2. `04_multiple.ipynb` — Multiple
  3. `05_scores.ipynb` — Scores
  4. `06_bleu.ipynb` — Bleu

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 10) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 10）是机器学习流水线中的基础构建块。

---
