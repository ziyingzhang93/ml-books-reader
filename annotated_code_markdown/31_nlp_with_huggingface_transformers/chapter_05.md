# HF Transformers
## Chapter 05

---

### Complete

# 04 — Complete / 04 Complete

**Chapter 05 — File 2 of 2 / 第05章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Encode the input text**.

本脚本演示 **Encode the input text**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
from functools import lru_cache
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class AutoComplete:
    def __init__(self, model_name="gpt2"):
        """Initialize the auto-complete system."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def get_completion(self, text, max_length=50):
        """Generate completion for the input text."""
        print("**** Completion:", text)
```

---
## Step 2 — Encode the input text

```python
inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attn_masks = inputs["attention_mask"].to(self.device)
```

---
## Step 3 — Generate completion

```python
with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attn_masks,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
```

---
## Step 4 — Decode and extract completion

```python
full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = full_text[len(text):]

        return completion

class CachedAutoComplete(AutoComplete):
    def __init__(self, cache_size=1000, **kwargs):
        """Initialize with caching support."""
        super().__init__(**kwargs)
        self.get_completion = lru_cache(maxsize=cache_size)(self.get_completion)

class OptimizedAutoComplete(CachedAutoComplete):
    def __init__(self, **kwargs):
        """Initialize with optimizations."""
        super().__init__(**kwargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda":
            self.model = self.model.half()  # Use FP16 on GPU
```

---
## Step 5 — use eval mode and cuda graphs

```python
self.model.eval()

    def preprocess_batch(self, texts):
        """Efficiently process multiple texts."""
```

---
## Step 6 — Tokenize all texts at once

```python
inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return inputs.to(self.device)

    def generate_batch(self, texts, max_length=50):
        """Generate completions for multiple texts."""
```

---
## Step 7 — Preprocess batch

```python
inputs = self.preprocess_batch(texts)
```

---
## Step 8 — Generate completions

```python
with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
```

---
## Step 9 — Decode completions

```python
completions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

---
## Step 10 — Extract new text

```python
results = []
        for text, completion in zip(texts, completions):
            results.append(completion[len(text):])

        return results
```

---
## Step 11 — Example: Optimized batch completion

```python
optimized_complete = OptimizedAutoComplete()
texts = [
    "Machine learning is",
    "Deep neural networks can",
    "The training process involves"
]
completions = optimized_complete.generate_batch(texts)
for text, completion in zip(texts, completions):
    print(f"\nInput: {text}")
    print(f"Completion: {completion}")
```

---
## Learning Notes / 学习笔记

- **概念**: Encode the input text 是机器学习中的常用技术。  
  *Encode the input text is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Complete / 04 Complete
# Complete Code / 完整代码
# ===============================

from functools import lru_cache
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class AutoComplete:
    def __init__(self, model_name="gpt2"):
        """Initialize the auto-complete system."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def get_completion(self, text, max_length=50):
        """Generate completion for the input text."""
        print("**** Completion:", text)
        # Encode the input text
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attn_masks = inputs["attention_mask"].to(self.device)

        # Generate completion
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attn_masks,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )

        # Decode and extract completion
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = full_text[len(text):]

        return completion

class CachedAutoComplete(AutoComplete):
    def __init__(self, cache_size=1000, **kwargs):
        """Initialize with caching support."""
        super().__init__(**kwargs)
        self.get_completion = lru_cache(maxsize=cache_size)(self.get_completion)

class OptimizedAutoComplete(CachedAutoComplete):
    def __init__(self, **kwargs):
        """Initialize with optimizations."""
        super().__init__(**kwargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda":
            self.model = self.model.half()  # Use FP16 on GPU

        # use eval mode and cuda graphs
        self.model.eval()

    def preprocess_batch(self, texts):
        """Efficiently process multiple texts."""
        # Tokenize all texts at once
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return inputs.to(self.device)

    def generate_batch(self, texts, max_length=50):
        """Generate completions for multiple texts."""
        # Preprocess batch
        inputs = self.preprocess_batch(texts)

        # Generate completions
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )

        # Decode completions
        completions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract new text
        results = []
        for text, completion in zip(texts, completions):
            results.append(completion[len(text):])

        return results

# Example: Optimized batch completion
optimized_complete = OptimizedAutoComplete()
texts = [
    "Machine learning is",
    "Deep neural networks can",
    "The training process involves"
]
completions = optimized_complete.generate_batch(texts)
for text, completion in zip(texts, completions):
    print(f"\nInput: {text}")
    print(f"Completion: {completion}")
```

---

### Chapter Summary

# Chapter 05 Summary / 第05章总结

## Theme / 主题: Chapter 05 / Chapter 05

This chapter contains **2 code files** demonstrating chapter 05.

本章包含 **2 个代码文件**，演示Chapter 05。

---
## Evolution / 演化路线

  1. `01_autocomplete.ipynb` — Autocomplete
  2. `04_complete.ipynb` — Complete

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 05) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 05）是机器学习流水线中的基础构建块。

---
