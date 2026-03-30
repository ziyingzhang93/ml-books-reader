# HF Transformers
## Chapter 07

---

### Greedy

# 05 — Greedy / 05 Greedy

**Chapter 07 — File 5 of 11 / 第07章 — 第5个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Generate text with greedy decoding vs. sampling**.

本脚本演示 **Generate text with greedy decoding vs. sampling**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The secret to happiness is"
inputs = tokenizer(prompt, return_tensors="pt")
```

---
## Step 2 — Generate text with greedy decoding vs. sampling

```python
print(f"Prompt: {prompt}\n")
print("Greedy Decoding (do_sample=False):")
output = model.generate(
    **inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.0,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
print()
print("Sampling (do_sample=True):")
output = model.generate(
    **inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.0,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
```

---
## Learning Notes / 学习笔记

- **概念**: Generate text with greedy decoding vs. sampling 是机器学习中的常用技术。  
  *Generate text with greedy decoding vs. sampling is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Greedy / 05 Greedy
# Complete Code / 完整代码
# ===============================

from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The secret to happiness is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text with greedy decoding vs. sampling
print(f"Prompt: {prompt}\n")
print("Greedy Decoding (do_sample=False):")
output = model.generate(
    **inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.0,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
print()
print("Sampling (do_sample=True):")
output = model.generate(
    **inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.0,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
```

---

➡️ **Next / 下一步**: File 6 of 11

---

### Beam

# 06 — Beam / 06 Beam

**Chapter 07 — File 6 of 11 / 第07章 — 第6个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Generate text with greedy decoding vs. sampling**.

本脚本演示 **Generate text with greedy decoding vs. sampling**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The key to successful machine learning is"
inputs = tokenizer(prompt, return_tensors="pt")
```

---
## Step 2 — Generate text with greedy decoding vs. sampling

```python
print(f"Prompt: {prompt}\n")
outputs = model.generate(
    **inputs,
    num_beams=5,             # Number of beams to use
    early_stopping=True,     # Stop when all beams have finished
    no_repeat_ngram_size=2,  # Avoid repeating n-grams
    num_return_sequences=3,  # Return multiple sequences
    max_length=100,
    temperature=1.5,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
for idx, output in enumerate(outputs):
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Generated Text ({idx+1}):")
    print(generated_text)
```

---
## Learning Notes / 学习笔记

- **概念**: Generate text with greedy decoding vs. sampling 是机器学习中的常用技术。  
  *Generate text with greedy decoding vs. sampling is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Beam / 06 Beam
# Complete Code / 完整代码
# ===============================

from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The key to successful machine learning is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text with greedy decoding vs. sampling
print(f"Prompt: {prompt}\n")
outputs = model.generate(
    **inputs,
    num_beams=5,             # Number of beams to use
    early_stopping=True,     # Stop when all beams have finished
    no_repeat_ngram_size=2,  # Avoid repeating n-grams
    num_return_sequences=3,  # Return multiple sequences
    max_length=100,
    temperature=1.5,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
for idx, output in enumerate(outputs):
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Generated Text ({idx+1}):")
    print(generated_text)
```

---

➡️ **Next / 下一步**: File 7 of 11

---

### 

# 7 —  / 7

**Chapter 07 — File 9 of 11 / 第07章 — 第9个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Encode the input prompt**.

本脚本演示 **Encode the input prompt**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TextGenerator:
    def __init__(self, model_name='gpt2'):
        """Initialize the text generator with a pre-trained model.

        Args:
            model_name (str): Name of the pre-trained model to use.
                              Any of: 'gpt2', 'gpt2-medium', 'gpt2-large'
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def generate_text(self, prompt, max_length=100, temperature=0.7,
                      top_k=50, top_p=0.95):
        """Generate text based on the input prompt.

        Args:
            prompt (str): Input text to continue from
            max_length (int): Maximum length of generated text
            temperature (float): Controls randomness in generation
            top_k (int): Number of highest probability tokens to consider
            top_p (float): Cumulative probability threshold for token filtering

        Returns:
            str: Generated text including the prompt
        """
        try:
```

---
## Step 2 — Encode the input prompt

```python
inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
```

---
## Step 3 — Configure generation parameters

```python
gen_kwargs = {
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "pad_token_id": self.tokenizer.eos_token_id,
                "no_repeat_ngram_size": 2,
                "do_sample": True,
            }
```

---
## Step 4 — Generate text

```python
with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )
```

---
## Step 5 — Decode and return the generated text

```python
generated_text = self.tokenizer.decode(
                output_sequences[0],
                skip_special_tokens=True
            )
            return generated_text
        except Exception as e:
            print(f"Error during text generation: {str(e)}")
            return prompt
```

---
## Learning Notes / 学习笔记

- **概念**: Encode the input prompt 是机器学习中的常用技术。  
  *Encode the input prompt is a common technique in machine learning.*

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
#  / 7
# Complete Code / 完整代码
# ===============================

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TextGenerator:
    def __init__(self, model_name='gpt2'):
        """Initialize the text generator with a pre-trained model.

        Args:
            model_name (str): Name of the pre-trained model to use.
                              Any of: 'gpt2', 'gpt2-medium', 'gpt2-large'
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def generate_text(self, prompt, max_length=100, temperature=0.7,
                      top_k=50, top_p=0.95):
        """Generate text based on the input prompt.

        Args:
            prompt (str): Input text to continue from
            max_length (int): Maximum length of generated text
            temperature (float): Controls randomness in generation
            top_k (int): Number of highest probability tokens to consider
            top_p (float): Cumulative probability threshold for token filtering

        Returns:
            str: Generated text including the prompt
        """
        try:
            # Encode the input prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Configure generation parameters
            gen_kwargs = {
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "pad_token_id": self.tokenizer.eos_token_id,
                "no_repeat_ngram_size": 2,
                "do_sample": True,
            }

            # Generate text
            with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )

            # Decode and return the generated text
            generated_text = self.tokenizer.decode(
                output_sequences[0],
                skip_special_tokens=True
            )
            return generated_text
        except Exception as e:
            print(f"Error during text generation: {str(e)}")
            return prompt
```

---

➡️ **Next / 下一步**: File 10 of 11

---

### Chapter Summary

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **11 code files** demonstrating chapter 07.

本章包含 **11 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `01_generate.ipynb` — Generate
  2. `02_temperature.ipynb` — Temperature
  3. `03_top.ipynb` — Top
  4. `04_repetition.ipynb` — Repetition
  5. `05_greedy.ipynb` — Greedy
  6. `06_beam.ipynb` — Beam
  7. `10.ipynb` — 
  8. `11.ipynb` — 
  9. `7.ipynb` — 
  10. `8.ipynb` — 
  11. `9.ipynb` — 

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---
