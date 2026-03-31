# HuggingFace Transformers NLP / NLP with HF Transformers
## Chapter 08

---

### Basic



---

### Length

# 05 — Length / 05 Length

**Chapter 08 — File 2 of 3 / 第08章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Length**.

本脚本演示 **05 Length**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained("sshleifer/distilbart-cnn-12-6",
                                    trust_remote_code=True)
# 打印输出 / Print output
print(config.max_position_embeddings)

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6",
                                          trust_remote_code=True)
# 打印输出 / Print output
print(tokenizer.model_max_length)
```

---
## Learning Notes / 学习笔记

- **概念**: Length 是机器学习中的常用技术。  
  *Length is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Length / 05 Length
# Complete Code / 完整代码
# ===============================

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained("sshleifer/distilbart-cnn-12-6",
                                    trust_remote_code=True)
# 打印输出 / Print output
print(config.max_position_embeddings)

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6",
                                          trust_remote_code=True)
# 打印输出 / Print output
print(tokenizer.model_max_length)
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Complete

# 07 — Complete / 07 Complete

**Chapter 08 — File 3 of 3 / 第08章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Remove extra whitespace**.

本脚本演示 **Remove extra whitespace**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
import functools
import pprint
# 导入正则表达式模块 / Import regex module
import re

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class AdvancedTextSummarizer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6", quantize=False):
        """Initialize the advanced summarizer with additional features.

        Args:
            model_name (str): Name of the pre-trained model to use
            quantize (bool): Whether to quantize the model for faster inference
        """
        self.device = "cuda" if torch.cuda.is_available() and not quantize else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if quantize:
            self.model = torch.quantization.quantize_dynamic(
                self.model, dtype=torch.qint8)
        self.model.to(self.device)

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize input text.

        Args:
            text (str): Raw input text

        Returns:
            str: Cleaned and normalized text
        """
```

---
## Step 2 — Remove extra whitespace

```python
text = re.sub(r"\s+", " ", text.strip())
```

---
## Step 3 — Remove URLs

```python
text = re.sub(r"^https?://[^\s/$.?#].[^\s]*$", "", text)
```

---
## Step 4 — Remove special characters but keep punctuation

```python
text = re.sub(r"[^\w\s.,!?-]", "", text)

        return text

    def split_long_text(self, text: str, max_tokens: int = 1024) -> list[str]:
        """Split long text into chunks that fit within model's max token limit.

        Args:
            text (str): Input text
            max_tokens (int): Maximum tokens per chunk

        Returns:
            list[str]: List of text chunks
        """
```

---
## Step 5 — Tokenize the full text

```python
tokens = self.tokenizer.tokenize(text)
```

---
## Step 6 — Split into chunks, then convert back to strings

```python
# 获取长度 / Get length
chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
        return [self.tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

    @functools.lru_cache(maxsize=200)
    def cached_summarize(self, text: str, max_length: int = 130, min_length: int = 30,
                         length_penalty: float = 2.0, repetition_penalty: float = 2.0,
                         num_beams: int = 4, early_stopping: bool = True) -> str:
        """Cached version of the summarization function."""
        try:
```

---
## Step 7 — Tokenize the input text

```python
inputs = self.tokenizer(text, max_length=1024, truncation=True,
                                    padding="max_length", return_tensors="pt"
                                    ).to(self.device)
```

---
## Step 8 — Generate summary

```python
summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
                early_stopping=early_stopping,
                no_repeat_ngram_size=3,   # Prevent repetition of phrases
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            # 打印输出 / Print output
            print(f"Error during summarization: {str(e)}")
            return text

    def summarize_batch(self, texts: list[str], batch_size: int = 4, **kwargs
                        ) -> list[str]:
        """Summarize multiple texts efficiently in batches.

        Args:
            texts (list[str]): List of input texts
            batch_size (int): Number of texts to process at once
            **kwargs: Additional arguments for summarization

        Returns:
            list[str]: List of generated summaries
        """
        summaries = []

        # 获取长度 / Get length
        for i in range(0, len(texts), batch_size):
```

---
## Step 9 — Create batch and process each text in the batch

```python
batch = texts[i:i + batch_size]
            processed_batch = [self.preprocess_text(text) for text in batch]
```

---
## Step 10 — Tokenize batch

```python
inputs = self.tokenizer(processed_batch, max_length=1024, truncation=True,
                                    padding=True, return_tensors="pt"
                                    ).to(self.device)
```

---
## Step 11 — Generate summaries for batch

```python
summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **kwargs
            )
```

---
## Step 12 — Decode summaries

```python
summaries.extend([self.tokenizer.decode(ids, skip_special_tokens=True)
                              for ids in summary_ids])
        return summaries

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30,
                  length_penalty: float = 2.0, repetition_penalty: float = 2.0,
                  num_beams: int = 4, early_stopping: bool = True) -> dict[str, str]:
        """Generate a summary with advanced features.

        Args:
            text (str): The text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            length_penalty (float): Penalty for longer summaries
            repetition_penalty (float): Penalty for repeated tokens
            num_beams (int): Number of beams for beam search
            early_stopping (bool): Whether to stop when all beams are finished

        Returns:
            dict[str, str]: Dictionary containing original and summarized text
        """
```

---
## Step 13 — Preprocess the text

```python
cleaned_text = self.preprocess_text(text)
```

---
## Step 14 — Handle long texts

```python
chunks = self.split_long_text(cleaned_text)
        chunk_summaries = []
        for chunk in chunks:
            summary = self.cached_summarize(
                chunk,
                max_length=max_length // len(chunks),  # Adjust length for chunks
                # 获取长度 / Get length
                min_length=min_length // len(chunks),
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
                early_stopping=early_stopping
            )
            # 添加元素到列表末尾 / Append element to list end
            chunk_summaries.append(summary)

        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "summary": " ".join(chunk_summaries)
        }
```

---
## Step 15 — Initialize the advanced summarizer with caching enabled and quantization

```python
adv_summarizer = AdvancedTextSummarizer(quantize=True)
```

---
## Step 16 — Sample text

```python
long_text = """
The development of artificial intelligence (AI) has significantly impacted various
industries worldwide. From healthcare to finance, AI-powered applications have
streamlined operations, improved accuracy, and unlocked new possibilities. In healthcare,
AI assists in diagnostics, personalized treatment plans, and drug discovery. In finance,
it aids in fraud detection, algorithmic trading, and customer service. Despite its
benefits, AI raises concerns about data privacy, ethical implications, and job
displacement.
"""
```

---
## Step 17 — Generate a summary with default settings

```python
adv_summary = adv_summarizer.summarize(long_text)
# 打印输出 / Print output
print("Advanced Summary:")
# 打印输出 / Print output
pprint.pprint(adv_summary)
```

---
## Step 18 — Batch summarization

```python
texts = [
  "AI is revolutionizing healthcare with better diagnostics and personalized treatments.",
  "Self-driving cars are powered by machine learning algorithms that continuously " \
      "learn from traffic patterns.",
  "Natural language processing helps computers understand and communicate with humans " \
    "more effectively.",
  "Climate change is being studied using AI models to predict future environmental " \
    "patterns."
]
```

---
## Step 19 — Summarize multiple texts in a batch

```python
batch_summaries = adv_summarizer.summarize_batch(texts, batch_size=2)
# 打印输出 / Print output
print("\nBatch Summaries:")
# 同时获取索引和值 / Get both index and value
for i, s in enumerate(batch_summaries, 1):
    # 打印输出 / Print output
    pprint.pprint(f"{i}. {s}")
```

---
## Learning Notes / 学习笔记

- **概念**: Remove extra whitespace 是机器学习中的常用技术。  
  *Remove extra whitespace is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Complete / 07 Complete
# Complete Code / 完整代码
# ===============================

import functools
import pprint
# 导入正则表达式模块 / Import regex module
import re

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class AdvancedTextSummarizer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6", quantize=False):
        """Initialize the advanced summarizer with additional features.

        Args:
            model_name (str): Name of the pre-trained model to use
            quantize (bool): Whether to quantize the model for faster inference
        """
        self.device = "cuda" if torch.cuda.is_available() and not quantize else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if quantize:
            self.model = torch.quantization.quantize_dynamic(
                self.model, dtype=torch.qint8)
        self.model.to(self.device)

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize input text.

        Args:
            text (str): Raw input text

        Returns:
            str: Cleaned and normalized text
        """
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())
        # Remove URLs
        text = re.sub(r"^https?://[^\s/$.?#].[^\s]*$", "", text)
        # Remove special characters but keep punctuation
        text = re.sub(r"[^\w\s.,!?-]", "", text)

        return text

    def split_long_text(self, text: str, max_tokens: int = 1024) -> list[str]:
        """Split long text into chunks that fit within model's max token limit.

        Args:
            text (str): Input text
            max_tokens (int): Maximum tokens per chunk

        Returns:
            list[str]: List of text chunks
        """
        # Tokenize the full text
        tokens = self.tokenizer.tokenize(text)

        # Split into chunks, then convert back to strings
        # 获取长度 / Get length
        chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
        return [self.tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

    @functools.lru_cache(maxsize=200)
    def cached_summarize(self, text: str, max_length: int = 130, min_length: int = 30,
                         length_penalty: float = 2.0, repetition_penalty: float = 2.0,
                         num_beams: int = 4, early_stopping: bool = True) -> str:
        """Cached version of the summarization function."""
        try:
            # Tokenize the input text
            inputs = self.tokenizer(text, max_length=1024, truncation=True,
                                    padding="max_length", return_tensors="pt"
                                    ).to(self.device)
            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
                early_stopping=early_stopping,
                no_repeat_ngram_size=3,   # Prevent repetition of phrases
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            # 打印输出 / Print output
            print(f"Error during summarization: {str(e)}")
            return text

    def summarize_batch(self, texts: list[str], batch_size: int = 4, **kwargs
                        ) -> list[str]:
        """Summarize multiple texts efficiently in batches.

        Args:
            texts (list[str]): List of input texts
            batch_size (int): Number of texts to process at once
            **kwargs: Additional arguments for summarization

        Returns:
            list[str]: List of generated summaries
        """
        summaries = []

        # 获取长度 / Get length
        for i in range(0, len(texts), batch_size):
            # Create batch and process each text in the batch
            batch = texts[i:i + batch_size]
            processed_batch = [self.preprocess_text(text) for text in batch]

            # Tokenize batch
            inputs = self.tokenizer(processed_batch, max_length=1024, truncation=True,
                                    padding=True, return_tensors="pt"
                                    ).to(self.device)
            # Generate summaries for batch
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **kwargs
            )
            # Decode summaries
            summaries.extend([self.tokenizer.decode(ids, skip_special_tokens=True)
                              for ids in summary_ids])
        return summaries

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30,
                  length_penalty: float = 2.0, repetition_penalty: float = 2.0,
                  num_beams: int = 4, early_stopping: bool = True) -> dict[str, str]:
        """Generate a summary with advanced features.

        Args:
            text (str): The text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            length_penalty (float): Penalty for longer summaries
            repetition_penalty (float): Penalty for repeated tokens
            num_beams (int): Number of beams for beam search
            early_stopping (bool): Whether to stop when all beams are finished

        Returns:
            dict[str, str]: Dictionary containing original and summarized text
        """
        # Preprocess the text
        cleaned_text = self.preprocess_text(text)

        # Handle long texts
        chunks = self.split_long_text(cleaned_text)
        chunk_summaries = []
        for chunk in chunks:
            summary = self.cached_summarize(
                chunk,
                max_length=max_length // len(chunks),  # Adjust length for chunks
                # 获取长度 / Get length
                min_length=min_length // len(chunks),
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
                early_stopping=early_stopping
            )
            # 添加元素到列表末尾 / Append element to list end
            chunk_summaries.append(summary)

        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "summary": " ".join(chunk_summaries)
        }

# Initialize the advanced summarizer with caching enabled and quantization
adv_summarizer = AdvancedTextSummarizer(quantize=True)

# Sample text
long_text = """
The development of artificial intelligence (AI) has significantly impacted various
industries worldwide. From healthcare to finance, AI-powered applications have
streamlined operations, improved accuracy, and unlocked new possibilities. In healthcare,
AI assists in diagnostics, personalized treatment plans, and drug discovery. In finance,
it aids in fraud detection, algorithmic trading, and customer service. Despite its
benefits, AI raises concerns about data privacy, ethical implications, and job
displacement.
"""

# Generate a summary with default settings
adv_summary = adv_summarizer.summarize(long_text)
# 打印输出 / Print output
print("Advanced Summary:")
# 打印输出 / Print output
pprint.pprint(adv_summary)

# Batch summarization
texts = [
  "AI is revolutionizing healthcare with better diagnostics and personalized treatments.",
  "Self-driving cars are powered by machine learning algorithms that continuously " \
      "learn from traffic patterns.",
  "Natural language processing helps computers understand and communicate with humans " \
    "more effectively.",
  "Climate change is being studied using AI models to predict future environmental " \
    "patterns."
]

# Summarize multiple texts in a batch
batch_summaries = adv_summarizer.summarize_batch(texts, batch_size=2)
# 打印输出 / Print output
print("\nBatch Summaries:")
# 同时获取索引和值 / Get both index and value
for i, s in enumerate(batch_summaries, 1):
    # 打印输出 / Print output
    pprint.pprint(f"{i}. {s}")
```

---

### Chapter Summary / 章节总结

# Chapter 08 Summary / 第08章总结

## Theme / 主题: Chapter 08 / Chapter 08

This chapter contains **3 code files** demonstrating chapter 08.

本章包含 **3 个代码文件**，演示Chapter 08。

---
## Evolution / 演化路线

  1. `01_basic.ipynb` — Basic
  2. `05_length.ipynb` — Length
  3. `07_complete.ipynb` — Complete

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 08) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 08）是机器学习流水线中的基础构建块。

---
