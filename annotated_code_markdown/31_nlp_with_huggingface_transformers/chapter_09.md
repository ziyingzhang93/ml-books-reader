# HuggingFace Transformers NLP / NLP with HF Transformers
## Chapter 09

---

### Arch

# 01 — Arch / 01 Arch

**Chapter 09 — File 1 of 6 / 第09章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Load model configuration**.

本脚本演示 **Load model configuration**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — Step 1

```python
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoConfig, AutoModelForSeq2SeqLM

def explore_model_architecture():
    """Examine DistilBart's configuration and architecture."""
    model_name = "sshleifer/distilbart-cnn-12-6"
```

---
## Step 2 — Load model configuration

```python
config = AutoConfig.from_pretrained(model_name)
    # 打印输出 / Print output
    print("Model Architecture:")
    # 打印输出 / Print output
    print(f"- Encoder layers: {config.encoder_layers}")
    # 打印输出 / Print output
    print(f"- Decoder layers: {config.decoder_layers}")
    # 打印输出 / Print output
    print(f"- Hidden size: {config.hidden_size}")
    # 打印输出 / Print output
    print(f"- Attention heads: {config.encoder_attention_heads}")
```

---
## Step 3 — Verify encoder-decoder structure

```python
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # 打印输出 / Print output
    print("\nModel Components:")
    # 打印输出 / Print output
    print(f"- Encoder: {type(model.model.encoder).__name__}")
    # 打印输出 / Print output
    print(f"- Decoder: {type(model.model.decoder).__name__}")
    return model, config
```

---
## Step 4 — Example usage

```python
model, config = explore_model_architecture()
```

---
## Learning Notes / 学习笔记

- **概念**: Load model configuration 是机器学习中的常用技术。  
  *Load model configuration is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Arch / 01 Arch
# Complete Code / 完整代码
# ===============================

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoConfig, AutoModelForSeq2SeqLM

def explore_model_architecture():
    """Examine DistilBart's configuration and architecture."""
    model_name = "sshleifer/distilbart-cnn-12-6"

    # Load model configuration
    config = AutoConfig.from_pretrained(model_name)
    # 打印输出 / Print output
    print("Model Architecture:")
    # 打印输出 / Print output
    print(f"- Encoder layers: {config.encoder_layers}")
    # 打印输出 / Print output
    print(f"- Decoder layers: {config.decoder_layers}")
    # 打印输出 / Print output
    print(f"- Hidden size: {config.hidden_size}")
    # 打印输出 / Print output
    print(f"- Attention heads: {config.encoder_attention_heads}")

    # Verify encoder-decoder structure
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # 打印输出 / Print output
    print("\nModel Components:")
    # 打印输出 / Print output
    print(f"- Encoder: {type(model.model.encoder).__name__}")
    # 打印输出 / Print output
    print(f"- Decoder: {type(model.model.decoder).__name__}")
    return model, config

# Example usage
model, config = explore_model_architecture()
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Print

# 02 — Print / 02 Print

**Chapter 09 — File 2 of 6 / 第09章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Load model configuration**.

本脚本演示 **Load model configuration**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — Step 1

```python
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoConfig, AutoModelForSeq2SeqLM

def explore_model_architecture():
    """Examine DistilBart's configuration and architecture."""
    model_name = "sshleifer/distilbart-cnn-12-6"
```

---
## Step 2 — Load model configuration

```python
config = AutoConfig.from_pretrained(model_name)
    # 打印输出 / Print output
    print("Model Architecture:")
    # 打印输出 / Print output
    print(f"- Encoder layers: {config.encoder_layers}")
    # 打印输出 / Print output
    print(f"- Decoder layers: {config.decoder_layers}")
    # 打印输出 / Print output
    print(f"- Hidden size: {config.hidden_size}")
    # 打印输出 / Print output
    print(f"- Attention heads: {config.encoder_attention_heads}")
```

---
## Step 3 — Verify encoder-decoder structure

```python
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # 打印输出 / Print output
    print("\nModel Components:")
    # 打印输出 / Print output
    print(f"- Encoder: {type(model.model.encoder).__name__}")
    # 打印输出 / Print output
    print(f"- Decoder: {type(model.model.decoder).__name__}")
    return model, config
```

---
## Step 4 — Example usage

```python
model, config = explore_model_architecture()
# 打印输出 / Print output
print(model)
```

---
## Learning Notes / 学习笔记

- **概念**: Load model configuration 是机器学习中的常用技术。  
  *Load model configuration is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Print / 02 Print
# Complete Code / 完整代码
# ===============================

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoConfig, AutoModelForSeq2SeqLM

def explore_model_architecture():
    """Examine DistilBart's configuration and architecture."""
    model_name = "sshleifer/distilbart-cnn-12-6"

    # Load model configuration
    config = AutoConfig.from_pretrained(model_name)
    # 打印输出 / Print output
    print("Model Architecture:")
    # 打印输出 / Print output
    print(f"- Encoder layers: {config.encoder_layers}")
    # 打印输出 / Print output
    print(f"- Decoder layers: {config.decoder_layers}")
    # 打印输出 / Print output
    print(f"- Hidden size: {config.hidden_size}")
    # 打印输出 / Print output
    print(f"- Attention heads: {config.encoder_attention_heads}")

    # Verify encoder-decoder structure
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # 打印输出 / Print output
    print("\nModel Components:")
    # 打印输出 / Print output
    print(f"- Encoder: {type(model.model.encoder).__name__}")
    # 打印输出 / Print output
    print(f"- Decoder: {type(model.model.decoder).__name__}")
    return model, config

# Example usage
model, config = explore_model_architecture()
# 打印输出 / Print output
print(model)
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Generate

# 03 — Generate / 03 Generate

**Chapter 09 — File 3 of 6 / 第09章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Generate summary using only the input tokens**.

本脚本演示 **Generate summary using only the input tokens**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """Initialize the summarizer with model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def summarize(self, text, context_weight=0.5, max_length=150, min_length=50,
                  num_beams=4, length_penalty=2.0, repetition_penalty=1.0,
                  do_sample=False, temperature=1.0, early_stopping=True):
        """Generate a summary with context awareness."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024
                                ).to(self.device)
```

---
## Step 2 — Generate summary using only the input tokens

```python
summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            temperature=temperature,
            early_stopping=early_stopping,
        )
```

---
## Step 3 — Decode and return the summary

```python
summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
```

---
## Step 4 — Let's run an example to see how it works

```python
summarizer = Summarizer()
text = """
The development of artificial intelligence has revolutionized numerous industries.
Machine learning algorithms now power everything from recommendation systems to
autonomous vehicles. Deep learning, in particular, has shown remarkable success
in tasks like image recognition and natural language processing. However, these
advances also raise important ethical considerations about AI's impact on society,
privacy, and employment.
"""

summary = summarizer.summarize(text)
# 打印输出 / Print output
print(f"Summary:\n{summary}")
```

---
## Learning Notes / 学习笔记

- **概念**: Generate summary using only the input tokens 是机器学习中的常用技术。  
  *Generate summary using only the input tokens is a common technique in machine learning.*

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
# Generate / 03 Generate
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """Initialize the summarizer with model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def summarize(self, text, context_weight=0.5, max_length=150, min_length=50,
                  num_beams=4, length_penalty=2.0, repetition_penalty=1.0,
                  do_sample=False, temperature=1.0, early_stopping=True):
        """Generate a summary with context awareness."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024
                                ).to(self.device)
        # Generate summary using only the input tokens
        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            temperature=temperature,
            early_stopping=early_stopping,
        )
        # Decode and return the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

# Let's run an example to see how it works
summarizer = Summarizer()
text = """
The development of artificial intelligence has revolutionized numerous industries.
Machine learning algorithms now power everything from recommendation systems to
autonomous vehicles. Deep learning, in particular, has shown remarkable success
in tasks like image recognition and natural language processing. However, these
advances also raise important ethical considerations about AI's impact on society,
privacy, and employment.
"""

summary = summarizer.summarize(text)
# 打印输出 / Print output
print(f"Summary:\n{summary}")
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Styles

# 04 — Styles / 04 Styles

**Chapter 09 — File 4 of 6 / 第09章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Generate summary using only the input tokens**.

本脚本演示 **Generate summary using only the input tokens**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """Initialize the summarizer with model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def summarize(self, text, context_weight=0.5, max_length=150, min_length=50,
                  num_beams=4, length_penalty=2.0, repetition_penalty=1.0,
                  do_sample=False, temperature=1.0, early_stopping=True):
        """Generate a summary with context awareness."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024
                                ).to(self.device)
```

---
## Step 2 — Generate summary using only the input tokens

```python
summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            temperature=temperature,
            early_stopping=early_stopping,
        )
```

---
## Step 3 — Decode and return the summary

```python
summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

class StyleControlledSummarizer(Summarizer):
    def summarize_with_style(self, text, style="concise"):
        """Generate summaries with different styles.

        Args:
            text (str): Input text to summarize
            style (str): Summary style ('concise', 'detailed', 'technical', 'simple')
        Returns:
            str: Generated summary with specified style
        """
        style_params = {
            "concise": {
                "max_length": 80,
                "min_length": 30,
                "length_penalty": 3.0,
                "num_beams": 4,
                "early_stopping": True
            },
            "detailed": {
                "max_length": 200,
                "min_length": 100,
                "length_penalty": 1.0,
                "num_beams": 6,
                "early_stopping": False
            },
            "technical": {
                "max_length": 150,
                "min_length": 50,
                "length_penalty": 2.0,
                "num_beams": 5,
                "repetition_penalty": 1.5
            },
            "simple": {
                "max_length": 100,
                "min_length": 30,
                "length_penalty": 2.0,
                "num_beams": 3,
                "do_sample": True,
                "temperature": 0.7
            }
        }
        params = style_params[style]
        return self.summarize(text, **params)
```

---
## Step 4 — Let's run an example to see how it works

```python
style_summarizer = StyleControlledSummarizer()
text = """
Quantum computing leverages the principles of quantum mechanics to perform
computations. Unlike classical computers that use bits, quantum computers
use quantum bits or qubits. These qubits can exist in multiple states
simultaneously through superposition, potentially allowing quantum computers
to solve certain problems exponentially faster than classical computers.
However, maintaining quantum coherence and minimizing errors remains a
significant challenge in building practical quantum computers.
"""

styles = ["concise", "detailed", "technical", "simple"]
for style in styles:
    summary = style_summarizer.summarize_with_style(text, style=style)
    # 打印输出 / Print output
    print(f"\n{style.capitalize()} Summary:")
    # 打印输出 / Print output
    print(summary)
```

---
## Learning Notes / 学习笔记

- **概念**: Generate summary using only the input tokens 是机器学习中的常用技术。  
  *Generate summary using only the input tokens is a common technique in machine learning.*

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
# Styles / 04 Styles
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """Initialize the summarizer with model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def summarize(self, text, context_weight=0.5, max_length=150, min_length=50,
                  num_beams=4, length_penalty=2.0, repetition_penalty=1.0,
                  do_sample=False, temperature=1.0, early_stopping=True):
        """Generate a summary with context awareness."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024
                                ).to(self.device)
        # Generate summary using only the input tokens
        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            temperature=temperature,
            early_stopping=early_stopping,
        )
        # Decode and return the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

class StyleControlledSummarizer(Summarizer):
    def summarize_with_style(self, text, style="concise"):
        """Generate summaries with different styles.

        Args:
            text (str): Input text to summarize
            style (str): Summary style ('concise', 'detailed', 'technical', 'simple')
        Returns:
            str: Generated summary with specified style
        """
        style_params = {
            "concise": {
                "max_length": 80,
                "min_length": 30,
                "length_penalty": 3.0,
                "num_beams": 4,
                "early_stopping": True
            },
            "detailed": {
                "max_length": 200,
                "min_length": 100,
                "length_penalty": 1.0,
                "num_beams": 6,
                "early_stopping": False
            },
            "technical": {
                "max_length": 150,
                "min_length": 50,
                "length_penalty": 2.0,
                "num_beams": 5,
                "repetition_penalty": 1.5
            },
            "simple": {
                "max_length": 100,
                "min_length": 30,
                "length_penalty": 2.0,
                "num_beams": 3,
                "do_sample": True,
                "temperature": 0.7
            }
        }
        params = style_params[style]
        return self.summarize(text, **params)

# Let's run an example to see how it works
style_summarizer = StyleControlledSummarizer()
text = """
Quantum computing leverages the principles of quantum mechanics to perform
computations. Unlike classical computers that use bits, quantum computers
use quantum bits or qubits. These qubits can exist in multiple states
simultaneously through superposition, potentially allowing quantum computers
to solve certain problems exponentially faster than classical computers.
However, maintaining quantum coherence and minimizing errors remains a
significant challenge in building practical quantum computers.
"""

styles = ["concise", "detailed", "technical", "simple"]
for style in styles:
    summary = style_summarizer.summarize_with_style(text, style=style)
    # 打印输出 / Print output
    print(f"\n{style.capitalize()} Summary:")
    # 打印输出 / Print output
    print(summary)
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Eval

# 05 — Eval / 模型评估

**Chapter 09 — File 5 of 6 / 第09章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Generate summary using only the input tokens**.

本脚本演示 **Generate summary using only the input tokens**。

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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

class Summarizer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """Initialize the summarizer with model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def summarize(self, text, context_weight=0.5, max_length=150, min_length=50,
                  num_beams=4, length_penalty=2.0, repetition_penalty=1.0,
                  do_sample=False, temperature=1.0, early_stopping=True):
        """Generate a summary with context awareness."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024
                                ).to(self.device)
```

---
## Step 2 — Generate summary using only the input tokens

```python
summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            temperature=temperature,
            early_stopping=early_stopping,
        )
```

---
## Step 3 — Decode and return the summary

```python
summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

class StyleControlledSummarizer(Summarizer):
    def summarize_with_style(self, text, style="concise"):
        """Generate summaries with different styles.

        Args:
            text (str): Input text to summarize
            style (str): Summary style ('concise', 'detailed', 'technical', 'simple')
        Returns:
            str: Generated summary with specified style
        """
        style_params = {
            "concise": {
                "max_length": 80,
                "min_length": 30,
                "length_penalty": 3.0,
                "num_beams": 4,
                "early_stopping": True
            },
            "detailed": {
                "max_length": 200,
                "min_length": 100,
                "length_penalty": 1.0,
                "num_beams": 6,
                "early_stopping": False
            },
            "technical": {
                "max_length": 150,
                "min_length": 50,
                "length_penalty": 2.0,
                "num_beams": 5,
                "repetition_penalty": 1.5
            },
            "simple": {
                "max_length": 100,
                "min_length": 30,
                "length_penalty": 2.0,
                "num_beams": 3,
                "do_sample": True,
                "temperature": 0.7
            }
        }
        params = style_params[style]
        return self.summarize(text, **params)

class SummaryEvaluator:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        """Initialize with ROUGE metrics."""
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    def evaluate_summary(self, reference, candidate):
        """Calculate ROUGE scores for a summary.

        Args:
            reference (str): Reference summary
            candidate (str): Generated summary

        Returns:
            dict: ROUGE scores for different metrics
        """
        scores = self.scorer.score(reference, candidate)

        # 打印输出 / Print output
        print("Summary Quality Metrics:")
        # 打印输出 / Print output
        print(f"ROUGE-1: {scores['rouge1'].fmeasure:.3f}")
        # 打印输出 / Print output
        print(f"ROUGE-2: {scores['rouge2'].fmeasure:.3f}")
        # 打印输出 / Print output
        print(f"ROUGE-L: {scores['rougeL'].fmeasure:.3f}")

        return scores

text = """
Quantum computing leverages the principles of quantum mechanics to perform
computations. Unlike classical computers that use bits, quantum computers
use quantum bits or qubits. These qubits can exist in multiple states
simultaneously through superposition, potentially allowing quantum computers
to solve certain problems exponentially faster than classical computers.
However, maintaining quantum coherence and minimizing errors remains a
significant challenge in building practical quantum computers.
"""
```

---
## Step 4 — Checking the metrics implementation

```python
summarizer = StyleControlledSummarizer()
evaluator = SummaryEvaluator()
reference = "Quantum computing uses qubits for faster computation but faces " \
            "coherence challenges."
for style in ["concise", "detailed", "technical", "simple"]:
    candidate = summarizer.summarize_with_style(text, style=style)
    scores = evaluator.evaluate_summary(reference, candidate)
```

---
## Learning Notes / 学习笔记

- **概念**: Generate summary using only the input tokens 是机器学习中的常用技术。  
  *Generate summary using only the input tokens is a common technique in machine learning.*

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
# Eval / 模型评估
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

class Summarizer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """Initialize the summarizer with model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def summarize(self, text, context_weight=0.5, max_length=150, min_length=50,
                  num_beams=4, length_penalty=2.0, repetition_penalty=1.0,
                  do_sample=False, temperature=1.0, early_stopping=True):
        """Generate a summary with context awareness."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024
                                ).to(self.device)
        # Generate summary using only the input tokens
        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            temperature=temperature,
            early_stopping=early_stopping,
        )
        # Decode and return the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

class StyleControlledSummarizer(Summarizer):
    def summarize_with_style(self, text, style="concise"):
        """Generate summaries with different styles.

        Args:
            text (str): Input text to summarize
            style (str): Summary style ('concise', 'detailed', 'technical', 'simple')
        Returns:
            str: Generated summary with specified style
        """
        style_params = {
            "concise": {
                "max_length": 80,
                "min_length": 30,
                "length_penalty": 3.0,
                "num_beams": 4,
                "early_stopping": True
            },
            "detailed": {
                "max_length": 200,
                "min_length": 100,
                "length_penalty": 1.0,
                "num_beams": 6,
                "early_stopping": False
            },
            "technical": {
                "max_length": 150,
                "min_length": 50,
                "length_penalty": 2.0,
                "num_beams": 5,
                "repetition_penalty": 1.5
            },
            "simple": {
                "max_length": 100,
                "min_length": 30,
                "length_penalty": 2.0,
                "num_beams": 3,
                "do_sample": True,
                "temperature": 0.7
            }
        }
        params = style_params[style]
        return self.summarize(text, **params)

class SummaryEvaluator:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        """Initialize with ROUGE metrics."""
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    def evaluate_summary(self, reference, candidate):
        """Calculate ROUGE scores for a summary.

        Args:
            reference (str): Reference summary
            candidate (str): Generated summary

        Returns:
            dict: ROUGE scores for different metrics
        """
        scores = self.scorer.score(reference, candidate)

        # 打印输出 / Print output
        print("Summary Quality Metrics:")
        # 打印输出 / Print output
        print(f"ROUGE-1: {scores['rouge1'].fmeasure:.3f}")
        # 打印输出 / Print output
        print(f"ROUGE-2: {scores['rouge2'].fmeasure:.3f}")
        # 打印输出 / Print output
        print(f"ROUGE-L: {scores['rougeL'].fmeasure:.3f}")

        return scores

text = """
Quantum computing leverages the principles of quantum mechanics to perform
computations. Unlike classical computers that use bits, quantum computers
use quantum bits or qubits. These qubits can exist in multiple states
simultaneously through superposition, potentially allowing quantum computers
to solve certain problems exponentially faster than classical computers.
However, maintaining quantum coherence and minimizing errors remains a
significant challenge in building practical quantum computers.
"""

# Checking the metrics implementation
summarizer = StyleControlledSummarizer()
evaluator = SummaryEvaluator()
reference = "Quantum computing uses qubits for faster computation but faces " \
            "coherence challenges."
for style in ["concise", "detailed", "technical", "simple"]:
    candidate = summarizer.summarize_with_style(text, style=style)
    scores = evaluator.evaluate_summary(reference, candidate)
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Complete

# 07 — Complete / 07 Complete

**Chapter 09 — File 6 of 6 / 第09章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Generate summary using only the input tokens**.

本脚本演示 **Generate summary using only the input tokens**。

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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
from rouge_score import rouge_scorer
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """Initialize the summarizer with model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def summarize(self, text, context_weight=0.5, max_length=150, min_length=50,
                  num_beams=4, length_penalty=2.0, repetition_penalty=1.0,
                  do_sample=False, temperature=1.0, early_stopping=True):
        """Generate a summary with context awareness."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024
                                ).to(self.device)
```

---
## Step 2 — Generate summary using only the input tokens

```python
summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            temperature=temperature,
            early_stopping=early_stopping,
        )
```

---
## Step 3 — Decode and return the summary

```python
summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

class StyleControlledSummarizer(Summarizer):
    def summarize_with_style(self, text, style="concise"):
        """Generate summaries with different styles.

        Args:
            text (str): Input text to summarize
            style (str): Summary style ('concise', 'detailed', 'technical', 'simple')
        Returns:
            str: Generated summary with specified style
        """
        style_params = {
            "concise": {
                "max_length": 80,
                "min_length": 30,
                "length_penalty": 3.0,
                "num_beams": 4,
                "early_stopping": True
            },
            "detailed": {
                "max_length": 200,
                "min_length": 100,
                "length_penalty": 1.0,
                "num_beams": 6,
                "early_stopping": False
            },
            "technical": {
                "max_length": 150,
                "min_length": 50,
                "length_penalty": 2.0,
                "num_beams": 5,
                "repetition_penalty": 1.5
            },
            "simple": {
                "max_length": 100,
                "min_length": 30,
                "length_penalty": 2.0,
                "num_beams": 3,
                "do_sample": True,
                "temperature": 0.7
            }
        }
        params = style_params[style]
        return self.summarize(text, **params)

class SummaryEvaluator:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        """Initialize with ROUGE metrics."""
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    def evaluate_summary(self, reference, candidate):
        """Calculate ROUGE scores for a summary.

        Args:
            reference (str): Reference summary
            candidate (str): Generated summary

        Returns:
            dict: ROUGE scores for different metrics
        """
        scores = self.scorer.score(reference, candidate)

        # 打印输出 / Print output
        print("Summary Quality Metrics:")
        # 打印输出 / Print output
        print(f"ROUGE-1: {scores['rouge1'].fmeasure:.3f}")
        # 打印输出 / Print output
        print(f"ROUGE-2: {scores['rouge2'].fmeasure:.3f}")
        # 打印输出 / Print output
        print(f"ROUGE-L: {scores['rougeL'].fmeasure:.3f}")

        return scores
```

---
## Step 4 — Checking the metrics implementation

```python
summarizer = StyleControlledSummarizer()
evaluator = SummaryEvaluator()
text = """
Quantum computing leverages the principles of quantum mechanics to perform
computations. Unlike classical computers that use bits, quantum computers
use quantum bits or qubits. These qubits can exist in multiple states
simultaneously through superposition, potentially allowing quantum computers
to solve certain problems exponentially faster than classical computers.
However, maintaining quantum coherence and minimizing errors remains a
significant challenge in building practical quantum computers.
"""
reference = "Quantum computing uses qubits for faster computation but faces coherence challenges."
for style in ["concise", "detailed", "technical", "simple"]:
    summary = summarizer.summarize_with_style(text, style=style)
    # 打印输出 / Print output
    print(f"\n{style.capitalize()} Summary:")
    # 打印输出 / Print output
    print(summary)
    scores = evaluator.evaluate_summary(reference, summary)
```

---
## Learning Notes / 学习笔记

- **概念**: Generate summary using only the input tokens 是机器学习中的常用技术。  
  *Generate summary using only the input tokens is a common technique in machine learning.*

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
# Complete / 07 Complete
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
from rouge_score import rouge_scorer
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """Initialize the summarizer with model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def summarize(self, text, context_weight=0.5, max_length=150, min_length=50,
                  num_beams=4, length_penalty=2.0, repetition_penalty=1.0,
                  do_sample=False, temperature=1.0, early_stopping=True):
        """Generate a summary with context awareness."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024
                                ).to(self.device)
        # Generate summary using only the input tokens
        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            temperature=temperature,
            early_stopping=early_stopping,
        )
        # Decode and return the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

class StyleControlledSummarizer(Summarizer):
    def summarize_with_style(self, text, style="concise"):
        """Generate summaries with different styles.

        Args:
            text (str): Input text to summarize
            style (str): Summary style ('concise', 'detailed', 'technical', 'simple')
        Returns:
            str: Generated summary with specified style
        """
        style_params = {
            "concise": {
                "max_length": 80,
                "min_length": 30,
                "length_penalty": 3.0,
                "num_beams": 4,
                "early_stopping": True
            },
            "detailed": {
                "max_length": 200,
                "min_length": 100,
                "length_penalty": 1.0,
                "num_beams": 6,
                "early_stopping": False
            },
            "technical": {
                "max_length": 150,
                "min_length": 50,
                "length_penalty": 2.0,
                "num_beams": 5,
                "repetition_penalty": 1.5
            },
            "simple": {
                "max_length": 100,
                "min_length": 30,
                "length_penalty": 2.0,
                "num_beams": 3,
                "do_sample": True,
                "temperature": 0.7
            }
        }
        params = style_params[style]
        return self.summarize(text, **params)

class SummaryEvaluator:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        """Initialize with ROUGE metrics."""
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    def evaluate_summary(self, reference, candidate):
        """Calculate ROUGE scores for a summary.

        Args:
            reference (str): Reference summary
            candidate (str): Generated summary

        Returns:
            dict: ROUGE scores for different metrics
        """
        scores = self.scorer.score(reference, candidate)

        # 打印输出 / Print output
        print("Summary Quality Metrics:")
        # 打印输出 / Print output
        print(f"ROUGE-1: {scores['rouge1'].fmeasure:.3f}")
        # 打印输出 / Print output
        print(f"ROUGE-2: {scores['rouge2'].fmeasure:.3f}")
        # 打印输出 / Print output
        print(f"ROUGE-L: {scores['rougeL'].fmeasure:.3f}")

        return scores

# Checking the metrics implementation
summarizer = StyleControlledSummarizer()
evaluator = SummaryEvaluator()
text = """
Quantum computing leverages the principles of quantum mechanics to perform
computations. Unlike classical computers that use bits, quantum computers
use quantum bits or qubits. These qubits can exist in multiple states
simultaneously through superposition, potentially allowing quantum computers
to solve certain problems exponentially faster than classical computers.
However, maintaining quantum coherence and minimizing errors remains a
significant challenge in building practical quantum computers.
"""
reference = "Quantum computing uses qubits for faster computation but faces coherence challenges."
for style in ["concise", "detailed", "technical", "simple"]:
    summary = summarizer.summarize_with_style(text, style=style)
    # 打印输出 / Print output
    print(f"\n{style.capitalize()} Summary:")
    # 打印输出 / Print output
    print(summary)
    scores = evaluator.evaluate_summary(reference, summary)
```

---

### Chapter Summary / 章节总结

# Chapter 09 Summary / 第09章总结

## Theme / 主题: Chapter 09 / Chapter 09

This chapter contains **6 code files** demonstrating chapter 09.

本章包含 **6 个代码文件**，演示Chapter 09。

---
## Evolution / 演化路线

  1. `01_arch.ipynb` — Arch
  2. `02_print.ipynb` — Print
  3. `03_generate.ipynb` — Generate
  4. `04_styles.ipynb` — Styles
  5. `05_eval.ipynb` — Eval
  6. `07_complete.ipynb` — Complete

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 09) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 09）是机器学习流水线中的基础构建块。

---
