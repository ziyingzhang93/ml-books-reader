# 从零构建Transformer / Building Transformers from Scratch
## Chapter 03

---

### Whitespace

# 01 — Whitespace / 01 Whitespace

**Chapter 03 — File 1 of 12 / 第03章 — 第1个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Whitespace**.

本脚本演示 **01 Whitespace**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
text = "Hello, world! This is a test."
tokens = text.split()
# 打印输出 / Print output
print(f"Tokens: {tokens}")
```

---
## Learning Notes / 学习笔记

- **概念**: Whitespace 是机器学习中的常用技术。  
  *Whitespace is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Whitespace / 01 Whitespace
# Complete Code / 完整代码
# ===============================

text = "Hello, world! This is a test."
tokens = text.split()
# 打印输出 / Print output
print(f"Tokens: {tokens}")
```

---

➡️ **Next / 下一步**: File 2 of 12

---

### Regex

# 02 — Regex / 02 Regex

**Chapter 03 — File 2 of 12 / 第03章 — 第2个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Regex**.

本脚本演示 **02 Regex**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入正则表达式模块 / Import regex module
import re

text = "Hello, world! This is a test."
tokens = re.findall(r'\w+|[^\w\s]', text)
# 打印输出 / Print output
print(f"Tokens: {tokens}")
```

---
## Learning Notes / 学习笔记

- **概念**: Regex 是机器学习中的常用技术。  
  *Regex is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Regex / 02 Regex
# Complete Code / 完整代码
# ===============================

# 导入正则表达式模块 / Import regex module
import re

text = "Hello, world! This is a test."
tokens = re.findall(r'\w+|[^\w\s]', text)
# 打印输出 / Print output
print(f"Tokens: {tokens}")
```

---

➡️ **Next / 下一步**: File 3 of 12

---

### Lower

# 03 — Lower / 03 Lower

**Chapter 03 — File 3 of 12 / 第03章 — 第3个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Lower**.

本脚本演示 **03 Lower**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入正则表达式模块 / Import regex module
import re

text = "Hello, world! This is a test."
tokens = re.findall(r'\w+|[^\w\s]', text.lower())
# 打印输出 / Print output
print(f"Tokens: {tokens}")
```

---
## Learning Notes / 学习笔记

- **概念**: Lower 是机器学习中的常用技术。  
  *Lower is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lower / 03 Lower
# Complete Code / 完整代码
# ===============================

# 导入正则表达式模块 / Import regex module
import re

text = "Hello, world! This is a test."
tokens = re.findall(r'\w+|[^\w\s]', text.lower())
# 打印输出 / Print output
print(f"Tokens: {tokens}")
```

---

➡️ **Next / 下一步**: File 4 of 12

---

### Porter

# 04 — Porter / 04 Porter

**Chapter 03 — File 4 of 12 / 第03章 — 第4个文件（共12个）**

---

## Summary / 总结

This script demonstrates **download the necessary resources if haven't done so**.

本脚本演示 **download the necessary resources if haven't done so**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
```

---
## Step 2 — download the necessary resources if haven't done so

```python
nltk.download('punkt_tab')

text = "These models may become unstable quickly if not initialized."
stemmer = PorterStemmer()
words = word_tokenize(text)
stemmed_words = [stemmer.stem(word) for word in words]
# 打印输出 / Print output
print(stemmed_words)
```

---
## Learning Notes / 学习笔记

- **概念**: download the necessary resources if haven't done so 是机器学习中的常用技术。  
  *download the necessary resources if haven't done so is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Porter / 04 Porter
# Complete Code / 完整代码
# ===============================

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# download the necessary resources if haven't done so
nltk.download('punkt_tab')

text = "These models may become unstable quickly if not initialized."
stemmer = PorterStemmer()
words = word_tokenize(text)
stemmed_words = [stemmer.stem(word) for word in words]
# 打印输出 / Print output
print(stemmed_words)
```

---

➡️ **Next / 下一步**: File 5 of 12

---

### Lemmatize

# 05 — Lemmatize / 05 Lemmatize

**Chapter 03 — File 5 of 12 / 第03章 — 第5个文件（共12个）**

---

## Summary / 总结

This script demonstrates **download the necessary resources if haven't done so**.

本脚本演示 **download the necessary resources if haven't done so**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
```

---
## Step 2 — download the necessary resources if haven't done so

```python
nltk.download('wordnet')

text = "These models may become unstable quickly if not initialized."
lemmatizer = WordNetLemmatizer()
words = word_tokenize(text)
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
# 打印输出 / Print output
print(lemmatized_words)
```

---
## Learning Notes / 学习笔记

- **概念**: download the necessary resources if haven't done so 是机器学习中的常用技术。  
  *download the necessary resources if haven't done so is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lemmatize / 05 Lemmatize
# Complete Code / 完整代码
# ===============================

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# download the necessary resources if haven't done so
nltk.download('wordnet')

text = "These models may become unstable quickly if not initialized."
lemmatizer = WordNetLemmatizer()
words = word_tokenize(text)
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
# 打印输出 / Print output
print(lemmatized_words)
```

---

➡️ **Next / 下一步**: File 6 of 12

---

### Gpt2 Bpe

# 06 — Gpt2 Bpe / 06 Gpt2 Bpe

**Chapter 03 — File 6 of 12 / 第03章 — 第6个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Load the GPT-2 tokenizer (which uses BPE)**.

本脚本演示 **Load the GPT-2 tokenizer (which uses BPE)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import GPT2Tokenizer
```

---
## Step 2 — Load the GPT-2 tokenizer (which uses BPE)

```python
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
```

---
## Step 3 — Tokenize a text

```python
text = "Pre-trained models are available."
tokens = tokenizer.encode(text)
# 打印输出 / Print output
print(f"Token IDs: {tokens}")
# 打印输出 / Print output
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
# 打印输出 / Print output
print(f"Decoded: {tokenizer.decode(tokens)}")
```

---
## Learning Notes / 学习笔记

- **概念**: Load the GPT-2 tokenizer (which uses BPE) 是机器学习中的常用技术。  
  *Load the GPT-2 tokenizer (which uses BPE) is a common technique in machine learning.*

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
# Gpt2 Bpe / 06 Gpt2 Bpe
# Complete Code / 完整代码
# ===============================

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import GPT2Tokenizer

# Load the GPT-2 tokenizer (which uses BPE)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tokenize a text
text = "Pre-trained models are available."
tokens = tokenizer.encode(text)
# 打印输出 / Print output
print(f"Token IDs: {tokens}")
# 打印输出 / Print output
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
# 打印输出 / Print output
print(f"Decoded: {tokenizer.decode(tokens)}")
```

---

➡️ **Next / 下一步**: File 7 of 12

---

### Tiktoken

# 07 — Tiktoken / 07 Tiktoken

**Chapter 03 — File 7 of 12 / 第03章 — 第7个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Tiktoken**.

本脚本演示 **07 Tiktoken**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
text = "Pre-trained models are available."
tokens = encoding.encode(text)
# 打印输出 / Print output
print(f"Token IDs: {tokens}")
# 打印输出 / Print output
print(f"Tokens: {[encoding.decode_single_token_bytes(t) for t in tokens]}")
# 打印输出 / Print output
print(f"Decoded: {encoding.decode(tokens)}")
```

---
## Learning Notes / 学习笔记

- **概念**: Tiktoken 是机器学习中的常用技术。  
  *Tiktoken is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Tiktoken / 07 Tiktoken
# Complete Code / 完整代码
# ===============================

import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
text = "Pre-trained models are available."
tokens = encoding.encode(text)
# 打印输出 / Print output
print(f"Token IDs: {tokens}")
# 打印输出 / Print output
print(f"Tokens: {[encoding.decode_single_token_bytes(t) for t in tokens]}")
# 打印输出 / Print output
print(f"Decoded: {encoding.decode(tokens)}")
```

---

➡️ **Next / 下一步**: File 8 of 12

---

### Train Bpe

# 08 — Train Bpe / 08 Train Bpe

**Chapter 03 — File 8 of 12 / 第03章 — 第8个文件（共12个）**

---

## Summary / 总结

This script demonstrates **reload the trained tokenizer**.

本脚本演示 **reload the trained tokenizer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
# 打印输出 / Print output
print(ds)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
# 打印输出 / Print output
print(tokenizer)

tokenizer.train_from_iterator(ds["train"]["text"], trainer)
# 打印输出 / Print output
print(tokenizer)
tokenizer.save("my-tokenizer.json")
```

---
## Step 2 — reload the trained tokenizer

```python
tokenizer = Tokenizer.from_file("my-tokenizer.json")
```

---
## Learning Notes / 学习笔记

- **概念**: reload the trained tokenizer 是机器学习中的常用技术。  
  *reload the trained tokenizer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Bpe / 08 Train Bpe
# Complete Code / 完整代码
# ===============================

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
# 打印输出 / Print output
print(ds)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
# 打印输出 / Print output
print(tokenizer)

tokenizer.train_from_iterator(ds["train"]["text"], trainer)
# 打印输出 / Print output
print(tokenizer)
tokenizer.save("my-tokenizer.json")

# reload the trained tokenizer
tokenizer = Tokenizer.from_file("my-tokenizer.json")
```

---

➡️ **Next / 下一步**: File 9 of 12

---

### Bert Wordpiece

# 09 — Bert Wordpiece / 09 Bert Wordpiece

**Chapter 03 — File 9 of 12 / 第03章 — 第9个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Load the WordPiece tokenizer from BERT**.

本脚本演示 **Load the WordPiece tokenizer from BERT**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import BertTokenizer
```

---
## Step 2 — Load the WordPiece tokenizer from BERT

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```

---
## Step 3 — Tokenize a text

```python
text = "These models are usually initialized with Gaussian random values."
tokens = tokenizer.encode(text)
# 打印输出 / Print output
print(f"Token IDs: {tokens}")
# 打印输出 / Print output
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
# 打印输出 / Print output
print(f"Decoded: {tokenizer.decode(tokens)}")
```

---
## Learning Notes / 学习笔记

- **概念**: Load the WordPiece tokenizer from BERT 是机器学习中的常用技术。  
  *Load the WordPiece tokenizer from BERT is a common technique in machine learning.*

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
# Bert Wordpiece / 09 Bert Wordpiece
# Complete Code / 完整代码
# ===============================

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import BertTokenizer

# Load the WordPiece tokenizer from BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize a text
text = "These models are usually initialized with Gaussian random values."
tokens = tokenizer.encode(text)
# 打印输出 / Print output
print(f"Token IDs: {tokens}")
# 打印输出 / Print output
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
# 打印输出 / Print output
print(f"Decoded: {tokenizer.decode(tokens)}")
```

---

➡️ **Next / 下一步**: File 10 of 12

---

### Train Wordpiece

# 10 — Train Wordpiece / 10 Train Wordpiece

**Chapter 03 — File 10 of 12 / 第03章 — 第10个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Train Wordpiece**.

本脚本演示 **10 Train Wordpiece**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer.train_from_iterator(ds["train"]["text"], trainer)
tokenizer.save("my-tokenizer.json")
```

---
## Learning Notes / 学习笔记

- **概念**: Train Wordpiece 是机器学习中的常用技术。  
  *Train Wordpiece is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Wordpiece / 10 Train Wordpiece
# Complete Code / 完整代码
# ===============================

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer.train_from_iterator(ds["train"]["text"], trainer)
tokenizer.save("my-tokenizer.json")
```

---

➡️ **Next / 下一步**: File 11 of 12

---

### T5 Sentencepiece

# 11 — T5 Sentencepiece / 11 T5 Sentencepiece

**Chapter 03 — File 11 of 12 / 第03章 — 第11个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Load the T5 tokenizer (which uses SentencePiece+Unigram)**.

本脚本演示 **Load the T5 tokenizer (which uses SentencePiece+Unigram)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import T5Tokenizer
```

---
## Step 2 — Load the T5 tokenizer (which uses SentencePiece+Unigram)

```python
tokenizer = T5Tokenizer.from_pretrained("t5-small")
```

---
## Step 3 — Tokenize a text

```python
text = "SentencePiece is a subword tokenizer used in models such as XLNet and T5."
tokens = tokenizer.encode(text)
# 打印输出 / Print output
print(f"Token IDs: {tokens}")
# 打印输出 / Print output
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
# 打印输出 / Print output
print(f"Decoded: {tokenizer.decode(tokens)}")
```

---
## Learning Notes / 学习笔记

- **概念**: Load the T5 tokenizer (which uses SentencePiece+Unigram) 是机器学习中的常用技术。  
  *Load the T5 tokenizer (which uses SentencePiece+Unigram) is a common technique in machine learning.*

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
# T5 Sentencepiece / 11 T5 Sentencepiece
# Complete Code / 完整代码
# ===============================

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import T5Tokenizer

# Load the T5 tokenizer (which uses SentencePiece+Unigram)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Tokenize a text
text = "SentencePiece is a subword tokenizer used in models such as XLNet and T5."
tokens = tokenizer.encode(text)
# 打印输出 / Print output
print(f"Token IDs: {tokens}")
# 打印输出 / Print output
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
# 打印输出 / Print output
print(f"Decoded: {tokenizer.decode(tokens)}")
```

---

➡️ **Next / 下一步**: File 12 of 12

---

### Train Sentencepiece

# 12 — Train Sentencepiece / 12 Train Sentencepiece

**Chapter 03 — File 12 of 12 / 第03章 — 第12个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Train Sentencepiece**.

本脚本演示 **12 Train Sentencepiece**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
from datasets import load_dataset
from tokenizers import SentencePieceUnigramTokenizer

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
tokenizer = SentencePieceUnigramTokenizer()

tokenizer.train_from_iterator(ds["train"]["text"])
tokenizer.save("my-tokenizer.json")
```

---
## Learning Notes / 学习笔记

- **概念**: Train Sentencepiece 是机器学习中的常用技术。  
  *Train Sentencepiece is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Sentencepiece / 12 Train Sentencepiece
# Complete Code / 完整代码
# ===============================

from datasets import load_dataset
from tokenizers import SentencePieceUnigramTokenizer

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
tokenizer = SentencePieceUnigramTokenizer()

tokenizer.train_from_iterator(ds["train"]["text"])
tokenizer.save("my-tokenizer.json")
```

---

### Chapter Summary / 章节总结

# Chapter 03 Summary / 第03章总结

## Theme / 主题: Chapter 03 / Chapter 03

This chapter contains **12 code files** demonstrating chapter 03.

本章包含 **12 个代码文件**，演示Chapter 03。

---
## Evolution / 演化路线

  1. `01_whitespace.ipynb` — Whitespace
  2. `02_regex.ipynb` — Regex
  3. `03_lower.ipynb` — Lower
  4. `04_porter.ipynb` — Porter
  5. `05_lemmatize.ipynb` — Lemmatize
  6. `06_gpt2_bpe.ipynb` — Gpt2 Bpe
  7. `07_tiktoken.ipynb` — Tiktoken
  8. `08_train_bpe.ipynb` — Train Bpe
  9. `09_bert_wordpiece.ipynb` — Bert Wordpiece
  10. `10_train_wordpiece.ipynb` — Train Wordpiece
  11. `11_t5_sentencepiece.ipynb` — T5 Sentencepiece
  12. `12_train_sentencepiece.ipynb` — Train Sentencepiece

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 03) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 03）是机器学习流水线中的基础构建块。

---
