# 从零构建Transformer
## Chapter 14

---

### Download

# 01 — Download / 01 Download

**Chapter 14 — File 1 of 11 / 第14章 — 第1个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Download**.

本脚本演示 **01 Download**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import os
import requests

if not os.path.exists("fra-eng.zip"):
    url = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
    response = requests.get(url)
    with open("fra-eng.zip", "wb") as f:
        f.write(response.content)
```

---
## Learning Notes / 学习笔记

- **概念**: Download 是机器学习中的常用技术。  
  *Download is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Download / 01 Download
# Complete Code / 完整代码
# ===============================

import os
import requests

if not os.path.exists("fra-eng.zip"):
    url = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
    response = requests.get(url)
    with open("fra-eng.zip", "wb") as f:
        f.write(response.content)
```

---

➡️ **Next / 下一步**: File 2 of 11

---

### Read

# 02 — Read / 02 Read

**Chapter 14 — File 2 of 11 / 第14章 — 第2个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Read**.

本脚本演示 **02 Read**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
import unicodedata
import zipfile

def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()

text_pairs = []
with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
    for line in zip_ref.read("fra.txt").decode("utf-8").splitlines():
        eng, fra = normalize(line)
        text_pairs.append((eng, fra))
```

---
## Learning Notes / 学习笔记

- **概念**: Read 是机器学习中的常用技术。  
  *Read is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Read / 02 Read
# Complete Code / 完整代码
# ===============================

import unicodedata
import zipfile

def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()

text_pairs = []
with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
    for line in zip_ref.read("fra.txt").decode("utf-8").splitlines():
        eng, fra = normalize(line)
        text_pairs.append((eng, fra))
```

---

➡️ **Next / 下一步**: File 3 of 11

---

### Test Tokenizer

# 04 — Test Tokenizer / 分词

**Chapter 14 — File 4 of 11 / 第14章 — 第4个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning**.

本脚本演示 **Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
import random
import unicodedata
import zipfile

import tokenizers

def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()

text_pairs = []
with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
    for line in zip_ref.read("fra.txt").decode("utf-8").splitlines():
        eng, fra = normalize(line)
        text_pairs.append((eng, fra))

en_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
fr_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
```

---
## Step 2 — Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning
of the sentence

```python
en_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
fr_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
```

---
## Step 3 — Configure decoder: So that word boundary symbol "Ġ" will be removed

```python
en_tokenizer.decoder = tokenizers.decoders.ByteLevel()
fr_tokenizer.decoder = tokenizers.decoders.ByteLevel()
```

---
## Step 4 — Train BPE for English and French using the same trainer

```python
VOCAB_SIZE = 8000
trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[start]", "[end]", "[pad]"],
    show_progress=True
)
en_tokenizer.train_from_iterator([x[0] for x in text_pairs], trainer=trainer)
fr_tokenizer.train_from_iterator([x[1] for x in text_pairs], trainer=trainer)

en_tokenizer.enable_padding(pad_id=en_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
fr_tokenizer.enable_padding(pad_id=fr_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
```

---
## Step 5 — Save the trained tokenizers

```python
en_tokenizer.save("en_tokenizer.json", pretty=True)
fr_tokenizer.save("fr_tokenizer.json", pretty=True)
```

---
## Step 6 — Test the tokenizer

```python
print("Sample tokenization:")
en_sample, fr_sample = random.choice(text_pairs)
encoded = en_tokenizer.encode(en_sample)
print(f"Original: {en_sample}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Decoded: {en_tokenizer.decode(encoded.ids)}")
print()

encoded = fr_tokenizer.encode("[start] " + fr_sample + " [end]")
print(f"Original: {fr_sample}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Decoded: {fr_tokenizer.decode(encoded.ids)}")
print()
```

---
## Learning Notes / 学习笔记

- **概念**: Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning 是机器学习中的常用技术。  
  *Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Test Tokenizer / 分词
# Complete Code / 完整代码
# ===============================

import random
import unicodedata
import zipfile

import tokenizers

def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()

text_pairs = []
with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
    for line in zip_ref.read("fra.txt").decode("utf-8").splitlines():
        eng, fra = normalize(line)
        text_pairs.append((eng, fra))

en_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
fr_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())

# Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning
# of the sentence
en_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
fr_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)

# Configure decoder: So that word boundary symbol "Ġ" will be removed
en_tokenizer.decoder = tokenizers.decoders.ByteLevel()
fr_tokenizer.decoder = tokenizers.decoders.ByteLevel()

# Train BPE for English and French using the same trainer
VOCAB_SIZE = 8000
trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[start]", "[end]", "[pad]"],
    show_progress=True
)
en_tokenizer.train_from_iterator([x[0] for x in text_pairs], trainer=trainer)
fr_tokenizer.train_from_iterator([x[1] for x in text_pairs], trainer=trainer)

en_tokenizer.enable_padding(pad_id=en_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
fr_tokenizer.enable_padding(pad_id=fr_tokenizer.token_to_id("[pad]"), pad_token="[pad]")

# Save the trained tokenizers
en_tokenizer.save("en_tokenizer.json", pretty=True)
fr_tokenizer.save("fr_tokenizer.json", pretty=True)

# Test the tokenizer
print("Sample tokenization:")
en_sample, fr_sample = random.choice(text_pairs)
encoded = en_tokenizer.encode(en_sample)
print(f"Original: {en_sample}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Decoded: {en_tokenizer.decode(encoded.ids)}")
print()

encoded = fr_tokenizer.encode("[start] " + fr_sample + " [end]")
print(f"Original: {fr_sample}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Decoded: {fr_tokenizer.decode(encoded.ids)}")
print()
```

---

➡️ **Next / 下一步**: File 5 of 11

---

### Encoder

# 05 — Encoder / 数据编码

**Chapter 14 — File 5 of 11 / 第14章 — 第5个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Encoder**.

本脚本演示 **数据编码**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
import torch.nn as nn

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell
```

---
## Learning Notes / 学习笔记

- **概念**: Encoder 是机器学习中的常用技术。  
  *Encoder is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `nn.Embedding` | 将整数ID映射为稠密向量 | Map integer IDs to dense vectors |
| `nn.LSTM` | 长短期记忆网络，处理序列数据 | Long Short-Term Memory for sequences |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Encoder / 数据编码
# Complete Code / 完整代码
# ===============================

import torch.nn as nn

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell
```

---

➡️ **Next / 下一步**: File 6 of 11

---

### Model

# 07 — Model / 07 Model

**Chapter 14 — File 7 of 11 / 第14章 — 第7个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Model**.

本脚本演示 **07 Model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
import torch
import torch.nn as nn

class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        batch_size, target_len = target_seq.shape
        outputs = []
        _enc_out, hidden, cell = self.encoder(input_seq)
        dec_in = target_seq[:, :1]
        for t in range(target_len-1):
            pred, hidden, cell = self.decoder(dec_in, hidden, cell)
            pred = pred[:, -1:, :]
            outputs.append(pred)
            dec_in = torch.cat([dec_in, pred.argmax(dim=2)], dim=1)
        outputs = torch.cat(outputs, dim=1)
        return outputs
```

---
## Learning Notes / 学习笔记

- **概念**: Model 是机器学习中的常用技术。  
  *Model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model / 07 Model
# Complete Code / 完整代码
# ===============================

import torch
import torch.nn as nn

class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        batch_size, target_len = target_seq.shape
        outputs = []
        _enc_out, hidden, cell = self.encoder(input_seq)
        dec_in = target_seq[:, :1]
        for t in range(target_len-1):
            pred, hidden, cell = self.decoder(dec_in, hidden, cell)
            pred = pred[:, -1:, :]
            outputs.append(pred)
            dec_in = torch.cat([dec_in, pred.argmax(dim=2)], dim=1)
        outputs = torch.cat(outputs, dim=1)
        return outputs
```

---

➡️ **Next / 下一步**: File 8 of 11

---

### Dataset

# 08 — Dataset / 08 Dataset

**Chapter 14 — File 8 of 11 / 第14章 — 第8个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning**.

本脚本演示 **Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
import unicodedata
import zipfile

import torch
import tokenizers
from torch.utils.data import Dataset, DataLoader

def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()

text_pairs = []
with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
    for line in zip_ref.read("fra.txt").decode("utf-8").splitlines():
        eng, fra = normalize(line)
        text_pairs.append((eng, fra))

en_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
fr_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
```

---
## Step 2 — Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning
of the sentence

```python
en_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
fr_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
```

---
## Step 3 — Configure decoder: So that word boundary symbol "Ġ" will be removed

```python
en_tokenizer.decoder = tokenizers.decoders.ByteLevel()
fr_tokenizer.decoder = tokenizers.decoders.ByteLevel()
```

---
## Step 4 — Train BPE for English and French using the same trainer

```python
VOCAB_SIZE = 8000
trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[start]", "[end]", "[pad]"],
    show_progress=True
)
en_tokenizer.train_from_iterator([x[0] for x in text_pairs], trainer=trainer)
fr_tokenizer.train_from_iterator([x[1] for x in text_pairs], trainer=trainer)

en_tokenizer.enable_padding(pad_id=en_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
fr_tokenizer.enable_padding(pad_id=fr_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
```

---
## Step 5 — Save the trained tokenizers

```python
en_tokenizer.save("en_tokenizer.json", pretty=True)
fr_tokenizer.save("fr_tokenizer.json", pretty=True)

class TranslationDataset(Dataset):
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        eng, fra = self.text_pairs[idx]
        return eng, "[start] " + fra + " [end]"

def collate_fn(batch):
    en_str, fr_str = zip(*batch)
    en_enc = en_tokenizer.encode_batch(en_str, add_special_tokens=True)
    fr_enc = fr_tokenizer.encode_batch(fr_str, add_special_tokens=True)
    en_ids = [enc.ids for enc in en_enc]
    fr_ids = [enc.ids for enc in fr_enc]
    return torch.tensor(en_ids), torch.tensor(fr_ids)

BATCH_SIZE = 32
dataset = TranslationDataset(text_pairs)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn)
```

---
## Learning Notes / 学习笔记

- **概念**: Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning 是机器学习中的常用技术。  
  *Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataLoader` | 数据加载器，批量读取数据 | Loads data in batches for training |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dataset / 08 Dataset
# Complete Code / 完整代码
# ===============================

import unicodedata
import zipfile

import torch
import tokenizers
from torch.utils.data import Dataset, DataLoader

def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()

text_pairs = []
with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
    for line in zip_ref.read("fra.txt").decode("utf-8").splitlines():
        eng, fra = normalize(line)
        text_pairs.append((eng, fra))

en_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
fr_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())

# Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning
# of the sentence
en_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
fr_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)

# Configure decoder: So that word boundary symbol "Ġ" will be removed
en_tokenizer.decoder = tokenizers.decoders.ByteLevel()
fr_tokenizer.decoder = tokenizers.decoders.ByteLevel()

# Train BPE for English and French using the same trainer
VOCAB_SIZE = 8000
trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[start]", "[end]", "[pad]"],
    show_progress=True
)
en_tokenizer.train_from_iterator([x[0] for x in text_pairs], trainer=trainer)
fr_tokenizer.train_from_iterator([x[1] for x in text_pairs], trainer=trainer)

en_tokenizer.enable_padding(pad_id=en_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
fr_tokenizer.enable_padding(pad_id=fr_tokenizer.token_to_id("[pad]"), pad_token="[pad]")

# Save the trained tokenizers
en_tokenizer.save("en_tokenizer.json", pretty=True)
fr_tokenizer.save("fr_tokenizer.json", pretty=True)

class TranslationDataset(Dataset):
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        eng, fra = self.text_pairs[idx]
        return eng, "[start] " + fra + " [end]"

def collate_fn(batch):
    en_str, fr_str = zip(*batch)
    en_enc = en_tokenizer.encode_batch(en_str, add_special_tokens=True)
    fr_enc = fr_tokenizer.encode_batch(fr_str, add_special_tokens=True)
    en_ids = [enc.ids for enc in en_enc]
    fr_ids = [enc.ids for enc in fr_enc]
    return torch.tensor(en_ids), torch.tensor(fr_ids)

BATCH_SIZE = 32
dataset = TranslationDataset(text_pairs)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn)
```

---

➡️ **Next / 下一步**: File 9 of 11

---

### One Sample

# 09 — One Sample / 09 One Sample

**Chapter 14 — File 9 of 11 / 第14章 — 第9个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning**.

本脚本演示 **Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
import unicodedata
import zipfile

import torch
import tokenizers
from torch.utils.data import Dataset, DataLoader

def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()

text_pairs = []
with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
    for line in zip_ref.read("fra.txt").decode("utf-8").splitlines():
        eng, fra = normalize(line)
        text_pairs.append((eng, fra))

en_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
fr_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
```

---
## Step 2 — Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning
of the sentence

```python
en_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
fr_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
```

---
## Step 3 — Configure decoder: So that word boundary symbol "Ġ" will be removed

```python
en_tokenizer.decoder = tokenizers.decoders.ByteLevel()
fr_tokenizer.decoder = tokenizers.decoders.ByteLevel()
```

---
## Step 4 — Train BPE for English and French using the same trainer

```python
VOCAB_SIZE = 8000
trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[start]", "[end]", "[pad]"],
    show_progress=True
)
en_tokenizer.train_from_iterator([x[0] for x in text_pairs], trainer=trainer)
fr_tokenizer.train_from_iterator([x[1] for x in text_pairs], trainer=trainer)

en_tokenizer.enable_padding(pad_id=en_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
fr_tokenizer.enable_padding(pad_id=fr_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
```

---
## Step 5 — Save the trained tokenizers

```python
en_tokenizer.save("en_tokenizer.json", pretty=True)
fr_tokenizer.save("fr_tokenizer.json", pretty=True)

class TranslationDataset(Dataset):
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        eng, fra = self.text_pairs[idx]
        return eng, "[start] " + fra + " [end]"

def collate_fn(batch):
    en_str, fr_str = zip(*batch)
    en_enc = en_tokenizer.encode_batch(en_str, add_special_tokens=True)
    fr_enc = fr_tokenizer.encode_batch(fr_str, add_special_tokens=True)
    en_ids = [enc.ids for enc in en_enc]
    fr_ids = [enc.ids for enc in fr_enc]
    return torch.tensor(en_ids), torch.tensor(fr_ids)

BATCH_SIZE = 32
dataset = TranslationDataset(text_pairs)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn)

for en_ids, fr_ids in dataloader:
    print(f"English: {en_ids}")
    print(f"French: {fr_ids}")
    break
```

---
## Learning Notes / 学习笔记

- **概念**: Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning 是机器学习中的常用技术。  
  *Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataLoader` | 数据加载器，批量读取数据 | Loads data in batches for training |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# One Sample / 09 One Sample
# Complete Code / 完整代码
# ===============================

import unicodedata
import zipfile

import torch
import tokenizers
from torch.utils.data import Dataset, DataLoader

def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()

text_pairs = []
with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
    for line in zip_ref.read("fra.txt").decode("utf-8").splitlines():
        eng, fra = normalize(line)
        text_pairs.append((eng, fra))

en_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
fr_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())

# Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning
# of the sentence
en_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
fr_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)

# Configure decoder: So that word boundary symbol "Ġ" will be removed
en_tokenizer.decoder = tokenizers.decoders.ByteLevel()
fr_tokenizer.decoder = tokenizers.decoders.ByteLevel()

# Train BPE for English and French using the same trainer
VOCAB_SIZE = 8000
trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[start]", "[end]", "[pad]"],
    show_progress=True
)
en_tokenizer.train_from_iterator([x[0] for x in text_pairs], trainer=trainer)
fr_tokenizer.train_from_iterator([x[1] for x in text_pairs], trainer=trainer)

en_tokenizer.enable_padding(pad_id=en_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
fr_tokenizer.enable_padding(pad_id=fr_tokenizer.token_to_id("[pad]"), pad_token="[pad]")

# Save the trained tokenizers
en_tokenizer.save("en_tokenizer.json", pretty=True)
fr_tokenizer.save("fr_tokenizer.json", pretty=True)

class TranslationDataset(Dataset):
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        eng, fra = self.text_pairs[idx]
        return eng, "[start] " + fra + " [end]"

def collate_fn(batch):
    en_str, fr_str = zip(*batch)
    en_enc = en_tokenizer.encode_batch(en_str, add_special_tokens=True)
    fr_enc = fr_tokenizer.encode_batch(fr_str, add_special_tokens=True)
    en_ids = [enc.ids for enc in en_enc]
    fr_ids = [enc.ids for enc in fr_enc]
    return torch.tensor(en_ids), torch.tensor(fr_ids)

BATCH_SIZE = 32
dataset = TranslationDataset(text_pairs)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn)

for en_ids, fr_ids in dataloader:
    print(f"English: {en_ids}")
    print(f"French: {fr_ids}")
    break
```

---

➡️ **Next / 下一步**: File 10 of 11

---

### Create Model

# 10 — Create Model / 10 Create Model

**Chapter 14 — File 10 of 11 / 第14章 — 第10个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Create Model**.

本脚本演示 **10 Create Model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
import torch
import torch.nn as nn
import tokenizers

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_seq, hidden, cell):
        embedded = self.embedding(input_seq)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell

class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        batch_size, target_len = target_seq.shape
        outputs = []
        _enc_out, hidden, cell = self.encoder(input_seq)
        dec_in = target_seq[:, :1]
        for t in range(target_len-1):
            pred, hidden, cell = self.decoder(dec_in, hidden, cell)
            pred = pred[:, -1:, :]
            outputs.append(pred)
            dec_in = torch.cat([dec_in, pred.argmax(dim=2)], dim=1)
        outputs = torch.cat(outputs, dim=1)
        return outputs

en_tokenizer = tokenizers.Tokenizer.from_file("en_tokenizer.json")
fr_tokenizer = tokenizers.Tokenizer.from_file("fr_tokenizer.json")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emb_dim = 256
hidden_dim = 256
num_layers = 1
enc_vocab = en_tokenizer.get_vocab_size()
dec_vocab = fr_tokenizer.get_vocab_size()


encoder = EncoderLSTM(enc_vocab, emb_dim, hidden_dim, num_layers).to(device)
decoder = DecoderLSTM(dec_vocab, emb_dim, hidden_dim, num_layers).to(device)
model = Seq2SeqLSTM(encoder, decoder).to(device)
print(model)
```

---
## Learning Notes / 学习笔记

- **概念**: Create Model 是机器学习中的常用技术。  
  *Create Model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `nn.Embedding` | 将整数ID映射为稠密向量 | Map integer IDs to dense vectors |
| `nn.LSTM` | 长短期记忆网络，处理序列数据 | Long Short-Term Memory for sequences |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Create Model / 10 Create Model
# Complete Code / 完整代码
# ===============================

import torch
import torch.nn as nn
import tokenizers

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_seq, hidden, cell):
        embedded = self.embedding(input_seq)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell

class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        batch_size, target_len = target_seq.shape
        outputs = []
        _enc_out, hidden, cell = self.encoder(input_seq)
        dec_in = target_seq[:, :1]
        for t in range(target_len-1):
            pred, hidden, cell = self.decoder(dec_in, hidden, cell)
            pred = pred[:, -1:, :]
            outputs.append(pred)
            dec_in = torch.cat([dec_in, pred.argmax(dim=2)], dim=1)
        outputs = torch.cat(outputs, dim=1)
        return outputs

en_tokenizer = tokenizers.Tokenizer.from_file("en_tokenizer.json")
fr_tokenizer = tokenizers.Tokenizer.from_file("fr_tokenizer.json")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emb_dim = 256
hidden_dim = 256
num_layers = 1
enc_vocab = en_tokenizer.get_vocab_size()
dec_vocab = fr_tokenizer.get_vocab_size()


encoder = EncoderLSTM(enc_vocab, emb_dim, hidden_dim, num_layers).to(device)
decoder = DecoderLSTM(dec_vocab, emb_dim, hidden_dim, num_layers).to(device)
model = Seq2SeqLSTM(encoder, decoder).to(device)
print(model)
```

---

➡️ **Next / 下一步**: File 11 of 11

---

### Complete

# 14 — Complete / 14 Complete

**Chapter 14 — File 11 of 11 / 第14章 — 第11个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Complete**.

本脚本演示 **14 Complete**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────┐
│  定义模型 Define Model  │
└───────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — Step 1

```python
import random
import os
import unicodedata
import zipfile

import requests
import torch
import torch.nn as nn
import torch.optim as optim
import tokenizers
import tqdm
```

---
## Step 2 — Data preparation

Download dataset provided by Anki: https://www.manythings.org/anki/ with requests

```python
if not os.path.exists("fra-eng.zip"):
    url = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
    response = requests.get(url)
    with open("fra-eng.zip", "wb") as f:
        f.write(response.content)
```

---
## Step 3 — Normalize text
each line of the file is in the format "<english>\t<french>"
We convert text to lowercasee, normalize unicode (UFKC)

```python
def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()

text_pairs = []
with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
    for line in zip_ref.read("fra.txt").decode("utf-8").splitlines():
        eng, fra = normalize(line)
        text_pairs.append((eng, fra))
```

---
## Step 4 — Tokenization with BPE


```python
if os.path.exists("en_tokenizer.json") and os.path.exists("fr_tokenizer.json"):
    en_tokenizer = tokenizers.Tokenizer.from_file("en_tokenizer.json")
    fr_tokenizer = tokenizers.Tokenizer.from_file("fr_tokenizer.json")
else:
    en_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    fr_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
```

---
## Step 5 — Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning
of the sentence

```python
en_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    fr_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
```

---
## Step 6 — Configure decoder: So that word boundary symbol "Ġ" will be removed

```python
en_tokenizer.decoder = tokenizers.decoders.ByteLevel()
    fr_tokenizer.decoder = tokenizers.decoders.ByteLevel()
```

---
## Step 7 — Train BPE for English and French using the same trainer

```python
VOCAB_SIZE = 8000
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[start]", "[end]", "[pad]"],
        show_progress=True
    )
    en_tokenizer.train_from_iterator([x[0] for x in text_pairs], trainer=trainer)
    fr_tokenizer.train_from_iterator([x[1] for x in text_pairs], trainer=trainer)

    en_tokenizer.enable_padding(pad_id=en_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
    fr_tokenizer.enable_padding(pad_id=fr_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
```

---
## Step 8 — Save the trained tokenizers

```python
en_tokenizer.save("en_tokenizer.json", pretty=True)
    fr_tokenizer.save("fr_tokenizer.json", pretty=True)
```

---
## Step 9 — Test the tokenizer

```python
print("Sample tokenization:")
en_sample, fr_sample = random.choice(text_pairs)
encoded = en_tokenizer.encode(en_sample)
print(f"Original: {en_sample}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Decoded: {en_tokenizer.decode(encoded.ids)}")
print()

encoded = fr_tokenizer.encode("[start] " + fr_sample + " [end]")
print(f"Original: {fr_sample}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Decoded: {fr_tokenizer.decode(encoded.ids)}")
print()
```

---
## Step 10 — Create PyTorch dataset for the BPE-encoded translation pairs


```python
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        eng, fra = self.text_pairs[idx]
        return eng, "[start] " + fra + " [end]"


def collate_fn(batch):
    en_str, fr_str = zip(*batch)
    en_enc = en_tokenizer.encode_batch(en_str, add_special_tokens=True)
    fr_enc = fr_tokenizer.encode_batch(fr_str, add_special_tokens=True)
    en_ids = [enc.ids for enc in en_enc]
    fr_ids = [enc.ids for enc in fr_enc]
    return torch.tensor(en_ids), torch.tensor(fr_ids)


BATCH_SIZE = 32
dataset = TranslationDataset(text_pairs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         collate_fn=collate_fn)
```

---
## Step 11 — Test the dataset

```python
for en_ids, fr_ids in dataloader:
    print(f"English: {en_ids}")
    print(f"French: {fr_ids}")
    break
```

---
## Step 12 — Create LSTM seq2seq model for translation


```python
class EncoderLSTM(nn.Module):
    """A stacked LSTM encoder with an embedding layer"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        """
        Plain LSTM is used. No bidirectional LSTM.

        Args:
            vocab_size: The size of the input vocabulary
            embedding_dim: The dimension of the embedding vector
            hidden_dim: The dimension of the hidden state
            num_layers: The number of recurrent layers (layers of stacked LSTM)
            dropout: The dropout rate, applied to all LSTM layers except the last one
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, input_seq):
```

---
## Step 13 — input seq = [batch_size, seq_len] -> embedded = [batch_size, seq_len, embedding_dim]

```python
embedded = self.embedding(input_seq)
```

---
## Step 14 — outputs = [batch_size, seq_len, embedding_dim]
hidden = cell = [n_layers, batch_size, hidden_dim]

```python
outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_seq, hidden, cell):
```

---
## Step 15 — input seq = [batch_size, seq_len] -> embedded = [batch_size, seq_len, embedding_dim]
hidden = cell = [n_layers, batch_size, hidden_dim]

```python
embedded = self.embedding(input_seq)
```

---
## Step 16 — output = [batch_size, seq_len, embedding_dim]

```python
output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        """Given the partial target sequence, predict the next token"""
```

---
## Step 17 — input seq = [batch_size, seq_len]
target seq = [batch_size, seq_len]

```python
batch_size, target_len = target_seq.shape
```

---
## Step 18 — storing output logits

```python
outputs = []
```

---
## Step 19 — encoder forward pass

```python
_enc_out, hidden, cell = self.encoder(input_seq)
        dec_in = target_seq[:, :1]
```

---
## Step 20 — decoder forward pass

```python
for t in range(target_len-1):
```

---
## Step 21 — last target token and hidden states -> next token

```python
pred, hidden, cell = self.decoder(dec_in, hidden, cell)
```

---
## Step 22 — store the prediction

```python
pred = pred[:, -1:, :]
            outputs.append(pred)
```

---
## Step 23 — use the predicted token as the next input

```python
dec_in = torch.cat([dec_in, pred.argmax(dim=2)], dim=1)
        outputs = torch.cat(outputs, dim=1)
        return outputs
```

---
## Step 24 — Initialize model parameters

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enc_vocab = len(en_tokenizer.get_vocab())
dec_vocab = len(fr_tokenizer.get_vocab())
emb_dim = 256
hidden_dim = 256
num_layers = 2
dropout = 0.1
```

---
## Step 25 — Create model

```python
encoder = EncoderLSTM(enc_vocab, emb_dim, hidden_dim, num_layers, dropout).to(device)
decoder = DecoderLSTM(dec_vocab, emb_dim, hidden_dim, num_layers, dropout).to(device)
model = Seq2SeqLSTM(encoder, decoder).to(device)
print(model)

print("Model created with:")
print(f"  Input vocabulary size: {enc_vocab}")
print(f"  Output vocabulary size: {dec_vocab}")
print(f"  Embedding dimension: {emb_dim}")
print(f"  Hidden dimension: {hidden_dim}")
print(f"  Number of layers: {num_layers}")
print(f"  Dropout: {dropout}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
```

---
## Step 26 — Train unless model.pth exists

```python
if os.path.exists("seq2seq.pth"):
    model.load_state_dict(torch.load("seq2seq.pth"))
else:
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=fr_tokenizer.token_to_id("[pad]"))
    N_EPOCHS = 30

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0
        for en_ids, fr_ids in tqdm.tqdm(dataloader, desc="Training"):
```

---
## Step 27 — Move the "sentences" to device

```python
en_ids = en_ids.to(device)
            fr_ids = fr_ids.to(device)
```

---
## Step 28 — zero the grad, then forward pass

```python
optimizer.zero_grad()
            outputs = model(en_ids, fr_ids)
```

---
## Step 29 — compute the loss: compare 3D logits to 2D targets

```python
loss = loss_fn(outputs.reshape(-1, dec_vocab), fr_ids[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{N_EPOCHS}; Avg loss {epoch_loss/len(dataloader)}"
              f"; Latest loss {loss.item()}")
        torch.save(model.state_dict(), f"seq2seq-epoch-{epoch+1}.pth")
```

---
## Step 30 — Test once every 5 epochs

```python
if (epoch+1) % 5 != 0:
            continue
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for en_ids, fr_ids in tqdm.tqdm(dataloader, desc="Evaluating"):
                en_ids = en_ids.to(device)
                fr_ids = fr_ids.to(device)
                outputs = model(en_ids, fr_ids)
                loss = loss_fn(outputs.reshape(-1, dec_vocab), fr_ids[:, 1:].reshape(-1))
                epoch_loss += loss.item()
        print(f"Eval loss: {epoch_loss/len(dataloader)}")
```

---
## Step 31 — Save the final model

```python
torch.save(model.state_dict(), "seq2seq.pth")
```

---
## Step 32 — Test for a few samples

```python
model.eval()
N_SAMPLES = 5
MAX_LEN = 60
with torch.no_grad():
    start_token = torch.tensor([fr_tokenizer.token_to_id("[start]")]).to(device)
    for en, true_fr in random.sample(text_pairs, N_SAMPLES):
        en_ids = torch.tensor(en_tokenizer.encode(en).ids).unsqueeze(0).to(device)
        _output, hidden, cell = model.encoder(en_ids)
        pred_ids = [start_token]
        for _ in range(MAX_LEN):
            decoder_input = torch.tensor(pred_ids).unsqueeze(0).to(device)
            output, hidden, cell = model.decoder(decoder_input, hidden, cell)
            output = output[:, -1, :].argmax(dim=1)
            pred_ids.append(output.item())
```

---
## Step 33 — early stop if the predicted token is the end token

```python
if pred_ids[-1] == fr_tokenizer.token_to_id("[end]"):
                break
```

---
## Step 34 — Decode the predicted IDs

```python
pred_fr = fr_tokenizer.decode(pred_ids)
        print(f"English: {en}")
        print(f"French: {true_fr}")
        print(f"Predicted: {pred_fr}")
        print()
```

---
## Learning Notes / 学习笔记

- **概念**: Complete 是机器学习中的常用技术。  
  *Complete is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataLoader` | 数据加载器，批量读取数据 | Loads data in batches for training |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `nn.Embedding` | 将整数ID映射为稠密向量 | Map integer IDs to dense vectors |
| `nn.LSTM` | 长短期记忆网络，处理序列数据 | Long Short-Term Memory for sequences |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Complete / 14 Complete
# Complete Code / 完整代码
# ===============================

import random
import os
import unicodedata
import zipfile

import requests
import torch
import torch.nn as nn
import torch.optim as optim
import tokenizers
import tqdm


#
# Data preparation
#

# Download dataset provided by Anki: https://www.manythings.org/anki/ with requests
if not os.path.exists("fra-eng.zip"):
    url = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
    response = requests.get(url)
    with open("fra-eng.zip", "wb") as f:
        f.write(response.content)

# Normalize text
# each line of the file is in the format "<english>\t<french>"
# We convert text to lowercasee, normalize unicode (UFKC)
def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()

text_pairs = []
with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
    for line in zip_ref.read("fra.txt").decode("utf-8").splitlines():
        eng, fra = normalize(line)
        text_pairs.append((eng, fra))

#
# Tokenization with BPE
#

if os.path.exists("en_tokenizer.json") and os.path.exists("fr_tokenizer.json"):
    en_tokenizer = tokenizers.Tokenizer.from_file("en_tokenizer.json")
    fr_tokenizer = tokenizers.Tokenizer.from_file("fr_tokenizer.json")
else:
    en_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    fr_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())

    # Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning
    # of the sentence
    en_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    fr_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)

    # Configure decoder: So that word boundary symbol "Ġ" will be removed
    en_tokenizer.decoder = tokenizers.decoders.ByteLevel()
    fr_tokenizer.decoder = tokenizers.decoders.ByteLevel()

    # Train BPE for English and French using the same trainer
    VOCAB_SIZE = 8000
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[start]", "[end]", "[pad]"],
        show_progress=True
    )
    en_tokenizer.train_from_iterator([x[0] for x in text_pairs], trainer=trainer)
    fr_tokenizer.train_from_iterator([x[1] for x in text_pairs], trainer=trainer)

    en_tokenizer.enable_padding(pad_id=en_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
    fr_tokenizer.enable_padding(pad_id=fr_tokenizer.token_to_id("[pad]"), pad_token="[pad]")

    # Save the trained tokenizers
    en_tokenizer.save("en_tokenizer.json", pretty=True)
    fr_tokenizer.save("fr_tokenizer.json", pretty=True)

# Test the tokenizer
print("Sample tokenization:")
en_sample, fr_sample = random.choice(text_pairs)
encoded = en_tokenizer.encode(en_sample)
print(f"Original: {en_sample}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Decoded: {en_tokenizer.decode(encoded.ids)}")
print()

encoded = fr_tokenizer.encode("[start] " + fr_sample + " [end]")
print(f"Original: {fr_sample}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Decoded: {fr_tokenizer.decode(encoded.ids)}")
print()

#
# Create PyTorch dataset for the BPE-encoded translation pairs
#

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        eng, fra = self.text_pairs[idx]
        return eng, "[start] " + fra + " [end]"


def collate_fn(batch):
    en_str, fr_str = zip(*batch)
    en_enc = en_tokenizer.encode_batch(en_str, add_special_tokens=True)
    fr_enc = fr_tokenizer.encode_batch(fr_str, add_special_tokens=True)
    en_ids = [enc.ids for enc in en_enc]
    fr_ids = [enc.ids for enc in fr_enc]
    return torch.tensor(en_ids), torch.tensor(fr_ids)


BATCH_SIZE = 32
dataset = TranslationDataset(text_pairs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         collate_fn=collate_fn)

# Test the dataset
for en_ids, fr_ids in dataloader:
    print(f"English: {en_ids}")
    print(f"French: {fr_ids}")
    break

#
# Create LSTM seq2seq model for translation
#

class EncoderLSTM(nn.Module):
    """A stacked LSTM encoder with an embedding layer"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        """
        Plain LSTM is used. No bidirectional LSTM.

        Args:
            vocab_size: The size of the input vocabulary
            embedding_dim: The dimension of the embedding vector
            hidden_dim: The dimension of the hidden state
            num_layers: The number of recurrent layers (layers of stacked LSTM)
            dropout: The dropout rate, applied to all LSTM layers except the last one
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, input_seq):
        # input seq = [batch_size, seq_len] -> embedded = [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(input_seq)
        # outputs = [batch_size, seq_len, embedding_dim]
        # hidden = cell = [n_layers, batch_size, hidden_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_seq, hidden, cell):
        # input seq = [batch_size, seq_len] -> embedded = [batch_size, seq_len, embedding_dim]
        # hidden = cell = [n_layers, batch_size, hidden_dim]
        embedded = self.embedding(input_seq)
        # output = [batch_size, seq_len, embedding_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        """Given the partial target sequence, predict the next token"""
        # input seq = [batch_size, seq_len]
        # target seq = [batch_size, seq_len]
        batch_size, target_len = target_seq.shape
        # storing output logits
        outputs = []
        # encoder forward pass
        _enc_out, hidden, cell = self.encoder(input_seq)
        dec_in = target_seq[:, :1]
        # decoder forward pass
        for t in range(target_len-1):
            # last target token and hidden states -> next token
            pred, hidden, cell = self.decoder(dec_in, hidden, cell)
            # store the prediction
            pred = pred[:, -1:, :]
            outputs.append(pred)
            # use the predicted token as the next input
            dec_in = torch.cat([dec_in, pred.argmax(dim=2)], dim=1)
        outputs = torch.cat(outputs, dim=1)
        return outputs


# Initialize model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enc_vocab = len(en_tokenizer.get_vocab())
dec_vocab = len(fr_tokenizer.get_vocab())
emb_dim = 256
hidden_dim = 256
num_layers = 2
dropout = 0.1

# Create model
encoder = EncoderLSTM(enc_vocab, emb_dim, hidden_dim, num_layers, dropout).to(device)
decoder = DecoderLSTM(dec_vocab, emb_dim, hidden_dim, num_layers, dropout).to(device)
model = Seq2SeqLSTM(encoder, decoder).to(device)
print(model)

print("Model created with:")
print(f"  Input vocabulary size: {enc_vocab}")
print(f"  Output vocabulary size: {dec_vocab}")
print(f"  Embedding dimension: {emb_dim}")
print(f"  Hidden dimension: {hidden_dim}")
print(f"  Number of layers: {num_layers}")
print(f"  Dropout: {dropout}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Train unless model.pth exists
if os.path.exists("seq2seq.pth"):
    model.load_state_dict(torch.load("seq2seq.pth"))
else:
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=fr_tokenizer.token_to_id("[pad]"))
    N_EPOCHS = 30

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0
        for en_ids, fr_ids in tqdm.tqdm(dataloader, desc="Training"):
            # Move the "sentences" to device
            en_ids = en_ids.to(device)
            fr_ids = fr_ids.to(device)
            # zero the grad, then forward pass
            optimizer.zero_grad()
            outputs = model(en_ids, fr_ids)
            # compute the loss: compare 3D logits to 2D targets
            loss = loss_fn(outputs.reshape(-1, dec_vocab), fr_ids[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{N_EPOCHS}; Avg loss {epoch_loss/len(dataloader)}"
              f"; Latest loss {loss.item()}")
        torch.save(model.state_dict(), f"seq2seq-epoch-{epoch+1}.pth")
        # Test once every 5 epochs
        if (epoch+1) % 5 != 0:
            continue
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for en_ids, fr_ids in tqdm.tqdm(dataloader, desc="Evaluating"):
                en_ids = en_ids.to(device)
                fr_ids = fr_ids.to(device)
                outputs = model(en_ids, fr_ids)
                loss = loss_fn(outputs.reshape(-1, dec_vocab), fr_ids[:, 1:].reshape(-1))
                epoch_loss += loss.item()
        print(f"Eval loss: {epoch_loss/len(dataloader)}")

    # Save the final model
    torch.save(model.state_dict(), "seq2seq.pth")

# Test for a few samples
model.eval()
N_SAMPLES = 5
MAX_LEN = 60
with torch.no_grad():
    start_token = torch.tensor([fr_tokenizer.token_to_id("[start]")]).to(device)
    for en, true_fr in random.sample(text_pairs, N_SAMPLES):
        en_ids = torch.tensor(en_tokenizer.encode(en).ids).unsqueeze(0).to(device)
        _output, hidden, cell = model.encoder(en_ids)
        pred_ids = [start_token]
        for _ in range(MAX_LEN):
            decoder_input = torch.tensor(pred_ids).unsqueeze(0).to(device)
            output, hidden, cell = model.decoder(decoder_input, hidden, cell)
            output = output[:, -1, :].argmax(dim=1)
            pred_ids.append(output.item())
            # early stop if the predicted token is the end token
            if pred_ids[-1] == fr_tokenizer.token_to_id("[end]"):
                break
        # Decode the predicted IDs
        pred_fr = fr_tokenizer.decode(pred_ids)
        print(f"English: {en}")
        print(f"French: {true_fr}")
        print(f"Predicted: {pred_fr}")
        print()
```

---
