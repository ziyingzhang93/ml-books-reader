# 从零构建Transformer / Building Transformers from Scratch
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
# 导入操作系统接口 / Import OS interface
import os
import requests

if not os.path.exists("fra-eng.zip"):
    url = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
    response = requests.get(url)
    # 打开文件（自动关闭） / Open file (auto-close)
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

# 导入操作系统接口 / Import OS interface
import os
import requests

if not os.path.exists("fra-eng.zip"):
    url = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
    response = requests.get(url)
    # 打开文件（自动关闭） / Open file (auto-close)
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
        # 添加元素到列表末尾 / Append element to list end
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
        # 添加元素到列表末尾 / Append element to list end
        text_pairs.append((eng, fra))
```

---

➡️ **Next / 下一步**: File 3 of 11

---

### Tokenizer



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
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
# 导入随机数生成模块 / Import random number module
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
        # 添加元素到列表末尾 / Append element to list end
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
# 打印输出 / Print output
print("Sample tokenization:")
en_sample, fr_sample = random.choice(text_pairs)
encoded = en_tokenizer.encode(en_sample)
# 打印输出 / Print output
print(f"Original: {en_sample}")
# 打印输出 / Print output
print(f"Tokens: {encoded.tokens}")
# 打印输出 / Print output
print(f"IDs: {encoded.ids}")
# 打印输出 / Print output
print(f"Decoded: {en_tokenizer.decode(encoded.ids)}")
# 打印输出 / Print output
print()

encoded = fr_tokenizer.encode("[start] " + fr_sample + " [end]")
# 打印输出 / Print output
print(f"Original: {fr_sample}")
# 打印输出 / Print output
print(f"Tokens: {encoded.tokens}")
# 打印输出 / Print output
print(f"IDs: {encoded.ids}")
# 打印输出 / Print output
print(f"Decoded: {fr_tokenizer.decode(encoded.ids)}")
# 打印输出 / Print output
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

# 导入随机数生成模块 / Import random number module
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
        # 添加元素到列表末尾 / Append element to list end
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
# 打印输出 / Print output
print("Sample tokenization:")
en_sample, fr_sample = random.choice(text_pairs)
encoded = en_tokenizer.encode(en_sample)
# 打印输出 / Print output
print(f"Original: {en_sample}")
# 打印输出 / Print output
print(f"Tokens: {encoded.tokens}")
# 打印输出 / Print output
print(f"IDs: {encoded.ids}")
# 打印输出 / Print output
print(f"Decoded: {en_tokenizer.decode(encoded.ids)}")
# 打印输出 / Print output
print()

encoded = fr_tokenizer.encode("[start] " + fr_sample + " [end]")
# 打印输出 / Print output
print(f"Original: {fr_sample}")
# 打印输出 / Print output
print(f"Tokens: {encoded.tokens}")
# 打印输出 / Print output
print(f"IDs: {encoded.ids}")
# 打印输出 / Print output
print(f"Decoded: {fr_tokenizer.decode(encoded.ids)}")
# 打印输出 / Print output
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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class EncoderLSTM(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
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

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class EncoderLSTM(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell
```

---

➡️ **Next / 下一步**: File 6 of 11

---

### Decoder



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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Seq2SeqLSTM(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, encoder, decoder):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, input_seq, target_seq):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        batch_size, target_len = target_seq.shape
        outputs = []
        _enc_out, hidden, cell = self.encoder(input_seq)
        dec_in = target_seq[:, :1]
        # 生成整数序列 / Generate integer sequence
        for t in range(target_len-1):
            pred, hidden, cell = self.decoder(dec_in, hidden, cell)
            pred = pred[:, -1:, :]
            # 添加元素到列表末尾 / Append element to list end
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

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Seq2SeqLSTM(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, encoder, decoder):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, input_seq, target_seq):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        batch_size, target_len = target_seq.shape
        outputs = []
        _enc_out, hidden, cell = self.encoder(input_seq)
        dec_in = target_seq[:, :1]
        # 生成整数序列 / Generate integer sequence
        for t in range(target_len-1):
            pred, hidden, cell = self.decoder(dec_in, hidden, cell)
            pred = pred[:, -1:, :]
            # 添加元素到列表末尾 / Append element to list end
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
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
import unicodedata
import zipfile

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
import tokenizers
# 导入数据加载工具 / Import data loading utilities
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
        # 添加元素到列表末尾 / Append element to list end
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

# 定义数据集 / Define dataset
class TranslationDataset(Dataset):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        # 获取长度 / Get length
        return len(self.text_pairs)

    def __getitem__(self, idx):
        eng, fra = self.text_pairs[idx]
        return eng, "[start] " + fra + " [end]"

def collate_fn(batch):
    # 将多个序列配对 / Pair multiple sequences
    en_str, fr_str = zip(*batch)
    en_enc = en_tokenizer.encode_batch(en_str, add_special_tokens=True)
    fr_enc = fr_tokenizer.encode_batch(fr_str, add_special_tokens=True)
    en_ids = [enc.ids for enc in en_enc]
    fr_ids = [enc.ids for enc in fr_enc]
    return torch.tensor(en_ids), torch.tensor(fr_ids)

BATCH_SIZE = 32
# 定义数据集 / Define dataset
dataset = TranslationDataset(text_pairs)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
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

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
import tokenizers
# 导入数据加载工具 / Import data loading utilities
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
        # 添加元素到列表末尾 / Append element to list end
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

# 定义数据集 / Define dataset
class TranslationDataset(Dataset):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        # 获取长度 / Get length
        return len(self.text_pairs)

    def __getitem__(self, idx):
        eng, fra = self.text_pairs[idx]
        return eng, "[start] " + fra + " [end]"

def collate_fn(batch):
    # 将多个序列配对 / Pair multiple sequences
    en_str, fr_str = zip(*batch)
    en_enc = en_tokenizer.encode_batch(en_str, add_special_tokens=True)
    fr_enc = fr_tokenizer.encode_batch(fr_str, add_special_tokens=True)
    en_ids = [enc.ids for enc in en_enc]
    fr_ids = [enc.ids for enc in fr_enc]
    return torch.tensor(en_ids), torch.tensor(fr_ids)

BATCH_SIZE = 32
# 定义数据集 / Define dataset
dataset = TranslationDataset(text_pairs)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
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
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — Step 1

```python
import unicodedata
import zipfile

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
import tokenizers
# 导入数据加载工具 / Import data loading utilities
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
        # 添加元素到列表末尾 / Append element to list end
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

# 定义数据集 / Define dataset
class TranslationDataset(Dataset):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        # 获取长度 / Get length
        return len(self.text_pairs)

    def __getitem__(self, idx):
        eng, fra = self.text_pairs[idx]
        return eng, "[start] " + fra + " [end]"

def collate_fn(batch):
    # 将多个序列配对 / Pair multiple sequences
    en_str, fr_str = zip(*batch)
    en_enc = en_tokenizer.encode_batch(en_str, add_special_tokens=True)
    fr_enc = fr_tokenizer.encode_batch(fr_str, add_special_tokens=True)
    en_ids = [enc.ids for enc in en_enc]
    fr_ids = [enc.ids for enc in fr_enc]
    return torch.tensor(en_ids), torch.tensor(fr_ids)

BATCH_SIZE = 32
# 定义数据集 / Define dataset
dataset = TranslationDataset(text_pairs)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn)

for en_ids, fr_ids in dataloader:
    # 打印输出 / Print output
    print(f"English: {en_ids}")
    # 打印输出 / Print output
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

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
import tokenizers
# 导入数据加载工具 / Import data loading utilities
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
        # 添加元素到列表末尾 / Append element to list end
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

# 定义数据集 / Define dataset
class TranslationDataset(Dataset):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        # 获取长度 / Get length
        return len(self.text_pairs)

    def __getitem__(self, idx):
        eng, fra = self.text_pairs[idx]
        return eng, "[start] " + fra + " [end]"

def collate_fn(batch):
    # 将多个序列配对 / Pair multiple sequences
    en_str, fr_str = zip(*batch)
    en_enc = en_tokenizer.encode_batch(en_str, add_special_tokens=True)
    fr_enc = fr_tokenizer.encode_batch(fr_str, add_special_tokens=True)
    en_ids = [enc.ids for enc in en_enc]
    fr_ids = [enc.ids for enc in fr_enc]
    return torch.tensor(en_ids), torch.tensor(fr_ids)

BATCH_SIZE = 32
# 定义数据集 / Define dataset
dataset = TranslationDataset(text_pairs)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn)

for en_ids, fr_ids in dataloader:
    # 打印输出 / Print output
    print(f"English: {en_ids}")
    # 打印输出 / Print output
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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
import tokenizers

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class EncoderLSTM(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class DecoderLSTM(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out = nn.Linear(embedding_dim, vocab_size)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, input_seq, hidden, cell):
        embedded = self.embedding(input_seq)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Seq2SeqLSTM(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, encoder, decoder):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, input_seq, target_seq):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        batch_size, target_len = target_seq.shape
        outputs = []
        _enc_out, hidden, cell = self.encoder(input_seq)
        dec_in = target_seq[:, :1]
        # 生成整数序列 / Generate integer sequence
        for t in range(target_len-1):
            pred, hidden, cell = self.decoder(dec_in, hidden, cell)
            pred = pred[:, -1:, :]
            # 添加元素到列表末尾 / Append element to list end
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


# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
encoder = EncoderLSTM(enc_vocab, emb_dim, hidden_dim, num_layers).to(device)
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
decoder = DecoderLSTM(dec_vocab, emb_dim, hidden_dim, num_layers).to(device)
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
model = Seq2SeqLSTM(encoder, decoder).to(device)
# 打印输出 / Print output
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

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
import tokenizers

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class EncoderLSTM(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class DecoderLSTM(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out = nn.Linear(embedding_dim, vocab_size)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, input_seq, hidden, cell):
        embedded = self.embedding(input_seq)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Seq2SeqLSTM(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, encoder, decoder):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, input_seq, target_seq):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        batch_size, target_len = target_seq.shape
        outputs = []
        _enc_out, hidden, cell = self.encoder(input_seq)
        dec_in = target_seq[:, :1]
        # 生成整数序列 / Generate integer sequence
        for t in range(target_len-1):
            pred, hidden, cell = self.decoder(dec_in, hidden, cell)
            pred = pred[:, -1:, :]
            # 添加元素到列表末尾 / Append element to list end
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


# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
encoder = EncoderLSTM(enc_vocab, emb_dim, hidden_dim, num_layers).to(device)
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
decoder = DecoderLSTM(dec_vocab, emb_dim, hidden_dim, num_layers).to(device)
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
model = Seq2SeqLSTM(encoder, decoder).to(device)
# 打印输出 / Print output
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
# 导入随机数生成模块 / Import random number module
import random
# 导入操作系统接口 / Import OS interface
import os
import unicodedata
import zipfile

import requests
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
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
    # 打开文件（自动关闭） / Open file (auto-close)
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
        # 添加元素到列表末尾 / Append element to list end
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
# 打印输出 / Print output
print("Sample tokenization:")
en_sample, fr_sample = random.choice(text_pairs)
encoded = en_tokenizer.encode(en_sample)
# 打印输出 / Print output
print(f"Original: {en_sample}")
# 打印输出 / Print output
print(f"Tokens: {encoded.tokens}")
# 打印输出 / Print output
print(f"IDs: {encoded.ids}")
# 打印输出 / Print output
print(f"Decoded: {en_tokenizer.decode(encoded.ids)}")
# 打印输出 / Print output
print()

encoded = fr_tokenizer.encode("[start] " + fr_sample + " [end]")
# 打印输出 / Print output
print(f"Original: {fr_sample}")
# 打印输出 / Print output
print(f"Tokens: {encoded.tokens}")
# 打印输出 / Print output
print(f"IDs: {encoded.ids}")
# 打印输出 / Print output
print(f"Decoded: {fr_tokenizer.decode(encoded.ids)}")
# 打印输出 / Print output
print()
```

---
## Step 10 — Create PyTorch dataset for the BPE-encoded translation pairs


```python
# 定义数据集 / Define dataset
class TranslationDataset(torch.utils.data.Dataset):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        # 获取长度 / Get length
        return len(self.text_pairs)

    def __getitem__(self, idx):
        eng, fra = self.text_pairs[idx]
        return eng, "[start] " + fra + " [end]"


def collate_fn(batch):
    # 将多个序列配对 / Pair multiple sequences
    en_str, fr_str = zip(*batch)
    en_enc = en_tokenizer.encode_batch(en_str, add_special_tokens=True)
    fr_enc = fr_tokenizer.encode_batch(fr_str, add_special_tokens=True)
    en_ids = [enc.ids for enc in en_enc]
    fr_ids = [enc.ids for enc in fr_enc]
    return torch.tensor(en_ids), torch.tensor(fr_ids)


BATCH_SIZE = 32
# 定义数据集 / Define dataset
dataset = TranslationDataset(text_pairs)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         collate_fn=collate_fn)
```

---
## Step 11 — Test the dataset

```python
for en_ids, fr_ids in dataloader:
    # 打印输出 / Print output
    print(f"English: {en_ids}")
    # 打印输出 / Print output
    print(f"French: {fr_ids}")
    break
```

---
## Step 12 — Create LSTM seq2seq model for translation


```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class EncoderLSTM(nn.Module):
    """A stacked LSTM encoder with an embedding layer"""
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
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
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
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


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class DecoderLSTM(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out = nn.Linear(embedding_dim, vocab_size)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
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


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Seq2SeqLSTM(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, encoder, decoder):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, input_seq, target_seq):
        """Given the partial target sequence, predict the next token"""
```

---
## Step 17 — input seq = [batch_size, seq_len]
target seq = [batch_size, seq_len]

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
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
# 生成整数序列 / Generate integer sequence
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
            # 添加元素到列表末尾 / Append element to list end
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
# 获取长度 / Get length
enc_vocab = len(en_tokenizer.get_vocab())
# 获取长度 / Get length
dec_vocab = len(fr_tokenizer.get_vocab())
emb_dim = 256
hidden_dim = 256
num_layers = 2
dropout = 0.1
```

---
## Step 25 — Create model

```python
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
encoder = EncoderLSTM(enc_vocab, emb_dim, hidden_dim, num_layers, dropout).to(device)
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
decoder = DecoderLSTM(dec_vocab, emb_dim, hidden_dim, num_layers, dropout).to(device)
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
model = Seq2SeqLSTM(encoder, decoder).to(device)
# 打印输出 / Print output
print(model)

# 打印输出 / Print output
print("Model created with:")
# 打印输出 / Print output
print(f"  Input vocabulary size: {enc_vocab}")
# 打印输出 / Print output
print(f"  Output vocabulary size: {dec_vocab}")
# 打印输出 / Print output
print(f"  Embedding dimension: {emb_dim}")
# 打印输出 / Print output
print(f"  Hidden dimension: {hidden_dim}")
# 打印输出 / Print output
print(f"  Number of layers: {num_layers}")
# 打印输出 / Print output
print(f"  Dropout: {dropout}")
# 打印输出 / Print output
print(f"  Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
```

---
## Step 26 — Train unless model.pth exists

```python
if os.path.exists("seq2seq.pth"):
    # 加载模型参数 / Load model parameters
    model.load_state_dict(torch.load("seq2seq.pth"))
else:
    # Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
    loss_fn = nn.CrossEntropyLoss(ignore_index=fr_tokenizer.token_to_id("[pad]"))
    N_EPOCHS = 30

    # 生成整数序列 / Generate integer sequence
    for epoch in range(N_EPOCHS):
        # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
        model.train()
        epoch_loss = 0
        for en_ids, fr_ids in tqdm.tqdm(dataloader, desc="Training"):
```

---
## Step 27 — Move the "sentences" to device

```python
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
en_ids = en_ids.to(device)
            # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
            fr_ids = fr_ids.to(device)
```

---
## Step 28 — zero the grad, then forward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
            outputs = model(en_ids, fr_ids)
```

---
## Step 29 — compute the loss: compare 3D logits to 2D targets

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
loss = loss_fn(outputs.reshape(-1, dec_vocab), fr_ids[:, 1:].reshape(-1))
            # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
            loss.backward()
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            optimizer.step()
            epoch_loss += loss.item()
        # 打印输出 / Print output
        print(f"Epoch {epoch+1}/{N_EPOCHS}; Avg loss {epoch_loss/len(dataloader)}"
              f"; Latest loss {loss.item()}")
        # 获取模型参数字典 / Get model parameter dictionary
        torch.save(model.state_dict(), f"seq2seq-epoch-{epoch+1}.pth")
```

---
## Step 30 — Test once every 5 epochs

```python
if (epoch+1) % 5 != 0:
            continue
        # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
        model.eval()
        epoch_loss = 0
        # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
        with torch.no_grad():
            for en_ids, fr_ids in tqdm.tqdm(dataloader, desc="Evaluating"):
                # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
                en_ids = en_ids.to(device)
                # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
                fr_ids = fr_ids.to(device)
                outputs = model(en_ids, fr_ids)
                # 改变数组形状（不改变数据） / Reshape array (data unchanged)
                loss = loss_fn(outputs.reshape(-1, dec_vocab), fr_ids[:, 1:].reshape(-1))
                epoch_loss += loss.item()
        # 打印输出 / Print output
        print(f"Eval loss: {epoch_loss/len(dataloader)}")
```

---
## Step 31 — Save the final model

```python
# 获取模型参数字典 / Get model parameter dictionary
torch.save(model.state_dict(), "seq2seq.pth")
```

---
## Step 32 — Test for a few samples

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
N_SAMPLES = 5
MAX_LEN = 60
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
    start_token = torch.tensor([fr_tokenizer.token_to_id("[start]")]).to(device)
    for en, true_fr in random.sample(text_pairs, N_SAMPLES):
        # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
        en_ids = torch.tensor(en_tokenizer.encode(en).ids).unsqueeze(0).to(device)
        _output, hidden, cell = model.encoder(en_ids)
        pred_ids = [start_token]
        # 生成整数序列 / Generate integer sequence
        for _ in range(MAX_LEN):
            # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
            decoder_input = torch.tensor(pred_ids).unsqueeze(0).to(device)
            output, hidden, cell = model.decoder(decoder_input, hidden, cell)
            output = output[:, -1, :].argmax(dim=1)
            # 添加元素到列表末尾 / Append element to list end
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
        # 打印输出 / Print output
        print(f"English: {en}")
        # 打印输出 / Print output
        print(f"French: {true_fr}")
        # 打印输出 / Print output
        print(f"Predicted: {pred_fr}")
        # 打印输出 / Print output
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

# 导入随机数生成模块 / Import random number module
import random
# 导入操作系统接口 / Import OS interface
import os
import unicodedata
import zipfile

import requests
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
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
    # 打开文件（自动关闭） / Open file (auto-close)
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
        # 添加元素到列表末尾 / Append element to list end
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
# 打印输出 / Print output
print("Sample tokenization:")
en_sample, fr_sample = random.choice(text_pairs)
encoded = en_tokenizer.encode(en_sample)
# 打印输出 / Print output
print(f"Original: {en_sample}")
# 打印输出 / Print output
print(f"Tokens: {encoded.tokens}")
# 打印输出 / Print output
print(f"IDs: {encoded.ids}")
# 打印输出 / Print output
print(f"Decoded: {en_tokenizer.decode(encoded.ids)}")
# 打印输出 / Print output
print()

encoded = fr_tokenizer.encode("[start] " + fr_sample + " [end]")
# 打印输出 / Print output
print(f"Original: {fr_sample}")
# 打印输出 / Print output
print(f"Tokens: {encoded.tokens}")
# 打印输出 / Print output
print(f"IDs: {encoded.ids}")
# 打印输出 / Print output
print(f"Decoded: {fr_tokenizer.decode(encoded.ids)}")
# 打印输出 / Print output
print()

#
# Create PyTorch dataset for the BPE-encoded translation pairs
#

# 定义数据集 / Define dataset
class TranslationDataset(torch.utils.data.Dataset):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        # 获取长度 / Get length
        return len(self.text_pairs)

    def __getitem__(self, idx):
        eng, fra = self.text_pairs[idx]
        return eng, "[start] " + fra + " [end]"


def collate_fn(batch):
    # 将多个序列配对 / Pair multiple sequences
    en_str, fr_str = zip(*batch)
    en_enc = en_tokenizer.encode_batch(en_str, add_special_tokens=True)
    fr_enc = fr_tokenizer.encode_batch(fr_str, add_special_tokens=True)
    en_ids = [enc.ids for enc in en_enc]
    fr_ids = [enc.ids for enc in fr_enc]
    return torch.tensor(en_ids), torch.tensor(fr_ids)


BATCH_SIZE = 32
# 定义数据集 / Define dataset
dataset = TranslationDataset(text_pairs)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         collate_fn=collate_fn)

# Test the dataset
for en_ids, fr_ids in dataloader:
    # 打印输出 / Print output
    print(f"English: {en_ids}")
    # 打印输出 / Print output
    print(f"French: {fr_ids}")
    break

#
# Create LSTM seq2seq model for translation
#

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class EncoderLSTM(nn.Module):
    """A stacked LSTM encoder with an embedding layer"""
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
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
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, input_seq):
        # input seq = [batch_size, seq_len] -> embedded = [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(input_seq)
        # outputs = [batch_size, seq_len, embedding_dim]
        # hidden = cell = [n_layers, batch_size, hidden_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class DecoderLSTM(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out = nn.Linear(embedding_dim, vocab_size)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, input_seq, hidden, cell):
        # input seq = [batch_size, seq_len] -> embedded = [batch_size, seq_len, embedding_dim]
        # hidden = cell = [n_layers, batch_size, hidden_dim]
        embedded = self.embedding(input_seq)
        # output = [batch_size, seq_len, embedding_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Seq2SeqLSTM(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, encoder, decoder):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, input_seq, target_seq):
        """Given the partial target sequence, predict the next token"""
        # input seq = [batch_size, seq_len]
        # target seq = [batch_size, seq_len]
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        batch_size, target_len = target_seq.shape
        # storing output logits
        outputs = []
        # encoder forward pass
        _enc_out, hidden, cell = self.encoder(input_seq)
        dec_in = target_seq[:, :1]
        # decoder forward pass
        # 生成整数序列 / Generate integer sequence
        for t in range(target_len-1):
            # last target token and hidden states -> next token
            pred, hidden, cell = self.decoder(dec_in, hidden, cell)
            # store the prediction
            pred = pred[:, -1:, :]
            # 添加元素到列表末尾 / Append element to list end
            outputs.append(pred)
            # use the predicted token as the next input
            dec_in = torch.cat([dec_in, pred.argmax(dim=2)], dim=1)
        outputs = torch.cat(outputs, dim=1)
        return outputs


# Initialize model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 获取长度 / Get length
enc_vocab = len(en_tokenizer.get_vocab())
# 获取长度 / Get length
dec_vocab = len(fr_tokenizer.get_vocab())
emb_dim = 256
hidden_dim = 256
num_layers = 2
dropout = 0.1

# Create model
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
encoder = EncoderLSTM(enc_vocab, emb_dim, hidden_dim, num_layers, dropout).to(device)
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
decoder = DecoderLSTM(dec_vocab, emb_dim, hidden_dim, num_layers, dropout).to(device)
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
model = Seq2SeqLSTM(encoder, decoder).to(device)
# 打印输出 / Print output
print(model)

# 打印输出 / Print output
print("Model created with:")
# 打印输出 / Print output
print(f"  Input vocabulary size: {enc_vocab}")
# 打印输出 / Print output
print(f"  Output vocabulary size: {dec_vocab}")
# 打印输出 / Print output
print(f"  Embedding dimension: {emb_dim}")
# 打印输出 / Print output
print(f"  Hidden dimension: {hidden_dim}")
# 打印输出 / Print output
print(f"  Number of layers: {num_layers}")
# 打印输出 / Print output
print(f"  Dropout: {dropout}")
# 打印输出 / Print output
print(f"  Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Train unless model.pth exists
if os.path.exists("seq2seq.pth"):
    # 加载模型参数 / Load model parameters
    model.load_state_dict(torch.load("seq2seq.pth"))
else:
    # Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
    loss_fn = nn.CrossEntropyLoss(ignore_index=fr_tokenizer.token_to_id("[pad]"))
    N_EPOCHS = 30

    # 生成整数序列 / Generate integer sequence
    for epoch in range(N_EPOCHS):
        # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
        model.train()
        epoch_loss = 0
        for en_ids, fr_ids in tqdm.tqdm(dataloader, desc="Training"):
            # Move the "sentences" to device
            # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
            en_ids = en_ids.to(device)
            # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
            fr_ids = fr_ids.to(device)
            # zero the grad, then forward pass
            # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
            optimizer.zero_grad()
            outputs = model(en_ids, fr_ids)
            # compute the loss: compare 3D logits to 2D targets
            # 改变数组形状（不改变数据） / Reshape array (data unchanged)
            loss = loss_fn(outputs.reshape(-1, dec_vocab), fr_ids[:, 1:].reshape(-1))
            # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
            loss.backward()
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            optimizer.step()
            epoch_loss += loss.item()
        # 打印输出 / Print output
        print(f"Epoch {epoch+1}/{N_EPOCHS}; Avg loss {epoch_loss/len(dataloader)}"
              f"; Latest loss {loss.item()}")
        # 获取模型参数字典 / Get model parameter dictionary
        torch.save(model.state_dict(), f"seq2seq-epoch-{epoch+1}.pth")
        # Test once every 5 epochs
        if (epoch+1) % 5 != 0:
            continue
        # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
        model.eval()
        epoch_loss = 0
        # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
        with torch.no_grad():
            for en_ids, fr_ids in tqdm.tqdm(dataloader, desc="Evaluating"):
                # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
                en_ids = en_ids.to(device)
                # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
                fr_ids = fr_ids.to(device)
                outputs = model(en_ids, fr_ids)
                # 改变数组形状（不改变数据） / Reshape array (data unchanged)
                loss = loss_fn(outputs.reshape(-1, dec_vocab), fr_ids[:, 1:].reshape(-1))
                epoch_loss += loss.item()
        # 打印输出 / Print output
        print(f"Eval loss: {epoch_loss/len(dataloader)}")

    # Save the final model
    # 获取模型参数字典 / Get model parameter dictionary
    torch.save(model.state_dict(), "seq2seq.pth")

# Test for a few samples
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
N_SAMPLES = 5
MAX_LEN = 60
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
    start_token = torch.tensor([fr_tokenizer.token_to_id("[start]")]).to(device)
    for en, true_fr in random.sample(text_pairs, N_SAMPLES):
        # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
        en_ids = torch.tensor(en_tokenizer.encode(en).ids).unsqueeze(0).to(device)
        _output, hidden, cell = model.encoder(en_ids)
        pred_ids = [start_token]
        # 生成整数序列 / Generate integer sequence
        for _ in range(MAX_LEN):
            # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
            decoder_input = torch.tensor(pred_ids).unsqueeze(0).to(device)
            output, hidden, cell = model.decoder(decoder_input, hidden, cell)
            output = output[:, -1, :].argmax(dim=1)
            # 添加元素到列表末尾 / Append element to list end
            pred_ids.append(output.item())
            # early stop if the predicted token is the end token
            if pred_ids[-1] == fr_tokenizer.token_to_id("[end]"):
                break
        # Decode the predicted IDs
        pred_fr = fr_tokenizer.decode(pred_ids)
        # 打印输出 / Print output
        print(f"English: {en}")
        # 打印输出 / Print output
        print(f"French: {true_fr}")
        # 打印输出 / Print output
        print(f"Predicted: {pred_fr}")
        # 打印输出 / Print output
        print()
```

---

### Chapter Summary / 章节总结



---
