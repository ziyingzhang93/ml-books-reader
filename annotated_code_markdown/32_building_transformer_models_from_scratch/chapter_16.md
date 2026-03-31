# 从零构建Transformer / Building Transformers from Scratch
## Chapter 16

---

### Download



---

### Rope



---

### Gqa

# 05 — Gqa / 05 Gqa

**Chapter 16 — File 3 of 5 / 第16章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **projection**.

本脚本演示 **projection**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class GQA(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = dropout
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, q, k, v, mask=None, rope=None):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        q_batch_size, q_seq_len, hidden_dim = q.shape
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        k_batch_size, k_seq_len, hidden_dim = k.shape
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        v_batch_size, v_seq_len, hidden_dim = v.shape
```

---
## Step 2 — projection

```python
q = self.q_proj(q) \
            .view(q_batch_size, q_seq_len, -1, self.head_dim) \
            .transpose(1, 2)
        k = self.k_proj(k) \
            .view(k_batch_size, k_seq_len, -1, self.head_dim) \
            .transpose(1, 2)
        v = self.v_proj(v) \
            .view(v_batch_size, v_seq_len, -1, self.head_dim) \
            .transpose(1, 2)
```

---
## Step 3 — apply rotary positional encoding

```python
if rope:
            q = rope(q)
            k = rope(k)
```

---
## Step 4 — compute grouped query attention

```python
q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        output = F.scaled_dot_product_attention(q, k, v,
                                                attn_mask=mask,
                                                dropout_p=self.dropout,
                                                enable_gqa=True)
        output = output.transpose(1, 2) \
                       # 改变数组形状（不改变数据） / Reshape array (data unchanged)
                       .reshape(q_batch_size, q_seq_len, hidden_dim) \
                       .contiguous()
        output = self.out_proj(output)
        return output
```

---
## Learning Notes / 学习笔记

- **概念**: projection 是机器学习中的常用技术。  
  *projection is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gqa / 05 Gqa
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class GQA(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = dropout
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, q, k, v, mask=None, rope=None):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        q_batch_size, q_seq_len, hidden_dim = q.shape
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        k_batch_size, k_seq_len, hidden_dim = k.shape
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        v_batch_size, v_seq_len, hidden_dim = v.shape

        # projection
        q = self.q_proj(q) \
            .view(q_batch_size, q_seq_len, -1, self.head_dim) \
            .transpose(1, 2)
        k = self.k_proj(k) \
            .view(k_batch_size, k_seq_len, -1, self.head_dim) \
            .transpose(1, 2)
        v = self.v_proj(v) \
            .view(v_batch_size, v_seq_len, -1, self.head_dim) \
            .transpose(1, 2)

        # apply rotary positional encoding
        if rope:
            q = rope(q)
            k = rope(k)

        # compute grouped query attention
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        output = F.scaled_dot_product_attention(q, k, v,
                                                attn_mask=mask,
                                                dropout_p=self.dropout,
                                                enable_gqa=True)
        output = output.transpose(1, 2) \
                       # 改变数组形状（不改变数据） / Reshape array (data unchanged)
                       .reshape(q_batch_size, q_seq_len, hidden_dim) \
                       .contiguous()
        output = self.out_proj(output)
        return output
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Swiglu



---

### Transformer

# 18 — Transformer / 数据变换

**Chapter 16 — File 5 of 5 / 第16章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Transformer model implementation in PyTorch**.

本脚本演示 **Transformer model implementation in PyTorch**。

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
## Step 1 — Transformer model implementation in PyTorch

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
import torch.nn.functional as F
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
    def __init__(self, text_pairs, en_tokenizer, fr_tokenizer):
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
dataset = TranslationDataset(text_pairs, en_tokenizer, fr_tokenizer)
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
## Step 12 — Transformer model components


```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class RotaryPositionalEncoding(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, max_seq_len=1024):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        N = 10000
        # 生成整数序列 / Generate integer sequence
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        # 生成整数序列 / Generate integer sequence
        position = torch.arange(max_seq_len).float()
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class SwiGLU(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, hidden_dim, intermediate_dim):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.gate = nn.Linear(hidden_dim, intermediate_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.down = nn.Linear(intermediate_dim, hidden_dim)
        self.act = nn.SiLU()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.gate(x)) * self.up(x)
        x = self.down(x)
        return x


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class GQA(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = dropout
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, q, k, v, mask=None, rope=None):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        q_batch_size, q_seq_len, hidden_dim = q.shape
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        k_batch_size, k_seq_len, hidden_dim = k.shape
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        v_batch_size, v_seq_len, hidden_dim = v.shape
```

---
## Step 13 — projection

```python
q = self.q_proj(q).view(q_batch_size, q_seq_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(k_batch_size, k_seq_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(v_batch_size, v_seq_len, -1, self.head_dim).transpose(1, 2)
```

---
## Step 14 — apply rotary positional encoding

```python
if rope:
            q = rope(q)
            k = rope(k)
```

---
## Step 15 — compute grouped query attention

```python
q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        output = F.scaled_dot_product_attention(q, k, v,
                                                attn_mask=mask,
                                                dropout_p=self.dropout,
                                                enable_gqa=True)
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
        output = output.transpose(1, 2).reshape(q_batch_size, q_seq_len, hidden_dim).contiguous()
        output = self.out_proj(output)
        return output


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class EncoderLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x, mask=None, rope=None):
```

---
## Step 16 — self-attention sublayer

```python
out = x
        out = self.norm1(x)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
```

---
## Step 17 — MLP sublayer

```python
out = self.norm2(x)
        out = self.mlp(out)
        return out + x


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class DecoderLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.cross_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.norm3 = nn.RMSNorm(hidden_dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x, enc_out, mask=None, rope=None):
```

---
## Step 18 — self-attention sublayer

```python
out = x
        out = self.norm1(out)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
```

---
## Step 19 — cross-attention sublayer

```python
out = self.norm2(x)
        out = self.cross_attn(out, enc_out, enc_out, None, rope)
        x = out + x
```

---
## Step 20 — MLP sublayer

```python
x = out + x
        out = self.norm3(x)
        out = self.mlp(out)
        return out + x


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Transformer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, num_layers, num_heads, num_kv_heads, hidden_dim,
                 max_seq_len, vocab_size_src, vocab_size_tgt, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.rope = RotaryPositionalEncoding(hidden_dim // num_heads, max_seq_len)
        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.src_embedding = nn.Embedding(vocab_size_src, hidden_dim)
        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.tgt_embedding = nn.Embedding(vocab_size_tgt, hidden_dim)
        self.encoders = nn.ModuleList([
            # 生成整数序列 / Generate integer sequence
            EncoderLayer(hidden_dim, num_heads, num_kv_heads, dropout) for _ in range(num_layers)
        ])
        self.decoders = nn.ModuleList([
            # 生成整数序列 / Generate integer sequence
            DecoderLayer(hidden_dim, num_heads, num_kv_heads, dropout) for _ in range(num_layers)
        ])
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out = nn.Linear(hidden_dim, vocab_size_tgt)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
```

---
## Step 21 — Encoder

```python
x = self.src_embedding(src_ids)
        for encoder in self.encoders:
            x = encoder(x, src_mask, self.rope)
        enc_out = x
```

---
## Step 22 — Decoder

```python
x = self.tgt_embedding(tgt_ids)
        for decoder in self.decoders:
            x = decoder(x, enc_out, tgt_mask, self.rope)
        return self.out(x)


model_config = {
    "num_layers": 4,
    "num_heads": 8,
    "num_kv_heads": 4,
    "hidden_dim": 128,
    "max_seq_len": 768,
    # 获取长度 / Get length
    "vocab_size_src": len(en_tokenizer.get_vocab()),
    # 获取长度 / Get length
    "vocab_size_tgt": len(fr_tokenizer.get_vocab()),
    "dropout": 0.1,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
model = Transformer(**model_config).to(device)
# 打印输出 / Print output
print(model)
```

---
## Step 23 — Training

```python
# 打印输出 / Print output
print("Model created with:")
# 打印输出 / Print output
print(f"  Input vocabulary size: {model_config['vocab_size_src']}")
# 打印输出 / Print output
print(f"  Output vocabulary size: {model_config['vocab_size_tgt']}")
# 打印输出 / Print output
print(f"  Number of layers: {model_config['num_layers']}")
# 打印输出 / Print output
print(f"  Number of heads: {model_config['num_heads']}")
# 打印输出 / Print output
print(f"  Number of KV heads: {model_config['num_kv_heads']}")
# 打印输出 / Print output
print(f"  Hidden dimension: {model_config['hidden_dim']}")
# 打印输出 / Print output
print(f"  Max sequence length: {model_config['max_seq_len']}")
# 打印输出 / Print output
print(f"  Dropout: {model_config['dropout']}")
# 打印输出 / Print output
print(f"  Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

def create_causal_mask(seq_len, device):
    """
    Create a causal mask for autoregressive attention.

    Args:
        seq_len: Length of the sequence

    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask


def create_padding_mask(batch, padding_token_id):
    """
    Create a padding mask for a batch of sequences.

    Args:
        batch: Batch of sequences, shape (batch_size, seq_len)
        padding_token_id: ID of the padding token

    Returns:
        Padding mask of shape (batch_size, seq_len, seq_len)
    """
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    batch_size, seq_len = batch.shape
    device = batch.device
    padded = torch.zeros_like(batch, device=device).float() \
                  .masked_fill(batch == padding_token_id, float('-inf'))
    # 创建全零张量 / Create tensor of zeros
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device) + \
           padded[:, :, None] + \
           padded[:, None, :]
    return mask[:, None, :, :]
```

---
## Step 24 — Train unless model.pth exists

```python
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss(ignore_index=fr_tokenizer.token_to_id("[pad]"))
if os.path.exists("transformer.pth"):
    # 加载模型参数 / Load model parameters
    model.load_state_dict(torch.load("transformer.pth"))
else:
    N_EPOCHS = 60
    LR = 0.005
    WARMUP_STEPS = 1000
    CLIP_NORM = 5.0
    best_loss = float('inf')
    # Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
    optimizer = optim.Adam(model.parameters(), lr=LR)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        # 获取长度 / Get length
        optimizer, T_max=N_EPOCHS * len(dataloader) - WARMUP_STEPS, eta_min=0)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_STEPS])
    # 打印输出 / Print output
    print(f"Training for {N_EPOCHS} epochs with {len(dataloader)} steps per epoch")

    # 生成整数序列 / Generate integer sequence
    for epoch in range(N_EPOCHS):
        # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
        model.train()
        epoch_loss = 0
        for en_ids, fr_ids in tqdm.tqdm(dataloader, desc="Training"):
```

---
## Step 25 — Move the "sentences" to device

```python
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
en_ids = en_ids.to(device)
            # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
            fr_ids = fr_ids.to(device)
```

---
## Step 26 — create source mask as padding mask, target mask as causal mask

```python
src_mask = create_padding_mask(en_ids, en_tokenizer.token_to_id("[pad]"))
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            tgt_mask = create_causal_mask(fr_ids.shape[1], device).unsqueeze(0) + \
                       create_padding_mask(fr_ids, fr_tokenizer.token_to_id("[pad]"))
```

---
## Step 27 — zero the grad, then forward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
            outputs = model(en_ids, fr_ids, src_mask, tgt_mask)
```

---
## Step 28 — compute the loss: compare 3D logits to 2D targets

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
loss = loss_fn(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]),
                           # 改变数组形状（不改变数据） / Reshape array (data unchanged)
                           fr_ids[:, 1:].reshape(-1))
            # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
            loss.backward()
            # 获取所有可训练参数 / Get all trainable parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=False)
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            optimizer.step()
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            scheduler.step()
            epoch_loss += loss.item()
        # 打印输出 / Print output
        print(f"Epoch {epoch+1}/{N_EPOCHS}; Avg loss {epoch_loss/len(dataloader)}"
              f"; Latest loss {loss.item()}")
```

---
## Step 29 — Test

```python
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
                src_mask = create_padding_mask(en_ids, en_tokenizer.token_to_id("[pad]"))
                # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
                tgt_mask = create_causal_mask(fr_ids.shape[1], device).unsqueeze(0) + \
                           create_padding_mask(fr_ids, fr_tokenizer.token_to_id("[pad]"))
                outputs = model(en_ids, fr_ids, src_mask, tgt_mask)
                # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
                loss = loss_fn(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]),
                               # 改变数组形状（不改变数据） / Reshape array (data unchanged)
                               fr_ids[:, 1:].reshape(-1))
                epoch_loss += loss.item()
        # 打印输出 / Print output
        print(f"Eval loss: {epoch_loss/len(dataloader)}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # 获取模型参数字典 / Get model parameter dictionary
            torch.save(model.state_dict(), f"transformer-epoch-{epoch+1}.pth")
```

---
## Step 30 — Save the final model after training

```python
# 获取模型参数字典 / Get model parameter dictionary
torch.save(model.state_dict(), "transformer.pth")
```

---
## Step 31 — Test for a few samples

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
N_SAMPLES = 5
MAX_LEN = 60
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
    start_token = torch.tensor([fr_tokenizer.token_to_id("[start]")]).to(device)
    for en, true_fr in random.sample(dataset.text_pairs, N_SAMPLES):
        # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
        en_ids = torch.tensor(en_tokenizer.encode(en).ids).unsqueeze(0).to(device)
```

---
## Step 32 — get context from encoder

```python
src_mask = create_padding_mask(en_ids, en_tokenizer.token_to_id("[pad]"))
        x = model.src_embedding(en_ids)
        for encoder in model.encoders:
            x = encoder(x, src_mask, model.rope)
        enc_out = x
```

---
## Step 33 — generate output from decoder

```python
fr_ids = start_token.unsqueeze(0)
        # 生成整数序列 / Generate integer sequence
        for _ in range(MAX_LEN):
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            tgt_mask = create_causal_mask(fr_ids.shape[1], device).unsqueeze(0)
            tgt_mask = tgt_mask + create_padding_mask(fr_ids, fr_tokenizer.token_to_id("[pad]"))
            x = model.tgt_embedding(fr_ids)
            for decoder in model.decoders:
                x = decoder(x, enc_out, tgt_mask, model.rope)
            outputs = model.out(x)

            outputs = outputs.argmax(dim=-1)
            fr_ids = torch.cat([fr_ids, outputs[:, -1:]], axis=-1)
            if fr_ids[0, -1] == fr_tokenizer.token_to_id("[end]"):
                break
```

---
## Step 34 — Decode the predicted IDs

```python
pred_fr = fr_tokenizer.decode(fr_ids[0].tolist())
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

- **概念**: Transformer model implementation in PyTorch 是机器学习中的常用技术。  
  *Transformer model implementation in PyTorch is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataLoader` | 数据加载器，批量读取数据 | Loads data in batches for training |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `nn.Embedding` | 将整数ID映射为稠密向量 | Map integer IDs to dense vectors |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Transformer / 数据变换
# Complete Code / 完整代码
# ===============================

# Transformer model implementation in PyTorch

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
import torch.nn.functional as F
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
    def __init__(self, text_pairs, en_tokenizer, fr_tokenizer):
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
dataset = TranslationDataset(text_pairs, en_tokenizer, fr_tokenizer)
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
# Transformer model components
#

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class RotaryPositionalEncoding(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dim, max_seq_len=1024):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        N = 10000
        # 生成整数序列 / Generate integer sequence
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        # 生成整数序列 / Generate integer sequence
        position = torch.arange(max_seq_len).float()
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class SwiGLU(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, hidden_dim, intermediate_dim):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.gate = nn.Linear(hidden_dim, intermediate_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.down = nn.Linear(intermediate_dim, hidden_dim)
        self.act = nn.SiLU()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.gate(x)) * self.up(x)
        x = self.down(x)
        return x


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class GQA(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = dropout
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, q, k, v, mask=None, rope=None):
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        q_batch_size, q_seq_len, hidden_dim = q.shape
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        k_batch_size, k_seq_len, hidden_dim = k.shape
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        v_batch_size, v_seq_len, hidden_dim = v.shape

        # projection
        q = self.q_proj(q).view(q_batch_size, q_seq_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(k_batch_size, k_seq_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(v_batch_size, v_seq_len, -1, self.head_dim).transpose(1, 2)

        # apply rotary positional encoding
        if rope:
            q = rope(q)
            k = rope(k)

        # compute grouped query attention
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        output = F.scaled_dot_product_attention(q, k, v,
                                                attn_mask=mask,
                                                dropout_p=self.dropout,
                                                enable_gqa=True)
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
        output = output.transpose(1, 2).reshape(q_batch_size, q_seq_len, hidden_dim).contiguous()
        output = self.out_proj(output)
        return output


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class EncoderLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x, mask=None, rope=None):
        # self-attention sublayer
        out = x
        out = self.norm1(x)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
        # MLP sublayer
        out = self.norm2(x)
        out = self.mlp(out)
        return out + x


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class DecoderLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.cross_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.norm3 = nn.RMSNorm(hidden_dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x, enc_out, mask=None, rope=None):
        # self-attention sublayer
        out = x
        out = self.norm1(out)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
        # cross-attention sublayer
        out = self.norm2(x)
        out = self.cross_attn(out, enc_out, enc_out, None, rope)
        x = out + x
        # MLP sublayer
        x = out + x
        out = self.norm3(x)
        out = self.mlp(out)
        return out + x


# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Transformer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, num_layers, num_heads, num_kv_heads, hidden_dim,
                 max_seq_len, vocab_size_src, vocab_size_tgt, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.rope = RotaryPositionalEncoding(hidden_dim // num_heads, max_seq_len)
        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.src_embedding = nn.Embedding(vocab_size_src, hidden_dim)
        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.tgt_embedding = nn.Embedding(vocab_size_tgt, hidden_dim)
        self.encoders = nn.ModuleList([
            # 生成整数序列 / Generate integer sequence
            EncoderLayer(hidden_dim, num_heads, num_kv_heads, dropout) for _ in range(num_layers)
        ])
        self.decoders = nn.ModuleList([
            # 生成整数序列 / Generate integer sequence
            DecoderLayer(hidden_dim, num_heads, num_kv_heads, dropout) for _ in range(num_layers)
        ])
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out = nn.Linear(hidden_dim, vocab_size_tgt)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        # Encoder
        x = self.src_embedding(src_ids)
        for encoder in self.encoders:
            x = encoder(x, src_mask, self.rope)
        enc_out = x
        # Decoder
        x = self.tgt_embedding(tgt_ids)
        for decoder in self.decoders:
            x = decoder(x, enc_out, tgt_mask, self.rope)
        return self.out(x)


model_config = {
    "num_layers": 4,
    "num_heads": 8,
    "num_kv_heads": 4,
    "hidden_dim": 128,
    "max_seq_len": 768,
    # 获取长度 / Get length
    "vocab_size_src": len(en_tokenizer.get_vocab()),
    # 获取长度 / Get length
    "vocab_size_tgt": len(fr_tokenizer.get_vocab()),
    "dropout": 0.1,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
model = Transformer(**model_config).to(device)
# 打印输出 / Print output
print(model)

# Training

# 打印输出 / Print output
print("Model created with:")
# 打印输出 / Print output
print(f"  Input vocabulary size: {model_config['vocab_size_src']}")
# 打印输出 / Print output
print(f"  Output vocabulary size: {model_config['vocab_size_tgt']}")
# 打印输出 / Print output
print(f"  Number of layers: {model_config['num_layers']}")
# 打印输出 / Print output
print(f"  Number of heads: {model_config['num_heads']}")
# 打印输出 / Print output
print(f"  Number of KV heads: {model_config['num_kv_heads']}")
# 打印输出 / Print output
print(f"  Hidden dimension: {model_config['hidden_dim']}")
# 打印输出 / Print output
print(f"  Max sequence length: {model_config['max_seq_len']}")
# 打印输出 / Print output
print(f"  Dropout: {model_config['dropout']}")
# 打印输出 / Print output
print(f"  Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

def create_causal_mask(seq_len, device):
    """
    Create a causal mask for autoregressive attention.

    Args:
        seq_len: Length of the sequence

    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask


def create_padding_mask(batch, padding_token_id):
    """
    Create a padding mask for a batch of sequences.

    Args:
        batch: Batch of sequences, shape (batch_size, seq_len)
        padding_token_id: ID of the padding token

    Returns:
        Padding mask of shape (batch_size, seq_len, seq_len)
    """
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    batch_size, seq_len = batch.shape
    device = batch.device
    padded = torch.zeros_like(batch, device=device).float() \
                  .masked_fill(batch == padding_token_id, float('-inf'))
    # 创建全零张量 / Create tensor of zeros
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device) + \
           padded[:, :, None] + \
           padded[:, None, :]
    return mask[:, None, :, :]


# Train unless model.pth exists
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss(ignore_index=fr_tokenizer.token_to_id("[pad]"))
if os.path.exists("transformer.pth"):
    # 加载模型参数 / Load model parameters
    model.load_state_dict(torch.load("transformer.pth"))
else:
    N_EPOCHS = 60
    LR = 0.005
    WARMUP_STEPS = 1000
    CLIP_NORM = 5.0
    best_loss = float('inf')
    # Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
    optimizer = optim.Adam(model.parameters(), lr=LR)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        # 获取长度 / Get length
        optimizer, T_max=N_EPOCHS * len(dataloader) - WARMUP_STEPS, eta_min=0)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_STEPS])
    # 打印输出 / Print output
    print(f"Training for {N_EPOCHS} epochs with {len(dataloader)} steps per epoch")

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
            # create source mask as padding mask, target mask as causal mask
            src_mask = create_padding_mask(en_ids, en_tokenizer.token_to_id("[pad]"))
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            tgt_mask = create_causal_mask(fr_ids.shape[1], device).unsqueeze(0) + \
                       create_padding_mask(fr_ids, fr_tokenizer.token_to_id("[pad]"))
            # zero the grad, then forward pass
            # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
            optimizer.zero_grad()
            outputs = model(en_ids, fr_ids, src_mask, tgt_mask)
            # compute the loss: compare 3D logits to 2D targets
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            loss = loss_fn(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]),
                           # 改变数组形状（不改变数据） / Reshape array (data unchanged)
                           fr_ids[:, 1:].reshape(-1))
            # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
            loss.backward()
            # 获取所有可训练参数 / Get all trainable parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=False)
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            optimizer.step()
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            scheduler.step()
            epoch_loss += loss.item()
        # 打印输出 / Print output
        print(f"Epoch {epoch+1}/{N_EPOCHS}; Avg loss {epoch_loss/len(dataloader)}"
              f"; Latest loss {loss.item()}")
        # Test
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
                src_mask = create_padding_mask(en_ids, en_tokenizer.token_to_id("[pad]"))
                # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
                tgt_mask = create_causal_mask(fr_ids.shape[1], device).unsqueeze(0) + \
                           create_padding_mask(fr_ids, fr_tokenizer.token_to_id("[pad]"))
                outputs = model(en_ids, fr_ids, src_mask, tgt_mask)
                # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
                loss = loss_fn(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]),
                               # 改变数组形状（不改变数据） / Reshape array (data unchanged)
                               fr_ids[:, 1:].reshape(-1))
                epoch_loss += loss.item()
        # 打印输出 / Print output
        print(f"Eval loss: {epoch_loss/len(dataloader)}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # 获取模型参数字典 / Get model parameter dictionary
            torch.save(model.state_dict(), f"transformer-epoch-{epoch+1}.pth")

    # Save the final model after training
    # 获取模型参数字典 / Get model parameter dictionary
    torch.save(model.state_dict(), "transformer.pth")

# Test for a few samples
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
N_SAMPLES = 5
MAX_LEN = 60
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
    start_token = torch.tensor([fr_tokenizer.token_to_id("[start]")]).to(device)
    for en, true_fr in random.sample(dataset.text_pairs, N_SAMPLES):
        # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
        en_ids = torch.tensor(en_tokenizer.encode(en).ids).unsqueeze(0).to(device)

        # get context from encoder
        src_mask = create_padding_mask(en_ids, en_tokenizer.token_to_id("[pad]"))
        x = model.src_embedding(en_ids)
        for encoder in model.encoders:
            x = encoder(x, src_mask, model.rope)
        enc_out = x

        # generate output from decoder
        fr_ids = start_token.unsqueeze(0)
        # 生成整数序列 / Generate integer sequence
        for _ in range(MAX_LEN):
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            tgt_mask = create_causal_mask(fr_ids.shape[1], device).unsqueeze(0)
            tgt_mask = tgt_mask + create_padding_mask(fr_ids, fr_tokenizer.token_to_id("[pad]"))
            x = model.tgt_embedding(fr_ids)
            for decoder in model.decoders:
                x = decoder(x, enc_out, tgt_mask, model.rope)
            outputs = model.out(x)

            outputs = outputs.argmax(dim=-1)
            fr_ids = torch.cat([fr_ids, outputs[:, -1:]], axis=-1)
            if fr_ids[0, -1] == fr_tokenizer.token_to_id("[end]"):
                break

        # Decode the predicted IDs
        pred_fr = fr_tokenizer.decode(fr_ids[0].tolist())
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
