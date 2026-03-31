# 从零构建Transformer / Building Transformers from Scratch
## Chapter 17

---

### Download

# 02 — Download / 02 Download

**Chapter 17 — File 1 of 2 / 第17章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Download novels from Project Gutenberg**.

本脚本演示 **Download novels from Project Gutenberg**。

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
```

---
## Step 2 — Download novels from Project Gutenberg

```python
DATASOURCE = {
    "moby_dick": "https://www.gutenberg.org/ebooks/2701.txt.utf-8",
    "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
    "dracula": "https://www.gutenberg.org/ebooks/345.txt.utf-8",
    "little_women": "https://www.gutenberg.org/ebooks/37106.txt.utf-8",
    "pride_and_prejudice": "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
    "alice_in_wonderland": "https://www.gutenberg.org/ebooks/11.txt.utf-8",
    "crime_and_punishment": "https://www.gutenberg.org/ebooks/2554.txt.utf-8",
    "tom_sawyer": "https://www.gutenberg.org/ebooks/74.txt.utf-8",
    "tale_of_two_cities": "https://www.gutenberg.org/ebooks/98.txt.utf-8",
    "sherlock_holmes": "https://www.gutenberg.org/ebooks/1661.txt.utf-8",
    "war_and_peace": "https://www.gutenberg.org/ebooks/2600.txt.utf-8",
}
# 获取字典的键值对 / Get dict key-value pairs
for filename, url in DATASOURCE.items():
    if not os.path.exists(f"{filename}.txt"):
        response = requests.get(url)
        # 打开文件（自动关闭） / Open file (auto-close)
        with open(f"{filename}.txt", "wb") as f:
            f.write(response.content)
```

---
## Learning Notes / 学习笔记

- **概念**: Download novels from Project Gutenberg 是机器学习中的常用技术。  
  *Download novels from Project Gutenberg is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Download / 02 Download
# Complete Code / 完整代码
# ===============================

# 导入操作系统接口 / Import OS interface
import os
import requests

# Download novels from Project Gutenberg
DATASOURCE = {
    "moby_dick": "https://www.gutenberg.org/ebooks/2701.txt.utf-8",
    "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
    "dracula": "https://www.gutenberg.org/ebooks/345.txt.utf-8",
    "little_women": "https://www.gutenberg.org/ebooks/37106.txt.utf-8",
    "pride_and_prejudice": "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
    "alice_in_wonderland": "https://www.gutenberg.org/ebooks/11.txt.utf-8",
    "crime_and_punishment": "https://www.gutenberg.org/ebooks/2554.txt.utf-8",
    "tom_sawyer": "https://www.gutenberg.org/ebooks/74.txt.utf-8",
    "tale_of_two_cities": "https://www.gutenberg.org/ebooks/98.txt.utf-8",
    "sherlock_holmes": "https://www.gutenberg.org/ebooks/1661.txt.utf-8",
    "war_and_peace": "https://www.gutenberg.org/ebooks/2600.txt.utf-8",
}
# 获取字典的键值对 / Get dict key-value pairs
for filename, url in DATASOURCE.items():
    if not os.path.exists(f"{filename}.txt"):
        response = requests.get(url)
        # 打开文件（自动关闭） / Open file (auto-close)
        with open(f"{filename}.txt", "wb") as f:
            f.write(response.content)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Textgen

# 08 — Textgen / 08 Textgen

**Chapter 17 — File 2 of 2 / 第17章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Download novels from Project Gutenberg**.

本脚本演示 **Download novels from Project Gutenberg**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
# 导入操作系统接口 / Import OS interface
import os
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
## Step 2 — Download novels from Project Gutenberg

```python
DATASOURCE = {
    "moby_dick": "https://www.gutenberg.org/ebooks/2701.txt.utf-8",
    "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
    "dracula": "https://www.gutenberg.org/ebooks/345.txt.utf-8",
    "little_women": "https://www.gutenberg.org/ebooks/37106.txt.utf-8",
    "pride_and_prejudice": "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
    "alice_in_wonderland": "https://www.gutenberg.org/ebooks/11.txt.utf-8",
    "crime_and_punishment": "https://www.gutenberg.org/ebooks/2554.txt.utf-8",
    "tom_sawyer": "https://www.gutenberg.org/ebooks/74.txt.utf-8",
    "tale_of_two_cities": "https://www.gutenberg.org/ebooks/98.txt.utf-8",
    "sherlock_holmes": "https://www.gutenberg.org/ebooks/1661.txt.utf-8",
    "war_and_peace": "https://www.gutenberg.org/ebooks/2600.txt.utf-8",
}
# 获取字典的键值对 / Get dict key-value pairs
for filename, url in DATASOURCE.items():
    if not os.path.exists(f"{filename}.txt"):
        response = requests.get(url)
        # 打开文件（自动关闭） / Open file (auto-close)
        with open(f"{filename}.txt", "wb") as f:
            f.write(response.content)
```

---
## Step 3 — Read and preprocess the text

```python
def preprocess_gutenberg(filename):
    # 打开文件（自动关闭） / Open file (auto-close)
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
```

---
## Step 4 — Find the start and end of the actual content

```python
start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = text.find("\n", start) + 1
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
```

---
## Step 5 — Extract the main content

```python
text = text[start:end].strip()
```

---
## Step 6 — Basic preprocessing
Remove multiple newlines and spaces

```python
text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text

def get_dataset_text():
    all_text = []
    for filename in DATASOURCE:
        text = preprocess_gutenberg(f"{filename}.txt")
        # 添加元素到列表末尾 / Append element to list end
        all_text.append(text)
    return all_text
```

---
## Step 7 — Tokenization with BPE

```python
if os.path.exists("gutenberg_tokenizer.json"):
    tokenizer = tokenizers.Tokenizer.from_file("gutenberg_tokenizer.json")
else:
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
```

---
## Step 8 — Configure pre-tokenizer add space at beginning of the sentence

```python
tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
```

---
## Step 9 — Configure decoder so that the boundary symbols will be removed

```python
tokenizer.decoder = tokenizers.decoders.ByteLevel()
```

---
## Step 10 — Train BPE

```python
VOCAB_SIZE = 10000
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[pad]", "[eos]"],
        show_progress=True
    )
    text = get_dataset_text()
    tokenizer.train_from_iterator(text, trainer=trainer)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[pad]"), pad_token="[pad]")
```

---
## Step 11 — Save the trained tokenizer

```python
tokenizer.save("gutenberg_tokenizer.json", pretty=True)
```

---
## Step 12 — Create PyTorch dataset

```python
# 定义数据集 / Define dataset
class GutenbergDataset(torch.utils.data.Dataset):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, text, tokenizer, seq_len=512):
        self.seq_len = seq_len
```

---
## Step 13 — Encode the entire text

```python
self.encoded = tokenizer.encode(text).ids

    def __len__(self):
        # 获取长度 / Get length
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.seq_len + 1]  # +1 for target
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return x, y

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
## Step 14 — projection

```python
q = self.q_proj(q).view(q_batch_size, q_seq_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(k_batch_size, k_seq_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(v_batch_size, v_seq_len, -1, self.head_dim).transpose(1, 2)
```

---
## Step 15 — apply rotary positional encoding

```python
if rope:
            q = rope(q)
            k = rope(k)
```

---
## Step 16 — compute grouped query attention

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
class DecoderLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, hidden_dim, num_heads, num_kv_heads, dropout=0.1):
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
## Step 17 — self-attention sublayer

```python
out = self.norm1(x)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
```

---
## Step 18 — MLP sublayer

```python
out = self.norm2(x)
        out = self.mlp(out)
        return out + x

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class TextGenerationModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, num_layers, num_heads, num_kv_heads, hidden_dim,
                 max_seq_len, vocab_size, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.rope = RotaryPositionalEncoding(hidden_dim // num_heads, max_seq_len)
        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoders = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, num_kv_heads, dropout)
            # 生成整数序列 / Generate integer sequence
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out = nn.Linear(hidden_dim, vocab_size)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, ids, mask=None):
        x = self.embedding(ids)
        for decoder in self.decoders:
            x = decoder(x, mask, self.rope)
        x = self.norm(x)
        return self.out(x)

def create_causal_mask(seq_len, device):
    """Create a causal mask for autoregressive attention."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask
```

---
## Step 19 — Training configuration

```python
model_config = {
    "num_layers": 8,
    "num_heads": 8,
    "num_kv_heads": 4,
    "hidden_dim": 768,
    "max_seq_len": 512,
    # 获取长度 / Get length
    "vocab_size": len(tokenizer.get_vocab()),
    "dropout": 0.1,
}
```

---
## Step 20 — Initialize model, optimizer, etc.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
model = TextGenerationModel(**model_config).to(device)
```

---
## Step 21 — Create dataset and dataloader

```python
BATCH_SIZE = 32
text = "\n".join(get_dataset_text())
# 定义数据集 / Define dataset
dataset = GutenbergDataset(text, tokenizer, seq_len=model_config["max_seq_len"])
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
```

---
## Step 22 — Training loop

```python
if os.path.exists("textgen_model.pth"):
    # 加载模型参数 / Load model parameters
    model.load_state_dict(torch.load("textgen_model.pth"))
else:
    N_EPOCHS = 2
    LR = 0.0005
    WARMUP_STEPS = 2000
    CLIP_NORM = 6.0

    # 获取所有可训练参数 / Get all trainable parameters
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    # 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[pad]"))
```

---
## Step 23 — Learning rate scheduling

```python
warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        # 获取长度 / Get length
        optimizer, T_max=N_EPOCHS * len(dataloader) - WARMUP_STEPS, eta_min=0)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_STEPS])

    # 打印输出 / Print output
    print(f"Training for {N_EPOCHS} epochs with {len(dataloader)} steps per epoch")
    best_loss = float('inf')

    # 生成整数序列 / Generate integer sequence
    for epoch in range(N_EPOCHS):
        # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
        model.train()
        epoch_loss = 0

        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        for x, y in progress_bar:
            # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
            x = x.to(device)
            # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
            y = y.to(device)
```

---
## Step 24 — Create causal mask

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
mask = create_causal_mask(x.shape[1], device)
```

---
## Step 25 — Forward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
            outputs = model(x, mask.unsqueeze(0))
```

---
## Step 26 — Compute loss

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
loss = loss_fn(outputs.view(-1, outputs.shape[-1]), y.view(-1))
```

---
## Step 27 — Backward pass

```python
# 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
loss.backward()
            torch.nn.utils.clip_grad_norm_(
                # 获取所有可训练参数 / Get all trainable parameters
                model.parameters(), CLIP_NORM, error_if_nonfinite=True
            )
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            optimizer.step()
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            scheduler.step()
            epoch_loss += loss.item()
```

---
## Step 28 — Show loss in tqdm

```python
progress_bar.set_postfix(loss=loss.item())

        # 获取长度 / Get length
        avg_loss = epoch_loss / len(dataloader)
        # 打印输出 / Print output
        print(f"Epoch {epoch+1}/{N_EPOCHS}; Avg loss: {avg_loss:.4f}")
```

---
## Step 29 — Save checkpoint if loss improved

```python
if avg_loss < best_loss:
            best_loss = avg_loss
            # 获取模型参数字典 / Get model parameter dictionary
            torch.save(model.state_dict(), "textgen_model.pth")
```

---
## Step 30 — Generation function

```python
def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    # 获取所有可训练参数 / Get all trainable parameters
    device = next(model.parameters()).device
```

---
## Step 31 — Encode the prompt

```python
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
input_ids = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0).to(device)

    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        # 生成整数序列 / Generate integer sequence
        for _ in range(max_length):
```

---
## Step 32 — Get model predictions for the next token as the last element of the output

```python
outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
```

---
## Step 33 — Sample from the distribution

```python
probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
```

---
## Step 34 — Append to input_ids

```python
input_ids = torch.cat([input_ids, next_token], dim=1)
```

---
## Step 35 — Stop if we predict the end token

```python
if next_token[0].item() == tokenizer.token_to_id("[eos]"):
                break

    return tokenizer.decode(input_ids[0].tolist())
```

---
## Step 36 — Test the model with some prompts

```python
test_prompts = [
    "Once upon a time,",
    "We the people of the",
    "In the beginning was the",
]

# 打印输出 / Print output
print("\nGenerating sample texts:")
for prompt in test_prompts:
    generated = generate_text(model, tokenizer, prompt)
    # 打印输出 / Print output
    print(f"\nPrompt: {prompt}")
    # 打印输出 / Print output
    print(f"Generated: {generated}")
    # 打印输出 / Print output
    print("-" * 80)
```

---
## Learning Notes / 学习笔记

- **概念**: Download novels from Project Gutenberg 是机器学习中的常用技术。  
  *Download novels from Project Gutenberg is a common technique in machine learning.*

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
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Textgen / 08 Textgen
# Complete Code / 完整代码
# ===============================

# 导入操作系统接口 / Import OS interface
import os
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

# Download novels from Project Gutenberg
DATASOURCE = {
    "moby_dick": "https://www.gutenberg.org/ebooks/2701.txt.utf-8",
    "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
    "dracula": "https://www.gutenberg.org/ebooks/345.txt.utf-8",
    "little_women": "https://www.gutenberg.org/ebooks/37106.txt.utf-8",
    "pride_and_prejudice": "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
    "alice_in_wonderland": "https://www.gutenberg.org/ebooks/11.txt.utf-8",
    "crime_and_punishment": "https://www.gutenberg.org/ebooks/2554.txt.utf-8",
    "tom_sawyer": "https://www.gutenberg.org/ebooks/74.txt.utf-8",
    "tale_of_two_cities": "https://www.gutenberg.org/ebooks/98.txt.utf-8",
    "sherlock_holmes": "https://www.gutenberg.org/ebooks/1661.txt.utf-8",
    "war_and_peace": "https://www.gutenberg.org/ebooks/2600.txt.utf-8",
}
# 获取字典的键值对 / Get dict key-value pairs
for filename, url in DATASOURCE.items():
    if not os.path.exists(f"{filename}.txt"):
        response = requests.get(url)
        # 打开文件（自动关闭） / Open file (auto-close)
        with open(f"{filename}.txt", "wb") as f:
            f.write(response.content)

# Read and preprocess the text
def preprocess_gutenberg(filename):
    # 打开文件（自动关闭） / Open file (auto-close)
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # Find the start and end of the actual content
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = text.find("\n", start) + 1
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")

    # Extract the main content
    text = text[start:end].strip()

    # Basic preprocessing
    # Remove multiple newlines and spaces
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text

def get_dataset_text():
    all_text = []
    for filename in DATASOURCE:
        text = preprocess_gutenberg(f"{filename}.txt")
        # 添加元素到列表末尾 / Append element to list end
        all_text.append(text)
    return all_text

# Tokenization with BPE
if os.path.exists("gutenberg_tokenizer.json"):
    tokenizer = tokenizers.Tokenizer.from_file("gutenberg_tokenizer.json")
else:
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    # Configure pre-tokenizer add space at beginning of the sentence
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    # Configure decoder so that the boundary symbols will be removed
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    # Train BPE
    VOCAB_SIZE = 10000
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[pad]", "[eos]"],
        show_progress=True
    )
    text = get_dataset_text()
    tokenizer.train_from_iterator(text, trainer=trainer)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[pad]"), pad_token="[pad]")
    # Save the trained tokenizer
    tokenizer.save("gutenberg_tokenizer.json", pretty=True)

# Create PyTorch dataset
# 定义数据集 / Define dataset
class GutenbergDataset(torch.utils.data.Dataset):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, text, tokenizer, seq_len=512):
        self.seq_len = seq_len
        # Encode the entire text
        self.encoded = tokenizer.encode(text).ids

    def __len__(self):
        # 获取长度 / Get length
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.seq_len + 1]  # +1 for target
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return x, y

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
class DecoderLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, hidden_dim, num_heads, num_kv_heads, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x, mask=None, rope=None):
        # self-attention sublayer
        out = self.norm1(x)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
        # MLP sublayer
        out = self.norm2(x)
        out = self.mlp(out)
        return out + x

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class TextGenerationModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, num_layers, num_heads, num_kv_heads, hidden_dim,
                 max_seq_len, vocab_size, dropout=0.1):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.rope = RotaryPositionalEncoding(hidden_dim // num_heads, max_seq_len)
        # 嵌入层：将整数ID映射为稠密向量 / Embedding: map integer IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoders = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, num_kv_heads, dropout)
            # 生成整数序列 / Generate integer sequence
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_dim)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.out = nn.Linear(hidden_dim, vocab_size)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, ids, mask=None):
        x = self.embedding(ids)
        for decoder in self.decoders:
            x = decoder(x, mask, self.rope)
        x = self.norm(x)
        return self.out(x)

def create_causal_mask(seq_len, device):
    """Create a causal mask for autoregressive attention."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask

# Training configuration
model_config = {
    "num_layers": 8,
    "num_heads": 8,
    "num_kv_heads": 4,
    "hidden_dim": 768,
    "max_seq_len": 512,
    # 获取长度 / Get length
    "vocab_size": len(tokenizer.get_vocab()),
    "dropout": 0.1,
}

# Initialize model, optimizer, etc.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
model = TextGenerationModel(**model_config).to(device)

# Create dataset and dataloader
BATCH_SIZE = 32
text = "\n".join(get_dataset_text())
# 定义数据集 / Define dataset
dataset = GutenbergDataset(text, tokenizer, seq_len=model_config["max_seq_len"])
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
if os.path.exists("textgen_model.pth"):
    # 加载模型参数 / Load model parameters
    model.load_state_dict(torch.load("textgen_model.pth"))
else:
    N_EPOCHS = 2
    LR = 0.0005
    WARMUP_STEPS = 2000
    CLIP_NORM = 6.0

    # 获取所有可训练参数 / Get all trainable parameters
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    # 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[pad]"))

    # Learning rate scheduling
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        # 获取长度 / Get length
        optimizer, T_max=N_EPOCHS * len(dataloader) - WARMUP_STEPS, eta_min=0)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_STEPS])

    # 打印输出 / Print output
    print(f"Training for {N_EPOCHS} epochs with {len(dataloader)} steps per epoch")
    best_loss = float('inf')

    # 生成整数序列 / Generate integer sequence
    for epoch in range(N_EPOCHS):
        # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
        model.train()
        epoch_loss = 0

        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        for x, y in progress_bar:
            # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
            x = x.to(device)
            # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
            y = y.to(device)

            # Create causal mask
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            mask = create_causal_mask(x.shape[1], device)

            # Forward pass
            # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
            optimizer.zero_grad()
            outputs = model(x, mask.unsqueeze(0))

            # Compute loss
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), y.view(-1))

            # Backward pass
            # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                # 获取所有可训练参数 / Get all trainable parameters
                model.parameters(), CLIP_NORM, error_if_nonfinite=True
            )
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            optimizer.step()
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            scheduler.step()
            epoch_loss += loss.item()

            # Show loss in tqdm
            progress_bar.set_postfix(loss=loss.item())

        # 获取长度 / Get length
        avg_loss = epoch_loss / len(dataloader)
        # 打印输出 / Print output
        print(f"Epoch {epoch+1}/{N_EPOCHS}; Avg loss: {avg_loss:.4f}")

        # Save checkpoint if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 获取模型参数字典 / Get model parameter dictionary
            torch.save(model.state_dict(), "textgen_model.pth")

# Generation function
def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    # 获取所有可训练参数 / Get all trainable parameters
    device = next(model.parameters()).device

    # Encode the prompt
    # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
    input_ids = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0).to(device)

    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        # 生成整数序列 / Generate integer sequence
        for _ in range(max_length):
            # Get model predictions for the next token as the last element of the output
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            # Stop if we predict the end token
            if next_token[0].item() == tokenizer.token_to_id("[eos]"):
                break

    return tokenizer.decode(input_ids[0].tolist())

# Test the model with some prompts
test_prompts = [
    "Once upon a time,",
    "We the people of the",
    "In the beginning was the",
]

# 打印输出 / Print output
print("\nGenerating sample texts:")
for prompt in test_prompts:
    generated = generate_text(model, tokenizer, prompt)
    # 打印输出 / Print output
    print(f"\nPrompt: {prompt}")
    # 打印输出 / Print output
    print(f"Generated: {generated}")
    # 打印输出 / Print output
    print("-" * 80)
```

---

### Chapter Summary / 章节总结



---
