# 从零构建Transformer / Building Transformers from Scratch
## Chapter 01

---

### Encoder Postnorm

# 01 — Encoder Postnorm / Transformer 编码器层（Post-Norm）

**Chapter 01 — File 1 of 3 / 第01章 — 第1个文件（共3个）**

---

## Background / 背景导读

**这段代码在做什么？为什么要学它？**

Transformer 是当今最强大的 AI 模型架构，ChatGPT、BERT、GPT-4 都基于它。本文件手写了 Transformer **编码器**的一个核心层。

**生活比喻：** 想象一个班级在讨论一篇文章：
1. 每个学生（= 每个词/token）先**环顾四周**，看看其他同学怎么理解这篇文章（= **自注意力 Self-Attention**）
2. 然后每个学生**独立思考**，结合收集到的信息形成自己的理解（= **前馈网络 Feed-Forward**）
3. 每一轮讨论后，老师会帮大家**整理思路**，防止理解跑偏（= **LayerNorm 归一化**）

这就是一个编码器层做的事。本文件的 "Post-Norm" 指的是整理思路（LayerNorm）放在讨论**之后**。

---

## Architecture Flowchart / 架构流程图

```
输入 x (batch=3, seq_len=7, d_model=16)
  │
  ▼
┌─────────────────────────────┐
│   Self-Attention 自注意力    │ ← 每个词去"看"其他所有词
│   Q = K = V = x             │   计算相关性，收集上下文信息
└──────────┬──────────────────┘
           │
           ▼
  x_new + x_original  ← 残差连接（保留原始信息）
           │
           ▼
      LayerNorm1       ← 归一化（稳定数值范围）
           │
           ▼
┌─────────────────────────────┐
│   Feed-Forward 前馈网络      │ ← 两层线性变换 + ReLU
│   16 → 32 → 16              │   独立思考，增加表达能力
└──────────┬──────────────────┘
           │
           ▼
  x_new + x_residual  ← 残差连接
           │
           ▼
      LayerNorm2       ← 归一化
           │
           ▼
输出 (batch=3, seq_len=7, d_model=16)
形状不变，但每个词已融合了上下文信息
```

---

## Summary / 总结

This script builds a **Transformer Encoder Layer** with Post-Norm architecture. It has two sub-layers: Multi-Head Self-Attention and Feed-Forward Network, each followed by residual connection + LayerNorm.

本脚本构建了一个 **Post-Norm 架构的 Transformer 编码器层**。它包含两个子层：多头自注意力和前馈网络，每个子层后面都有残差连接 + LayerNorm。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — Define the Encoder Layer / 定义编码器层

**这一步在干什么？** 定义编码器层的所有"零件"。就像组装一台机器前，先把所有零件摆出来。

**What does this step do?** Define all components of the encoder layer — like laying out parts before assembling a machine.

```python
import torch          # PyTorch 深度学习框架 / PyTorch deep learning framework
import torch.nn as nn  # 神经网络模块，包含所有层的定义 / Neural network module with all layer definitions

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class TransformerEncoderLayer(nn.Module):
    # 继承 nn.Module —— 所有 PyTorch 模型的基类
    # Inherits nn.Module — the base class for all PyTorch models
    
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, d_model, d_ff, num_heads):
        # d_model = 16  每个词的向量维度（词的"信息容量"）
        #                Dimension of each token's vector ("information capacity")
        # d_ff = 32     前馈网络中间层维度（临时升维，增加表达力）
        #                Feed-forward hidden dimension (expand then compress)
        # num_heads = 4  注意力头数（从4个不同角度去关注）
        #                Number of attention heads (4 different perspectives)
        
        super().__init__()  # 调用父类初始化 / Call parent class init
        
        # 零件1：多头自注意力
        # Component 1: Multi-Head Self-Attention
        # 让每个词计算和其他词的"相关性分数"，然后加权聚合信息
        # Each token computes "relevance scores" with all other tokens, then aggregates
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # 零件2：前馈网络（两层线性变换）
        # Component 2: Feed-Forward Network (two linear layers)
        self.ff_proj = nn.Linear(d_model, d_ff)      # 升维 16→32 / Expand 16→32
        self.output_proj = nn.Linear(d_ff, d_model)   # 降维 32→16 / Compress 32→16
        # 为什么要升维再降维？就像先展开一张纸（看到更多细节），再折回去（保持尺寸一致）
        # Why expand then compress? Like unfolding paper to see more detail, then folding back
        
        # 零件3：两个 LayerNorm（每个子层一个）
        # Component 3: Two LayerNorms (one per sub-layer)
        # 作用：把数值拉回正常范围，防止数字越算越大或越算越小
        # Purpose: Normalize values to prevent numbers from exploding or vanishing
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力子层后 / After self-attention
        self.norm2 = nn.LayerNorm(d_model)  # 前馈子层后 / After feed-forward
        
        # 零件4：激活函数 ReLU
        # Component 4: ReLU activation
        # 作用：引入非线性，负数变0，正数不变。没有它，多层网络等于一层
        # Purpose: Non-linearity. Negative→0, positive→unchanged. Without it, deep = shallow
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        """
        处理输入序列 / Process the input sequence
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
               Input tensor of shape (batch_size, seq_len, d_model)
               例如 / Example: (3个句子, 每句7个词, 每词16维向量)
               
        Returns:
            形状不变的输出张量，但每个词已融合上下文信息
            Same-shape tensor, but each token now contains contextual information
        """
```

---
## Step 2 — Self-Attention Sublayer / 自注意力子层

**这一步在干什么？** 让每个词去"看"句子里的其他所有词，计算相关性，收集有用信息。

**比喻：** 你读到"苹果发布了新手机"中的"苹果"，需要看后面的"发布""手机"才知道这里说的是公司不是水果。自注意力就是这个"看上下文"的过程。

```
"苹果" → 看到 "发布" "手机" → 哦，是 Apple 公司
"苹果" → 看到 "很甜" "红色" → 哦，是水果
```

**What does this step do?** Each token "looks at" all other tokens to compute relevance and gather context.

```python
        # ---- 自注意力子层 / Self-Attention Sub-layer ----
        
        residual = x  # 先保存原始输入（用于残差连接）
                      # Save original input (for residual connection later)
        
        # attention(Q, K, V) — Q=K=V=x 就是"自"注意力
        # Q(Query): 我想找什么信息？ / What am I looking for?
        # K(Key):   我有什么信息？   / What information do I have?
        # V(Value): 我的实际内容     / My actual content
        # 返回 (output, attention_weights)，我们只要 output 即 x[0]
        x = self.attention(x, x, x)
        
        # 残差连接 + LayerNorm（Post-Norm：先加再归一化）
        # Residual connection + LayerNorm (Post-Norm: add first, then normalize)
        # x[0] + residual：新信息 + 原始信息，防止深层网络梯度消失
        # 就像学新知识时不会忘掉旧知识
        x = self.norm1(x[0] + residual)
```

---
## Step 3 — Feed-Forward Sublayer / 前馈网络子层

**这一步在干什么？** 每个词独立地进行"深度思考"。注意力子层是"群体讨论"，前馈网络是"个人消化"。

**比喻：** 开完会后，每个人回到工位，独自整理会议笔记，形成自己的理解。

**What does this step do?** Each token independently processes its information. Attention = group discussion, FFN = individual reflection.

```python
        # ---- 前馈网络子层 / Feed-Forward Sub-layer ----
        
        residual = x  # 再次保存输入 / Save input again
        
        # 第一层：升维 16→32，然后 ReLU 激活
        # Layer 1: Expand 16→32, then ReLU activation
        x = self.act(self.ff_proj(x))
        
        # 第二层：降维 32→16，然后 ReLU 激活
        # Layer 2: Compress 32→16, then ReLU activation
        x = self.act(self.output_proj(x))
        
        # 残差连接 + LayerNorm（同 Step 2）
        # Residual + LayerNorm (same pattern as Step 2)
        x = self.norm2(x + residual)
        
        return x  # 输出形状不变 (3, 7, 16)，但语义更丰富
                  # Output shape unchanged (3,7,16), but semantically richer

# ---- 测试代码 / Test Code ----

# 生成随机输入：3个句子，每句7个词，每词16维
# Generate random input: 3 sentences, 7 tokens each, 16-dim vectors
# 生成正态分布随机张量 / Generate random tensor from normal distribution
seq = torch.randn(3, 7, 16)

# 创建编码器层：d_model=16, d_ff=32, num_heads=4
# Create encoder layer: d_model=16, d_ff=32, num_heads=4
layer = TransformerEncoderLayer(16, 32, 4)

# 前向传播
# Forward pass
out_seq = layer(seq)

# 打印所有可学习参数的名称和形状
# Print all learnable parameter names and shapes
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print({name: weight.shape for name, weight in layer.state_dict().items()})

# 打印输出形状 → torch.Size([3, 7, 16])（和输入一样！）
# Print output shape → torch.Size([3, 7, 16]) (same as input!)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(out_seq.shape)
```

---
## Expected Output / 预期输出

运行上面的代码，你会看到类似这样的结果：

```
{'attention.in_proj_weight': torch.Size([48, 16]),     ← 注意力投影权重 Q/K/V 各16维×3=48
 'attention.in_proj_bias': torch.Size([48]),
 'attention.out_proj.weight': torch.Size([16, 16]),    ← 注意力输出投影
 'attention.out_proj.bias': torch.Size([16]),
 'ff_proj.weight': torch.Size([32, 16]),               ← 前馈层升维 16→32
 'ff_proj.bias': torch.Size([32]),
 'output_proj.weight': torch.Size([16, 32]),            ← 前馈层降维 32→16
 'output_proj.bias': torch.Size([16]),
 'norm1.weight': torch.Size([16]),                      ← LayerNorm 参数
 'norm1.bias': torch.Size([16]),
 'norm2.weight': torch.Size([16]),
 'norm2.bias': torch.Size([16])}
torch.Size([3, 7, 16])  ← 输出形状 = 输入形状，说明层结构正确！
```

**关键观察 / Key Observations:**
- 输入 `(3, 7, 16)` → 输出 `(3, 7, 16)`，形状完全一致
- 这意味着编码器层可以**堆叠多层**（一层的输出直接作为下一层的输入）
- 真实的 BERT 用 12 层，GPT-3 用 96 层

---
## Learning Notes / 学习笔记

### Key Concepts / 核心概念

- **Self-Attention（自注意力）**: 每个词计算与其他所有词的相关性，然后加权聚合信息。Q=K=V=x 就是"自己跟自己算"。
  *Each token computes relevance with all other tokens. Q=K=V=x means "attending to itself".*

- **Residual Connection（残差连接）**: `output = f(x) + x`。把新结果加上原始输入。作用是让梯度能直接"穿透"回传，解决深层网络训练困难的问题。
  *Add the original input back to the output. Allows gradients to flow directly, solving the vanishing gradient problem.*

- **Post-Norm vs Pre-Norm**: Post-Norm 是先做残差相加再归一化（本文件）。Pre-Norm 是先归一化再进子层（GPT 系列用这种）。Pre-Norm 训练更稳定，Post-Norm 效果可能略好。
  *Post-Norm: add then normalize (this file). Pre-Norm: normalize then process (used by GPT). Pre-Norm trains more stably.*

- **Multi-Head（多头）**: 4 个头 = 4 种不同的"关注方式"。一个头可能关注语法关系，另一个关注语义关系。
  *4 heads = 4 different "ways of paying attention". One might focus on syntax, another on semantics.*

### Glossary / 术语表

| 术语 Term | 解释 Explanation |
|-----------|-----------------|
| **Tensor** | 多维数组。标量=0维，向量=1维，矩阵=2维，3维以上都叫张量 |
| **d_model** | 每个 token 的向量维度，决定模型的"信息容量" |
| **d_ff** | 前馈网络中间层维度，通常是 d_model 的 2~4 倍 |
| **num_heads** | 注意力头数，d_model 必须能被 num_heads 整除 |
| **batch_size** | 一次处理多少个句子（并行计算，加速训练） |
| **seq_len** | 序列长度，即每个句子有多少个 token |
| **LayerNorm** | 对每个样本的特征维度做归一化，稳定训练过程 |
| **ReLU** | 激活函数，f(x)=max(0,x)，引入非线性 |
| **nn.Module** | PyTorch 所有模型的基类，定义 `__init__` 和 `forward` |

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `nn.LayerNorm` | 层归一化，Transformer常用 | Layer normalization, common in Transformers |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.MultiheadAttention` | 多头注意力机制 | Multi-head attention mechanism |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Transformer Encoder Layer (Post-Norm)
# Transformer 编码器层（后归一化）
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class TransformerEncoderLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, d_model, d_ff, num_heads):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.ff_proj = nn.Linear(d_model, d_ff)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output_proj = nn.Linear(d_ff, d_model)
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm1 = nn.LayerNorm(d_model)
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm2 = nn.LayerNorm(d_model)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # Self-attention sublayer / 自注意力子层
        residual = x
        x = self.attention(x, x, x)
        x = self.norm1(x[0] + residual)

        # Feed-forward sublayer / 前馈网络子层
        residual = x
        x = self.act(self.ff_proj(x))
        x = self.act(self.output_proj(x))
        x = self.norm2(x + residual)

        return x

# 生成正态分布随机张量 / Generate random tensor from normal distribution
seq = torch.randn(3, 7, 16)
layer = TransformerEncoderLayer(16, 32, 4)
out_seq = layer(seq)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print({name: weight.shape for name, weight in layer.state_dict().items()})
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(out_seq.shape)
```

---

➡️ **Next / 下一步**: File 2 of 3 — Encoder Pre-Norm（先归一化再进子层的变体，对比学习效果更好）

---

### Encoder Prenorm

# 02 — Encoder Prenorm / 数据编码

**Chapter 01 — File 2 of 3 / 第01章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Self-attention sublayer**.

本脚本演示 **Self-attention sublayer**。

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
class TransformerEncoderLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, d_model, d_ff, num_heads):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.ff_proj = nn.Linear(d_model, d_ff)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output_proj = nn.Linear(d_ff, d_model)
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm1 = nn.LayerNorm(d_model)
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm2 = nn.LayerNorm(d_model)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        """Process the input sequence x

        Args:
            x: The input sequence of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The processed sequence of shape (batch_size, seq_len, d_model).
        """
```

---
## Step 2 — Self-attention sublayer

```python
residual = x
        x = self.norm1(x)
        x = self.attention(x, x, x)
        x = x[0] + residual
```

---
## Step 3 — Feed-forward sublayer

```python
residual = x
        x = self.norm2(x)
        x = self.act(self.ff_proj(x))
        x = self.act(self.output_proj(x))
        x = x + residual

        return x

# 生成正态分布随机张量 / Generate random tensor from normal distribution
seq = torch.randn(3, 7, 16)
layer = TransformerEncoderLayer(16, 32, 4)
out_seq = layer(seq)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print({name: weight.shape for name, weight in layer.state_dict().items()})
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(out_seq.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: Self-attention sublayer 是机器学习中的常用技术。  
  *Self-attention sublayer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `nn.LayerNorm` | 层归一化，Transformer常用 | Layer normalization, common in Transformers |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.MultiheadAttention` | 多头注意力机制 | Multi-head attention mechanism |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Encoder Prenorm / 数据编码
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class TransformerEncoderLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, d_model, d_ff, num_heads):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.ff_proj = nn.Linear(d_model, d_ff)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output_proj = nn.Linear(d_ff, d_model)
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm1 = nn.LayerNorm(d_model)
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm2 = nn.LayerNorm(d_model)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        """Process the input sequence x

        Args:
            x: The input sequence of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The processed sequence of shape (batch_size, seq_len, d_model).
        """
        # Self-attention sublayer
        residual = x
        x = self.norm1(x)
        x = self.attention(x, x, x)
        x = x[0] + residual

        # Feed-forward sublayer
        residual = x
        x = self.norm2(x)
        x = self.act(self.ff_proj(x))
        x = self.act(self.output_proj(x))
        x = x + residual

        return x

# 生成正态分布随机张量 / Generate random tensor from normal distribution
seq = torch.randn(3, 7, 16)
layer = TransformerEncoderLayer(16, 32, 4)
out_seq = layer(seq)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print({name: weight.shape for name, weight in layer.state_dict().items()})
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(out_seq.shape)
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Decoder Layer

# 04 — Decoder Layer / 04 Decoder Layer

**Chapter 01 — File 3 of 3 / 第01章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Self-attention sublayer**.

本脚本演示 **Self-attention sublayer**。

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
class TransformerDecoderLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, d_model, d_ff, num_heads):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.xattention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.ff_proj = nn.Linear(d_model, d_ff)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output_proj = nn.Linear(d_ff, d_model)
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm1 = nn.LayerNorm(d_model)
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm2 = nn.LayerNorm(d_model)
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm3 = nn.LayerNorm(d_model)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x, y):
        """Process the input sequence x with decoder input y

        Args:
            x: The input sequence of shape (batch_size, seq_len, d_model).
            y: The output sequence from encoder of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The processed sequence of shape (batch_size, seq_len, d_model).
        """
```

---
## Step 2 — Self-attention sublayer

```python
residual = x
        x = self.norm1(x)
        x = self.attention(x, x, x)
        x = x[0] + residual
```

---
## Step 3 — Cross-attention sublayer

```python
residual = x
        x = self.norm2(x)
        x = self.xattention(x, y, y)
        x = x[0] + residual
```

---
## Step 4 — Feed-forward sublayer

```python
residual = x
        x = self.norm3(x)
        x = self.act(self.ff_proj(x))
        x = self.act(self.output_proj(x))
        x = x + residual

        return x

# 生成正态分布随机张量 / Generate random tensor from normal distribution
dec_seq = torch.randn(3, 7, 16)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
enc_seq = torch.randn(3, 11, 16)
layer = TransformerDecoderLayer(16, 32, 4)
out_seq = layer(dec_seq, enc_seq)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print({name: weight.shape for name, weight in layer.state_dict().items()})
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(out_seq.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: Self-attention sublayer 是机器学习中的常用技术。  
  *Self-attention sublayer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `attention` | 注意力机制：让模型关注重要部分 | Attention: focus on important parts |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `nn.LayerNorm` | 层归一化，Transformer常用 | Layer normalization, common in Transformers |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.MultiheadAttention` | 多头注意力机制 | Multi-head attention mechanism |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.randn` | 生成标准正态分布随机张量 | Generate random tensor from normal distribution |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Decoder Layer / 04 Decoder Layer
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class TransformerDecoderLayer(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, d_model, d_ff, num_heads):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # 多头注意力：从多个角度关注输入 / Multi-head attention: attend from multiple perspectives
        self.xattention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.ff_proj = nn.Linear(d_model, d_ff)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output_proj = nn.Linear(d_ff, d_model)
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm1 = nn.LayerNorm(d_model)
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm2 = nn.LayerNorm(d_model)
        # 层归一化：稳定训练，防止数值爆炸 / LayerNorm: stabilize training, prevent value explosion
        self.norm3 = nn.LayerNorm(d_model)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x, y):
        """Process the input sequence x with decoder input y

        Args:
            x: The input sequence of shape (batch_size, seq_len, d_model).
            y: The output sequence from encoder of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The processed sequence of shape (batch_size, seq_len, d_model).
        """
        # Self-attention sublayer
        residual = x
        x = self.norm1(x)
        x = self.attention(x, x, x)
        x = x[0] + residual

        # Cross-attention sublayer
        residual = x
        x = self.norm2(x)
        x = self.xattention(x, y, y)
        x = x[0] + residual

        # Feed-forward sublayer
        residual = x
        x = self.norm3(x)
        x = self.act(self.ff_proj(x))
        x = self.act(self.output_proj(x))
        x = x + residual

        return x

# 生成正态分布随机张量 / Generate random tensor from normal distribution
dec_seq = torch.randn(3, 7, 16)
# 生成正态分布随机张量 / Generate random tensor from normal distribution
enc_seq = torch.randn(3, 11, 16)
layer = TransformerDecoderLayer(16, 32, 4)
out_seq = layer(dec_seq, enc_seq)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print({name: weight.shape for name, weight in layer.state_dict().items()})
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(out_seq.shape)
```

---

### Chapter Summary / 章节总结

# Chapter 01 Summary / 第01章总结

## Theme / 主题: Chapter 01 / Chapter 01

This chapter contains **3 code files** demonstrating chapter 01.

本章包含 **3 个代码文件**，演示Chapter 01。

---
## Evolution / 演化路线

  1. `01_encoder_postnorm.ipynb` — Encoder Postnorm
  2. `02_encoder_prenorm.ipynb` — Encoder Prenorm
  3. `04_decoder_layer.ipynb` — Decoder Layer

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 01) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 01）是机器学习流水线中的基础构建块。

---
