# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 28

---

### Count

# 02 — Count / 02 Count

**Chapter 28 — File 1 of 11 / 第28章 — 第1个文件（共11个）**

---

## Summary / 总结

This script demonstrates **load ascii text and covert to lowercase**.

本脚本演示 **load ascii text and covert to lowercase**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — load ascii text and covert to lowercase

```python
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
```

---
## Step 2 — create mapping of unique chars to integers

```python
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))
```

---
## Step 3 — summarize the loaded data

```python
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)
```

---
## Learning Notes / 学习笔记

- **概念**: load ascii text and covert to lowercase 是机器学习中的常用技术。  
  *load ascii text and covert to lowercase is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Count / 02 Count
# Complete Code / 完整代码
# ===============================

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)
```

---

➡️ **Next / 下一步**: File 2 of 11

---

### Slidingwin



---

### Convert



---

### Model

# 05 — Model / 05 Model

**Chapter 28 — File 4 of 11 / 第28章 — 第4个文件（共11个）**

---

## Summary / 总结

This script demonstrates **take only the last output**.

本脚本演示 **take only the last output**。

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
class CharModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(256, n_vocab)
    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
```

---
## Step 2 — take only the last output

```python
x = x[:, -1, :]
```

---
## Step 3 — produce output

```python
x = self.linear(self.dropout(x))
        return x
```

---
## Learning Notes / 学习笔记

- **概念**: take only the last output 是机器学习中的常用技术。  
  *take only the last output is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `nn.Dropout` | 随机丢弃神经元防止过拟合 | Randomly drop neurons to prevent overfitting |
| `nn.LSTM` | 长短期记忆网络，处理序列数据 | Long Short-Term Memory for sequences |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model / 05 Model
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CharModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(256, n_vocab)
    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x
```

---

➡️ **Next / 下一步**: File 5 of 11

---

### Train

# 06 — Train / 06 Train

**Chapter 28 — File 5 of 11 / 第28章 — 第5个文件（共11个）**

---

## Summary / 总结

This script demonstrates **load ascii text and covert to lowercase**.

本脚本演示 **load ascii text and covert to lowercase**。

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.utils.data as data
```

---
## Step 2 — load ascii text and covert to lowercase

```python
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
```

---
## Step 3 — create mapping of unique chars to integers

```python
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))
```

---
## Step 4 — summarize the loaded data

```python
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
```

---
## Step 5 — prepare the dataset of input to output pairs encoded as integers

```python
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
```

---
## Step 6 — reshape X to be [samples, time steps, features]

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CharModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(256, n_vocab)
    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
```

---
## Step 7 — take only the last output

```python
x = x[:, -1, :]
```

---
## Step 8 — produce output

```python
x = self.linear(self.dropout(x))
        return x

n_epochs = 40
batch_size = 128
model = CharModel()

# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss(reduction="sum")
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
```

---
## Step 9 — Validation

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    loss = 0
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            # 获取模型参数字典 / Get model parameter dictionary
            best_model = model.state_dict()
        # 打印输出 / Print output
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model, char_to_int], "single-char.pth")
```

---
## Learning Notes / 学习笔记

- **概念**: load ascii text and covert to lowercase 是机器学习中的常用技术。  
  *load ascii text and covert to lowercase is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `nn.Dropout` | 随机丢弃神经元防止过拟合 | Randomly drop neurons to prevent overfitting |
| `nn.LSTM` | 长短期记忆网络，处理序列数据 | Long Short-Term Memory for sequences |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train / 06 Train
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.utils.data as data

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)

# reshape X to be [samples, time steps, features]
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CharModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(256, n_vocab)
    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

n_epochs = 40
batch_size = 128
model = CharModel()

# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss(reduction="sum")
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
    # Validation
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    loss = 0
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            # 获取模型参数字典 / Get model parameter dictionary
            best_model = model.state_dict()
        # 打印输出 / Print output
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model, char_to_int], "single-char.pth")
```

---

➡️ **Next / 下一步**: File 6 of 11

---

### Charpredict

# 07 — Charpredict / 07 Charpredict

**Chapter 28 — File 6 of 11 / 第28章 — 第6个文件（共11个）**

---

## Summary / 总结

This script demonstrates **load ascii text and covert to lowercase**.

本脚本演示 **load ascii text and covert to lowercase**。

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.utils.data as data
```

---
## Step 2 — load ascii text and covert to lowercase

```python
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
```

---
## Step 3 — create mapping of unique chars to integers

```python
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))
```

---
## Step 4 — summarize the loaded data

```python
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)
```

---
## Step 5 — prepare the dataset of input to output pairs encoded as integers

```python
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
# 打印输出 / Print output
print("Total Patterns: ", n_patterns)
```

---
## Step 6 — reshape X to be [samples, time steps, features]

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CharModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(256, n_vocab)
    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
```

---
## Step 7 — take only the last output

```python
x = x[:, -1, :]
```

---
## Step 8 — produce output

```python
x = self.linear(self.dropout(x))
        return x

n_epochs = 40
batch_size = 128
model = CharModel()

# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss(reduction="sum")
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
```

---
## Step 9 — Validation

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    loss = 0
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            # 获取模型参数字典 / Get model parameter dictionary
            best_model = model.state_dict()
        # 打印输出 / Print output
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model, char_to_int], "single-char.pth")
```

---
## Learning Notes / 学习笔记

- **概念**: load ascii text and covert to lowercase 是机器学习中的常用技术。  
  *load ascii text and covert to lowercase is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `nn.Dropout` | 随机丢弃神经元防止过拟合 | Randomly drop neurons to prevent overfitting |
| `nn.LSTM` | 长短期记忆网络，处理序列数据 | Long Short-Term Memory for sequences |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Charpredict / 07 Charpredict
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.utils.data as data

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
# 打印输出 / Print output
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CharModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(256, n_vocab)
    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

n_epochs = 40
batch_size = 128
model = CharModel()

# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss(reduction="sum")
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
    # Validation
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    loss = 0
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            # 获取模型参数字典 / Get model parameter dictionary
            best_model = model.state_dict()
        # 打印输出 / Print output
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model, char_to_int], "single-char.pth")
```

---

➡️ **Next / 下一步**: File 7 of 11

---

### Generate

# 09 — Generate / 09 Generate

**Chapter 28 — File 7 of 11 / 第28章 — 第7个文件（共11个）**

---

## Summary / 总结

This script demonstrates **reload the model**.

本脚本演示 **reload the model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

best_model, char_to_int = torch.load("single-char.pth")
# 获取长度 / Get length
n_vocab = len(char_to_int)
# 获取字典的键值对 / Get dict key-value pairs
int_to_char = dict((i, c) for c, i in char_to_int.items())
```

---
## Step 2 — reload the model

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CharModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(256, n_vocab)
    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
```

---
## Step 3 — take only the last output

```python
x = x[:, -1, :]
```

---
## Step 4 — produce output

```python
x = self.linear(self.dropout(x))
        return x
model = CharModel()
# 加载模型参数 / Load model parameters
model.load_state_dict(best_model)
```

---
## Step 5 — randomly generate a prompt

```python
filename = "wonderland.txt"
seq_length = 100
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# 生成随机数 / Generate random numbers
start = np.random.randint(0, len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
pattern = [char_to_int[c] for c in prompt]

# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
# 打印输出 / Print output
print('Prompt: "%s"' % prompt)
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # 生成整数序列 / Generate integer sequence
    for i in range(1000):
```

---
## Step 6 — format input array of int into PyTorch tensor

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
```

---
## Step 7 — generate logits as output from the model

```python
prediction = model(x)
```

---
## Step 8 — convert logits into one character

```python
index = int(prediction.argmax())
        result = int_to_char[index]
        # 打印输出 / Print output
        print(result, end="")
```

---
## Step 9 — append the new character into the prompt for the next iteration

```python
# 添加元素到列表末尾 / Append element to list end
pattern.append(index)
        pattern = pattern[1:]
# 打印输出 / Print output
print()
# 打印输出 / Print output
print("Done.")
```

---
## Learning Notes / 学习笔记

- **概念**: reload the model 是机器学习中的常用技术。  
  *reload the model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `nn.Dropout` | 随机丢弃神经元防止过拟合 | Randomly drop neurons to prevent overfitting |
| `nn.LSTM` | 长短期记忆网络，处理序列数据 | Long Short-Term Memory for sequences |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `np.random` | 随机数生成 | Random number generation |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Generate / 09 Generate
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

best_model, char_to_int = torch.load("single-char.pth")
# 获取长度 / Get length
n_vocab = len(char_to_int)
# 获取字典的键值对 / Get dict key-value pairs
int_to_char = dict((i, c) for c, i in char_to_int.items())

# reload the model
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CharModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(256, n_vocab)
    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x
model = CharModel()
# 加载模型参数 / Load model parameters
model.load_state_dict(best_model)

# randomly generate a prompt
filename = "wonderland.txt"
seq_length = 100
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# 生成随机数 / Generate random numbers
start = np.random.randint(0, len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
pattern = [char_to_int[c] for c in prompt]

# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
# 打印输出 / Print output
print('Prompt: "%s"' % prompt)
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # 生成整数序列 / Generate integer sequence
    for i in range(1000):
        # format input array of int into PyTorch tensor
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        # generate logits as output from the model
        prediction = model(x)
        # convert logits into one character
        index = int(prediction.argmax())
        result = int_to_char[index]
        # 打印输出 / Print output
        print(result, end="")
        # append the new character into the prompt for the next iteration
        # 添加元素到列表末尾 / Append element to list end
        pattern.append(index)
        pattern = pattern[1:]
# 打印输出 / Print output
print()
# 打印输出 / Print output
print("Done.")
```

---

➡️ **Next / 下一步**: File 8 of 11

---

### Twolayers



---

### Complete

# 11 — Complete / 11 Complete

**Chapter 28 — File 9 of 11 / 第28章 — 第9个文件（共11个）**

---

## Summary / 总结

This script demonstrates **load ascii text and covert to lowercase**.

本脚本演示 **load ascii text and covert to lowercase**。

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.utils.data as data
```

---
## Step 2 — load ascii text and covert to lowercase

```python
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
```

---
## Step 3 — create mapping of unique chars to integers

```python
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))
```

---
## Step 4 — summarize the loaded data

```python
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)
```

---
## Step 5 — prepare the dataset of input to output pairs encoded as integers

```python
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
# 打印输出 / Print output
print("Total Patterns: ", n_patterns)
```

---
## Step 6 — reshape X to be [samples, time steps, features]

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CharModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2,
                            batch_first=True, dropout=0.2)
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(256, n_vocab)
    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
```

---
## Step 7 — take only the last output

```python
x = x[:, -1, :]
```

---
## Step 8 — produce output

```python
x = self.linear(self.dropout(x))
        return x

n_epochs = 40
batch_size = 128
model = CharModel()

# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss(reduction="sum")
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
```

---
## Step 9 — Validation

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    loss = 0
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            # 获取模型参数字典 / Get model parameter dictionary
            best_model = model.state_dict()
        # 打印输出 / Print output
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model, char_to_int], "single-char.pth")
```

---
## Step 10 — Generation using the trained model

```python
best_model, char_to_int = torch.load("single-char.pth")
# 获取长度 / Get length
n_vocab = len(char_to_int)
# 获取字典的键值对 / Get dict key-value pairs
int_to_char = dict((i, c) for c, i in char_to_int.items())
# 加载模型参数 / Load model parameters
model.load_state_dict(best_model)
```

---
## Step 11 — randomly generate a prompt

```python
filename = "wonderland.txt"
seq_length = 100
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# 生成随机数 / Generate random numbers
start = np.random.randint(0, len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
pattern = [char_to_int[c] for c in prompt]

# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
# 打印输出 / Print output
print('Prompt: "%s"' % prompt)
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # 生成整数序列 / Generate integer sequence
    for i in range(1000):
```

---
## Step 12 — format input array of int into PyTorch tensor

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
```

---
## Step 13 — generate logits as output from the model

```python
prediction = model(x)
```

---
## Step 14 — convert logits into one character

```python
index = int(prediction.argmax())
        result = int_to_char[index]
        # 打印输出 / Print output
        print(result, end="")
```

---
## Step 15 — append the new character into the prompt for the next iteration

```python
# 添加元素到列表末尾 / Append element to list end
pattern.append(index)
        pattern = pattern[1:]
# 打印输出 / Print output
print()
# 打印输出 / Print output
print("Done.")
```

---
## Learning Notes / 学习笔记

- **概念**: load ascii text and covert to lowercase 是机器学习中的常用技术。  
  *load ascii text and covert to lowercase is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `nn.Dropout` | 随机丢弃神经元防止过拟合 | Randomly drop neurons to prevent overfitting |
| `nn.LSTM` | 长短期记忆网络，处理序列数据 | Long Short-Term Memory for sequences |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `np.random` | 随机数生成 | Random number generation |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Complete / 11 Complete
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.utils.data as data

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
# 打印输出 / Print output
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CharModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2,
                            batch_first=True, dropout=0.2)
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(256, n_vocab)
    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

n_epochs = 40
batch_size = 128
model = CharModel()

# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss(reduction="sum")
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
    # Validation
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    loss = 0
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            # 获取模型参数字典 / Get model parameter dictionary
            best_model = model.state_dict()
        # 打印输出 / Print output
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model, char_to_int], "single-char.pth")

# Generation using the trained model
best_model, char_to_int = torch.load("single-char.pth")
# 获取长度 / Get length
n_vocab = len(char_to_int)
# 获取字典的键值对 / Get dict key-value pairs
int_to_char = dict((i, c) for c, i in char_to_int.items())
# 加载模型参数 / Load model parameters
model.load_state_dict(best_model)

# randomly generate a prompt
filename = "wonderland.txt"
seq_length = 100
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# 生成随机数 / Generate random numbers
start = np.random.randint(0, len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
pattern = [char_to_int[c] for c in prompt]

# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
# 打印输出 / Print output
print('Prompt: "%s"' % prompt)
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # 生成整数序列 / Generate integer sequence
    for i in range(1000):
        # format input array of int into PyTorch tensor
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        # generate logits as output from the model
        prediction = model(x)
        # convert logits into one character
        index = int(prediction.argmax())
        result = int_to_char[index]
        # 打印输出 / Print output
        print(result, end="")
        # append the new character into the prompt for the next iteration
        # 添加元素到列表末尾 / Append element to list end
        pattern.append(index)
        pattern = pattern[1:]
# 打印输出 / Print output
print()
# 打印输出 / Print output
print("Done.")
```

---

➡️ **Next / 下一步**: File 10 of 11

---

### Cuda

# 12 — Cuda / 12 Cuda

**Chapter 28 — File 10 of 11 / 第28章 — 第10个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Cuda**.

本脚本演示 **12 Cuda**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 打印输出 / Print output
print(torch.cuda.is_available())
```

---
## Learning Notes / 学习笔记

- **概念**: Cuda 是机器学习中的常用技术。  
  *Cuda is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cuda / 12 Cuda
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 打印输出 / Print output
print(torch.cuda.is_available())
```

---

➡️ **Next / 下一步**: File 11 of 11

---

### Traincuda

# 15 — Traincuda / 15 Traincuda

**Chapter 28 — File 11 of 11 / 第28章 — 第11个文件（共11个）**

---

## Summary / 总结

This script demonstrates **load ascii text and covert to lowercase**.

本脚本演示 **load ascii text and covert to lowercase**。

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.utils.data as data
```

---
## Step 2 — load ascii text and covert to lowercase

```python
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
```

---
## Step 3 — create mapping of unique chars to integers

```python
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))
```

---
## Step 4 — summarize the loaded data

```python
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)
```

---
## Step 5 — prepare the dataset of input to output pairs encoded as integers

```python
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
# 打印输出 / Print output
print("Total Patterns: ", n_patterns)
```

---
## Step 6 — reshape X to be [samples, time steps, features]

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CharModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2,
                            batch_first=True, dropout=0.2)
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(256, n_vocab)
    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
```

---
## Step 7 — take only the last output

```python
x = x[:, -1, :]
```

---
## Step 8 — produce output

```python
x = self.linear(self.dropout(x))
        return x

n_epochs = 40
batch_size = 128
model = CharModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
model.to(device)

# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss(reduction="sum")
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    for X_batch, y_batch in loader:
        # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
        y_pred = model(X_batch.to(device))
        # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
        loss = loss_fn(y_pred, y_batch.to(device))
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
```

---
## Step 9 — Validation

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    loss = 0
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
            y_pred = model(X_batch.to(device))
            # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
            loss += loss_fn(y_pred, y_batch.to(device))
        if loss < best_loss:
            best_loss = loss
            # 获取模型参数字典 / Get model parameter dictionary
            best_model = model.state_dict()
        # 打印输出 / Print output
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model, char_to_int], "single-char.pth")
```

---
## Step 10 — Generation using the trained model

```python
best_model, char_to_int = torch.load("single-char.pth")
# 获取长度 / Get length
n_vocab = len(char_to_int)
# 获取字典的键值对 / Get dict key-value pairs
int_to_char = dict((i, c) for c, i in char_to_int.items())
# 加载模型参数 / Load model parameters
model.load_state_dict(best_model)
```

---
## Step 11 — randomly generate a prompt

```python
filename = "wonderland.txt"
seq_length = 100
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# 生成随机数 / Generate random numbers
start = np.random.randint(0, len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
pattern = [char_to_int[c] for c in prompt]

# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
# 打印输出 / Print output
print('Prompt: "%s"' % prompt)
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # 生成整数序列 / Generate integer sequence
    for i in range(1000):
```

---
## Step 12 — format input array of int into PyTorch tensor

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
```

---
## Step 13 — generate logits as output from the model

```python
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
prediction = model(x.to(device))
```

---
## Step 14 — convert logits into one character

```python
index = int(prediction.argmax())
        result = int_to_char[index]
        # 打印输出 / Print output
        print(result, end="")
```

---
## Step 15 — append the new character into the prompt for the next iteration

```python
# 添加元素到列表末尾 / Append element to list end
pattern.append(index)
        pattern = pattern[1:]
# 打印输出 / Print output
print()
# 打印输出 / Print output
print("Done.")
```

---
## Learning Notes / 学习笔记

- **概念**: load ascii text and covert to lowercase 是机器学习中的常用技术。  
  *load ascii text and covert to lowercase is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `nn.Dropout` | 随机丢弃神经元防止过拟合 | Randomly drop neurons to prevent overfitting |
| `nn.LSTM` | 长短期记忆网络，处理序列数据 | Long Short-Term Memory for sequences |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `np.random` | 随机数生成 | Random number generation |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Traincuda / 15 Traincuda
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.utils.data as data

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
# 同时获取索引和值 / Get both index and value
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
# 获取长度 / Get length
n_chars = len(raw_text)
# 获取长度 / Get length
n_vocab = len(chars)
# 打印输出 / Print output
print("Total Characters: ", n_chars)
# 打印输出 / Print output
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
# 生成整数序列 / Generate integer sequence
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # 添加元素到列表末尾 / Append element to list end
    dataX.append([char_to_int[char] for char in seq_in])
    # 添加元素到列表末尾 / Append element to list end
    dataY.append(char_to_int[seq_out])
# 获取长度 / Get length
n_patterns = len(dataX)
# 打印输出 / Print output
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CharModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2,
                            batch_first=True, dropout=0.2)
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(256, n_vocab)
    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

n_epochs = 40
batch_size = 128
model = CharModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
model.to(device)

# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss(reduction="sum")
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    for X_batch, y_batch in loader:
        # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
        y_pred = model(X_batch.to(device))
        # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
        loss = loss_fn(y_pred, y_batch.to(device))
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
    # Validation
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    loss = 0
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
            y_pred = model(X_batch.to(device))
            # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
            loss += loss_fn(y_pred, y_batch.to(device))
        if loss < best_loss:
            best_loss = loss
            # 获取模型参数字典 / Get model parameter dictionary
            best_model = model.state_dict()
        # 打印输出 / Print output
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

torch.save([best_model, char_to_int], "single-char.pth")

# Generation using the trained model
best_model, char_to_int = torch.load("single-char.pth")
# 获取长度 / Get length
n_vocab = len(char_to_int)
# 获取字典的键值对 / Get dict key-value pairs
int_to_char = dict((i, c) for c, i in char_to_int.items())
# 加载模型参数 / Load model parameters
model.load_state_dict(best_model)

# randomly generate a prompt
filename = "wonderland.txt"
seq_length = 100
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# 生成随机数 / Generate random numbers
start = np.random.randint(0, len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
pattern = [char_to_int[c] for c in prompt]

# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
# 打印输出 / Print output
print('Prompt: "%s"' % prompt)
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # 生成整数序列 / Generate integer sequence
    for i in range(1000):
        # format input array of int into PyTorch tensor
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        # generate logits as output from the model
        # 将模型/数据移到GPU或CPU / Move model/data to GPU or CPU
        prediction = model(x.to(device))
        # convert logits into one character
        index = int(prediction.argmax())
        result = int_to_char[index]
        # 打印输出 / Print output
        print(result, end="")
        # append the new character into the prompt for the next iteration
        # 添加元素到列表末尾 / Append element to list end
        pattern.append(index)
        pattern = pattern[1:]
# 打印输出 / Print output
print()
# 打印输出 / Print output
print("Done.")
```

---

### Chapter Summary / 章节总结

# Chapter 28 Summary / 第28章总结

## Theme / 主题: Chapter 28 / Chapter 28

This chapter contains **11 code files** demonstrating chapter 28.

本章包含 **11 个代码文件**，演示Chapter 28。

---
## Evolution / 演化路线

  1. `02_count.ipynb` — Count
  2. `03_slidingwin.ipynb` — Slidingwin
  3. `04_convert.ipynb` — Convert
  4. `05_model.ipynb` — Model
  5. `06_train.ipynb` — Train
  6. `07_charpredict.ipynb` — Charpredict
  7. `09_generate.ipynb` — Generate
  8. `10_twolayers.ipynb` — Twolayers
  9. `11_complete.ipynb` — Complete
  10. `12_cuda.ipynb` — Cuda
  11. `15_traincuda.ipynb` — Traincuda

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 28) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 28）是机器学习流水线中的基础构建块。

---
