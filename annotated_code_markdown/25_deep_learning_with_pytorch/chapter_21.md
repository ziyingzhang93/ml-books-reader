# PyTorch DL
## Chapter 21

---

### Func

# 01 — Func / 01 Func

**Chapter 21 — File 1 of 5 / 第21章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Func**.

本脚本演示 **01 Func**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import torch

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))
```

---
## Learning Notes / 学习笔记

- **概念**: Func 是机器学习中的常用技术。  
  *Func is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Func / 01 Func
# Complete Code / 完整代码
# ===============================

import torch

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Checkpointing

# 07 — Checkpointing / 07 Checkpointing

**Chapter 21 — File 4 of 5 / 第21章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Label encode the target, convert to float tensors**.

本脚本演示 **Label encode the target, convert to float tensors**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — Step 1

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

data = fetch_openml("electricity", version=1, parser="auto")
```

---
## Step 2 — Label encode the target, convert to float tensors

```python
X = data['data'].astype('float').values
y = data['target']
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 3 — train-test split for model evaluation

```python
trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])

def checkpoint(model, optimizer, filename):
    torch.save({
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
    }, filename)

def resume(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
```

---
## Step 4 — Define the model

```python
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 12),
    nn.ReLU(),
    nn.Linear(12, 1),
    nn.Sigmoid(),
)
```

---
## Step 5 — Train the model

```python
n_epochs = 100
start_epoch = 0
loader = DataLoader(trainset, shuffle=True, batch_size=32)
X_test, y_test = default_collate(testset)
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

if start_epoch > 0:
    resume_epoch = start_epoch - 1
    resume(model, optimizer, f"epoch-{resume_epoch}.pth")

for epoch in range(start_epoch, n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    print(f"End of epoch {epoch}: accuracy = {float(acc)*100:.2f}%")
    checkpoint(model, optimizer, f"epoch-{epoch}.pth")
```

---
## Learning Notes / 学习笔记

- **概念**: Label encode the target, convert to float tensors 是机器学习中的常用技术。  
  *Label encode the target, convert to float tensors is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataLoader` | 数据加载器，批量读取数据 | Loads data in batches for training |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Checkpointing / 07 Checkpointing
# Complete Code / 完整代码
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

data = fetch_openml("electricity", version=1, parser="auto")

# Label encode the target, convert to float tensors
X = data['data'].astype('float').values
y = data['target']
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# train-test split for model evaluation
trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])

def checkpoint(model, optimizer, filename):
    torch.save({
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
    }, filename)

def resume(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Define the model
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 12),
    nn.ReLU(),
    nn.Linear(12, 1),
    nn.Sigmoid(),
)

# Train the model
n_epochs = 100
start_epoch = 0
loader = DataLoader(trainset, shuffle=True, batch_size=32)
X_test, y_test = default_collate(testset)
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

if start_epoch > 0:
    resume_epoch = start_epoch - 1
    resume(model, optimizer, f"epoch-{resume_epoch}.pth")

for epoch in range(start_epoch, n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    print(f"End of epoch {epoch}: accuracy = {float(acc)*100:.2f}%")
    checkpoint(model, optimizer, f"epoch-{epoch}.pth")
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Earlystop

# 10 — Earlystop / 10 Earlystop

**Chapter 21 — File 5 of 5 / 第21章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Label encode the target, convert to float tensors**.

本脚本演示 **Label encode the target, convert to float tensors**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — Step 1

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

data = fetch_openml("electricity", version=1, parser="auto")
```

---
## Step 2 — Label encode the target, convert to float tensors

```python
X = data['data'].astype('float').values
y = data['target']
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 3 — train-test split for model evaluation

```python
trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))
```

---
## Step 4 — Define the model

```python
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 12),
    nn.ReLU(),
    nn.Linear(12, 1),
    nn.Sigmoid(),
)
```

---
## Step 5 — Train the model

```python
n_epochs = 10000  # more than we needed
loader = DataLoader(trainset, shuffle=True, batch_size=32)
X_test, y_test = default_collate(testset)
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

early_stop_thresh = 5
best_accuracy = -1
best_epoch = -1

for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc) * 100
    print(f"End of epoch {epoch}: accuracy = {acc:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_epoch = epoch
        checkpoint(model, "best_model.pth")
    elif epoch - best_epoch > early_stop_thresh:
        print("Early stopped training at epoch %d" % epoch)
        break  # terminate the training loop

resume(model, "best_model.pth")
```

---
## Learning Notes / 学习笔记

- **概念**: Label encode the target, convert to float tensors 是机器学习中的常用技术。  
  *Label encode the target, convert to float tensors is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataLoader` | 数据加载器，批量读取数据 | Loads data in batches for training |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Earlystop / 10 Earlystop
# Complete Code / 完整代码
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

data = fetch_openml("electricity", version=1, parser="auto")

# Label encode the target, convert to float tensors
X = data['data'].astype('float').values
y = data['target']
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# train-test split for model evaluation
trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))

# Define the model
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 12),
    nn.ReLU(),
    nn.Linear(12, 1),
    nn.Sigmoid(),
)

# Train the model
n_epochs = 10000  # more than we needed
loader = DataLoader(trainset, shuffle=True, batch_size=32)
X_test, y_test = default_collate(testset)
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

early_stop_thresh = 5
best_accuracy = -1
best_epoch = -1

for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc) * 100
    print(f"End of epoch {epoch}: accuracy = {acc:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_epoch = epoch
        checkpoint(model, "best_model.pth")
    elif epoch - best_epoch > early_stop_thresh:
        print("Early stopped training at epoch %d" % epoch)
        break  # terminate the training loop

resume(model, "best_model.pth")
```

---
