# PyTorch 深度学习 / Deep Learning with PyTorch
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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

def checkpoint(model, filename):
    # 获取模型参数字典 / Get model parameter dictionary
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    # 加载模型参数 / Load model parameters
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

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

def checkpoint(model, filename):
    # 获取模型参数字典 / Get model parameter dictionary
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    # 加载模型参数 / Load model parameters
    model.load_state_dict(torch.load(filename))
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Train



---

### Resume



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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入数据加载工具 / Import data loading utilities
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder

data = fetch_openml("electricity", version=1, parser="auto")
```

---
## Step 2 — Label encode the target, convert to float tensors

```python
# 转换为NumPy数组 / Convert to NumPy array
X = data['data'].astype('float').values
y = data['target']
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 3 — train-test split for model evaluation

```python
# 定义数据集 / Define dataset
trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])

def checkpoint(model, optimizer, filename):
    torch.save({
        # 获取模型参数字典 / Get model parameter dictionary
        'optimizer': optimizer.state_dict(),
        # 获取模型参数字典 / Get model parameter dictionary
        'model': model.state_dict(),
    }, filename)

def resume(model, optimizer, filename):
    checkpoint = torch.load(filename)
    # 加载模型参数 / Load model parameters
    model.load_state_dict(checkpoint['model'])
    # 加载模型参数 / Load model parameters
    optimizer.load_state_dict(checkpoint['optimizer'])
```

---
## Step 4 — Define the model

```python
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid(),
)
```

---
## Step 5 — Train the model

```python
n_epochs = 100
start_epoch = 0
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = DataLoader(trainset, shuffle=True, batch_size=32)
X_test, y_test = default_collate(testset)
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()
# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.1)

if start_epoch > 0:
    resume_epoch = start_epoch - 1
    resume(model, optimizer, f"epoch-{resume_epoch}.pth")

# 生成整数序列 / Generate integer sequence
for epoch in range(start_epoch, n_epochs):
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
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    # 打印输出 / Print output
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

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入数据加载工具 / Import data loading utilities
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder

data = fetch_openml("electricity", version=1, parser="auto")

# Label encode the target, convert to float tensors
# 转换为NumPy数组 / Convert to NumPy array
X = data['data'].astype('float').values
y = data['target']
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# train-test split for model evaluation
# 定义数据集 / Define dataset
trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])

def checkpoint(model, optimizer, filename):
    torch.save({
        # 获取模型参数字典 / Get model parameter dictionary
        'optimizer': optimizer.state_dict(),
        # 获取模型参数字典 / Get model parameter dictionary
        'model': model.state_dict(),
    }, filename)

def resume(model, optimizer, filename):
    checkpoint = torch.load(filename)
    # 加载模型参数 / Load model parameters
    model.load_state_dict(checkpoint['model'])
    # 加载模型参数 / Load model parameters
    optimizer.load_state_dict(checkpoint['optimizer'])

# Define the model
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid(),
)

# Train the model
n_epochs = 100
start_epoch = 0
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = DataLoader(trainset, shuffle=True, batch_size=32)
X_test, y_test = default_collate(testset)
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()
# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.1)

if start_epoch > 0:
    resume_epoch = start_epoch - 1
    resume(model, optimizer, f"epoch-{resume_epoch}.pth")

# 生成整数序列 / Generate integer sequence
for epoch in range(start_epoch, n_epochs):
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
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    # 打印输出 / Print output
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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入数据加载工具 / Import data loading utilities
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder

data = fetch_openml("electricity", version=1, parser="auto")
```

---
## Step 2 — Label encode the target, convert to float tensors

```python
# 转换为NumPy数组 / Convert to NumPy array
X = data['data'].astype('float').values
y = data['target']
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 3 — train-test split for model evaluation

```python
# 定义数据集 / Define dataset
trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])

def checkpoint(model, filename):
    # 获取模型参数字典 / Get model parameter dictionary
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    # 加载模型参数 / Load model parameters
    model.load_state_dict(torch.load(filename))
```

---
## Step 4 — Define the model

```python
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid(),
)
```

---
## Step 5 — Train the model

```python
n_epochs = 10000  # more than we needed
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = DataLoader(trainset, shuffle=True, batch_size=32)
X_test, y_test = default_collate(testset)
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()
# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.1)

early_stop_thresh = 5
best_accuracy = -1
best_epoch = -1

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
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc) * 100
    # 打印输出 / Print output
    print(f"End of epoch {epoch}: accuracy = {acc:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_epoch = epoch
        checkpoint(model, "best_model.pth")
    elif epoch - best_epoch > early_stop_thresh:
        # 打印输出 / Print output
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

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入数据加载工具 / Import data loading utilities
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder

data = fetch_openml("electricity", version=1, parser="auto")

# Label encode the target, convert to float tensors
# 转换为NumPy数组 / Convert to NumPy array
X = data['data'].astype('float').values
y = data['target']
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# train-test split for model evaluation
# 定义数据集 / Define dataset
trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])

def checkpoint(model, filename):
    # 获取模型参数字典 / Get model parameter dictionary
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    # 加载模型参数 / Load model parameters
    model.load_state_dict(torch.load(filename))

# Define the model
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid(),
)

# Train the model
n_epochs = 10000  # more than we needed
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = DataLoader(trainset, shuffle=True, batch_size=32)
X_test, y_test = default_collate(testset)
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()
# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.1)

early_stop_thresh = 5
best_accuracy = -1
best_epoch = -1

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
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc) * 100
    # 打印输出 / Print output
    print(f"End of epoch {epoch}: accuracy = {acc:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_epoch = epoch
        checkpoint(model, "best_model.pth")
    elif epoch - best_epoch > early_stop_thresh:
        # 打印输出 / Print output
        print("Early stopped training at epoch %d" % epoch)
        break  # terminate the training loop

resume(model, "best_model.pth")
```

---

### Chapter Summary / 章节总结



---
