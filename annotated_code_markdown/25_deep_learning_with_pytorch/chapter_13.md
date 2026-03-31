# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 13

---

### Example

# 01 — Example / 01 Example

**Chapter 13 — File 1 of 8 / 第13章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Load data into NumPy arrays**.

本脚本演示 **Load data into NumPy arrays**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
```

---
## Step 2 — Load data into NumPy arrays

```python
data = load_iris()
X, y = data["data"], data["target"]
```

---
## Step 3 — convert NumPy array into PyTorch tensors

```python
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
```

---
## Step 4 — split

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
```

---
## Step 5 — PyTorch model

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()
```

---
## Step 6 — loss metric and optimizer

```python
loss_fn = nn.NLLLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---
## Step 7 — prepare model and training parameters

```python
n_epochs = 100
batch_size = 5
# 获取长度 / Get length
batch_start = torch.arange(0, len(X), batch_size)
```

---
## Step 8 — training loop

```python
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
```

---
## Step 9 — take a batch

```python
X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
```

---
## Step 10 — forward pass

```python
y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
```

---
## Step 11 — backward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
```

---
## Step 12 — update weights

```python
# 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
optimizer.step()
```

---
## Learning Notes / 学习笔记

- **概念**: Load data into NumPy arrays 是机器学习中的常用技术。  
  *Load data into NumPy arrays is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example / 01 Example
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split

# Load data into NumPy arrays
data = load_iris()
X, y = data["data"], data["target"]

# convert NumPy array into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# split
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# PyTorch model
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()

# loss metric and optimizer
loss_fn = nn.NLLLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 100
batch_size = 5
# 获取长度 / Get length
batch_start = torch.arange(0, len(X), batch_size)

# training loop
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
        # take a batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # update weights
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Copy

# 04 — Copy / 04 Copy

**Chapter 13 — File 2 of 8 / 第13章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Load data into NumPy arrays**.

本脚本演示 **Load data into NumPy arrays**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
```

---
## Step 2 — Load data into NumPy arrays

```python
data = load_iris()
X, y = data["data"], data["target"]
```

---
## Step 3 — convert NumPy array into PyTorch tensors

```python
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
```

---
## Step 4 — split

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
```

---
## Step 5 — PyTorch model

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()
```

---
## Step 6 — loss metric and optimizer

```python
loss_fn = nn.NLLLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---
## Step 7 — prepare model and training parameters

```python
n_epochs = 100
batch_size = 5
# 获取长度 / Get length
batch_start = torch.arange(0, len(X), batch_size)
```

---
## Step 8 — training loop

```python
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
```

---
## Step 9 — take a batch

```python
X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
```

---
## Step 10 — forward pass

```python
y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
```

---
## Step 11 — backward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
```

---
## Step 12 — update weights

```python
# 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
optimizer.step()
```

---
## Step 13 — test for accuracy

```python
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)
```

---
## Step 14 — create a new model

```python
newmodel = Multiclass()
```

---
## Step 15 — ask PyTorch to ignore autograd on update and overwrite parameters

```python
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # 将多个序列配对 / Pair multiple sequences
    for newtensor, oldtensor in zip(newmodel.parameters(), model.parameters()):
        newtensor.copy_(oldtensor)
```

---
## Step 16 — test with new model using copied tensor

```python
y_pred = newmodel(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)
```

---
## Learning Notes / 学习笔记

- **概念**: Load data into NumPy arrays 是机器学习中的常用技术。  
  *Load data into NumPy arrays is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Copy / 04 Copy
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split

# Load data into NumPy arrays
data = load_iris()
X, y = data["data"], data["target"]

# convert NumPy array into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# split
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# PyTorch model
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()

# loss metric and optimizer
loss_fn = nn.NLLLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 100
batch_size = 5
# 获取长度 / Get length
batch_start = torch.arange(0, len(X), batch_size)

# training loop
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
        # take a batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # update weights
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()

# test for accuracy
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)

# create a new model
newmodel = Multiclass()
# ask PyTorch to ignore autograd on update and overwrite parameters
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # 将多个序列配对 / Pair multiple sequences
    for newtensor, oldtensor in zip(newmodel.parameters(), model.parameters()):
        newtensor.copy_(oldtensor)
# test with new model using copied tensor
y_pred = newmodel(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Print

# 05 — Print / 05 Print

**Chapter 13 — File 3 of 8 / 第13章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **PyTorch model**.

本脚本演示 **PyTorch model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
import pprint
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
```

---
## Step 2 — PyTorch model

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()

pp = pprint.PrettyPrinter(indent=4)
# 打印输出 / Print output
pp.pprint(model.state_dict())
```

---
## Learning Notes / 学习笔记

- **概念**: PyTorch model 是机器学习中的常用技术。  
  *PyTorch model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Print / 05 Print
# Complete Code / 完整代码
# ===============================

import pprint
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# PyTorch model
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()

pp = pprint.PrettyPrinter(indent=4)
# 打印输出 / Print output
pp.pprint(model.state_dict())
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Serialize

# 06 — Serialize / 06 Serialize

**Chapter 13 — File 4 of 8 / 第13章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Load data into NumPy arrays**.

本脚本演示 **Load data into NumPy arrays**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
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
# 导入对象序列化模块 / Import object serialization module
import pickle
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
```

---
## Step 2 — Load data into NumPy arrays

```python
data = load_iris()
X, y = data["data"], data["target"]
```

---
## Step 3 — convert NumPy array into PyTorch tensors

```python
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
```

---
## Step 4 — split

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
```

---
## Step 5 — PyTorch model

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()
```

---
## Step 6 — loss metric and optimizer

```python
loss_fn = nn.NLLLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---
## Step 7 — prepare model and training parameters

```python
n_epochs = 100
batch_size = 5
# 获取长度 / Get length
batch_start = torch.arange(0, len(X), batch_size)
```

---
## Step 8 — training loop

```python
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
```

---
## Step 9 — take a batch

```python
X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
```

---
## Step 10 — forward pass

```python
y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
```

---
## Step 11 — backward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
```

---
## Step 12 — update weights

```python
# 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
optimizer.step()
```

---
## Step 13 — test for accuracy

```python
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)
```

---
## Step 14 — Save model

```python
# 打开文件（自动关闭） / Open file (auto-close)
with open("iris-model.pickle", "wb") as fp:
    # 获取模型参数字典 / Get model parameter dictionary
    pickle.dump(model.state_dict(), fp)
```

---
## Step 15 — Create new model and load states

```python
newmodel = Multiclass()
# 打开文件（自动关闭） / Open file (auto-close)
with open("iris-model.pickle", "rb") as fp:
    # 加载模型参数 / Load model parameters
    newmodel.load_state_dict(pickle.load(fp))
```

---
## Step 16 — test with new model using copied tensor

```python
y_pred = newmodel(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)
```

---
## Learning Notes / 学习笔记

- **概念**: Load data into NumPy arrays 是机器学习中的常用技术。  
  *Load data into NumPy arrays is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Serialize / 06 Serialize
# Complete Code / 完整代码
# ===============================

# 导入对象序列化模块 / Import object serialization module
import pickle
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split

# Load data into NumPy arrays
data = load_iris()
X, y = data["data"], data["target"]

# convert NumPy array into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# split
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# PyTorch model
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()

# loss metric and optimizer
loss_fn = nn.NLLLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 100
batch_size = 5
# 获取长度 / Get length
batch_start = torch.arange(0, len(X), batch_size)

# training loop
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
        # take a batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # update weights
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()

# test for accuracy
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)

# Save model
# 打开文件（自动关闭） / Open file (auto-close)
with open("iris-model.pickle", "wb") as fp:
    # 获取模型参数字典 / Get model parameter dictionary
    pickle.dump(model.state_dict(), fp)

# Create new model and load states
newmodel = Multiclass()
# 打开文件（自动关闭） / Open file (auto-close)
with open("iris-model.pickle", "rb") as fp:
    # 加载模型参数 / Load model parameters
    newmodel.load_state_dict(pickle.load(fp))

# test with new model using copied tensor
y_pred = newmodel(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Saveload

# 07 — Saveload / 保存/加载模型

**Chapter 13 — File 5 of 8 / 第13章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Load data into NumPy arrays**.

本脚本演示 **Load data into NumPy arrays**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
```

---
## Step 2 — Load data into NumPy arrays

```python
data = load_iris()
X, y = data["data"], data["target"]
```

---
## Step 3 — convert NumPy array into PyTorch tensors

```python
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
```

---
## Step 4 — split

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
```

---
## Step 5 — PyTorch model

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()
```

---
## Step 6 — loss metric and optimizer

```python
loss_fn = nn.NLLLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---
## Step 7 — prepare model and training parameters

```python
n_epochs = 100
batch_size = 5
# 获取长度 / Get length
batch_start = torch.arange(0, len(X), batch_size)
```

---
## Step 8 — training loop

```python
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
```

---
## Step 9 — take a batch

```python
X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
```

---
## Step 10 — forward pass

```python
y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
```

---
## Step 11 — backward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
```

---
## Step 12 — update weights

```python
# 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
optimizer.step()
```

---
## Step 13 — test for accuracy

```python
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)
```

---
## Step 14 — Save model

```python
# 获取模型参数字典 / Get model parameter dictionary
torch.save(model.state_dict(), "iris-model.pth")
```

---
## Step 15 — Create new model and load states

```python
newmodel = Multiclass()
# 加载模型参数 / Load model parameters
newmodel.load_state_dict(torch.load("iris-model.pth"))
```

---
## Step 16 — test with new model using copied tensor

```python
y_pred = newmodel(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)
```

---
## Learning Notes / 学习笔记

- **概念**: Load data into NumPy arrays 是机器学习中的常用技术。  
  *Load data into NumPy arrays is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Saveload / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split

# Load data into NumPy arrays
data = load_iris()
X, y = data["data"], data["target"]

# convert NumPy array into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# split
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# PyTorch model
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()

# loss metric and optimizer
loss_fn = nn.NLLLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 100
batch_size = 5
# 获取长度 / Get length
batch_start = torch.arange(0, len(X), batch_size)

# training loop
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
        # take a batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # update weights
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()

# test for accuracy
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)

# Save model
# 获取模型参数字典 / Get model parameter dictionary
torch.save(model.state_dict(), "iris-model.pth")

# Create new model and load states
newmodel = Multiclass()
# 加载模型参数 / Load model parameters
newmodel.load_state_dict(torch.load("iris-model.pth"))

# test with new model using copied tensor
y_pred = newmodel(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Saveload

# 08 — Saveload / 保存/加载模型

**Chapter 13 — File 6 of 8 / 第13章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Load data into NumPy arrays**.

本脚本演示 **Load data into NumPy arrays**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
```

---
## Step 2 — Load data into NumPy arrays

```python
data = load_iris()
X, y = data["data"], data["target"]
```

---
## Step 3 — convert NumPy array into PyTorch tensors

```python
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
```

---
## Step 4 — split

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
```

---
## Step 5 — PyTorch model

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()
```

---
## Step 6 — loss metric and optimizer

```python
loss_fn = nn.NLLLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---
## Step 7 — prepare model and training parameters

```python
n_epochs = 100
batch_size = 5
# 获取长度 / Get length
batch_start = torch.arange(0, len(X), batch_size)
```

---
## Step 8 — training loop

```python
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
```

---
## Step 9 — take a batch

```python
X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
```

---
## Step 10 — forward pass

```python
y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
```

---
## Step 11 — backward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
```

---
## Step 12 — update weights

```python
# 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
optimizer.step()
```

---
## Step 13 — test for accuracy

```python
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)
```

---
## Step 14 — Save model

```python
torch.save(model, "iris-model-full.pth")
```

---
## Step 15 — Load model

```python
newmodel = torch.load("iris-model-full.pth")
```

---
## Step 16 — test with new model using copied tensor

```python
y_pred = newmodel(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)
```

---
## Learning Notes / 学习笔记

- **概念**: Load data into NumPy arrays 是机器学习中的常用技术。  
  *Load data into NumPy arrays is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Saveload / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split

# Load data into NumPy arrays
data = load_iris()
X, y = data["data"], data["target"]

# convert NumPy array into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# split
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# PyTorch model
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()

# loss metric and optimizer
loss_fn = nn.NLLLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 100
batch_size = 5
# 获取长度 / Get length
batch_start = torch.arange(0, len(X), batch_size)

# training loop
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
        # take a batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # update weights
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()

# test for accuracy
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)

# Save model
torch.save(model, "iris-model-full.pth")

# Load model
newmodel = torch.load("iris-model-full.pth")

# test with new model using copied tensor
y_pred = newmodel(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Save

# 09 — Save / 保存/加载模型

**Chapter 13 — File 7 of 8 / 第13章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Load data into NumPy arrays**.

本脚本演示 **Load data into NumPy arrays**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
```

---
## Step 2 — Load data into NumPy arrays

```python
data = load_iris()
X, y = data["data"], data["target"]
```

---
## Step 3 — convert NumPy array into PyTorch tensors

```python
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
```

---
## Step 4 — split

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
```

---
## Step 5 — PyTorch model

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()
```

---
## Step 6 — loss metric and optimizer

```python
loss_fn = nn.NLLLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---
## Step 7 — prepare model and training parameters

```python
n_epochs = 100
batch_size = 5
# 获取长度 / Get length
batch_start = torch.arange(0, len(X), batch_size)
```

---
## Step 8 — training loop

```python
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
```

---
## Step 9 — take a batch

```python
X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
```

---
## Step 10 — forward pass

```python
y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
```

---
## Step 11 — backward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
```

---
## Step 12 — update weights

```python
# 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
optimizer.step()
```

---
## Step 13 — Save model

```python
# 获取模型参数字典 / Get model parameter dictionary
torch.save(model.state_dict(), "iris-model.pth")
```

---
## Learning Notes / 学习笔记

- **概念**: Load data into NumPy arrays 是机器学习中的常用技术。  
  *Load data into NumPy arrays is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Save / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split

# Load data into NumPy arrays
data = load_iris()
X, y = data["data"], data["target"]

# convert NumPy array into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# split
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# PyTorch model
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()

# loss metric and optimizer
loss_fn = nn.NLLLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 100
batch_size = 5
# 获取长度 / Get length
batch_start = torch.arange(0, len(X), batch_size)

# training loop
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
        # take a batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # update weights
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()

# Save model
# 获取模型参数字典 / Get model parameter dictionary
torch.save(model.state_dict(), "iris-model.pth")
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Load

# 10 — Load / 10 Load

**Chapter 13 — File 8 of 8 / 第13章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Load data into NumPy arrays**.

本脚本演示 **Load data into NumPy arrays**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌───────────────────────┐
│  定义模型 Define Model  │
└───────────────────────┘
```

---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
```

---
## Step 2 — Load data into NumPy arrays

```python
data = load_iris()
X, y = data["data"], data["target"]
```

---
## Step 3 — convert NumPy array into PyTorch tensors

```python
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
```

---
## Step 4 — split

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
```

---
## Step 5 — PyTorch model

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x
```

---
## Step 6 — Create new model and load states

```python
model = Multiclass()
# 加载模型参数 / Load model parameters
model.load_state_dict(torch.load("iris-model.pth"))
```

---
## Step 7 — Run model for inference

```python
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)
```

---
## Learning Notes / 学习笔记

- **概念**: Load data into NumPy arrays 是机器学习中的常用技术。  
  *Load data into NumPy arrays is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `numpy` | 数值计算库 | Numerical computing library |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load / 10 Load
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_iris
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split

# Load data into NumPy arrays
data = load_iris()
X, y = data["data"], data["target"]

# convert NumPy array into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# split
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# PyTorch model
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Multiclass(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.hidden = nn.Linear(4, 8)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

# Create new model and load states
model = Multiclass()
# 加载模型参数 / Load model parameters
model.load_state_dict(torch.load("iris-model.pth"))

# Run model for inference
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
# 打印输出 / Print output
print("Accuracy: %.2f" % acc)
```

---

### Chapter Summary / 章节总结

# Chapter 13 Summary / 第13章总结

## Theme / 主题: Chapter 13 / Chapter 13

This chapter contains **8 code files** demonstrating chapter 13.

本章包含 **8 个代码文件**，演示Chapter 13。

---
## Evolution / 演化路线

  1. `01_example.ipynb` — Example
  2. `04_copy.ipynb` — Copy
  3. `05_print.ipynb` — Print
  4. `06_serialize.ipynb` — Serialize
  5. `07_saveload.ipynb` — Saveload
  6. `08_saveload.ipynb` — Saveload
  7. `09_save.ipynb` — Save
  8. `10_load.ipynb` — Load

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 13) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 13）是机器学习流水线中的基础构建块。

---
