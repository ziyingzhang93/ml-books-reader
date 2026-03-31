# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 22

---

### Train

# 01 — Train / 01 Train

**Chapter 22 — File 1 of 2 / 第22章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **split**.

本脚本演示 **split**。

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

data = load_iris()
X = data['data']
y = data['target']
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
```

---
## Step 2 — split

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class IrisModel(nn.Module):
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

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x
```

---
## Step 3 — loss metric and optimizer

```python
model = IrisModel()
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---
## Step 4 — prepare model and training parameters

```python
n_epochs = 100
batch_size = 10
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)
```

---
## Step 5 — training loop

```python
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
```

---
## Step 6 — take a batch

```python
X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
```

---
## Step 7 — forward pass

```python
y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
```

---
## Step 8 — backward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
```

---
## Step 9 — update weights

```python
# 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
optimizer.step()
```

---
## Step 10 — validating model

```python
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
acc = float(acc)*100
# 打印输出 / Print output
print("Model accuracy: %.2f%%" % acc)
```

---
## Learning Notes / 学习笔记

- **概念**: split 是机器学习中的常用技术。  
  *split is a common technique in machine learning.*

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
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
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
# Train / 01 Train
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

data = load_iris()
X = data['data']
y = data['target']
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# split
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class IrisModel(nn.Module):
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

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

# loss metric and optimizer
model = IrisModel()
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 100
batch_size = 10
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)

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

# validating model
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
acc = float(acc)*100
# 打印输出 / Print output
print("Model accuracy: %.2f%%" % acc)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Onnx

# 02 — Onnx / 02 Onnx

**Chapter 22 — File 2 of 2 / 第22章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **split**.

本脚本演示 **split**。

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

data = load_iris()
X = data['data']
y = data['target']
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
```

---
## Step 2 — split

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class IrisModel(nn.Module):
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

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x
```

---
## Step 3 — loss metric and optimizer

```python
model = IrisModel()
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---
## Step 4 — prepare model and training parameters

```python
n_epochs = 100
batch_size = 10
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)
```

---
## Step 5 — training loop

```python
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
```

---
## Step 6 — take a batch

```python
X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
```

---
## Step 7 — forward pass

```python
y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
```

---
## Step 8 — backward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
```

---
## Step 9 — update weights

```python
# 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
optimizer.step()
```

---
## Step 10 — validating model

```python
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
acc = float(acc)*100
# 打印输出 / Print output
print("Model accuracy: %.2f%%" % acc)
```

---
## Step 11 — export to ONNX

```python
torch.onnx.export(model, X_test, 'iris.onnx',
                  input_names=["features"], output_names=["logits"])
```

---
## Learning Notes / 学习笔记

- **概念**: split 是机器学习中的常用技术。  
  *split is a common technique in machine learning.*

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
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
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
# Onnx / 02 Onnx
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

data = load_iris()
X = data['data']
y = data['target']
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# split
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class IrisModel(nn.Module):
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

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

# loss metric and optimizer
model = IrisModel()
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 100
batch_size = 10
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)

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

# validating model
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
acc = float(acc)*100
# 打印输出 / Print output
print("Model accuracy: %.2f%%" % acc)
# export to ONNX
torch.onnx.export(model, X_test, 'iris.onnx',
                  input_names=["features"], output_names=["logits"])
```

---

### Chapter Summary / 章节总结



---
