# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 23

---

### Example

# 01 — Example / 01 Example

**Chapter 23 — File 1 of 3 / 第23章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Read data**.

本脚本演示 **Read data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
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
from sklearn.datasets import fetch_california_housing
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
```

---
## Step 2 — Read data

```python
data = fetch_california_housing()
X, y = data.data, data.target
```

---
## Step 3 — train-test split for model evaluation

```python
# 划分训练集和测试集 / Split into train and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                            shuffle=True)
```

---
## Step 4 — Standardizing data

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler()
scaler.fit(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train = scaler.transform(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test = scaler.transform(X_test_raw)
```

---
## Step 5 — Convert to 2D PyTorch tensors

```python
X_train = torch.tensor(X_train, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 6 — Define the model

```python
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 24),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(24, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 6),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(6, 1)
)
```

---
## Step 7 — loss function and optimizer

```python
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()  # mean square error
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100   # number of epochs to run
batch_size = 32  # size of each batch
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
```

---
## Step 8 — take a batch

```python
X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
```

---
## Step 9 — forward pass

```python
y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
```

---
## Step 10 — backward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
```

---
## Step 11 — update weights

```python
# 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
optimizer.step()
```

---
## Learning Notes / 学习笔记

- **概念**: Read data 是机器学习中的常用技术。  
  *Read data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.MSELoss` | 均方误差损失，回归任务常用 | Mean squared error loss for regression |
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
from sklearn.datasets import fetch_california_housing
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler

# Read data
data = fetch_california_housing()
X, y = data.data, data.target

# train-test split for model evaluation
# 划分训练集和测试集 / Split into train and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                            shuffle=True)
# Standardizing data
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler()
scaler.fit(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train = scaler.transform(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test = scaler.transform(X_test_raw)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define the model
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 24),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(24, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 6),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(6, 1)
)

# loss function and optimizer
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()  # mean square error
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100   # number of epochs to run
batch_size = 32  # size of each batch
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)

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

➡️ **Next / 下一步**: File 2 of 3

---

### Metrics

# 04 — Metrics / 04 Metrics

**Chapter 23 — File 2 of 3 / 第23章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Read data**.

本脚本演示 **Read data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
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
from sklearn.datasets import fetch_california_housing
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
```

---
## Step 2 — Read data

```python
data = fetch_california_housing()
X, y = data.data, data.target
```

---
## Step 3 — train-test split for model evaluation

```python
# 划分训练集和测试集 / Split into train and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                            shuffle=True)
```

---
## Step 4 — Standardizing data

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler()
scaler.fit(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train = scaler.transform(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test = scaler.transform(X_test_raw)
```

---
## Step 5 — Convert to 2D PyTorch tensors

```python
X_train = torch.tensor(X_train, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 6 — Define the model

```python
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 24),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(24, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 6),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(6, 1)
)
```

---
## Step 7 — loss function, metrics, and optimizer

```python
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()  # mean square error
mae_fn = nn.L1Loss()  # mean absolute error
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100   # number of epochs to run
batch_size = 32  # size of each batch
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)

train_mse_history = []
test_mse_history = []
test_mae_history = []

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    epoch_mse = []
    for start in batch_start:
```

---
## Step 8 — take a batch

```python
X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
```

---
## Step 9 — forward pass

```python
y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 添加元素到列表末尾 / Append element to list end
        epoch_mse.append(float(loss))
```

---
## Step 10 — backward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
```

---
## Step 11 — update weights

```python
# 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
optimizer.step()
    # 获取长度 / Get length
    mean_mse = sum(epoch_mse) / len(epoch_mse)
    # 添加元素到列表末尾 / Append element to list end
    train_mse_history.append(mean_mse)
```

---
## Step 12 — validate model on test set

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mae = mae_fn(y_pred, y_test)
        # 添加元素到列表末尾 / Append element to list end
        test_mse_history.append(float(mse))
        # 添加元素到列表末尾 / Append element to list end
        test_mae_history.append(float(mae))
```

---
## Learning Notes / 学习笔记

- **概念**: Read data 是机器学习中的常用技术。  
  *Read data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.MSELoss` | 均方误差损失，回归任务常用 | Mean squared error loss for regression |
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
# Metrics / 04 Metrics
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_california_housing
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler

# Read data
data = fetch_california_housing()
X, y = data.data, data.target

# train-test split for model evaluation
# 划分训练集和测试集 / Split into train and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                            shuffle=True)
# Standardizing data
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler()
scaler.fit(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train = scaler.transform(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test = scaler.transform(X_test_raw)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define the model
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 24),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(24, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 6),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(6, 1)
)

# loss function, metrics, and optimizer
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()  # mean square error
mae_fn = nn.L1Loss()  # mean absolute error
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100   # number of epochs to run
batch_size = 32  # size of each batch
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)

train_mse_history = []
test_mse_history = []
test_mae_history = []

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    epoch_mse = []
    for start in batch_start:
        # take a batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 添加元素到列表末尾 / Append element to list end
        epoch_mse.append(float(loss))
        # backward pass
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # update weights
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
    # 获取长度 / Get length
    mean_mse = sum(epoch_mse) / len(epoch_mse)
    # 添加元素到列表末尾 / Append element to list end
    train_mse_history.append(mean_mse)
    # validate model on test set
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mae = mae_fn(y_pred, y_test)
        # 添加元素到列表末尾 / Append element to list end
        test_mse_history.append(float(mse))
        # 添加元素到列表末尾 / Append element to list end
        test_mae_history.append(float(mae))
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Plot

# 06 — Plot / 06 Plot

**Chapter 23 — File 3 of 3 / 第23章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Read data**.

本脚本演示 **Read data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_california_housing
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
```

---
## Step 2 — Read data

```python
data = fetch_california_housing()
X, y = data.data, data.target
```

---
## Step 3 — train-test split for model evaluation

```python
# 划分训练集和测试集 / Split into train and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                            shuffle=True)
```

---
## Step 4 — Standardizing data

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler()
scaler.fit(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train = scaler.transform(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test = scaler.transform(X_test_raw)
```

---
## Step 5 — Convert to 2D PyTorch tensors

```python
X_train = torch.tensor(X_train, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 6 — Define the model

```python
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 24),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(24, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 6),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(6, 1)
)
```

---
## Step 7 — loss function, metrics, and optimizer

```python
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()  # mean square error
mae_fn = nn.L1Loss()  # mean absolute error
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100   # number of epochs to run
batch_size = 32  # size of each batch
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)

train_mse_history = []
test_mse_history = []
test_mae_history = []

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    epoch_mse = []
    for start in batch_start:
```

---
## Step 8 — take a batch

```python
X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
```

---
## Step 9 — forward pass

```python
y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 添加元素到列表末尾 / Append element to list end
        epoch_mse.append(float(loss))
```

---
## Step 10 — backward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
```

---
## Step 11 — update weights

```python
# 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
optimizer.step()
    # 获取长度 / Get length
    mean_mse = sum(epoch_mse) / len(epoch_mse)
    # 添加元素到列表末尾 / Append element to list end
    train_mse_history.append(mean_mse)
```

---
## Step 12 — validate model on test set

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mae = mae_fn(y_pred, y_test)
        # 添加元素到列表末尾 / Append element to list end
        test_mse_history.append(float(mse))
        # 添加元素到列表末尾 / Append element to list end
        test_mae_history.append(float(mae))

# 绘制折线图 / Draw line plot
plt.plot(np.sqrt(train_mse_history), label="Train RMSE")
# 绘制折线图 / Draw line plot
plt.plot(np.sqrt(test_mse_history), label="Test RMSE")
# 绘制折线图 / Draw line plot
plt.plot(test_mae_history, label="Test MAE")
# 设置X轴标签 / Set X-axis label
plt.xlabel("epochs")
# 显示图例 / Show legend
plt.legend()
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Read data 是机器学习中的常用技术。  
  *Read data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.MSELoss` | 均方误差损失，回归任务常用 | Mean squared error loss for regression |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot / 06 Plot
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_california_housing
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler

# Read data
data = fetch_california_housing()
X, y = data.data, data.target

# train-test split for model evaluation
# 划分训练集和测试集 / Split into train and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                            shuffle=True)

# Standardizing data
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler()
scaler.fit(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train = scaler.transform(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test = scaler.transform(X_test_raw)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define the model
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 24),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(24, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 6),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(6, 1)
)

# loss function, metrics, and optimizer
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()  # mean square error
mae_fn = nn.L1Loss()  # mean absolute error
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100   # number of epochs to run
batch_size = 32  # size of each batch
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)

train_mse_history = []
test_mse_history = []
test_mae_history = []

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    epoch_mse = []
    for start in batch_start:
        # take a batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 添加元素到列表末尾 / Append element to list end
        epoch_mse.append(float(loss))
        # backward pass
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # update weights
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
    # 获取长度 / Get length
    mean_mse = sum(epoch_mse) / len(epoch_mse)
    # 添加元素到列表末尾 / Append element to list end
    train_mse_history.append(mean_mse)
    # validate model on test set
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mae = mae_fn(y_pred, y_test)
        # 添加元素到列表末尾 / Append element to list end
        test_mse_history.append(float(mse))
        # 添加元素到列表末尾 / Append element to list end
        test_mae_history.append(float(mae))

# 绘制折线图 / Draw line plot
plt.plot(np.sqrt(train_mse_history), label="Train RMSE")
# 绘制折线图 / Draw line plot
plt.plot(np.sqrt(test_mse_history), label="Test RMSE")
# 绘制折线图 / Draw line plot
plt.plot(test_mae_history, label="Test MAE")
# 设置X轴标签 / Set X-axis label
plt.xlabel("epochs")
# 显示图例 / Show legend
plt.legend()
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 23 Summary / 第23章总结

## Theme / 主题: Chapter 23 / Chapter 23

This chapter contains **3 code files** demonstrating chapter 23.

本章包含 **3 个代码文件**，演示Chapter 23。

---
## Evolution / 演化路线

  1. `01_example.ipynb` — Example
  2. `04_metrics.ipynb` — Metrics
  3. `06_plot.ipynb` — Plot

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 23) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 23）是机器学习流水线中的基础构建块。

---
