# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 12

---

### Data

# 01 — Data / 01 Data

**Chapter 12 — File 1 of 7 / 第12章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Data**.

本脚本演示 **01 Data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
# 打印输出 / Print output
print(data.feature_names)

X, y = data.data, data.target
```

---
## Learning Notes / 学习笔记

- **概念**: Data 是机器学习中的常用技术。  
  *Data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Data / 01 Data
# Complete Code / 完整代码
# ===============================

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
# 打印输出 / Print output
print(data.feature_names)

X, y = data.data, data.target
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Model

# 02 — Model / 02 Model

**Chapter 12 — File 2 of 7 / 第12章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Define the model**.

本脚本演示 **Define the model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
```

---
## Step 2 — Define the model

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
## Learning Notes / 学习笔记

- **概念**: Define the model 是机器学习中的常用技术。  
  *Define the model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model / 02 Model
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

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
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Loss

# 03 — Loss / 损失函数

**Chapter 12 — File 3 of 7 / 第12章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Define the model**.

本脚本演示 **Define the model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
```

---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
```

---
## Step 2 — Define the model

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
## Step 3 — loss function and optimizer

```python
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()  # mean square error
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

---
## Learning Notes / 学习笔记

- **概念**: Define the model 是机器学习中的常用技术。  
  *Define the model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.MSELoss` | 均方误差损失，回归任务常用 | Mean squared error loss for regression |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Loss / 损失函数
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim

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
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Loop

# 04 — Loop / 04 Loop

**Chapter 12 — File 4 of 7 / 第12章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **train-test split of the dataset**.

本脚本演示 **train-test split of the dataset**。

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
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — Step 1

```python
import copy
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
import tqdm
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_california_housing
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X, y = data.data, data.target
```

---
## Step 2 — train-test split of the dataset

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 3 — training parameters

```python
n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)
```

---
## Step 4 — Hold the best model

```python
best_mse = np.inf   # init to infinity
best_weights = None
history = []
```

---
## Step 5 — Define the model

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
## Step 6 — loss function and optimizer

```python
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()  # mean square error
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

---
## Step 7 — training loop

```python
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
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
## Step 12 — print progress

```python
bar.set_postfix(mse=float(loss))
```

---
## Step 13 — evaluate accuracy at end of each epoch

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    # 添加元素到列表末尾 / Append element to list end
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        # 获取模型参数字典 / Get model parameter dictionary
        best_weights = copy.deepcopy(model.state_dict())
```

---
## Step 14 — restore model and return best accuracy

```python
# 加载模型参数 / Load model parameters
model.load_state_dict(best_weights)
```

---
## Learning Notes / 学习笔记

- **概念**: train-test split of the dataset 是机器学习中的常用技术。  
  *train-test split of the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.MSELoss` | 均方误差损失，回归任务常用 | Mean squared error loss for regression |
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
# Loop / 04 Loop
# Complete Code / 完整代码
# ===============================

import copy
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
import tqdm
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_california_housing
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X, y = data.data, data.target

# train-test split of the dataset
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# training parameters
n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []

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
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# training loop
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
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
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    # 添加元素到列表末尾 / Append element to list end
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        # 获取模型参数字典 / Get model parameter dictionary
        best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
# 加载模型参数 / Load model parameters
model.load_state_dict(best_weights)
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Complete

# 06 — Complete / 06 Complete

**Chapter 12 — File 5 of 7 / 第12章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Read data**.

本脚本演示 **Read data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
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
import copy

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
import tqdm
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_california_housing
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
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
```

---
## Step 4 — Convert to 2D PyTorch tensors

```python
X_train = torch.tensor(X_train, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 5 — Define the model

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
## Step 6 — loss function and optimizer

```python
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()  # mean square error
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)
```

---
## Step 7 — Hold the best model

```python
best_mse = np.inf   # init to infinity
best_weights = None
history = []

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
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
## Step 12 — print progress

```python
bar.set_postfix(mse=float(loss))
```

---
## Step 13 — evaluate accuracy at end of each epoch

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    # 添加元素到列表末尾 / Append element to list end
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        # 获取模型参数字典 / Get model parameter dictionary
        best_weights = copy.deepcopy(model.state_dict())
```

---
## Step 14 — restore model and return best accuracy

```python
# 加载模型参数 / Load model parameters
model.load_state_dict(best_weights)
# 打印输出 / Print output
print("MSE: %.2f" % best_mse)
# 打印输出 / Print output
print("RMSE: %.2f" % np.sqrt(best_mse))
# 绘制折线图 / Draw line plot
plt.plot(history)
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
# Complete / 06 Complete
# Complete Code / 完整代码
# ===============================

import copy

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
import tqdm
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_california_housing

# Read data
data = fetch_california_housing()
X, y = data.data, data.target

# train-test split for model evaluation
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

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
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
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
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    # 添加元素到列表末尾 / Append element to list end
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        # 获取模型参数字典 / Get model parameter dictionary
        best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
# 加载模型参数 / Load model parameters
model.load_state_dict(best_weights)
# 打印输出 / Print output
print("MSE: %.2f" % best_mse)
# 打印输出 / Print output
print("RMSE: %.2f" % np.sqrt(best_mse))
# 绘制折线图 / Draw line plot
plt.plot(history)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Preprocessing

# 07 — Preprocessing / 07 Preprocessing

**Chapter 12 — File 6 of 7 / 第12章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Read data**.

本脚本演示 **Read data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_california_housing
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
## Learning Notes / 学习笔记

- **概念**: Read data 是机器学习中的常用技术。  
  *Read data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Preprocessing / 07 Preprocessing
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_california_housing
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
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Scaled



---

### Chapter Summary / 章节总结

# Chapter 12 Summary / 第12章总结

## Theme / 主题: Chapter 12 / Chapter 12

This chapter contains **7 code files** demonstrating chapter 12.

本章包含 **7 个代码文件**，演示Chapter 12。

---
## Evolution / 演化路线

  1. `01_data.ipynb` — Data
  2. `02_model.ipynb` — Model
  3. `03_loss.ipynb` — Loss
  4. `04_loop.ipynb` — Loop
  5. `06_complete.ipynb` — Complete
  6. `07_preprocessing.ipynb` — Preprocessing
  7. `09_scaled.ipynb` — Scaled

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 12) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 12）是机器学习流水线中的基础构建块。

---
