# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 17

---

### Fixed



---

### Linear



---

### Exp



---

### Custom

# 07 — Custom / 07 Custom

**Chapter 17 — File 4 of 4 / 第17章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **load dataset, split into input (X) and output (y) variables**.

本脚本演示 **load dataset, split into input (X) and output (y) variables**。

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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim.lr_scheduler as lr_scheduler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
```

---
## Step 2 — load dataset, split into input (X) and output (y) variables

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = pd.read_csv("ionosphere.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
X = dataset[:,0:34].astype(float)
y = dataset[:,34]
```

---
## Step 3 — encode class values as integers

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)
```

---
## Step 4 — convert into PyTorch tensors

```python
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 5 — train-test split for evaluation of the model

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
```

---
## Step 6 — create model

```python
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(34, 34),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(34, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)

def lr_lambda(epoch):
```

---
## Step 7 — LR to be 0.1 * (1/1+0.01*epoch)

```python
base_lr = 0.1
    factor = 0.01
    return base_lr/(1+factor*epoch)
```

---
## Step 8 — Train the model

```python
n_epochs = 50
batch_size = 24
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)
lr = 0.1
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()
# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=lr)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
# 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
model.train()
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
    before_lr = optimizer.param_groups[0]["lr"]
    # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    # 打印输出 / Print output
    print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))
```

---
## Step 9 — evaluate accuracy after training

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
y_pred = model(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
# 打印输出 / Print output
print("Model accuracy: %.2f%%" % (acc*100))
```

---
## Learning Notes / 学习笔记

- **概念**: load dataset, split into input (X) and output (y) variables 是机器学习中的常用技术。  
  *load dataset, split into input (X) and output (y) variables is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
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
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Custom / 07 Custom
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim.lr_scheduler as lr_scheduler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split

# load dataset, split into input (X) and output (y) variables
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = pd.read_csv("ionosphere.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
X = dataset[:,0:34].astype(float)
y = dataset[:,34]

# encode class values as integers
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)

# convert into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# train-test split for evaluation of the model
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# create model
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(34, 34),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(34, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)

def lr_lambda(epoch):
    # LR to be 0.1 * (1/1+0.01*epoch)
    base_lr = 0.1
    factor = 0.01
    return base_lr/(1+factor*epoch)

# Train the model
n_epochs = 50
batch_size = 24
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)
lr = 0.1
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()
# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=lr)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
# 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
model.train()
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for start in batch_start:
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
    before_lr = optimizer.param_groups[0]["lr"]
    # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    # 打印输出 / Print output
    print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))

# evaluate accuracy after training
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
y_pred = model(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
# 打印输出 / Print output
print("Model accuracy: %.2f%%" % (acc*100))
```

---

### Chapter Summary / 章节总结

# Chapter 17 Summary / 第17章总结

## Theme / 主题: Chapter 17 / Chapter 17

This chapter contains **4 code files** demonstrating chapter 17.

本章包含 **4 个代码文件**，演示Chapter 17。

---
## Evolution / 演化路线

  1. `02_fixed.ipynb` — Fixed
  2. `03_linear.ipynb` — Linear
  3. `04_exp.ipynb` — Exp
  4. `07_custom.ipynb` — Custom

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 17) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 17）是机器学习流水线中的基础构建块。

---
