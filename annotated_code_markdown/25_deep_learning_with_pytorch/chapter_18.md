# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 18

---

### Load



---

### Dataloader

# 03 — Dataloader / 03 Dataloader

**Chapter 18 — File 2 of 4 / 第18章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Read data, convert to NumPy arrays**.

本脚本演示 **Read data, convert to NumPy arrays**。

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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入数据加载工具 / Import data loading utilities
from torch.utils.data import DataLoader
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
```

---
## Step 2 — Read data, convert to NumPy arrays

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("sonar.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
X = data.iloc[:, 0:60].values
# 转换为NumPy数组 / Convert to NumPy array
y = data.iloc[:, 60].values
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
## Step 6 — set up DataLoader for training set

```python
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=16)
```

---
## Step 7 — create model

```python
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(60, 60),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(60, 30),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(30, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
```

---
## Step 8 — Train the model

```python
n_epochs = 200
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()
# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.1)
# 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
model.train()
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
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

- **概念**: Read data, convert to NumPy arrays 是机器学习中的常用技术。  
  *Read data, convert to NumPy arrays is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataLoader` | 数据加载器，批量读取数据 | Loads data in batches for training |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `numpy` | 数值计算库 | Numerical computing library |
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
# Dataloader / 03 Dataloader
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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入数据加载工具 / Import data loading utilities
from torch.utils.data import DataLoader
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder

# Read data, convert to NumPy arrays
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("sonar.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
X = data.iloc[:, 0:60].values
# 转换为NumPy数组 / Convert to NumPy array
y = data.iloc[:, 60].values

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

# set up DataLoader for training set
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=16)

# create model
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(60, 60),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(60, 30),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(30, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)

# Train the model
n_epochs = 200
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()
# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.1)
# 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
model.train()
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()

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

➡️ **Next / 下一步**: File 3 of 4

---

### Dataset



---

### Training

# 06 — Training / 06 Training

**Chapter 18 — File 4 of 4 / 第18章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Read data, convert to NumPy arrays**.

本脚本演示 **Read data, convert to NumPy arrays**。

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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入数据加载工具 / Import data loading utilities
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
```

---
## Step 2 — Read data, convert to NumPy arrays

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("sonar.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
X = data.iloc[:, 0:60].values
# 转换为NumPy数组 / Convert to NumPy array
y = data.iloc[:, 60].values
```

---
## Step 3 — encode class values as integers

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = encoder.transform(y).reshape(-1, 1)

# 定义数据集 / Define dataset
class SonarDataset(Dataset):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, X, y):
```

---
## Step 4 — convert into PyTorch tensors and remember them

```python
self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
```

---
## Step 5 — this should return the size of the dataset

```python
# 获取长度 / Get length
return len(self.X)

    def __getitem__(self, idx):
```

---
## Step 6 — this should return one sample from the dataset

```python
features = self.X[idx]
        target = self.y[idx]
        return features, target
```

---
## Step 7 — set up DataLoader for data set

```python
# 定义数据集 / Define dataset
dataset = SonarDataset(X, y)
trainset, testset = random_split(dataset, [0.7, 0.3])
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = DataLoader(trainset, shuffle=True, batch_size=16)
```

---
## Step 8 — create model

```python
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(60, 60),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(60, 30),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(30, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
```

---
## Step 9 — Train the model

```python
n_epochs = 200
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()
# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.1)
# 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
model.train()
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
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
## Step 10 — create one test tensor from the testset

```python
X_test, y_test = default_collate(testset)
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

- **概念**: Read data, convert to NumPy arrays 是机器学习中的常用技术。  
  *Read data, convert to NumPy arrays is a common technique in machine learning.*

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
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Training / 06 Training
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
# 导入数据加载工具 / Import data loading utilities
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder

# Read data, convert to NumPy arrays
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("sonar.csv", header=None)
# 转换为NumPy数组 / Convert to NumPy array
X = data.iloc[:, 0:60].values
# 转换为NumPy数组 / Convert to NumPy array
y = data.iloc[:, 60].values

# encode class values as integers
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = encoder.transform(y).reshape(-1, 1)

# 定义数据集 / Define dataset
class SonarDataset(Dataset):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        # 获取长度 / Get length
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target

# set up DataLoader for data set
# 定义数据集 / Define dataset
dataset = SonarDataset(X, y)
trainset, testset = random_split(dataset, [0.7, 0.3])
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = DataLoader(trainset, shuffle=True, batch_size=16)

# create model
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(60, 60),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(60, 30),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(30, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)

# Train the model
n_epochs = 200
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()
# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.1)
# 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
model.train()
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()

# create one test tensor from the testset
X_test, y_test = default_collate(testset)
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

# Chapter 18 Summary / 第18章总结

## Theme / 主题: Chapter 18 / Chapter 18

This chapter contains **4 code files** demonstrating chapter 18.

本章包含 **4 个代码文件**，演示Chapter 18。

---
## Evolution / 演化路线

  1. `01_load.ipynb` — Load
  2. `03_dataloader.ipynb` — Dataloader
  3. `04_dataset.ipynb` — Dataset
  4. `06_training.ipynb` — Training

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 18) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 18）是机器学习流水线中的基础构建块。

---
