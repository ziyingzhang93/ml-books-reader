# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 10

---

### Csv

# 01 — Csv / 01 Csv

**Chapter 10 — File 1 of 7 / 第10章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Csv**.

本脚本演示 **01 Csv**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("iris.csv", header=None)
X = data.iloc[:, 0:4]
y = data.iloc[:, 4:]
```

---
## Learning Notes / 学习笔记

- **概念**: Csv 是机器学习中的常用技术。  
  *Csv is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Csv / 01 Csv
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("iris.csv", header=None)
X = data.iloc[:, 0:4]
y = data.iloc[:, 4:]
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Onehot



---

### Model



---

### Loss



---

### Loop

# 05 — Loop / 05 Loop

**Chapter 10 — File 5 of 7 / 第10章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors**.

本脚本演示 **convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
import tqdm
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("iris.csv", header=None)
X = data.iloc[:, 0:4]
y = data.iloc[:, 4:]

# 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = ohe.transform(y)
```

---
## Step 2 — convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors

```python
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

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

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

model = Multiclass()
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---
## Step 3 — training parameters

```python
n_epochs = 200
batch_size = 5
# 获取长度 / Get length
batches_per_epoch = len(X) // batch_size

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 生成整数序列 / Generate integer sequence
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
```

---
## Step 4 — take a batch

```python
start = i * batch_size
            X_batch = X[start:start+batch_size]
            y_batch = y[start:start+batch_size]
```

---
## Step 5 — forward pass

```python
y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
```

---
## Step 6 — backward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
            # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
            loss.backward()
```

---
## Step 7 — update weights

```python
# 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
optimizer.step()
```

---
## Learning Notes / 学习笔记

- **概念**: convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors 是机器学习中的常用技术。  
  *convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
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
# Loop / 05 Loop
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
import tqdm
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("iris.csv", header=None)
X = data.iloc[:, 0:4]
y = data.iloc[:, 4:]

# 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = ohe.transform(y)

# convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

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

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

model = Multiclass()
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training parameters
n_epochs = 200
batch_size = 5
# 获取长度 / Get length
batches_per_epoch = len(X) // batch_size

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 生成整数序列 / Generate integer sequence
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            X_batch = X[start:start+batch_size]
            y_batch = y[start:start+batch_size]
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

➡️ **Next / 下一步**: File 6 of 7

---

### Split

# 06 — Split / 06 Split

**Chapter 10 — File 6 of 7 / 第10章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **read data and apply one-hot encoding**.

本脚本演示 **read data and apply one-hot encoding**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 训练模型 / Train the model

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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder
```

---
## Step 2 — read data and apply one-hot encoding

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("iris.csv", header=None)
X = data.iloc[:, 0:4]
y = data.iloc[:, 4:]
# 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = ohe.transform(y)
```

---
## Step 3 — convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors

```python
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
```

---
## Step 4 — split

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
```

---
## Learning Notes / 学习笔记

- **概念**: read data and apply one-hot encoding 是机器学习中的常用技术。  
  *read data and apply one-hot encoding is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Split / 06 Split
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder

# read data and apply one-hot encoding
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("iris.csv", header=None)
X = data.iloc[:, 0:4]
y = data.iloc[:, 4:]
# 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = ohe.transform(y)

# convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# split
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Complete

# 10 — Complete / 10 Complete

**Chapter 10 — File 7 of 7 / 第10章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **read data and apply one-hot encoding**.

本脚本演示 **read data and apply one-hot encoding**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
┌───────────────────────┐
│  定义模型 Define Model  │
└───────────────────────┘
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
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
from sklearn.preprocessing import OneHotEncoder
```

---
## Step 2 — read data and apply one-hot encoding

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("iris.csv", header=None)
X = data.iloc[:, 0:4]
y = data.iloc[:, 4:]
# 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = ohe.transform(y)
```

---
## Step 3 — convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors

```python
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
```

---
## Step 4 — split

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

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

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x
```

---
## Step 5 — loss metric and optimizer

```python
model = Multiclass()
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---
## Step 6 — prepare model and training parameters

```python
n_epochs = 200
batch_size = 5
# 获取长度 / Get length
batches_per_epoch = len(X_train) // batch_size

best_acc = - np.inf   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []
```

---
## Step 7 — training loop

```python
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []
```

---
## Step 8 — set model in training mode and run through each batch

```python
# 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
model.train()
    # 生成整数序列 / Generate integer sequence
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
```

---
## Step 9 — take a batch

```python
start = i * batch_size
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
## Step 13 — compute and store metrics

```python
acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            # 添加元素到列表末尾 / Append element to list end
            epoch_loss.append(float(loss))
            # 添加元素到列表末尾 / Append element to list end
            epoch_acc.append(float(acc))
            bar.set_postfix(
                loss=float(loss),
                acc=float(acc)
            )
```

---
## Step 14 — set model in evaluation mode and run through the test set

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    y_pred = model(X_test)
    ce = loss_fn(y_pred, y_test)
    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
    ce = float(ce)
    acc = float(acc)
    # 计算均值 / Calculate mean
    train_loss_hist.append(np.mean(epoch_loss))
    # 计算均值 / Calculate mean
    train_acc_hist.append(np.mean(epoch_acc))
    # 添加元素到列表末尾 / Append element to list end
    test_loss_hist.append(ce)
    # 添加元素到列表末尾 / Append element to list end
    test_acc_hist.append(acc)
    if acc > best_acc:
        best_acc = acc
        # 获取模型参数字典 / Get model parameter dictionary
        best_weights = copy.deepcopy(model.state_dict())
    # 打印输出 / Print output
    print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")
```

---
## Step 15 — Restore best model

```python
# 加载模型参数 / Load model parameters
model.load_state_dict(best_weights)
```

---
## Step 16 — Plot the loss and accuracy

```python
# 绘制折线图 / Draw line plot
plt.plot(train_loss_hist, label="train")
# 绘制折线图 / Draw line plot
plt.plot(test_loss_hist, label="test")
# 设置X轴标签 / Set X-axis label
plt.xlabel("epochs")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("cross entropy")
# 显示图例 / Show legend
plt.legend()
# 显示图表 / Display the plot
plt.show()

# 绘制折线图 / Draw line plot
plt.plot(train_acc_hist, label="train")
# 绘制折线图 / Draw line plot
plt.plot(test_acc_hist, label="test")
# 设置X轴标签 / Set X-axis label
plt.xlabel("epochs")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("accuracy")
# 显示图例 / Show legend
plt.legend()
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: read data and apply one-hot encoding 是机器学习中的常用技术。  
  *read data and apply one-hot encoding is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `np.mean` | 计算均值 | Calculate mean |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
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
# Complete / 10 Complete
# Complete Code / 完整代码
# ===============================

import copy

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
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
from sklearn.preprocessing import OneHotEncoder

# read data and apply one-hot encoding
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("iris.csv", header=None)
X = data.iloc[:, 0:4]
y = data.iloc[:, 4:]
# 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = ohe.transform(y)

# convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# split
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

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

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

# loss metric and optimizer
model = Multiclass()
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 200
batch_size = 5
# 获取长度 / Get length
batches_per_epoch = len(X_train) // batch_size

best_acc = - np.inf   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

# training loop
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []
    # set model in training mode and run through each batch
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    # 生成整数序列 / Generate integer sequence
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
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
            # compute and store metrics
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            # 添加元素到列表末尾 / Append element to list end
            epoch_loss.append(float(loss))
            # 添加元素到列表末尾 / Append element to list end
            epoch_acc.append(float(acc))
            bar.set_postfix(
                loss=float(loss),
                acc=float(acc)
            )
    # set model in evaluation mode and run through the test set
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    y_pred = model(X_test)
    ce = loss_fn(y_pred, y_test)
    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
    ce = float(ce)
    acc = float(acc)
    # 计算均值 / Calculate mean
    train_loss_hist.append(np.mean(epoch_loss))
    # 计算均值 / Calculate mean
    train_acc_hist.append(np.mean(epoch_acc))
    # 添加元素到列表末尾 / Append element to list end
    test_loss_hist.append(ce)
    # 添加元素到列表末尾 / Append element to list end
    test_acc_hist.append(acc)
    if acc > best_acc:
        best_acc = acc
        # 获取模型参数字典 / Get model parameter dictionary
        best_weights = copy.deepcopy(model.state_dict())
    # 打印输出 / Print output
    print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")

# Restore best model
# 加载模型参数 / Load model parameters
model.load_state_dict(best_weights)

# Plot the loss and accuracy
# 绘制折线图 / Draw line plot
plt.plot(train_loss_hist, label="train")
# 绘制折线图 / Draw line plot
plt.plot(test_loss_hist, label="test")
# 设置X轴标签 / Set X-axis label
plt.xlabel("epochs")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("cross entropy")
# 显示图例 / Show legend
plt.legend()
# 显示图表 / Display the plot
plt.show()

# 绘制折线图 / Draw line plot
plt.plot(train_acc_hist, label="train")
# 绘制折线图 / Draw line plot
plt.plot(test_acc_hist, label="test")
# 设置X轴标签 / Set X-axis label
plt.xlabel("epochs")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("accuracy")
# 显示图例 / Show legend
plt.legend()
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 10 Summary / 第10章总结

## Theme / 主题: Chapter 10 / Chapter 10

This chapter contains **7 code files** demonstrating chapter 10.

本章包含 **7 个代码文件**，演示Chapter 10。

---
## Evolution / 演化路线

  1. `01_csv.ipynb` — Csv
  2. `02_onehot.ipynb` — Onehot
  3. `03_model.ipynb` — Model
  4. `04_loss.ipynb` — Loss
  5. `05_loop.ipynb` — Loop
  6. `06_split.ipynb` — Split
  7. `10_complete.ipynb` — Complete

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 10) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 10）是机器学习流水线中的基础构建块。

---
