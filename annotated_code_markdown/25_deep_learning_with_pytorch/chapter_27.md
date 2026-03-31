# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 27

---

### Plot

# 01 — Plot / 01 Plot

**Chapter 27 — File 1 of 7 / 第27章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Plot**.

本脚本演示 **01 Plot**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = pd.read_csv('airline-passengers.csv')
# 转换为NumPy数组 / Convert to NumPy array
timeseries = df[["Passengers"]].values.astype('float32')

# 绘制折线图 / Draw line plot
plt.plot(timeseries)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Plot 是机器学习中的常用技术。  
  *Plot is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot / 01 Plot
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = pd.read_csv('airline-passengers.csv')
# 转换为NumPy数组 / Convert to NumPy array
timeseries = df[["Passengers"]].values.astype('float32')

# 绘制折线图 / Draw line plot
plt.plot(timeseries)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Split



---

### Create



---

### Dataset

# 04 — Dataset / 04 Dataset

**Chapter 27 — File 4 of 7 / 第27章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **train-test split for time series**.

本脚本演示 **train-test split for time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
```

---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = pd.read_csv('airline-passengers.csv')
# 转换为NumPy数组 / Convert to NumPy array
timeseries = df[["Passengers"]].values.astype('float32')
```

---
## Step 2 — train-test split for time series

```python
# 获取长度 / Get length
train_size = int(len(timeseries) * 0.67)
# 获取长度 / Get length
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    # 获取长度 / Get length
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        # 添加元素到列表末尾 / Append element to list end
        X.append(feature)
        # 添加元素到列表末尾 / Append element to list end
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

lookback = 1
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X_train.shape, y_train.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X_test.shape, y_test.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: train-test split for time series 是机器学习中的常用技术。  
  *train-test split for time series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dataset / 04 Dataset
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = pd.read_csv('airline-passengers.csv')
# 转换为NumPy数组 / Convert to NumPy array
timeseries = df[["Passengers"]].values.astype('float32')

# train-test split for time series
# 获取长度 / Get length
train_size = int(len(timeseries) * 0.67)
# 获取长度 / Get length
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    # 获取长度 / Get length
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        # 添加元素到列表末尾 / Append element to list end
        X.append(feature)
        # 添加元素到列表末尾 / Append element to list end
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

lookback = 1
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X_train.shape, y_train.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X_test.shape, y_test.shape)
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Training

# 07 — Training / 07 Training

**Chapter 27 — File 5 of 7 / 第27章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **train-test split for time series**.

本脚本演示 **train-test split for time series**。

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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.utils.data as data

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = pd.read_csv('airline-passengers.csv')
# 转换为NumPy数组 / Convert to NumPy array
timeseries = df[["Passengers"]].values.astype('float32')
```

---
## Step 2 — train-test split for time series

```python
# 获取长度 / Get length
train_size = int(len(timeseries) * 0.67)
# 获取长度 / Get length
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    # 获取长度 / Get length
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        # 添加元素到列表末尾 / Append element to list end
        X.append(feature)
        # 添加元素到列表末尾 / Append element to list end
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

lookback = 1
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class AirModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(50, 1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = AirModel()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 2000
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
## Step 3 — Validation

```python
if epoch % 100 != 0:
        continue
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    # 打印输出 / Print output
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
```

---
## Learning Notes / 学习笔记

- **概念**: train-test split for time series 是机器学习中的常用技术。  
  *train-test split for time series is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.LSTM` | 长短期记忆网络，处理序列数据 | Long Short-Term Memory for sequences |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.MSELoss` | 均方误差损失，回归任务常用 | Mean squared error loss for regression |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Training / 07 Training
# Complete Code / 完整代码
# ===============================

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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.utils.data as data

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = pd.read_csv('airline-passengers.csv')
# 转换为NumPy数组 / Convert to NumPy array
timeseries = df[["Passengers"]].values.astype('float32')

# train-test split for time series
# 获取长度 / Get length
train_size = int(len(timeseries) * 0.67)
# 获取长度 / Get length
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    # 获取长度 / Get length
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        # 添加元素到列表末尾 / Append element to list end
        X.append(feature)
        # 添加元素到列表末尾 / Append element to list end
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

lookback = 1
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class AirModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(50, 1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = AirModel()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 2000
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
    if epoch % 100 != 0:
        continue
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    # 打印输出 / Print output
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Predict

# 08 — Predict / 08 Predict

**Chapter 27 — File 6 of 7 / 第27章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **train-test split for time series**.

本脚本演示 **train-test split for time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 可视化结果 / Visualize results

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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.utils.data as data

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = pd.read_csv('airline-passengers.csv')
# 转换为NumPy数组 / Convert to NumPy array
timeseries = df[["Passengers"]].values.astype('float32')
```

---
## Step 2 — train-test split for time series

```python
# 获取长度 / Get length
train_size = int(len(timeseries) * 0.67)
# 获取长度 / Get length
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    # 获取长度 / Get length
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        # 添加元素到列表末尾 / Append element to list end
        X.append(feature)
        # 添加元素到列表末尾 / Append element to list end
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

lookback = 1
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class AirModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(50, 1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = AirModel()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 2000
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
## Step 3 — Validation

```python
if epoch % 100 != 0:
        continue
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    # 打印输出 / Print output
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
```

---
## Step 4 — shift train predictions for plotting

```python
train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train)[:, -1, :]
```

---
## Step 5 — shift test predictions for plotting

```python
test_plot = np.ones_like(timeseries) * np.nan
    # 获取长度 / Get length
    test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
```

---
## Step 6 — plot

```python
# 绘制折线图 / Draw line plot
plt.plot(timeseries, c='b')
# 绘制折线图 / Draw line plot
plt.plot(train_plot, c='r')
# 绘制折线图 / Draw line plot
plt.plot(test_plot, c='g')
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: train-test split for time series 是机器学习中的常用技术。  
  *train-test split for time series is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `nn.LSTM` | 长短期记忆网络，处理序列数据 | Long Short-Term Memory for sequences |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.MSELoss` | 均方误差损失，回归任务常用 | Mean squared error loss for regression |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `np.ones` | 全一数组 | Array filled with ones |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Predict / 08 Predict
# Complete Code / 完整代码
# ===============================

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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.utils.data as data

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = pd.read_csv('airline-passengers.csv')
# 转换为NumPy数组 / Convert to NumPy array
timeseries = df[["Passengers"]].values.astype('float32')

# train-test split for time series
# 获取长度 / Get length
train_size = int(len(timeseries) * 0.67)
# 获取长度 / Get length
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    # 获取长度 / Get length
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        # 添加元素到列表末尾 / Append element to list end
        X.append(feature)
        # 添加元素到列表末尾 / Append element to list end
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

lookback = 1
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class AirModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(50, 1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = AirModel()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 2000
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
    if epoch % 100 != 0:
        continue
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    # 打印输出 / Print output
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train)[:, -1, :]
    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    # 获取长度 / Get length
    test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
# plot
# 绘制折线图 / Draw line plot
plt.plot(timeseries, c='b')
# 绘制折线图 / Draw line plot
plt.plot(train_plot, c='r')
# 绘制折线图 / Draw line plot
plt.plot(test_plot, c='g')
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### 4Step

# 09 — 4Step / 09 4Step

**Chapter 27 — File 7 of 7 / 第27章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **train-test split for time series**.

本脚本演示 **train-test split for time series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 可视化结果 / Visualize results

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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.utils.data as data

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = pd.read_csv('airline-passengers.csv')
# 转换为NumPy数组 / Convert to NumPy array
timeseries = df[["Passengers"]].values.astype('float32')
```

---
## Step 2 — train-test split for time series

```python
# 获取长度 / Get length
train_size = int(len(timeseries) * 0.67)
# 获取长度 / Get length
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    # 获取长度 / Get length
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        # 添加元素到列表末尾 / Append element to list end
        X.append(feature)
        # 添加元素到列表末尾 / Append element to list end
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

lookback = 4
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class AirModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(50, 1)
    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = AirModel()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 2000
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
## Step 3 — Validation

```python
if epoch % 100 != 0:
        continue
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    # 打印输出 / Print output
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
```

---
## Step 4 — shift train predictions for plotting

```python
train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train)[:, -1, :]
```

---
## Step 5 — shift test predictions for plotting

```python
test_plot = np.ones_like(timeseries) * np.nan
    # 获取长度 / Get length
    test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
```

---
## Step 6 — plot

```python
# 绘制折线图 / Draw line plot
plt.plot(timeseries)
# 绘制折线图 / Draw line plot
plt.plot(train_plot, c='r')
# 绘制折线图 / Draw line plot
plt.plot(test_plot, c='g')
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: train-test split for time series 是机器学习中的常用技术。  
  *train-test split for time series is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `nn.LSTM` | 长短期记忆网络，处理序列数据 | Long Short-Term Memory for sequences |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.MSELoss` | 均方误差损失，回归任务常用 | Mean squared error loss for regression |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `np.ones` | 全一数组 | Array filled with ones |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# 4Step / 09 4Step
# Complete Code / 完整代码
# ===============================

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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.utils.data as data

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = pd.read_csv('airline-passengers.csv')
# 转换为NumPy数组 / Convert to NumPy array
timeseries = df[["Passengers"]].values.astype('float32')

# train-test split for time series
# 获取长度 / Get length
train_size = int(len(timeseries) * 0.67)
# 获取长度 / Get length
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    # 获取长度 / Get length
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        # 添加元素到列表末尾 / Append element to list end
        X.append(feature)
        # 添加元素到列表末尾 / Append element to list end
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

lookback = 4
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class AirModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # LSTM层：处理序列，能记住长期信息 / LSTM: process sequences with long-term memory
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear = nn.Linear(50, 1)
    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = AirModel()
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
loss_fn = nn.MSELoss()
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 2000
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
    if epoch % 100 != 0:
        continue
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    # 打印输出 / Print output
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train)[:, -1, :]
    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    # 获取长度 / Get length
    test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
# plot
# 绘制折线图 / Draw line plot
plt.plot(timeseries)
# 绘制折线图 / Draw line plot
plt.plot(train_plot, c='r')
# 绘制折线图 / Draw line plot
plt.plot(test_plot, c='g')
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 27 Summary / 第27章总结

## Theme / 主题: Chapter 27 / Chapter 27

This chapter contains **7 code files** demonstrating chapter 27.

本章包含 **7 个代码文件**，演示Chapter 27。

---
## Evolution / 演化路线

  1. `01_plot.ipynb` — Plot
  2. `02_split.ipynb` — Split
  3. `03_create.ipynb` — Create
  4. `04_dataset.ipynb` — Dataset
  5. `07_training.ipynb` — Training
  6. `08_predict.ipynb` — Predict
  7. `09_4step.ipynb` — 4Step

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 27) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 27）是机器学习流水线中的基础构建块。

---
