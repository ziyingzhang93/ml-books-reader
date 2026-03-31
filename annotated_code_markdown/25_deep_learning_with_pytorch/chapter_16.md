# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 16

---

### Baseline



---

### Dropout

# 02 — Dropout / 随机失活

**Chapter 16 — File 2 of 3 / 第16章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Read data**.

本脚本演示 **Read data**。

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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
```

---
## Step 2 — Read data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]
```

---
## Step 3 — Label encode the target from string to integer

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)
```

---
## Step 4 — Convert to 2D PyTorch tensors

```python
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 5 — Define PyTorch model, with dropout at input

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class SonarModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act1 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(60, 30)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act2 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(30, 1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.sigmoid = nn.Sigmoid()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.dropout(x)
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x
```

---
## Step 6 — Helper function to train the model and return the validation result

```python
def model_train(model, X_train, y_train, X_val, y_val,
                n_epochs=300, batch_size=16):
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    loss_fn = nn.BCELoss()
    # SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    # 获取长度 / Get length
    batch_start = torch.arange(0, len(X_train), batch_size)

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
```

---
## Step 7 — evaluate accuracy after training

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    y_pred = model(X_val)
    acc = (y_pred.round() == y_val).float().mean()
    acc = float(acc)
    return acc
```

---
## Step 8 — run 10-fold cross validation

```python
kfold = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = []
for train, test in kfold.split(X, y):
```

---
## Step 9 — create model, train, and get accuracy

```python
model = SonarModel()
    acc = model_train(model, X[train], y[train], X[test], y[test])
    # 打印输出 / Print output
    print("Accuracy: %.2f" % acc)
    # 添加元素到列表末尾 / Append element to list end
    accuracies.append(acc)
```

---
## Step 10 — evaluate the model

```python
# 计算均值 / Calculate mean
mean = np.mean(accuracies)
# 计算标准差 / Calculate standard deviation
std = np.std(accuracies)
# 打印输出 / Print output
print("Baseline: %.2f%% (+/- %.2f%%)" % (mean*100, std*100))
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
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Dropout` | 随机丢弃神经元防止过拟合 | Randomly drop neurons to prevent overfitting |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `np.mean` | 计算均值 | Calculate mean |
| `np.std` | 计算标准差 | Calculate standard deviation |
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
# Dropout / 随机失活
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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold

# Read data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Label encode the target from string to integer
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Define PyTorch model, with dropout at input
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class SonarModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act1 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(60, 30)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act2 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(30, 1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.sigmoid = nn.Sigmoid()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.dropout(x)
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

# Helper function to train the model and return the validation result
def model_train(model, X_train, y_train, X_val, y_val,
                n_epochs=300, batch_size=16):
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    loss_fn = nn.BCELoss()
    # SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    # 获取长度 / Get length
    batch_start = torch.arange(0, len(X_train), batch_size)

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

    # evaluate accuracy after training
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    y_pred = model(X_val)
    acc = (y_pred.round() == y_val).float().mean()
    acc = float(acc)
    return acc

# run 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = []
for train, test in kfold.split(X, y):
    # create model, train, and get accuracy
    model = SonarModel()
    acc = model_train(model, X[train], y[train], X[test], y[test])
    # 打印输出 / Print output
    print("Accuracy: %.2f" % acc)
    # 添加元素到列表末尾 / Append element to list end
    accuracies.append(acc)

# evaluate the model
# 计算均值 / Calculate mean
mean = np.mean(accuracies)
# 计算标准差 / Calculate standard deviation
std = np.std(accuracies)
# 打印输出 / Print output
print("Baseline: %.2f%% (+/- %.2f%%)" % (mean*100, std*100))
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Dropout

# 03 — Dropout / 随机失活

**Chapter 16 — File 3 of 3 / 第16章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Read data**.

本脚本演示 **Read data**。

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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
```

---
## Step 2 — Read data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]
```

---
## Step 3 — Label encode the target from string to integer

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)
```

---
## Step 4 — Convert to 2D PyTorch tensors

```python
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 5 — Define PyTorch model, with dropout at hidden layers

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class SonarModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act1 = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout1 = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(60, 30)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act2 = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout2 = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(30, 1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.sigmoid = nn.Sigmoid()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.output(x))
        return x
```

---
## Step 6 — Helper function to train the model and return the validation result

```python
def model_train(model, X_train, y_train, X_val, y_val,
                n_epochs=300, batch_size=16):
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    loss_fn = nn.BCELoss()
    # SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    # 获取长度 / Get length
    batch_start = torch.arange(0, len(X_train), batch_size)

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
```

---
## Step 7 — evaluate accuracy after training

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    y_pred = model(X_val)
    acc = (y_pred.round() == y_val).float().mean()
    acc = float(acc)
    return acc
```

---
## Step 8 — run 10-fold cross validation

```python
kfold = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = []
for train, test in kfold.split(X, y):
```

---
## Step 9 — create model, train, and get accuracy

```python
model = SonarModel()
    acc = model_train(model, X[train], y[train], X[test], y[test])
    # 打印输出 / Print output
    print("Accuracy: %.2f" % acc)
    # 添加元素到列表末尾 / Append element to list end
    accuracies.append(acc)
```

---
## Step 10 — evaluate the model

```python
# 计算均值 / Calculate mean
mean = np.mean(accuracies)
# 计算标准差 / Calculate standard deviation
std = np.std(accuracies)
# 打印输出 / Print output
print("Baseline: %.2f%% (+/- %.2f%%)" % (mean*100, std*100))
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
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Dropout` | 随机丢弃神经元防止过拟合 | Randomly drop neurons to prevent overfitting |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `np.mean` | 计算均值 | Calculate mean |
| `np.std` | 计算标准差 | Calculate standard deviation |
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
# Dropout / 随机失活
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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold

# Read data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Label encode the target from string to integer
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Define PyTorch model, with dropout at hidden layers
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class SonarModel(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act1 = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout1 = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(60, 30)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act2 = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout2 = nn.Dropout(0.2)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(30, 1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.sigmoid = nn.Sigmoid()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.output(x))
        return x

# Helper function to train the model and return the validation result
def model_train(model, X_train, y_train, X_val, y_val,
                n_epochs=300, batch_size=16):
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    loss_fn = nn.BCELoss()
    # SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    # 获取长度 / Get length
    batch_start = torch.arange(0, len(X_train), batch_size)

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

    # evaluate accuracy after training
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    y_pred = model(X_val)
    acc = (y_pred.round() == y_val).float().mean()
    acc = float(acc)
    return acc

# run 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = []
for train, test in kfold.split(X, y):
    # create model, train, and get accuracy
    model = SonarModel()
    acc = model_train(model, X[train], y[train], X[test], y[test])
    # 打印输出 / Print output
    print("Accuracy: %.2f" % acc)
    # 添加元素到列表末尾 / Append element to list end
    accuracies.append(acc)

# evaluate the model
# 计算均值 / Calculate mean
mean = np.mean(accuracies)
# 计算标准差 / Calculate standard deviation
std = np.std(accuracies)
# 打印输出 / Print output
print("Baseline: %.2f%% (+/- %.2f%%)" % (mean*100, std*100))
```

---

### Chapter Summary / 章节总结

# Chapter 16 Summary / 第16章总结

## Theme / 主题: Chapter 16 / Chapter 16

This chapter contains **3 code files** demonstrating chapter 16.

本章包含 **3 个代码文件**，演示Chapter 16。

---
## Evolution / 演化路线

  1. `01_baseline.ipynb` — Baseline
  2. `02_dropout.ipynb` — Dropout
  3. `03_dropout.ipynb` — Dropout

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 16) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 16）是机器学习流水线中的基础构建块。

---
