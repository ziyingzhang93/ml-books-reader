# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 14

---

### Makedata

# 01 — Makedata / 01 Makedata

**Chapter 14 — File 1 of 11 / 第14章 — 第1个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Make data: Two circles on x-y plane as a classification problem**.

本脚本演示 **Make data: Two circles on x-y plane as a classification problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — Step 1

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
```

---
## Step 2 — Make data: Two circles on x-y plane as a classification problem

```python
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

# 创建画布 / Create figure canvas
plt.figure(figsize=(8,6))
# 绘制散点图 / Draw scatter plot
plt.scatter(X[:,0], X[:,1], c=y)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Make data: Two circles on x-y plane as a classification problem 是机器学习中的常用技术。  
  *Make data: Two circles on x-y plane as a classification problem is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Makedata / 01 Makedata
# Complete Code / 完整代码
# ===============================

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

# Make data: Two circles on x-y plane as a classification problem
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

# 创建画布 / Create figure canvas
plt.figure(figsize=(8,6))
# 绘制散点图 / Draw scatter plot
plt.scatter(X[:,0], X[:,1], c=y)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 11

---

### Model

# 02 — Model / 02 Model

**Chapter 14 — File 2 of 11 / 第14章 — 第2个文件（共11个）**

---

## Summary / 总结

This script demonstrates **train model with optimizer**.

本脚本演示 **train model with optimizer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Model(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, activation=nn.ReLU):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer0 = nn.Linear(2,5)
        self.act0 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(5,5)
        self.act1 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(5,5)
        self.act2 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer3 = nn.Linear(5,5)
        self.act3 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer4 = nn.Linear(5,1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.act4 = nn.Sigmoid()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act0(self.layer0(x))
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act4(self.layer4(x))
        return x

def train_loop(model, X, y, n_epochs=300, batch_size=32):
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    loss_fn = nn.BCELoss()
    # Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 获取长度 / Get length
    batch_start = torch.arange(0, len(X), batch_size)

    bce_hist = []
    acc_hist = []

    # 生成整数序列 / Generate integer sequence
    for epoch in range(n_epochs):
```

---
## Step 2 — train model with optimizer

```python
# 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
model.train()
        for start in batch_start:
            X_batch = X[start:start+batch_size]
            y_batch = y[start:start+batch_size]
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
## Step 3 — evaluate BCE and accuracy at end of each epoch

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
        # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
        with torch.no_grad():
            y_pred = model(X)
            bce = float(loss_fn(y_pred, y))
            acc = float((y_pred.round() == y).float().mean())
        # 添加元素到列表末尾 / Append element to list end
        bce_hist.append(bce)
        # 添加元素到列表末尾 / Append element to list end
        acc_hist.append(acc)
```

---
## Step 4 — print metrics every 10 epochs

```python
if (epoch+1) % 10 == 0:
            # 打印输出 / Print output
            print("Before epoch %d: BCE=%.4f, Accuracy=%.2f%%" % (epoch+1, bce, acc*100))
    return bce_hist, acc_hist
```

---
## Learning Notes / 学习笔记

- **概念**: train model with optimizer 是机器学习中的常用技术。  
  *train model with optimizer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model / 02 Model
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Model(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, activation=nn.ReLU):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer0 = nn.Linear(2,5)
        self.act0 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(5,5)
        self.act1 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(5,5)
        self.act2 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer3 = nn.Linear(5,5)
        self.act3 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer4 = nn.Linear(5,1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.act4 = nn.Sigmoid()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act0(self.layer0(x))
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act4(self.layer4(x))
        return x

def train_loop(model, X, y, n_epochs=300, batch_size=32):
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    loss_fn = nn.BCELoss()
    # Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 获取长度 / Get length
    batch_start = torch.arange(0, len(X), batch_size)

    bce_hist = []
    acc_hist = []

    # 生成整数序列 / Generate integer sequence
    for epoch in range(n_epochs):
        # train model with optimizer
        # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
        model.train()
        for start in batch_start:
            X_batch = X[start:start+batch_size]
            y_batch = y[start:start+batch_size]
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
            optimizer.zero_grad()
            # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
            loss.backward()
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            optimizer.step()
        # evaluate BCE and accuracy at end of each epoch
        # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
        model.eval()
        # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
        with torch.no_grad():
            y_pred = model(X)
            bce = float(loss_fn(y_pred, y))
            acc = float((y_pred.round() == y).float().mean())
        # 添加元素到列表末尾 / Append element to list end
        bce_hist.append(bce)
        # 添加元素到列表末尾 / Append element to list end
        acc_hist.append(acc)
        # print metrics every 10 epochs
        if (epoch+1) % 10 == 0:
            # 打印输出 / Print output
            print("Before epoch %d: BCE=%.4f, Accuracy=%.2f%%" % (epoch+1, bce, acc*100))
    return bce_hist, acc_hist
```

---

➡️ **Next / 下一步**: File 3 of 11

---

### Train



---

### Functions

# 04 — Functions / 04 Functions

**Chapter 14 — File 4 of 11 / 第14章 — 第4个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Functions**.

本脚本演示 **04 Functions**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — Step 1

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

x = torch.linspace(-4, 4, 200)
# ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
relu = nn.ReLU()(x)
# Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
tanh = nn.Tanh()(x)
# Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
sigmoid = nn.Sigmoid()(x)

# 绘制折线图 / Draw line plot
plt.plot(x, sigmoid, label="sigmoid")
# 绘制折线图 / Draw line plot
plt.plot(x, tanh, label="tanh")
# 绘制折线图 / Draw line plot
plt.plot(x, relu, label="ReLU")
plt.ylim(-1.5, 2)
# 显示图例 / Show legend
plt.legend()
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Functions 是机器学习中的常用技术。  
  *Functions is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Functions / 04 Functions
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

x = torch.linspace(-4, 4, 200)
# ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
relu = nn.ReLU()(x)
# Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
tanh = nn.Tanh()(x)
# Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
sigmoid = nn.Sigmoid()(x)

# 绘制折线图 / Draw line plot
plt.plot(x, sigmoid, label="sigmoid")
# 绘制折线图 / Draw line plot
plt.plot(x, tanh, label="tanh")
# 绘制折线图 / Draw line plot
plt.plot(x, relu, label="ReLU")
plt.ylim(-1.5, 2)
# 显示图例 / Show legend
plt.legend()
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 5 of 11

---

### Relu



---

### Sigmoid



---

### Tanh



---

### Plot

# 09 — Plot / 09 Plot

**Chapter 14 — File 8 of 11 / 第14章 — 第8个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Plot**.

本脚本演示 **09 Plot**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

x = torch.linspace(-8,8,200)
# ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
relu = nn.ReLU()(x)
relu6 = nn.ReLU6()(x)
leaky = nn.LeakyReLU()(x)

# 绘制折线图 / Draw line plot
plt.plot(x, relu, c="purple", lw=2, ls=":", label="ReLU")
# 绘制折线图 / Draw line plot
plt.plot(x, relu6, c="orange", lw=2, ls="--", alpha=0.5, label="ReLU6")
# 绘制折线图 / Draw line plot
plt.plot(x, leaky, c="darkblue", lw=2, alpha=0.5, label="LeakyReLU")
# 显示图例 / Show legend
plt.legend()
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
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot / 09 Plot
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

x = torch.linspace(-8,8,200)
# ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
relu = nn.ReLU()(x)
relu6 = nn.ReLU6()(x)
leaky = nn.LeakyReLU()(x)

# 绘制折线图 / Draw line plot
plt.plot(x, relu, c="purple", lw=2, ls=":", label="ReLU")
# 绘制折线图 / Draw line plot
plt.plot(x, relu6, c="orange", lw=2, ls="--", alpha=0.5, label="ReLU6")
# 绘制折线图 / Draw line plot
plt.plot(x, leaky, c="darkblue", lw=2, alpha=0.5, label="LeakyReLU")
# 显示图例 / Show legend
plt.legend()
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 9 of 11

---

### Relu6

# 10 — Relu6 / 10 Relu6

**Chapter 14 — File 9 of 11 / 第14章 — 第9个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Make data: Two circles on x-y plane as a classification problem**.

本脚本演示 **Make data: Two circles on x-y plane as a classification problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
```

---
## Step 2 — Make data: Two circles on x-y plane as a classification problem

```python
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Model(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, activation=nn.ReLU):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer0 = nn.Linear(2,5)
        self.act0 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(5,5)
        self.act1 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(5,5)
        self.act2 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer3 = nn.Linear(5,5)
        self.act3 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer4 = nn.Linear(5,1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.act4 = nn.Sigmoid()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act0(self.layer0(x))
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act4(self.layer4(x))
        return x

def train_loop(model, X, y, n_epochs=300, batch_size=32):
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    loss_fn = nn.BCELoss()
    # Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 获取长度 / Get length
    batch_start = torch.arange(0, len(X), batch_size)

    bce_hist = []
    acc_hist = []
    grad_hist = [[],[],[],[],[]]

    # 生成整数序列 / Generate integer sequence
    for epoch in range(n_epochs):
```

---
## Step 3 — train model with optimizer

```python
# 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
model.train()
        layer_grad = [[],[],[],[],[]]
        for start in batch_start:
            X_batch = X[start:start+batch_size]
            y_batch = y[start:start+batch_size]
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
## Step 4 — collect mean absolute value of gradients

```python
layers = [model.layer0, model.layer1, model.layer2, model.layer3,
                      model.layer4]
            # 同时获取索引和值 / Get both index and value
            for n,layer in enumerate(layers):
                mean_grad = float(layer.weight.grad.abs().mean())
                # 添加元素到列表末尾 / Append element to list end
                layer_grad[n].append(mean_grad)
```

---
## Step 5 — evaluate BCE and accuracy at end of each epoch

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
        # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
        with torch.no_grad():
            y_pred = model(X)
            bce = float(loss_fn(y_pred, y))
            acc = float((y_pred.round() == y).float().mean())
        # 添加元素到列表末尾 / Append element to list end
        bce_hist.append(bce)
        # 添加元素到列表末尾 / Append element to list end
        acc_hist.append(acc)
        # 同时获取索引和值 / Get both index and value
        for n, grads in enumerate(layer_grad):
            # 获取长度 / Get length
            grad_hist[n].append(sum(grads)/len(grads))
```

---
## Step 6 — print metrics every 10 epochs

```python
if epoch % 10 == 9:
            # 打印输出 / Print output
            print("Epoch %d: BCE=%.4f, Accuracy=%.2f%%" % (epoch, bce, acc*100))
    return bce_hist, acc_hist, layer_grad

activation = nn.ReLU6
model = Model(activation=activation)
bce_hist, acc_hist, grad_hist = train_loop(model, X, y)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(bce_hist, label="BCE")
ax[0].plot(acc_hist, label="Accuracy")
ax[0].set_xlabel("Epochs")
ax[0].set_ylim(0, 1)
# 同时获取索引和值 / Get both index and value
for n, grads in enumerate(grad_hist):
    ax[1].plot(grads, label="layer"+str(n))
ax[1].set_xlabel("Epochs")
fig.suptitle(str(activation))
ax[0].legend()
ax[1].legend()
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Make data: Two circles on x-y plane as a classification problem 是机器学习中的常用技术。  
  *Make data: Two circles on x-y plane as a classification problem is a common technique in machine learning.*

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
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Relu6 / 10 Relu6
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles

# Make data: Two circles on x-y plane as a classification problem
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Model(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, activation=nn.ReLU):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer0 = nn.Linear(2,5)
        self.act0 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(5,5)
        self.act1 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(5,5)
        self.act2 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer3 = nn.Linear(5,5)
        self.act3 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer4 = nn.Linear(5,1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.act4 = nn.Sigmoid()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act0(self.layer0(x))
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act4(self.layer4(x))
        return x

def train_loop(model, X, y, n_epochs=300, batch_size=32):
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    loss_fn = nn.BCELoss()
    # Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 获取长度 / Get length
    batch_start = torch.arange(0, len(X), batch_size)

    bce_hist = []
    acc_hist = []
    grad_hist = [[],[],[],[],[]]

    # 生成整数序列 / Generate integer sequence
    for epoch in range(n_epochs):
        # train model with optimizer
        # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
        model.train()
        layer_grad = [[],[],[],[],[]]
        for start in batch_start:
            X_batch = X[start:start+batch_size]
            y_batch = y[start:start+batch_size]
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
            optimizer.zero_grad()
            # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
            loss.backward()
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            optimizer.step()
            # collect mean absolute value of gradients
            layers = [model.layer0, model.layer1, model.layer2, model.layer3,
                      model.layer4]
            # 同时获取索引和值 / Get both index and value
            for n,layer in enumerate(layers):
                mean_grad = float(layer.weight.grad.abs().mean())
                # 添加元素到列表末尾 / Append element to list end
                layer_grad[n].append(mean_grad)
        # evaluate BCE and accuracy at end of each epoch
        # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
        model.eval()
        # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
        with torch.no_grad():
            y_pred = model(X)
            bce = float(loss_fn(y_pred, y))
            acc = float((y_pred.round() == y).float().mean())
        # 添加元素到列表末尾 / Append element to list end
        bce_hist.append(bce)
        # 添加元素到列表末尾 / Append element to list end
        acc_hist.append(acc)
        # 同时获取索引和值 / Get both index and value
        for n, grads in enumerate(layer_grad):
            # 获取长度 / Get length
            grad_hist[n].append(sum(grads)/len(grads))
        # print metrics every 10 epochs
        if epoch % 10 == 9:
            # 打印输出 / Print output
            print("Epoch %d: BCE=%.4f, Accuracy=%.2f%%" % (epoch, bce, acc*100))
    return bce_hist, acc_hist, layer_grad

activation = nn.ReLU6
model = Model(activation=activation)
bce_hist, acc_hist, grad_hist = train_loop(model, X, y)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(bce_hist, label="BCE")
ax[0].plot(acc_hist, label="Accuracy")
ax[0].set_xlabel("Epochs")
ax[0].set_ylim(0, 1)
# 同时获取索引和值 / Get both index and value
for n, grads in enumerate(grad_hist):
    ax[1].plot(grads, label="layer"+str(n))
ax[1].set_xlabel("Epochs")
fig.suptitle(str(activation))
ax[0].legend()
ax[1].legend()
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 10 of 11

---

### Leaky



---

### Activation

# 12 — Activation / 12 Activation

**Chapter 14 — File 11 of 11 / 第14章 — 第11个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Make data: Two circles on x-y plane as a classification problem**.

本脚本演示 **Make data: Two circles on x-y plane as a classification problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
```

---
## Step 2 — Make data: Two circles on x-y plane as a classification problem

```python
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
```

---
## Step 3 — Binary classification model

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Model(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, activation=nn.ReLU):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer0 = nn.Linear(2,5)
        self.act0 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(5,5)
        self.act1 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(5,5)
        self.act2 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer3 = nn.Linear(5,5)
        self.act3 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer4 = nn.Linear(5,1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.act4 = nn.Sigmoid()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act0(self.layer0(x))
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act4(self.layer4(x))
        return x
```

---
## Step 4 — train the model and produce history

```python
def train_loop(model, X, y, n_epochs=300, batch_size=32):
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    loss_fn = nn.BCELoss()
    # Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 获取长度 / Get length
    batch_start = torch.arange(0, len(X), batch_size)

    bce_hist = []
    acc_hist = []
    grad_hist = [[],[],[],[],[]]

    # 生成整数序列 / Generate integer sequence
    for epoch in range(n_epochs):
```

---
## Step 5 — train model with optimizer

```python
# 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
model.train()
        layer_grad = [[],[],[],[],[]]
        for start in batch_start:
            X_batch = X[start:start+batch_size]
            y_batch = y[start:start+batch_size]
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
## Step 6 — collect mean absolute value of gradients

```python
layers = [model.layer0, model.layer1, model.layer2, model.layer3,
                      model.layer4]
            # 同时获取索引和值 / Get both index and value
            for n,layer in enumerate(layers):
                mean_grad = float(layer.weight.grad.abs().mean())
                # 添加元素到列表末尾 / Append element to list end
                layer_grad[n].append(mean_grad)
```

---
## Step 7 — evaluate BCE and accuracy at end of each epoch

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
        # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
        with torch.no_grad():
            y_pred = model(X)
            bce = float(loss_fn(y_pred, y))
            acc = float((y_pred.round() == y).float().mean())
        # 添加元素到列表末尾 / Append element to list end
        bce_hist.append(bce)
        # 添加元素到列表末尾 / Append element to list end
        acc_hist.append(acc)
        # 同时获取索引和值 / Get both index and value
        for n, grads in enumerate(layer_grad):
            # 获取长度 / Get length
            grad_hist[n].append(sum(grads)/len(grads))
```

---
## Step 8 — print metrics every 10 epochs

```python
if epoch % 10 == 9:
            # 打印输出 / Print output
            print("Epoch %d: BCE=%.4f, Accuracy=%.2f%%" % (epoch, bce, acc*100))
    return bce_hist, acc_hist, layer_grad
```

---
## Step 9 — pick different activation functions and compare the result visually

```python
for activation in [nn.Sigmoid, nn.Tanh, nn.ReLU, nn.ReLU6, nn.LeakyReLU]:
    model = Model(activation=activation)
    bce_hist, acc_hist, grad_hist = train_loop(model, X, y)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(bce_hist, label="BCE")
    ax[0].plot(acc_hist, label="Accuracy")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylim(0, 1)
    # 同时获取索引和值 / Get both index and value
    for n, grads in enumerate(grad_hist):
        ax[1].plot(grads, label="layer"+str(n))
    ax[1].set_xlabel("Epochs")
    fig.suptitle(str(activation))
    ax[0].legend()
    ax[1].legend()
    # 显示图表 / Display the plot
    plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Make data: Two circles on x-y plane as a classification problem 是机器学习中的常用技术。  
  *Make data: Two circles on x-y plane as a classification problem is a common technique in machine learning.*

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
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Activation / 12 Activation
# Complete Code / 完整代码
# ===============================

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim

# Make data: Two circles on x-y plane as a classification problem
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

# Binary classification model
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Model(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, activation=nn.ReLU):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer0 = nn.Linear(2,5)
        self.act0 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(5,5)
        self.act1 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(5,5)
        self.act2 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer3 = nn.Linear(5,5)
        self.act3 = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer4 = nn.Linear(5,1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.act4 = nn.Sigmoid()

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act0(self.layer0(x))
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act4(self.layer4(x))
        return x

# train the model and produce history
def train_loop(model, X, y, n_epochs=300, batch_size=32):
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    loss_fn = nn.BCELoss()
    # Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 获取长度 / Get length
    batch_start = torch.arange(0, len(X), batch_size)

    bce_hist = []
    acc_hist = []
    grad_hist = [[],[],[],[],[]]

    # 生成整数序列 / Generate integer sequence
    for epoch in range(n_epochs):
        # train model with optimizer
        # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
        model.train()
        layer_grad = [[],[],[],[],[]]
        for start in batch_start:
            X_batch = X[start:start+batch_size]
            y_batch = y[start:start+batch_size]
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
            optimizer.zero_grad()
            # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
            loss.backward()
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            optimizer.step()
            # collect mean absolute value of gradients
            layers = [model.layer0, model.layer1, model.layer2, model.layer3,
                      model.layer4]
            # 同时获取索引和值 / Get both index and value
            for n,layer in enumerate(layers):
                mean_grad = float(layer.weight.grad.abs().mean())
                # 添加元素到列表末尾 / Append element to list end
                layer_grad[n].append(mean_grad)
        # evaluate BCE and accuracy at end of each epoch
        # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
        model.eval()
        # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
        with torch.no_grad():
            y_pred = model(X)
            bce = float(loss_fn(y_pred, y))
            acc = float((y_pred.round() == y).float().mean())
        # 添加元素到列表末尾 / Append element to list end
        bce_hist.append(bce)
        # 添加元素到列表末尾 / Append element to list end
        acc_hist.append(acc)
        # 同时获取索引和值 / Get both index and value
        for n, grads in enumerate(layer_grad):
            # 获取长度 / Get length
            grad_hist[n].append(sum(grads)/len(grads))
        # print metrics every 10 epochs
        if epoch % 10 == 9:
            # 打印输出 / Print output
            print("Epoch %d: BCE=%.4f, Accuracy=%.2f%%" % (epoch, bce, acc*100))
    return bce_hist, acc_hist, layer_grad

# pick different activation functions and compare the result visually
for activation in [nn.Sigmoid, nn.Tanh, nn.ReLU, nn.ReLU6, nn.LeakyReLU]:
    model = Model(activation=activation)
    bce_hist, acc_hist, grad_hist = train_loop(model, X, y)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(bce_hist, label="BCE")
    ax[0].plot(acc_hist, label="Accuracy")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylim(0, 1)
    # 同时获取索引和值 / Get both index and value
    for n, grads in enumerate(grad_hist):
        ax[1].plot(grads, label="layer"+str(n))
    ax[1].set_xlabel("Epochs")
    fig.suptitle(str(activation))
    ax[0].legend()
    ax[1].legend()
    # 显示图表 / Display the plot
    plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 14 Summary / 第14章总结

## Theme / 主题: Chapter 14 / Chapter 14

This chapter contains **11 code files** demonstrating chapter 14.

本章包含 **11 个代码文件**，演示Chapter 14。

---
## Evolution / 演化路线

  1. `01_makedata.ipynb` — Makedata
  2. `02_model.ipynb` — Model
  3. `03_train.ipynb` — Train
  4. `04_functions.ipynb` — Functions
  5. `06_relu.ipynb` — Relu
  6. `07_sigmoid.ipynb` — Sigmoid
  7. `08_tanh.ipynb` — Tanh
  8. `09_plot.ipynb` — Plot
  9. `10_relu6.ipynb` — Relu6
  10. `11_leaky.ipynb` — Leaky
  11. `12_activation.ipynb` — Activation

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 14) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 14）是机器学习流水线中的基础构建块。

---
