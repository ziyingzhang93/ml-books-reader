# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 26

---

### Display



---

### Shape



---

### Baseline

# 06 — Baseline / 06 Baseline

**Chapter 26 — File 3 of 5 / 第26章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Load MNIST data**.

本脚本演示 **Load MNIST data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
import torchvision
```

---
## Step 2 — Load MNIST data

```python
train = torchvision.datasets.MNIST('data', train=True, download=True)
test = torchvision.datasets.MNIST('data', train=False, download=True)
```

---
## Step 3 — each sample becomes a vector of values 0-1

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X_train = train.data.reshape(-1, 784).float() / 255.0
y_train = train.targets
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X_test = test.data.reshape(-1, 784).float() / 255.0
y_test = test.targets

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Baseline(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(784, 784)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act1 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(784, 10)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.layer2(x)
        return x

model = Baseline()

# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.01)
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)),
                                     shuffle=True, batch_size=100)

n_epochs = 10
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
## Step 4 — Validation

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    y_pred = model(X_test)
    acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
    # 打印输出 / Print output
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
```

---
## Learning Notes / 学习笔记

- **概念**: Load MNIST data 是机器学习中的常用技术。  
  *Load MNIST data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataLoader` | 数据加载器，批量读取数据 | Loads data in batches for training |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
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
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Baseline / 06 Baseline
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
import torchvision

# Load MNIST data
train = torchvision.datasets.MNIST('data', train=True, download=True)
test = torchvision.datasets.MNIST('data', train=False, download=True)

# each sample becomes a vector of values 0-1
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X_train = train.data.reshape(-1, 784).float() / 255.0
y_train = train.targets
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X_test = test.data.reshape(-1, 784).float() / 255.0
y_test = test.targets

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Baseline(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(784, 784)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act1 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(784, 10)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.layer2(x)
        return x

model = Baseline()

# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.01)
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)),
                                     shuffle=True, batch_size=100)

n_epochs = 10
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
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    y_pred = model(X_test)
    acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
    # 打印输出 / Print output
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Cnn

# 09 — Cnn / 卷积神经网络

**Chapter 26 — File 4 of 5 / 第26章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Load MNIST data**.

本脚本演示 **Load MNIST data**。

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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
import torchvision
```

---
## Step 2 — Load MNIST data

```python
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (128,)),
])
train = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST('data', train=False, download=True, transform=transform)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=100)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
testloader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=100)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CNN(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.relu1 = nn.ReLU()
        # 最大池化：缩小特征图，保留最大值 / MaxPool: downsample, keep max values
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 展平层：多维→一维 / Flatten: multi-dim → 1D
        self.flat = nn.Flatten()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc = nn.Linear(27*27*10, 128)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.relu2 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(128, 10)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.relu1(self.conv(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu2(self.fc(self.flat(x)))
        x = self.output(x)
        return x

model = CNN()

# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()

n_epochs = 10
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    for X_batch, y_batch in trainloader:
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
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    acc = 0
    count = 0
    for X_batch, y_batch in testloader:
        y_pred = model(X_batch)
        acc += (torch.argmax(y_pred, 1) == y_batch).float().sum()
        # 获取长度 / Get length
        count += len(y_batch)
    acc = acc / count
    # 打印输出 / Print output
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
```

---
## Learning Notes / 学习笔记

- **概念**: Load MNIST data 是机器学习中的常用技术。  
  *Load MNIST data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `DataLoader` | 数据加载器，批量读取数据 | Loads data in batches for training |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Conv2d` | 二维卷积层，提取图像特征 | 2D convolution layer for image features |
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `nn.Dropout` | 随机丢弃神经元防止过拟合 | Randomly drop neurons to prevent overfitting |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cnn / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
import torchvision

# Load MNIST data
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (128,)),
])
train = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST('data', train=False, download=True, transform=transform)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=100)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
testloader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=100)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CNN(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.relu1 = nn.ReLU()
        # 最大池化：缩小特征图，保留最大值 / MaxPool: downsample, keep max values
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        # 展平层：多维→一维 / Flatten: multi-dim → 1D
        self.flat = nn.Flatten()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc = nn.Linear(27*27*10, 128)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.relu2 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(128, 10)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.relu1(self.conv(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu2(self.fc(self.flat(x)))
        x = self.output(x)
        return x

model = CNN()

# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()

n_epochs = 10
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    for X_batch, y_batch in trainloader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
    # Validation
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    acc = 0
    count = 0
    for X_batch, y_batch in testloader:
        y_pred = model(X_batch)
        acc += (torch.argmax(y_pred, 1) == y_batch).float().sum()
        # 获取长度 / Get length
        count += len(y_batch)
    acc = acc / count
    # 打印输出 / Print output
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Lenet5

# 12 — Lenet5 / 12 Lenet5

**Chapter 26 — File 5 of 5 / 第26章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Load MNIST data**.

本脚本演示 **Load MNIST data**。

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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
import torchvision
```

---
## Step 2 — Load MNIST data

```python
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (128,)),
])
train = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST('data', train=False, download=True, transform=transform)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=100)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
testloader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=100)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class LeNet5(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
        self.act3 = nn.Tanh()

        # 展平层：多维→一维 / Flatten: multi-dim → 1D
        self.flat = nn.Flatten()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc1 = nn.Linear(1*1*120, 84)
        # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
        self.act4 = nn.Tanh()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc2 = nn.Linear(84, 10)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
```

---
## Step 3 — input 1x28x28, output 6x28x28

```python
x = self.act1(self.conv1(x))
```

---
## Step 4 — input 6x28x28, output 6x14x14

```python
x = self.pool1(x)
```

---
## Step 5 — input 6x14x14, output 16x10x10

```python
x = self.act2(self.conv2(x))
```

---
## Step 6 — input 16x10x10, output 16x5x5

```python
x = self.pool2(x)
```

---
## Step 7 — input 16x5x5, output 120x1x1

```python
x = self.act3(self.conv3(x))
```

---
## Step 8 — input 120x1x1, output 84

```python
x = self.act4(self.fc1(self.flat(x)))
```

---
## Step 9 — input 84, output 10

```python
x = self.fc2(x)
        return x

model = LeNet5()

# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()

n_epochs = 10
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    for X_batch, y_batch in trainloader:
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
## Step 10 — Validation

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    acc = 0
    count = 0
    for X_batch, y_batch in testloader:
        y_pred = model(X_batch)
        acc += (torch.argmax(y_pred, 1) == y_batch).float().sum()
        # 获取长度 / Get length
        count += len(y_batch)
    acc = acc / count
    # 打印输出 / Print output
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
```

---
## Learning Notes / 学习笔记

- **概念**: Load MNIST data 是机器学习中的常用技术。  
  *Load MNIST data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `DataLoader` | 数据加载器，批量读取数据 | Loads data in batches for training |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Conv2d` | 二维卷积层，提取图像特征 | 2D convolution layer for image features |
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lenet5 / 12 Lenet5
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
import torchvision

# Load MNIST data
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (128,)),
])
train = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST('data', train=False, download=True, transform=transform)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=100)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
testloader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=100)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class LeNet5(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
        self.act3 = nn.Tanh()

        # 展平层：多维→一维 / Flatten: multi-dim → 1D
        self.flat = nn.Flatten()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc1 = nn.Linear(1*1*120, 84)
        # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
        self.act4 = nn.Tanh()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc2 = nn.Linear(84, 10)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # input 1x28x28, output 6x28x28
        x = self.act1(self.conv1(x))
        # input 6x28x28, output 6x14x14
        x = self.pool1(x)
        # input 6x14x14, output 16x10x10
        x = self.act2(self.conv2(x))
        # input 16x10x10, output 16x5x5
        x = self.pool2(x)
        # input 16x5x5, output 120x1x1
        x = self.act3(self.conv3(x))
        # input 120x1x1, output 84
        x = self.act4(self.fc1(self.flat(x)))
        # input 84, output 10
        x = self.fc2(x)
        return x

model = LeNet5()

# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()

n_epochs = 10
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    for X_batch, y_batch in trainloader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
    # Validation
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    acc = 0
    count = 0
    for X_batch, y_batch in testloader:
        y_pred = model(X_batch)
        acc += (torch.argmax(y_pred, 1) == y_batch).float().sum()
        # 获取长度 / Get length
        count += len(y_batch)
    acc = acc / count
    # 打印输出 / Print output
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
```

---

### Chapter Summary / 章节总结



---
