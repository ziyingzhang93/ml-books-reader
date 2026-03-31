# Python 深度学习 / Deep Learning with Python
## Chapter 01

---

### Chainer

# 01 — Chainer / 01 Chainer

**Chapter 01 — File 1 of 6 / 第01章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **create model**.

本脚本演示 **create model**。

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
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import iterators, optimizer, training, Chain
from chainer.datasets import mnist

train, test = mnist.get_mnist()
batchsize = 128
max_epoch = 10

train_iter = iterators.SerialIterator(train, batchsize)

class MLP(Chain):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, n_mid_units=100, n_out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_out)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
```

---
## Step 2 — create model

```python
model = MLP()
model = L.Classifier(model)  # using softmax cross entropy
```

---
## Step 3 — set up optimizer

```python
optimizer = optimizers.MomentumSGD()
optimizer.setup(model)
```

---
## Step 4 — connect train iterator and optimizer to an updater

```python
updater = training.updaters.StandardUpdater(train_iter, optimizer)
```

---
## Step 5 — set up trainer and run

```python
trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')
trainer.run()
```

---
## Learning Notes / 学习笔记

- **概念**: create model 是机器学习中的常用技术。  
  *create model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Chainer / 01 Chainer
# Complete Code / 完整代码
# ===============================

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import iterators, optimizer, training, Chain
from chainer.datasets import mnist

train, test = mnist.get_mnist()
batchsize = 128
max_epoch = 10

train_iter = iterators.SerialIterator(train, batchsize)

class MLP(Chain):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, n_mid_units=100, n_out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_out)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

# create model
model = MLP()
model = L.Classifier(model)  # using softmax cross entropy

# set up optimizer
optimizer = optimizers.MomentumSGD()
optimizer.setup(model)

# connect train iterator and optimizer to an updater
updater = training.updaters.StandardUpdater(train_iter, optimizer)

# set up trainer and run
trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')
trainer.run()
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Pytorch

# 03 — Pytorch / PyTorch

**Chapter 01 — File 2 of 6 / 第01章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Pytorch**.

本脚本演示 **PyTorch**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Model(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5,5), stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        # 展平层：多维→一维 / Flatten: multi-dim → 1D
        self.flatten = nn.Flatten()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear4 = nn.Linear(120, 84)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear5 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftMax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = F.tanh(self.conv3(x))
        # 展平为一维数组 / Flatten to 1D array
        x = self.flatten(x)
        x = F.tanh(self.linear4(x))
        x = self.linear5(x)
        return self.softmax(x)

model = Model()
```

---
## Learning Notes / 学习笔记

- **概念**: Pytorch 是机器学习中的常用技术。  
  *Pytorch is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `nn.Conv2d` | 二维卷积层，提取图像特征 | 2D convolution layer for image features |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pytorch / PyTorch
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.functional as F

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class Model(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5,5), stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        # 展平层：多维→一维 / Flatten: multi-dim → 1D
        self.flatten = nn.Flatten()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear4 = nn.Linear(120, 84)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.linear5 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftMax(dim=1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = F.tanh(self.conv3(x))
        # 展平为一维数组 / Flatten to 1D array
        x = self.flatten(x)
        x = F.tanh(self.linear4(x))
        x = self.linear5(x)
        return self.softmax(x)

model = Model()
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Pytorch



---

### Tfkeras



---

### Fit



---

### Training

# 07 — Training / 07 Training

**Chapter 01 — File 6 of 6 / 第01章 — 第6个文件（共6个）**

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
import datetime

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
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
    torchvision.transforms.ToTensor(), # required, otherwise MNIST are in PIL format
```

---
## Step 3 — torchvision.transforms.Normalize((0.5,), (0.5,)),

```python
])
train = torchvision.datasets.MNIST('./datafiles/', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST('./datafiles/', train=True, download=True, transform=transform)
```

---
## Step 4 — For manual feed into the model

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X_train = train.data.reshape(-1,1,28,28)
y_train = train.targets
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X_test = test.data.reshape(-1,1,28,28)
y_test = test.targets
```

---
## Step 5 — As iterator for data and target

```python
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)
```

---
## Step 6 — Neural network model

```python
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
```

---
## Step 7 — assume input 1x28x28

```python
# 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
nn.Conv2d(1, 6, kernel_size=(5,5), stride=1, padding=2),
    # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
    nn.Tanh(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
    nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
    # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
    nn.Tanh(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
    nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
    # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
    nn.Tanh(),
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    nn.Flatten(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(120, 84),
    # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
    nn.Tanh(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(84, 10),
    nn.LogSoftmax(dim=1)
)
```

---
## Step 8 — self-defined training loop function

```python
def training_loop(model, optimizer, loss_fn, train_loader, val_loader=None, n_epochs=100):
    best_loss, best_epoch = np.inf, -1
    # 获取模型参数字典 / Get model parameter dictionary
    best_state = model.state_dict()

    # 生成整数序列 / Generate integer sequence
    for epoch in range(n_epochs):
```

---
## Step 9 — Training

```python
# 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
model.train()
        train_loss = 0
        for data, target in train_loader:
            output = model(data)
            loss = loss_fn(output, target)
            # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
            optimizer.zero_grad()
            # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
            loss.backward()
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            optimizer.step()
            train_loss += loss.item()
```

---
## Step 10 — Validation

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
        status = (f"{str(datetime.datetime.now())} End of epoch {epoch}, "
                  # 获取长度 / Get length
                  f"training loss={train_loss/len(train_loader)}")
        if val_loader:
            val_loss = 0
            for data, target in val_loader:
                output = model(data)
                loss = loss_fn(output, target)
                val_loss += loss.item()
            # 获取长度 / Get length
            status += f", validation loss={val_loss/len(val_loader)}"
        # 打印输出 / Print output
        print(status)

# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss()
training_loop(model, optimizer, criterion, train_loader, test_loader, n_epochs=100)
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
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Conv2d` | 二维卷积层，提取图像特征 | 2D convolution layer for image features |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Training / 07 Training
# Complete Code / 完整代码
# ===============================

import datetime

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
import torchvision

# Load MNIST data
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), # required, otherwise MNIST are in PIL format
    #torchvision.transforms.Normalize((0.5,), (0.5,)),
])
train = torchvision.datasets.MNIST('./datafiles/', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST('./datafiles/', train=True, download=True, transform=transform)

# For manual feed into the model
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X_train = train.data.reshape(-1,1,28,28)
y_train = train.targets
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X_test = test.data.reshape(-1,1,28,28)
y_test = test.targets

# As iterator for data and target
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

# Neural network model
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # assume input 1x28x28
    # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
    nn.Conv2d(1, 6, kernel_size=(5,5), stride=1, padding=2),
    # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
    nn.Tanh(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
    nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
    # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
    nn.Tanh(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
    nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
    # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
    nn.Tanh(),
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    nn.Flatten(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(120, 84),
    # Tanh激活：压缩到(-1,1)范围 / Tanh: compress to (-1,1) range
    nn.Tanh(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(84, 10),
    nn.LogSoftmax(dim=1)
)

# self-defined training loop function
def training_loop(model, optimizer, loss_fn, train_loader, val_loader=None, n_epochs=100):
    best_loss, best_epoch = np.inf, -1
    # 获取模型参数字典 / Get model parameter dictionary
    best_state = model.state_dict()

    # 生成整数序列 / Generate integer sequence
    for epoch in range(n_epochs):
        # Training
        # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
        model.train()
        train_loss = 0
        for data, target in train_loader:
            output = model(data)
            loss = loss_fn(output, target)
            # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
            optimizer.zero_grad()
            # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
            loss.backward()
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            optimizer.step()
            train_loss += loss.item()
        # Validation
        # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
        model.eval()
        status = (f"{str(datetime.datetime.now())} End of epoch {epoch}, "
                  # 获取长度 / Get length
                  f"training loss={train_loss/len(train_loader)}")
        if val_loader:
            val_loss = 0
            for data, target in val_loader:
                output = model(data)
                loss = loss_fn(output, target)
                val_loss += loss.item()
            # 获取长度 / Get length
            status += f", validation loss={val_loss/len(val_loader)}"
        # 打印输出 / Print output
        print(status)

# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss()
training_loop(model, optimizer, criterion, train_loader, test_loader, n_epochs=100)
```

---

### Chapter Summary / 章节总结



---
