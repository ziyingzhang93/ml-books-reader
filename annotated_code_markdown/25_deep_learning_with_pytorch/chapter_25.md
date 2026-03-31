# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 25

---

### Cifar

# 01 — Cifar / 01 Cifar

**Chapter 25 — File 1 of 3 / 第25章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **input 3x32x32, output 32x32x32**.

本脚本演示 **input 3x32x32, output 32x32x32**。

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

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

batch_size = 32
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CIFAR10Model(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act1 = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.drop1 = nn.Dropout(0.3)

        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act2 = nn.ReLU()
        # 最大池化：缩小特征图，保留最大值 / MaxPool: downsample, keep max values
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # 展平层：多维→一维 / Flatten: multi-dim → 1D
        self.flat = nn.Flatten()

        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc3 = nn.Linear(8192, 512)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act3 = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.drop3 = nn.Dropout(0.5)

        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc4 = nn.Linear(512, 10)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
```

---
## Step 2 — input 3x32x32, output 32x32x32

```python
x = self.act1(self.conv1(x))
        x = self.drop1(x)
```

---
## Step 3 — input 32x32x32, output 32x32x32

```python
x = self.act2(self.conv2(x))
```

---
## Step 4 — input 32x32x32, output 32x16x16

```python
x = self.pool2(x)
```

---
## Step 5 — input 32x16x16, output 8192

```python
x = self.flat(x)
```

---
## Step 6 — input 8192, output 512

```python
x = self.act3(self.fc3(x))
        x = self.drop3(x)
```

---
## Step 7 — input 512, output 10

```python
x = self.fc4(x)
        return x

model = CIFAR10Model()
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()
# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

n_epochs = 20
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for inputs, labels in trainloader:
```

---
## Step 8 — forward, backward, and then weight update

```python
y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()

    acc = 0
    count = 0
    for inputs, labels in testloader:
        y_pred = model(inputs)
        acc += (torch.argmax(y_pred, 1) == labels).float().sum()
        # 获取长度 / Get length
        count += len(labels)
    acc /= count
    # 打印输出 / Print output
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))

# 获取模型参数字典 / Get model parameter dictionary
torch.save(model.state_dict(), "cifar10model.pth")
```

---
## Learning Notes / 学习笔记

- **概念**: input 3x32x32, output 32x32x32 是机器学习中的常用技术。  
  *input 3x32x32, output 32x32x32 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `DataLoader` | 数据加载器，批量读取数据 | Loads data in batches for training |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
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
# Cifar / 01 Cifar
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
import torchvision

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

batch_size = 32
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
# 创建数据加载器：按批次读取数据 / Create DataLoader: load data in batches
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CIFAR10Model(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act1 = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.drop1 = nn.Dropout(0.3)

        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act2 = nn.ReLU()
        # 最大池化：缩小特征图，保留最大值 / MaxPool: downsample, keep max values
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # 展平层：多维→一维 / Flatten: multi-dim → 1D
        self.flat = nn.Flatten()

        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc3 = nn.Linear(8192, 512)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act3 = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.drop3 = nn.Dropout(0.5)

        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc4 = nn.Linear(512, 10)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)
        return x

model = CIFAR10Model()
# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
loss_fn = nn.CrossEntropyLoss()
# SGD优化器：随机梯度下降 / SGD optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

n_epochs = 20
# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    for inputs, labels in trainloader:
        # forward, backward, and then weight update
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()

    acc = 0
    count = 0
    for inputs, labels in testloader:
        y_pred = model(inputs)
        acc += (torch.argmax(y_pred, 1) == labels).float().sum()
        # 获取长度 / Get length
        count += len(labels)
    acc /= count
    # 打印输出 / Print output
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))

# 获取模型参数字典 / Get model parameter dictionary
torch.save(model.state_dict(), "cifar10model.pth")
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Showone



---

### Feature

# 08 — Feature / 特征工程

**Chapter 25 — File 3 of 3 / 第25章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **input 3x32x32, output 32x32x32**.

本脚本演示 **input 3x32x32, output 32x32x32**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌───────────────────────┐
│  定义模型 Define Model  │
└───────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
import torchvision
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CIFAR10Model(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act1 = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.drop1 = nn.Dropout(0.3)

        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act2 = nn.ReLU()
        # 最大池化：缩小特征图，保留最大值 / MaxPool: downsample, keep max values
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # 展平层：多维→一维 / Flatten: multi-dim → 1D
        self.flat = nn.Flatten()

        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc3 = nn.Linear(8192, 512)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act3 = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.drop3 = nn.Dropout(0.5)

        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc4 = nn.Linear(512, 10)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
```

---
## Step 2 — input 3x32x32, output 32x32x32

```python
x = self.act1(self.conv1(x))
        x = self.drop1(x)
```

---
## Step 3 — input 32x32x32, output 32x32x32

```python
x = self.act2(self.conv2(x))
```

---
## Step 4 — input 32x32x32, output 32x16x16

```python
x = self.pool2(x)
```

---
## Step 5 — input 32x16x16, output 8192

```python
x = self.flat(x)
```

---
## Step 6 — input 8192, output 512

```python
x = self.act3(self.fc3(x))
        x = self.drop3(x)
```

---
## Step 7 — input 512, output 10

```python
x = self.fc4(x)
        return x

model = CIFAR10Model()
# 加载模型参数 / Load model parameters
model.load_state_dict(torch.load("cifar10model.pth"))

# 显示图像 / Display image
plt.imshow(trainset.data[7])
# 显示图表 / Display the plot
plt.show()

X = torch.tensor([trainset.data[7]], dtype=torch.float32).permute(0,3,1,2)
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    feature_maps = model.conv1(X)
fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
# 生成整数序列 / Generate integer sequence
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(feature_maps[0][i])
# 显示图表 / Display the plot
plt.show()

# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    feature_maps = model.act1(model.conv1(X))
    feature_maps = model.drop1(feature_maps)
    feature_maps = model.conv2(feature_maps)
fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
# 生成整数序列 / Generate integer sequence
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(feature_maps[0][i])
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: input 3x32x32, output 32x32x32 是机器学习中的常用技术。  
  *input 3x32x32, output 32x32x32 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `matplotlib` | 绑图库 | Plotting library |
| `nn.Conv2d` | 二维卷积层，提取图像特征 | 2D convolution layer for image features |
| `nn.Dropout` | 随机丢弃神经元防止过拟合 | Randomly drop neurons to prevent overfitting |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Feature / 特征工程
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
import torchvision
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class CIFAR10Model(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act1 = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.drop1 = nn.Dropout(0.3)

        # 二维卷积层：提取图像局部特征 / 2D convolution: extract local image features
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act2 = nn.ReLU()
        # 最大池化：缩小特征图，保留最大值 / MaxPool: downsample, keep max values
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # 展平层：多维→一维 / Flatten: multi-dim → 1D
        self.flat = nn.Flatten()

        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc3 = nn.Linear(8192, 512)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act3 = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.drop3 = nn.Dropout(0.5)

        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.fc4 = nn.Linear(512, 10)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)
        return x

model = CIFAR10Model()
# 加载模型参数 / Load model parameters
model.load_state_dict(torch.load("cifar10model.pth"))

# 显示图像 / Display image
plt.imshow(trainset.data[7])
# 显示图表 / Display the plot
plt.show()

X = torch.tensor([trainset.data[7]], dtype=torch.float32).permute(0,3,1,2)
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    feature_maps = model.conv1(X)
fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
# 生成整数序列 / Generate integer sequence
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(feature_maps[0][i])
# 显示图表 / Display the plot
plt.show()

# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    feature_maps = model.act1(model.conv1(X))
    feature_maps = model.drop1(feature_maps)
    feature_maps = model.conv2(feature_maps)
fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
# 生成整数序列 / Generate integer sequence
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(feature_maps[0][i])
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 25 Summary / 第25章总结

## Theme / 主题: Chapter 25 / Chapter 25

This chapter contains **3 code files** demonstrating chapter 25.

本章包含 **3 个代码文件**，演示Chapter 25。

---
## Evolution / 演化路线

  1. `01_cifar.ipynb` — Cifar
  2. `04_showone.ipynb` — Showone
  3. `08_feature.ipynb` — Feature

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 25) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 25）是机器学习流水线中的基础构建块。

---
