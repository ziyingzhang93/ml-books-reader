# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 08

---

### Csv

# 01 — Csv / 01 Csv

**Chapter 08 — File 1 of 5 / 第08章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **load the dataset**.

本脚本演示 **load the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
```

---
## Step 2 — load the dataset

```python
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 3 — split the dataset into training and test sets

```python
Xtrain = X[:700]
ytrain = y[:700]
Xtest = X[700:]
ytest = y[700:]
```

---
## Learning Notes / 学习笔记

- **概念**: load the dataset 是机器学习中的常用技术。  
  *load the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Csv / 01 Csv
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

# load the dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# split the dataset into training and test sets
Xtrain = X[:700]
ytrain = y[:700]
Xtest = X[700:]
ytest = y[700:]
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Model



---

### Train

# 05 — Train / 05 Train

**Chapter 08 — File 3 of 5 / 第08章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **load the dataset**.

本脚本演示 **load the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  🏋️ 训练模型 / Train Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
```

---
## Step 2 — load the dataset

```python
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 3 — split the dataset into training and test sets

```python
Xtrain = X[:700]
ytrain = y[:700]
Xtest = X[700:]
ytest = y[700:]

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 8),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
# 打印输出 / Print output
print(model)
```

---
## Step 4 — loss function and optimizer

```python
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()  # binary cross entropy
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 50    # number of epochs to run
batch_size = 10  # size of each batch
# 获取长度 / Get length
batches_per_epoch = len(Xtrain) // batch_size

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 生成整数序列 / Generate integer sequence
    for i in range(batches_per_epoch):
        start = i * batch_size
```

---
## Step 5 — take a batch

```python
Xbatch = Xtrain[start:start+batch_size]
        ybatch = ytrain[start:start+batch_size]
```

---
## Step 6 — forward pass

```python
y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
```

---
## Step 7 — backward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
```

---
## Step 8 — update weights

```python
# 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
optimizer.step()
```

---
## Step 9 — evaluate trained model with test set

```python
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
# 打印输出 / Print output
print("Accuracy {:.2f}".format(accuracy * 100))
```

---
## Learning Notes / 学习笔记

- **概念**: load the dataset 是机器学习中的常用技术。  
  *load the dataset is a common technique in machine learning.*

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
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train / 05 Train
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim

# load the dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# split the dataset into training and test sets
Xtrain = X[:700]
ytrain = y[:700]
Xtest = X[700:]
ytest = y[700:]

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 8),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
# 打印输出 / Print output
print(model)

# loss function and optimizer
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()  # binary cross entropy
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 50    # number of epochs to run
batch_size = 10  # size of each batch
# 获取长度 / Get length
batches_per_epoch = len(Xtrain) // batch_size

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 生成整数序列 / Generate integer sequence
    for i in range(batches_per_epoch):
        start = i * batch_size
        # take a batch
        Xbatch = Xtrain[start:start+batch_size]
        ybatch = ytrain[start:start+batch_size]
        # forward pass
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        # backward pass
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # update weights
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()

# evaluate trained model with test set
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
# 打印输出 / Print output
print("Accuracy {:.2f}".format(accuracy * 100))
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Visualize

# 08 — Visualize / 08 Visualize

**Chapter 08 — File 4 of 5 / 第08章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **load the dataset**.

本脚本演示 **load the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
```

---
## Step 2 — load the dataset

```python
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
```

---
## Step 3 — split into input (X) and output (y) variables

```python
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 4 — split the dataset into training and test sets

```python
Xtrain = X[:700]
ytrain = y[:700]
Xtest = X[700:]
ytest = y[700:]

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 8),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
# 打印输出 / Print output
print(model)
```

---
## Step 5 — loss function and optimizer

```python
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()  # binary cross entropy
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 50    # number of epochs to run
batch_size = 10  # size of each batch
# 获取长度 / Get length
batches_per_epoch = len(Xtrain) // batch_size
```

---
## Step 6 — collect statistics

```python
train_loss = []
train_acc = []
test_acc = []

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 生成整数序列 / Generate integer sequence
    for i in range(batches_per_epoch):
```

---
## Step 7 — take a batch

```python
start = i * batch_size
        Xbatch = Xtrain[start:start+batch_size]
        ybatch = ytrain[start:start+batch_size]
```

---
## Step 8 — forward pass

```python
y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        acc = (y_pred.round() == ybatch).float().mean()
```

---
## Step 9 — store metrics

```python
# 添加元素到列表末尾 / Append element to list end
train_loss.append(float(loss))
        # 添加元素到列表末尾 / Append element to list end
        train_acc.append(float(acc))
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
# 打印输出 / Print output
print(f"epoch {epoch} step {i} loss {loss} accuracy {acc}")
```

---
## Step 13 — evaluate model at end of epoch

```python
y_pred = model(Xtest)
    acc = (y_pred.round() == ytest).float().mean()
    # 添加元素到列表末尾 / Append element to list end
    test_acc.append(float(acc))
    # 打印输出 / Print output
    print(f"End of {epoch}, accuracy {acc}")
```

---
## Step 14 — Plot the loss metrics

```python
# 绘制折线图 / Draw line plot
plt.plot(train_loss)
# 设置X轴标签 / Set X-axis label
plt.xlabel("steps")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("loss")
plt.ylim(0)
# 显示图表 / Display the plot
plt.show()
```

---
## Step 15 — plot the accuracy metrics

```python
avg_train_acc = []
# 生成整数序列 / Generate integer sequence
for i in range(n_epochs):
    start = i * batch_size
    average = sum(train_acc[start:start+batches_per_epoch]) / batches_per_epoch
    # 添加元素到列表末尾 / Append element to list end
    avg_train_acc.append(average)

# 绘制折线图 / Draw line plot
plt.plot(avg_train_acc, label="train")
# 绘制折线图 / Draw line plot
plt.plot(test_acc, label="test")
# 设置X轴标签 / Set X-axis label
plt.xlabel("epochs")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("accuracy")
plt.ylim(0)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load the dataset 是机器学习中的常用技术。  
  *load the dataset is a common technique in machine learning.*

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
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Visualize / 08 Visualize
# Complete Code / 完整代码
# ===============================

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

# load the dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# split the dataset into training and test sets
Xtrain = X[:700]
ytrain = y[:700]
Xtest = X[700:]
ytest = y[700:]

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 8),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
# 打印输出 / Print output
print(model)

# loss function and optimizer
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()  # binary cross entropy
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 50    # number of epochs to run
batch_size = 10  # size of each batch
# 获取长度 / Get length
batches_per_epoch = len(Xtrain) // batch_size

# collect statistics
train_loss = []
train_acc = []
test_acc = []

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 生成整数序列 / Generate integer sequence
    for i in range(batches_per_epoch):
        # take a batch
        start = i * batch_size
        Xbatch = Xtrain[start:start+batch_size]
        ybatch = ytrain[start:start+batch_size]
        # forward pass
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        acc = (y_pred.round() == ybatch).float().mean()
        # store metrics
        # 添加元素到列表末尾 / Append element to list end
        train_loss.append(float(loss))
        # 添加元素到列表末尾 / Append element to list end
        train_acc.append(float(acc))
        # backward pass
        # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
        optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
        # update weights
        # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
        optimizer.step()
        # print progress
        # 打印输出 / Print output
        print(f"epoch {epoch} step {i} loss {loss} accuracy {acc}")
    # evaluate model at end of epoch
    y_pred = model(Xtest)
    acc = (y_pred.round() == ytest).float().mean()
    # 添加元素到列表末尾 / Append element to list end
    test_acc.append(float(acc))
    # 打印输出 / Print output
    print(f"End of {epoch}, accuracy {acc}")

# Plot the loss metrics
# 绘制折线图 / Draw line plot
plt.plot(train_loss)
# 设置X轴标签 / Set X-axis label
plt.xlabel("steps")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("loss")
plt.ylim(0)
# 显示图表 / Display the plot
plt.show()

# plot the accuracy metrics
avg_train_acc = []
# 生成整数序列 / Generate integer sequence
for i in range(n_epochs):
    start = i * batch_size
    average = sum(train_acc[start:start+batches_per_epoch]) / batches_per_epoch
    # 添加元素到列表末尾 / Append element to list end
    avg_train_acc.append(average)

# 绘制折线图 / Draw line plot
plt.plot(avg_train_acc, label="train")
# 绘制折线图 / Draw line plot
plt.plot(test_acc, label="test")
# 设置X轴标签 / Set X-axis label
plt.xlabel("epochs")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("accuracy")
plt.ylim(0)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Tqdm

# 11 — Tqdm / 11 Tqdm

**Chapter 08 — File 5 of 5 / 第08章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **load the dataset**.

本脚本演示 **load the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  🏋️ 训练模型 / Train Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
import tqdm
```

---
## Step 2 — load the dataset

```python
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
```

---
## Step 3 — split into input (X) and output (y) variables

```python
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 4 — split the dataset into training and test sets

```python
Xtrain = X[:700]
ytrain = y[:700]
Xtest = X[700:]
ytest = y[700:]

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 8),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
# 打印输出 / Print output
print(model)
```

---
## Step 5 — loss function and optimizer

```python
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()  # binary cross entropy
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 50    # number of epochs to run
batch_size = 10  # size of each batch
# 获取长度 / Get length
batches_per_epoch = len(Xtrain) // batch_size
```

---
## Step 6 — collect statistics

```python
train_loss = []
train_acc = []
test_acc = []

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 生成整数序列 / Generate integer sequence
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
```

---
## Step 7 — take a batch

```python
start = i * batch_size
            Xbatch = Xtrain[start:start+batch_size]
            ybatch = ytrain[start:start+batch_size]
```

---
## Step 8 — forward pass

```python
y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            acc = (y_pred.round() == ybatch).float().mean()
```

---
## Step 9 — store metrics

```python
# 添加元素到列表末尾 / Append element to list end
train_loss.append(float(loss))
            # 添加元素到列表末尾 / Append element to list end
            train_acc.append(float(acc))
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
bar.set_postfix(
                loss=float(loss),
                acc=f"{float(acc)*100:.2f}%"
            )
```

---
## Step 13 — evaluate model at end of epoch

```python
y_pred = model(Xtest)
    acc = (y_pred.round() == ytest).float().mean()
    # 添加元素到列表末尾 / Append element to list end
    test_acc.append(float(acc))
    # 打印输出 / Print output
    print(f"End of {epoch}, accuracy {acc}")
```

---
## Learning Notes / 学习笔记

- **概念**: load the dataset 是机器学习中的常用技术。  
  *load the dataset is a common technique in machine learning.*

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
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Tqdm / 11 Tqdm
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
import tqdm

# load the dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# split the dataset into training and test sets
Xtrain = X[:700]
ytrain = y[:700]
Xtest = X[700:]
ytest = y[700:]

# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 8),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 1),
    # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
    nn.Sigmoid()
)
# 打印输出 / Print output
print(model)

# loss function and optimizer
# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
loss_fn = nn.BCELoss()  # binary cross entropy
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 50    # number of epochs to run
batch_size = 10  # size of each batch
# 获取长度 / Get length
batches_per_epoch = len(Xtrain) // batch_size

# collect statistics
train_loss = []
train_acc = []
test_acc = []

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 生成整数序列 / Generate integer sequence
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            Xbatch = Xtrain[start:start+batch_size]
            ybatch = ytrain[start:start+batch_size]
            # forward pass
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            acc = (y_pred.round() == ybatch).float().mean()
            # store metrics
            # 添加元素到列表末尾 / Append element to list end
            train_loss.append(float(loss))
            # 添加元素到列表末尾 / Append element to list end
            train_acc.append(float(acc))
            # backward pass
            # 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
            optimizer.zero_grad()
            # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
            loss.backward()
            # update weights
            # 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
            optimizer.step()
            # print progress
            bar.set_postfix(
                loss=float(loss),
                acc=f"{float(acc)*100:.2f}%"
            )
    # evaluate model at end of epoch
    y_pred = model(Xtest)
    acc = (y_pred.round() == ytest).float().mean()
    # 添加元素到列表末尾 / Append element to list end
    test_acc.append(float(acc))
    # 打印输出 / Print output
    print(f"End of {epoch}, accuracy {acc}")
```

---

### Chapter Summary / 章节总结

# Chapter 08 Summary / 第08章总结

## Theme / 主题: Chapter 08 / Chapter 08

This chapter contains **5 code files** demonstrating chapter 08.

本章包含 **5 个代码文件**，演示Chapter 08。

---
## Evolution / 演化路线

  1. `01_csv.ipynb` — Csv
  2. `02_model.ipynb` — Model
  3. `05_train.ipynb` — Train
  4. `08_visualize.ipynb` — Visualize
  5. `11_tqdm.ipynb` — Tqdm

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 08) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 08）是机器学习流水线中的基础构建块。

---
