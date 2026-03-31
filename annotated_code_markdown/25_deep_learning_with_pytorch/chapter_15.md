# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 15

---

### Msemae

# 01 — Msemae / 01 Msemae

**Chapter 15 — File 1 of 7 / 第15章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Msemae**.

本脚本演示 **01 Msemae**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

mae = nn.L1Loss()
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
mse = nn.MSELoss()

predict = torch.tensor([0., 3.])
target = torch.tensor([1., 0.])

# 打印输出 / Print output
print("MAE: %.3f" % mae(predict, target))
# 打印输出 / Print output
print("MSE: %.3f" % mse(predict, target))
```

---
## Learning Notes / 学习笔记

- **概念**: Msemae 是机器学习中的常用技术。  
  *Msemae is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.MSELoss` | 均方误差损失，回归任务常用 | Mean squared error loss for regression |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Msemae / 01 Msemae
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

mae = nn.L1Loss()
# 均方误差损失：回归任务的标准损失函数 / MSE: standard loss for regression
mse = nn.MSELoss()

predict = torch.tensor([0., 3.])
target = torch.tensor([1., 0.])

# 打印输出 / Print output
print("MAE: %.3f" % mae(predict, target))
# 打印输出 / Print output
print("MSE: %.3f" % mse(predict, target))
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Ce

# 02 — Ce / 02 Ce

**Chapter 15 — File 2 of 7 / 第15章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Ce**.

本脚本演示 **02 Ce**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
ce = nn.CrossEntropyLoss()

logits = torch.tensor([[-1.90, -0.29, -2.30], [-0.29, -1.90, -2.30]])
target = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
# 打印输出 / Print output
print("Cross entropy: %.3f" % ce(logits, target))
```

---
## Learning Notes / 学习笔记

- **概念**: Ce 是机器学习中的常用技术。  
  *Ce is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ce / 02 Ce
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
ce = nn.CrossEntropyLoss()

logits = torch.tensor([[-1.90, -0.29, -2.30], [-0.29, -1.90, -2.30]])
target = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
# 打印输出 / Print output
print("Cross entropy: %.3f" % ce(logits, target))
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Integer

# 04 — Integer / 04 Integer

**Chapter 15 — File 3 of 7 / 第15章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Integer**.

本脚本演示 **04 Integer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
ce = nn.CrossEntropyLoss()

logits = torch.tensor([[-1.90, -0.29, -2.30], [-0.29, -1.90, -2.30]])
indices = torch.tensor([1, 0])
# 打印输出 / Print output
print("Cross entropy: %.3f" % ce(logits, indices))
```

---
## Learning Notes / 学习笔记

- **概念**: Integer 是机器学习中的常用技术。  
  *Integer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.CrossEntropyLoss` | 交叉熵损失，分类任务常用 | Cross-entropy loss for classification |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Integer / 04 Integer
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 交叉熵损失：分类任务的标准损失函数 / CrossEntropy: standard loss for classification
ce = nn.CrossEntropyLoss()

logits = torch.tensor([[-1.90, -0.29, -2.30], [-0.29, -1.90, -2.30]])
indices = torch.tensor([1, 0])
# 打印输出 / Print output
print("Cross entropy: %.3f" % ce(logits, indices))
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Onehot

# 05 — Onehot / 05 Onehot

**Chapter 15 — File 4 of 7 / 第15章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Onehot**.

本脚本演示 **05 Onehot**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

target = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
indices = torch.argmax(target, dim=1)
# 打印输出 / Print output
print(indices)
```

---
## Learning Notes / 学习笔记

- **概念**: Onehot 是机器学习中的常用技术。  
  *Onehot is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Onehot / 05 Onehot
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

target = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
indices = torch.argmax(target, dim=1)
# 打印输出 / Print output
print(indices)
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Nllloss

# 06 — Nllloss / 损失函数

**Chapter 15 — File 5 of 7 / 第15章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **softmax to apply on dimension 1, i.e. per row**.

本脚本演示 **softmax to apply on dimension 1, i.e. per row**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

ce = nn.NLLLoss()
```

---
## Step 2 — softmax to apply on dimension 1, i.e. per row

```python
logsoftmax = nn.LogSoftmax(dim=1)

logits = torch.tensor([[-1.90, -0.29, -2.30], [-0.29, -1.90, -2.30]])
pred = logsoftmax(logits)
indices = torch.tensor([1, 0])
# 打印输出 / Print output
print("Cross entropy: %.3f" % ce(pred, indices))
```

---
## Learning Notes / 学习笔记

- **概念**: softmax to apply on dimension 1, i.e. per row 是机器学习中的常用技术。  
  *softmax to apply on dimension 1, i.e. per row is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nllloss / 损失函数
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

ce = nn.NLLLoss()

# softmax to apply on dimension 1, i.e. per row
logsoftmax = nn.LogSoftmax(dim=1)

logits = torch.tensor([[-1.90, -0.29, -2.30], [-0.29, -1.90, -2.30]])
pred = logsoftmax(logits)
indices = torch.tensor([1, 0])
# 打印输出 / Print output
print("Cross entropy: %.3f" % ce(pred, indices))
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Bce

# 07 — Bce / 07 Bce

**Chapter 15 — File 6 of 7 / 第15章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Bce**.

本脚本演示 **07 Bce**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
bce = nn.BCELoss()

pred = torch.tensor([0.75, 0.25])
target = torch.tensor([1., 0.])
# 打印输出 / Print output
print("Binary cross entropy: %.3f" % bce(pred, target))
```

---
## Learning Notes / 学习笔记

- **概念**: Bce 是机器学习中的常用技术。  
  *Bce is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Bce / 07 Bce
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# 二元交叉熵损失：二分类任务 / BCE: binary classification loss
bce = nn.BCELoss()

pred = torch.tensor([0.75, 0.25])
target = torch.tensor([1., 0.])
# 打印输出 / Print output
print("Binary cross entropy: %.3f" % bce(pred, target))
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Mape

# 08 — Mape / 08 Mape

**Chapter 15 — File 7 of 7 / 第15章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Read data**.

本脚本演示 **Read data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
import copy

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_california_housing
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
```

---
## Step 2 — Read data

```python
data = fetch_california_housing()
X, y = data.data, data.target
```

---
## Step 3 — train-test split for model evaluation

```python
# 划分训练集和测试集 / Split into train and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                            shuffle=True)
```

---
## Step 4 — Standardizing data

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler()
scaler.fit(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train = scaler.transform(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test = scaler.transform(X_test_raw)
```

---
## Step 5 — Convert to 2D PyTorch tensors

```python
X_train = torch.tensor(X_train, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 6 — Define the model

```python
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 24),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(24, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 6),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(6, 1)
)
```

---
## Step 7 — loss function and optimizer

```python
def loss_fn(output, target):
```

---
## Step 8 — MAPE loss

```python
return torch.mean(torch.abs((target - output) / target))
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)
```

---
## Step 9 — Hold the best model

```python
best_mape = np.inf   # init to infinity
best_weights = None

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    for start in batch_start:
```

---
## Step 10 — take a batch

```python
X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
```

---
## Step 11 — forward pass

```python
y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
```

---
## Step 12 — backward pass

```python
# 清零梯度（每轮训练前必须调用） / Zero gradients (must call before each training step)
optimizer.zero_grad()
        # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
        loss.backward()
```

---
## Step 13 — update weights

```python
# 更新参数：根据梯度调整权重 / Update parameters: adjust weights based on gradients
optimizer.step()
```

---
## Step 14 — evaluate accuracy at end of each epoch

```python
# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
    y_pred = model(X_test)
    mape = float(loss_fn(y_pred, y_test))
    if mape < best_mape:
        best_mape = mape
        # 获取模型参数字典 / Get model parameter dictionary
        best_weights = copy.deepcopy(model.state_dict())
```

---
## Step 15 — restore model and return best accuracy

```python
# 加载模型参数 / Load model parameters
model.load_state_dict(best_weights)
# 打印输出 / Print output
print("MAPE: %.2f" % best_mape)

# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
```

---
## Step 16 — Test out inference with 5 samples

```python
# 生成整数序列 / Generate integer sequence
for i in range(5):
        X_sample = X_test_raw[i: i+1]
        # 用已拟合的模型转换数据 / Transform data with fitted model
        X_sample = scaler.transform(X_sample)
        X_sample = torch.tensor(X_sample, dtype=torch.float32)
        y_pred = model(X_sample)
        # 打印输出 / Print output
        print(f"{X_test_raw[i]} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")
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
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |
| `zero_grad` | 清零梯度，每轮训练前必须调用 | Zero gradients before each training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mape / 08 Mape
# Complete Code / 完整代码
# ===============================

import copy

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_california_housing
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler

# Read data
data = fetch_california_housing()
X, y = data.data, data.target

# train-test split for model evaluation
# 划分训练集和测试集 / Split into train and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                            shuffle=True)

# Standardizing data
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler()
scaler.fit(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train = scaler.transform(X_train_raw)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test = scaler.transform(X_test_raw)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define the model
# 顺序容器：按顺序堆叠层 / Sequential: stack layers in order
model = nn.Sequential(
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(8, 24),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(24, 12),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(12, 6),
    # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
    nn.ReLU(),
    # 全连接层：y = xW + b / Fully connected layer: y = xW + b
    nn.Linear(6, 1)
)

# loss function and optimizer
def loss_fn(output, target):
    # MAPE loss
    return torch.mean(torch.abs((target - output) / target))
# Adam优化器：自适应学习率，最常用 / Adam optimizer: adaptive LR, most popular
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
# 获取长度 / Get length
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mape = np.inf   # init to infinity
best_weights = None

# 生成整数序列 / Generate integer sequence
for epoch in range(n_epochs):
    # 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
    model.train()
    for start in batch_start:
        # take a batch
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
    # evaluate accuracy at end of each epoch
    # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
    model.eval()
    y_pred = model(X_test)
    mape = float(loss_fn(y_pred, y_test))
    if mape < best_mape:
        best_mape = mape
        # 获取模型参数字典 / Get model parameter dictionary
        best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
# 加载模型参数 / Load model parameters
model.load_state_dict(best_weights)
# 打印输出 / Print output
print("MAPE: %.2f" % best_mape)

# 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
model.eval()
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
    # Test out inference with 5 samples
    # 生成整数序列 / Generate integer sequence
    for i in range(5):
        X_sample = X_test_raw[i: i+1]
        # 用已拟合的模型转换数据 / Transform data with fitted model
        X_sample = scaler.transform(X_sample)
        X_sample = torch.tensor(X_sample, dtype=torch.float32)
        y_pred = model(X_sample)
        # 打印输出 / Print output
        print(f"{X_test_raw[i]} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")
```

---

### Chapter Summary / 章节总结

# Chapter 15 Summary / 第15章总结

## Theme / 主题: Chapter 15 / Chapter 15

This chapter contains **7 code files** demonstrating chapter 15.

本章包含 **7 个代码文件**，演示Chapter 15。

---
## Evolution / 演化路线

  1. `01_msemae.ipynb` — Msemae
  2. `02_ce.ipynb` — Ce
  3. `04_integer.ipynb` — Integer
  4. `05_onehot.ipynb` — Onehot
  5. `06_nllloss.ipynb` — Nllloss
  6. `07_bce.ipynb` — Bce
  7. `08_mape.ipynb` — Mape

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 15) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 15）是机器学习流水线中的基础构建块。

---
