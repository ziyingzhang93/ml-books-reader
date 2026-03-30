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
## Step 1 — Step 1

```python
import torch
import torch.nn as nn

mae = nn.L1Loss()
mse = nn.MSELoss()

predict = torch.tensor([0., 3.])
target = torch.tensor([1., 0.])

print("MAE: %.3f" % mae(predict, target))
print("MSE: %.3f" % mse(predict, target))
```

---
## Learning Notes / 学习笔记

- **概念**: Msemae 是机器学习中的常用技术。  
  *Msemae is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Msemae / 01 Msemae
# Complete Code / 完整代码
# ===============================

import torch
import torch.nn as nn

mae = nn.L1Loss()
mse = nn.MSELoss()

predict = torch.tensor([0., 3.])
target = torch.tensor([1., 0.])

print("MAE: %.3f" % mae(predict, target))
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
## Step 1 — Step 1

```python
import torch
import torch.nn as nn

ce = nn.CrossEntropyLoss()

logits = torch.tensor([[-1.90, -0.29, -2.30], [-0.29, -1.90, -2.30]])
target = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
print("Cross entropy: %.3f" % ce(logits, target))
```

---
## Learning Notes / 学习笔记

- **概念**: Ce 是机器学习中的常用技术。  
  *Ce is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ce / 02 Ce
# Complete Code / 完整代码
# ===============================

import torch
import torch.nn as nn

ce = nn.CrossEntropyLoss()

logits = torch.tensor([[-1.90, -0.29, -2.30], [-0.29, -1.90, -2.30]])
target = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
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
## Step 1 — Step 1

```python
import torch
import torch.nn as nn

ce = nn.CrossEntropyLoss()

logits = torch.tensor([[-1.90, -0.29, -2.30], [-0.29, -1.90, -2.30]])
indices = torch.tensor([1, 0])
print("Cross entropy: %.3f" % ce(logits, indices))
```

---
## Learning Notes / 学习笔记

- **概念**: Integer 是机器学习中的常用技术。  
  *Integer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Integer / 04 Integer
# Complete Code / 完整代码
# ===============================

import torch
import torch.nn as nn

ce = nn.CrossEntropyLoss()

logits = torch.tensor([[-1.90, -0.29, -2.30], [-0.29, -1.90, -2.30]])
indices = torch.tensor([1, 0])
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
## Step 1 — Step 1

```python
import torch

target = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
indices = torch.argmax(target, dim=1)
print(indices)
```

---
## Learning Notes / 学习笔记

- **概念**: Onehot 是机器学习中的常用技术。  
  *Onehot is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Onehot / 05 Onehot
# Complete Code / 完整代码
# ===============================

import torch

target = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
indices = torch.argmax(target, dim=1)
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
## Step 1 — Step 1

```python
import torch
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
print("Cross entropy: %.3f" % ce(pred, indices))
```

---
## Learning Notes / 学习笔记

- **概念**: softmax to apply on dimension 1, i.e. per row 是机器学习中的常用技术。  
  *softmax to apply on dimension 1, i.e. per row is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nllloss / 损失函数
# Complete Code / 完整代码
# ===============================

import torch
import torch.nn as nn

ce = nn.NLLLoss()

# softmax to apply on dimension 1, i.e. per row
logsoftmax = nn.LogSoftmax(dim=1)

logits = torch.tensor([[-1.90, -0.29, -2.30], [-0.29, -1.90, -2.30]])
pred = logsoftmax(logits)
indices = torch.tensor([1, 0])
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
## Step 1 — Step 1

```python
import torch
import torch.nn as nn

bce = nn.BCELoss()

pred = torch.tensor([0.75, 0.25])
target = torch.tensor([1., 0.])
print("Binary cross entropy: %.3f" % bce(pred, target))
```

---
## Learning Notes / 学习笔记

- **概念**: Bce 是机器学习中的常用技术。  
  *Bce is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Bce / 07 Bce
# Complete Code / 完整代码
# ===============================

import torch
import torch.nn as nn

bce = nn.BCELoss()

pred = torch.tensor([0.75, 0.25])
target = torch.tensor([1., 0.])
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
## Step 1 — Step 1

```python
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
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
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                            shuffle=True)
```

---
## Step 4 — Standardizing data

```python
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
```

---
## Step 5 — Convert to 2D PyTorch tensors

```python
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 6 — Define the model

```python
model = nn.Sequential(
    nn.Linear(8, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
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
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)
```

---
## Step 9 — Hold the best model

```python
best_mape = np.inf   # init to infinity
best_weights = None

for epoch in range(n_epochs):
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
optimizer.zero_grad()
        loss.backward()
```

---
## Step 13 — update weights

```python
optimizer.step()
```

---
## Step 14 — evaluate accuracy at end of each epoch

```python
model.eval()
    y_pred = model(X_test)
    mape = float(loss_fn(y_pred, y_test))
    if mape < best_mape:
        best_mape = mape
        best_weights = copy.deepcopy(model.state_dict())
```

---
## Step 15 — restore model and return best accuracy

```python
model.load_state_dict(best_weights)
print("MAPE: %.2f" % best_mape)

model.eval()
with torch.no_grad():
```

---
## Step 16 — Test out inference with 5 samples

```python
for i in range(5):
        X_sample = X_test_raw[i: i+1]
        X_sample = scaler.transform(X_sample)
        X_sample = torch.tensor(X_sample, dtype=torch.float32)
        y_pred = model(X_sample)
        print(f"{X_test_raw[i]} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")
```

---
## Learning Notes / 学习笔记

- **概念**: Read data 是机器学习中的常用技术。  
  *Read data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mape / 08 Mape
# Complete Code / 完整代码
# ===============================

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Read data
data = fetch_california_housing()
X, y = data.data, data.target

# train-test split for model evaluation
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                            shuffle=True)

# Standardizing data
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define the model
model = nn.Sequential(
    nn.Linear(8, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)

# loss function and optimizer
def loss_fn(output, target):
    # MAPE loss
    return torch.mean(torch.abs((target - output) / target))
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mape = np.inf   # init to infinity
best_weights = None

for epoch in range(n_epochs):
    model.train()
    for start in batch_start:
        # take a batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mape = float(loss_fn(y_pred, y_test))
    if mape < best_mape:
        best_mape = mape
        best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MAPE: %.2f" % best_mape)

model.eval()
with torch.no_grad():
    # Test out inference with 5 samples
    for i in range(5):
        X_sample = X_test_raw[i: i+1]
        X_sample = scaler.transform(X_sample)
        X_sample = torch.tensor(X_sample, dtype=torch.float32)
        y_pred = model(X_sample)
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
