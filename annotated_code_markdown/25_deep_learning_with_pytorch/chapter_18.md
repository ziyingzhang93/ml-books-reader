# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 18

---

### Load

# 01 — Load / 01 Load

**Chapter 18 — File 1 of 4 / 第18章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Read data, convert to NumPy arrays**.

本脚本演示 **Read data, convert to NumPy arrays**。

---
## Step 1 — Step 1

```python
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
```

---
## Step 2 — Read data, convert to NumPy arrays

```python
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60].values
y = data.iloc[:, 60].values
```

---
## Step 3 — encode class values as integers

```python
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
```

---
## Step 4 — convert into PyTorch tensors

```python
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 5 — create DataLoader, then take one batch

```python
loader = DataLoader(list(zip(X,y)), shuffle=True, batch_size=16)
for X_batch, y_batch in loader:
    print(X_batch, y_batch)
    break
```

---
## Learning Notes / 学习笔记

- **概念**: Read data, convert to NumPy arrays 是机器学习中的常用技术。  
  *Read data, convert to NumPy arrays is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load / 01 Load
# Complete Code / 完整代码
# ===============================

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# Read data, convert to NumPy arrays
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60].values
y = data.iloc[:, 60].values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# convert into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# create DataLoader, then take one batch
loader = DataLoader(list(zip(X,y)), shuffle=True, batch_size=16)
for X_batch, y_batch in loader:
    print(X_batch, y_batch)
    break
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Dataloader

# 03 — Dataloader / 03 Dataloader

**Chapter 18 — File 2 of 4 / 第18章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Read data, convert to NumPy arrays**.

本脚本演示 **Read data, convert to NumPy arrays**。

---
## Step 1 — Step 1

```python
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
```

---
## Step 2 — Read data, convert to NumPy arrays

```python
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60].values
y = data.iloc[:, 60].values
```

---
## Step 3 — encode class values as integers

```python
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
```

---
## Step 4 — convert into PyTorch tensors

```python
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 5 — train-test split for evaluation of the model

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
```

---
## Step 6 — set up DataLoader for training set

```python
loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=16)
```

---
## Step 7 — create model

```python
model = nn.Sequential(
    nn.Linear(60, 60),
    nn.ReLU(),
    nn.Linear(60, 30),
    nn.ReLU(),
    nn.Linear(30, 1),
    nn.Sigmoid()
)
```

---
## Step 8 — Train the model

```python
n_epochs = 200
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
model.train()
for epoch in range(n_epochs):
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---
## Step 9 — evaluate accuracy after training

```python
model.eval()
y_pred = model(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
print("Model accuracy: %.2f%%" % (acc*100))
```

---
## Learning Notes / 学习笔记

- **概念**: Read data, convert to NumPy arrays 是机器学习中的常用技术。  
  *Read data, convert to NumPy arrays is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dataloader / 03 Dataloader
# Complete Code / 完整代码
# ===============================

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# Read data, convert to NumPy arrays
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60].values
y = data.iloc[:, 60].values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# convert into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# train-test split for evaluation of the model
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# set up DataLoader for training set
loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=16)

# create model
model = nn.Sequential(
    nn.Linear(60, 60),
    nn.ReLU(),
    nn.Linear(60, 30),
    nn.ReLU(),
    nn.Linear(30, 1),
    nn.Sigmoid()
)

# Train the model
n_epochs = 200
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
model.train()
for epoch in range(n_epochs):
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# evaluate accuracy after training
model.eval()
y_pred = model(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
print("Model accuracy: %.2f%%" % (acc*100))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Dataset

# 04 — Dataset / 04 Dataset

**Chapter 18 — File 3 of 4 / 第18章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **convert into PyTorch tensors and remember them**.

本脚本演示 **convert into PyTorch tensors and remember them**。

---
## Step 1 — Step 1

```python
from torch.utils.data import Dataset

class SonarDataset(Dataset):
    def __init__(self, X, y):
```

---
## Step 2 — convert into PyTorch tensors and remember them

```python
self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
```

---
## Step 3 — this should return the size of the dataset

```python
return len(self.X)

    def __getitem__(self, idx):
```

---
## Step 4 — this should return one sample from the dataset

```python
features = self.X[idx]
        target = self.y[idx]
        return features, target
```

---
## Learning Notes / 学习笔记

- **概念**: convert into PyTorch tensors and remember them 是机器学习中的常用技术。  
  *convert into PyTorch tensors and remember them is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dataset / 04 Dataset
# Complete Code / 完整代码
# ===============================

from torch.utils.data import Dataset

class SonarDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Training

# 06 — Training / 06 Training

**Chapter 18 — File 4 of 4 / 第18章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Read data, convert to NumPy arrays**.

本脚本演示 **Read data, convert to NumPy arrays**。

---
## Step 1 — Step 1

```python
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
from sklearn.preprocessing import LabelEncoder
```

---
## Step 2 — Read data, convert to NumPy arrays

```python
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60].values
y = data.iloc[:, 60].values
```

---
## Step 3 — encode class values as integers

```python
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y).reshape(-1, 1)

class SonarDataset(Dataset):
    def __init__(self, X, y):
```

---
## Step 4 — convert into PyTorch tensors and remember them

```python
self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
```

---
## Step 5 — this should return the size of the dataset

```python
return len(self.X)

    def __getitem__(self, idx):
```

---
## Step 6 — this should return one sample from the dataset

```python
features = self.X[idx]
        target = self.y[idx]
        return features, target
```

---
## Step 7 — set up DataLoader for data set

```python
dataset = SonarDataset(X, y)
trainset, testset = random_split(dataset, [0.7, 0.3])
loader = DataLoader(trainset, shuffle=True, batch_size=16)
```

---
## Step 8 — create model

```python
model = nn.Sequential(
    nn.Linear(60, 60),
    nn.ReLU(),
    nn.Linear(60, 30),
    nn.ReLU(),
    nn.Linear(30, 1),
    nn.Sigmoid()
)
```

---
## Step 9 — Train the model

```python
n_epochs = 200
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
model.train()
for epoch in range(n_epochs):
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---
## Step 10 — create one test tensor from the testset

```python
X_test, y_test = default_collate(testset)
model.eval()
y_pred = model(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
print("Model accuracy: %.2f%%" % (acc*100))
```

---
## Learning Notes / 学习笔记

- **概念**: Read data, convert to NumPy arrays 是机器学习中的常用技术。  
  *Read data, convert to NumPy arrays is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Training / 06 Training
# Complete Code / 完整代码
# ===============================

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
from sklearn.preprocessing import LabelEncoder

# Read data, convert to NumPy arrays
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60].values
y = data.iloc[:, 60].values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y).reshape(-1, 1)

class SonarDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target

# set up DataLoader for data set
dataset = SonarDataset(X, y)
trainset, testset = random_split(dataset, [0.7, 0.3])
loader = DataLoader(trainset, shuffle=True, batch_size=16)

# create model
model = nn.Sequential(
    nn.Linear(60, 60),
    nn.ReLU(),
    nn.Linear(60, 30),
    nn.ReLU(),
    nn.Linear(30, 1),
    nn.Sigmoid()
)

# Train the model
n_epochs = 200
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
model.train()
for epoch in range(n_epochs):
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# create one test tensor from the testset
X_test, y_test = default_collate(testset)
model.eval()
y_pred = model(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
print("Model accuracy: %.2f%%" % (acc*100))
```

---

### Chapter Summary / 章节总结

# Chapter 18 Summary / 第18章总结

## Theme / 主题: Chapter 18 / Chapter 18

This chapter contains **4 code files** demonstrating chapter 18.

本章包含 **4 个代码文件**，演示Chapter 18。

---
## Evolution / 演化路线

  1. `01_load.ipynb` — Load
  2. `03_dataloader.ipynb` — Dataloader
  3. `04_dataset.ipynb` — Dataset
  4. `06_training.ipynb` — Training

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 18) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 18）是机器学习流水线中的基础构建块。

---
