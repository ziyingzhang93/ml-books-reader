# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 19

---

### Sonar

# 03 — Sonar / 03 Sonar

**Chapter 19 — File 1 of 7 / 第19章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Read data**.

本脚本演示 **Read data**。

---
## Step 1 — Step 1

```python
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from skorch import NeuralNetBinaryClassifier
```

---
## Step 2 — Read data

```python
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]
```

---
## Step 3 — Binary encoding of labels

```python
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
```

---
## Step 4 — Convert to 2D PyTorch tensors

```python
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
```

---
## Step 5 — Define the model

```python
class SonarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.output(x)
        return x
```

---
## Step 6 — create the skorch wrapper

```python
model = NeuralNetBinaryClassifier(
    SonarClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.0001,
    max_epochs=150,
    batch_size=10
)
```

---
## Step 7 — run

```python
model.fit(X, y)
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
# Sonar / 03 Sonar
# Complete Code / 完整代码
# ===============================

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from skorch import NeuralNetBinaryClassifier

# Read data
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Binary encoding of labels
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define the model
class SonarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.output(x)
        return x

# create the skorch wrapper
model = NeuralNetBinaryClassifier(
    SonarClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.0001,
    max_epochs=150,
    batch_size=10
)

# run
model.fit(X, y)
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Kfold

# 04 — Kfold / 04 Kfold

**Chapter 19 — File 2 of 7 / 第19章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Read data**.

本脚本演示 **Read data**。

---
## Step 1 — Step 1

```python
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skorch import NeuralNetBinaryClassifier
```

---
## Step 2 — Read data

```python
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]
```

---
## Step 3 — Binary encoding of labels

```python
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
```

---
## Step 4 — Convert to 2D PyTorch tensors

```python
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
```

---
## Step 5 — Define the model

```python
class SonarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.output(x)
        return x


model = NeuralNetBinaryClassifier(
    SonarClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.0001,
    max_epochs=150,
    batch_size=10,
    verbose=False
)

kfold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results)
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
# Kfold / 04 Kfold
# Complete Code / 完整代码
# ===============================

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skorch import NeuralNetBinaryClassifier

# Read data
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Binary encoding of labels
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define the model
class SonarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.output(x)
        return x


model = NeuralNetBinaryClassifier(
    SonarClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.0001,
    max_epochs=150,
    batch_size=10,
    verbose=False
)

kfold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results)
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Kfold

# 06 — Kfold / 06 Kfold

**Chapter 19 — File 3 of 7 / 第19章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Read data**.

本脚本演示 **Read data**。

---
## Step 1 — Step 1

```python
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
```

---
## Step 2 — Read data

```python
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]
```

---
## Step 3 — Binary encoding of labels

```python
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
```

---
## Step 4 — Convert to 2D PyTorch tensors

```python
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
```

---
## Step 5 — Define the model

```python
class SonarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.output(x)
        return x
```

---
## Step 6 — create the skorch wrapper

```python
model = NeuralNetBinaryClassifier(
    SonarClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.0001,
    max_epochs=150,
    batch_size=10,
    verbose=False
)
```

---
## Step 7 — k-fold

```python
kfold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("mean = %.3f; std = %.3f" % (results.mean(), results.std()))
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
# Kfold / 06 Kfold
# Complete Code / 完整代码
# ===============================

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Read data
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Binary encoding of labels
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define the model
class SonarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.output(x)
        return x

# create the skorch wrapper
model = NeuralNetBinaryClassifier(
    SonarClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.0001,
    max_epochs=150,
    batch_size=10,
    verbose=False
)

# k-fold
kfold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("mean = %.3f; std = %.3f" % (results.mean(), results.std()))
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Sklearn

# 07 — Sklearn / 07 Sklearn

**Chapter 19 — File 4 of 7 / 第19章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **load dataset**.

本脚本演示 **load dataset**。

---
## Step 1 — Step 1

```python
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
```

---
## Step 2 — load dataset

```python
data = pd.read_csv("sonar.csv", header=None)
```

---
## Step 3 — split into input (X) and output (Y) variables, in numpy arrays

```python
X = data.iloc[:, 0:60].values
y = data.iloc[:, 60].values
```

---
## Step 4 — binary encoding of labels

```python
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
```

---
## Step 5 — create model

```python
model = MLPClassifier(hidden_layer_sizes=(60,60,60), activation='relu',
                      max_iter=150, batch_size=10, verbose=False)
```

---
## Step 6 — evaluate using 10-fold cross validation

```python
kfold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("mean = %.3f; std = %.3f" % (results.mean(), results.std()))
```

---
## Learning Notes / 学习笔记

- **概念**: load dataset 是机器学习中的常用技术。  
  *load dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sklearn / 07 Sklearn
# Complete Code / 完整代码
# ===============================

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# load dataset
data = pd.read_csv("sonar.csv", header=None)
# split into input (X) and output (Y) variables, in numpy arrays
X = data.iloc[:, 0:60].values
y = data.iloc[:, 60].values

# binary encoding of labels
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# create model
model = MLPClassifier(hidden_layer_sizes=(60,60,60), activation='relu',
                      max_iter=150, batch_size=10, verbose=False)

# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("mean = %.3f; std = %.3f" % (results.mean(), results.std()))
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Pytorch

# 08 — Pytorch / PyTorch

**Chapter 19 — File 5 of 7 / 第19章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Pytorch**.

本脚本演示 **PyTorch**。

---
## Step 1 — Step 1

```python
import torch.nn as nn

class SonarClassifier(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = []
        self.acts = []
        for i in range(n_layers):
            self.layers.append(nn.Linear(60, 60))
            self.acts.append(nn.ReLU())
            self.add_module(f"layer{i}", self.layers[-1])
            self.add_module(f"act{i}", self.acts[-1])
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
        x = self.output(x)
        return x
```

---
## Learning Notes / 学习笔记

- **概念**: Pytorch 是机器学习中的常用技术。  
  *Pytorch is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pytorch / PyTorch
# Complete Code / 完整代码
# ===============================

import torch.nn as nn

class SonarClassifier(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = []
        self.acts = []
        for i in range(n_layers):
            self.layers.append(nn.Linear(60, 60))
            self.acts.append(nn.ReLU())
            self.add_module(f"layer{i}", self.layers[-1])
            self.add_module(f"act{i}", self.acts[-1])
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
        x = self.output(x)
        return x
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Gridsearch

# 10 — Gridsearch / 10 Gridsearch

**Chapter 19 — File 6 of 7 / 第19章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Read data**.

本脚本演示 **Read data**。

---
## Step 1 — Step 1

```python
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import GridSearchCV
```

---
## Step 2 — Read data

```python
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]
```

---
## Step 3 — Binary encoding of labels

```python
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
```

---
## Step 4 — Convert to 2D PyTorch tensors

```python
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
```

---
## Step 5 — Define the model

```python
class SonarClassifier(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = []
        self.acts = []
        for i in range(n_layers):
            self.layers.append(nn.Linear(60, 60))
            self.acts.append(nn.ReLU())
            self.add_module(f"layer{i}", self.layers[-1])
            self.add_module(f"act{i}", self.acts[-1])
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
        x = self.output(x)
        return x

model = NeuralNetBinaryClassifier(
    SonarClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.0001,
    max_epochs=150,
    batch_size=10,
    verbose=False
)

param_grid = {
    'module__n_layers': [1, 3, 5],
    'lr': [0.1, 0.01, 0.001, 0.0001],
    'max_epochs': [100, 150],
}

grid_search = GridSearchCV(model, param_grid, scoring='accuracy', verbose=1, cv=3)
result = grid_search.fit(X, y)

print("Best: %f using %s" % (result.best_score_, result.best_params_))
means = result.cv_results_['mean_test_score']
stds = result.cv_results_['std_test_score']
params = result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
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
# Gridsearch / 10 Gridsearch
# Complete Code / 完整代码
# ===============================

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import GridSearchCV

# Read data
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Binary encoding of labels
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define the model
class SonarClassifier(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = []
        self.acts = []
        for i in range(n_layers):
            self.layers.append(nn.Linear(60, 60))
            self.acts.append(nn.ReLU())
            self.add_module(f"layer{i}", self.layers[-1])
            self.add_module(f"act{i}", self.acts[-1])
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
        x = self.output(x)
        return x

model = NeuralNetBinaryClassifier(
    SonarClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.0001,
    max_epochs=150,
    batch_size=10,
    verbose=False
)

param_grid = {
    'module__n_layers': [1, 3, 5],
    'lr': [0.1, 0.01, 0.001, 0.0001],
    'max_epochs': [100, 150],
}

grid_search = GridSearchCV(model, param_grid, scoring='accuracy', verbose=1, cv=3)
result = grid_search.fit(X, y)

print("Best: %f using %s" % (result.best_score_, result.best_params_))
means = result.cv_results_['mean_test_score']
stds = result.cv_results_['std_test_score']
params = result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Skorch

# 13 — Skorch / 13 Skorch

**Chapter 19 — File 7 of 7 / 第19章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Read data**.

本脚本演示 **Read data**。

---
## Step 1 — Step 1

```python
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skorch import NeuralNetBinaryClassifier
```

---
## Step 2 — Read data

```python
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]
```

---
## Step 3 — Binary encoding of labels

```python
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
```

---
## Step 4 — Convert to 2D PyTorch tensors

```python
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


class SonarClassifier(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = []
        self.acts = []
        for i in range(n_layers):
            self.layers.append(nn.Linear(60, 60))
            self.acts.append(nn.ReLU())
            self.add_module(f"layer{i}", self.layers[-1])
            self.add_module(f"act{i}", self.acts[-1])
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
        x = self.output(x)
        return x

model = NeuralNetBinaryClassifier(
    SonarClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.0001,
    max_epochs=150,
    batch_size=10,
    verbose=False
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('float32', FunctionTransformer(func=lambda X: torch.tensor(X, dtype=torch.float32),
                                    validate=False)),
    ('sonarmodel', model.initialize()),
])

param_grid = {
    'sonarmodel__module__n_layers': [1, 3, 5],
    'sonarmodel__lr': [0.1, 0.01, 0.001, 0.0001],
    'sonarmodel__max_epochs': [100, 150],
}

grid_search = GridSearchCV(pipe, param_grid, scoring='accuracy', verbose=1, cv=3)
result = grid_search.fit(X, y)
print("Best: %f using %s" % (result.best_score_, result.best_params_))
means = result.cv_results_['mean_test_score']
stds = result.cv_results_['std_test_score']
params = result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
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
# Skorch / 13 Skorch
# Complete Code / 完整代码
# ===============================

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skorch import NeuralNetBinaryClassifier

# Read data
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Binary encoding of labels
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


class SonarClassifier(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = []
        self.acts = []
        for i in range(n_layers):
            self.layers.append(nn.Linear(60, 60))
            self.acts.append(nn.ReLU())
            self.add_module(f"layer{i}", self.layers[-1])
            self.add_module(f"act{i}", self.acts[-1])
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
        x = self.output(x)
        return x

model = NeuralNetBinaryClassifier(
    SonarClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.0001,
    max_epochs=150,
    batch_size=10,
    verbose=False
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('float32', FunctionTransformer(func=lambda X: torch.tensor(X, dtype=torch.float32),
                                    validate=False)),
    ('sonarmodel', model.initialize()),
])

param_grid = {
    'sonarmodel__module__n_layers': [1, 3, 5],
    'sonarmodel__lr': [0.1, 0.01, 0.001, 0.0001],
    'sonarmodel__max_epochs': [100, 150],
}

grid_search = GridSearchCV(pipe, param_grid, scoring='accuracy', verbose=1, cv=3)
result = grid_search.fit(X, y)
print("Best: %f using %s" % (result.best_score_, result.best_params_))
means = result.cv_results_['mean_test_score']
stds = result.cv_results_['std_test_score']
params = result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---

### Chapter Summary / 章节总结

# Chapter 19 Summary / 第19章总结

## Theme / 主题: Chapter 19 / Chapter 19

This chapter contains **7 code files** demonstrating chapter 19.

本章包含 **7 个代码文件**，演示Chapter 19。

---
## Evolution / 演化路线

  1. `03_sonar.ipynb` — Sonar
  2. `04_kfold.ipynb` — Kfold
  3. `06_kfold.ipynb` — Kfold
  4. `07_sklearn.ipynb` — Sklearn
  5. `08_pytorch.ipynb` — Pytorch
  6. `10_gridsearch.ipynb` — Gridsearch
  7. `13_skorch.ipynb` — Skorch

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 19) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 19）是机器学习流水线中的基础构建块。

---
