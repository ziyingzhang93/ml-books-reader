# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 20

---

### Skorch



---

### Print

# 05 — Print / 05 Print

**Chapter 20 — File 2 of 10 / 第20章 — 第2个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Print**.

本脚本演示 **05 Print**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
from skorch import NeuralNetClassifier

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class SonarClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, n_layers=3):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.layers = []
        self.acts = []
        # 生成整数序列 / Generate integer sequence
        for i in range(n_layers):
            # 全连接层：y = xW + b / Fully connected layer: y = xW + b
            self.layers.append(nn.Linear(60, 60))
            # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
            self.acts.append(nn.ReLU())
            self.add_module(f"layer{i}", self.layers[-1])
            self.add_module(f"act{i}", self.acts[-1])
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(60, 1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # 将多个序列配对 / Pair multiple sequences
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
        x = self.output(x)
        return x

model = NeuralNetClassifier(
    module=SonarClassifier,
    max_epochs=150,
    batch_size=10,
    module__n_layers=2
)
# 打印输出 / Print output
print(model.initialize())
```

---
## Learning Notes / 学习笔记

- **概念**: Print 是机器学习中的常用技术。  
  *Print is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Print / 05 Print
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
from skorch import NeuralNetClassifier

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class SonarClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, n_layers=3):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        self.layers = []
        self.acts = []
        # 生成整数序列 / Generate integer sequence
        for i in range(n_layers):
            # 全连接层：y = xW + b / Fully connected layer: y = xW + b
            self.layers.append(nn.Linear(60, 60))
            # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
            self.acts.append(nn.ReLU())
            self.add_module(f"layer{i}", self.layers[-1])
            self.add_module(f"act{i}", self.acts[-1])
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(60, 1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # 将多个序列配对 / Pair multiple sequences
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
        x = self.output(x)
        return x

model = NeuralNetClassifier(
    module=SonarClassifier,
    max_epochs=150,
    batch_size=10,
    module__n_layers=2
)
# 打印输出 / Print output
print(model.initialize())
```

---

➡️ **Next / 下一步**: File 3 of 10

---

### Minibatch



---

### Optimizer



---

### Rate



---

### Init

# 11 — Init / 11 Init

**Chapter 20 — File 6 of 10 / 第20章 — 第6个文件（共10个）**

---

## Summary / 总结

This script demonstrates **PyTorch classifier**.

本脚本演示 **PyTorch classifier**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
```

---
## Step 2 — PyTorch classifier

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class PimaClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, weight_init=nn.init.xavier_uniform_):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer = nn.Linear(8, 12)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(12, 1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.prob = nn.Sigmoid()
```

---
## Step 3 — manually init weights

```python
weight_init(self.layer.weight)
        weight_init(self.output.weight)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x
```

---
## Learning Notes / 学习笔记

- **概念**: PyTorch classifier 是机器学习中的常用技术。  
  *PyTorch classifier is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Init / 11 Init
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn

# PyTorch classifier
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class PimaClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, weight_init=nn.init.xavier_uniform_):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer = nn.Linear(8, 12)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(12, 1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.prob = nn.Sigmoid()
        # manually init weights
        weight_init(self.layer.weight)
        weight_init(self.output.weight)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x
```

---

➡️ **Next / 下一步**: File 7 of 10

---

### Init

# 12 — Init / 12 Init

**Chapter 20 — File 7 of 10 / 第20章 — 第7个文件（共10个）**

---

## Summary / 总结

This script demonstrates **load the dataset, split into input (X) and output (y) variables**.

本脚本演示 **load the dataset, split into input (X) and output (y) variables**。

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.init as init
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
from skorch import NeuralNetClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
```

---
## Step 2 — load the dataset, split into input (X) and output (y) variables

```python
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 3 — PyTorch classifier

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class PimaClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, weight_init=init.xavier_uniform_):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer = nn.Linear(8, 12)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(12, 1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.prob = nn.Sigmoid()
```

---
## Step 4 — manually init weights

```python
weight_init(self.layer.weight)
        weight_init(self.output.weight)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x
```

---
## Step 5 — create model with skorch

```python
model = NeuralNetClassifier(
    PimaClassifier,
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    criterion=nn.BCELoss,
    optimizer=optim.Adamax,
    max_epochs=100,
    batch_size=10,
    verbose=False
)
```

---
## Step 6 — define the grid search parameters

```python
param_grid = {
    'module__weight_init': [init.uniform_, init.normal_, init.zeros_,
                           init.xavier_normal_, init.xavier_uniform_,
                           init.kaiming_normal_, init.kaiming_uniform_]
}
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)
```

---
## Step 7 — summarize results

```python
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---
## Learning Notes / 学习笔记

- **概念**: load the dataset, split into input (X) and output (y) variables 是机器学习中的常用技术。  
  *load the dataset, split into input (X) and output (y) variables is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Init / 12 Init
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.init as init
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
from skorch import NeuralNetClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# PyTorch classifier
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class PimaClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, weight_init=init.xavier_uniform_):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer = nn.Linear(8, 12)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(12, 1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.prob = nn.Sigmoid()
        # manually init weights
        weight_init(self.layer.weight)
        weight_init(self.output.weight)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x

# create model with skorch
model = NeuralNetClassifier(
    PimaClassifier,
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    criterion=nn.BCELoss,
    optimizer=optim.Adamax,
    max_epochs=100,
    batch_size=10,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'module__weight_init': [init.uniform_, init.normal_, init.zeros_,
                           init.xavier_normal_, init.xavier_uniform_,
                           init.kaiming_normal_, init.kaiming_uniform_]
}
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# summarize results
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---

➡️ **Next / 下一步**: File 8 of 10

---

### Activation

# 13 — Activation / 13 Activation

**Chapter 20 — File 8 of 10 / 第20章 — 第8个文件（共10个）**

---

## Summary / 总结

This script demonstrates **load the dataset, split into input (X) and output (y) variables**.

本脚本演示 **load the dataset, split into input (X) and output (y) variables**。

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.init as init
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
from skorch import NeuralNetClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
```

---
## Step 2 — load the dataset, split into input (X) and output (y) variables

```python
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 3 — PyTorch classifier

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class PimaClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, activation=nn.ReLU):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer = nn.Linear(8, 12)
        self.act = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(12, 1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.prob = nn.Sigmoid()
```

---
## Step 4 — manually init weights

```python
init.kaiming_uniform_(self.layer.weight)
        init.kaiming_uniform_(self.output.weight)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x
```

---
## Step 5 — create model with skorch

```python
model = NeuralNetClassifier(
    PimaClassifier,
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    criterion=nn.BCELoss,
    optimizer=optim.Adamax,
    max_epochs=100,
    batch_size=10,
    verbose=False
)
```

---
## Step 6 — define the grid search parameters

```python
param_grid = {
    'module__activation': [nn.Identity, nn.ReLU, nn.ELU, nn.ReLU6,
                           nn.GELU, nn.Softplus, nn.Softsign, nn.Tanh,
                           nn.Sigmoid, nn.Hardsigmoid]
}
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)
```

---
## Step 7 — summarize results

```python
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---
## Learning Notes / 学习笔记

- **概念**: load the dataset, split into input (X) and output (y) variables 是机器学习中的常用技术。  
  *load the dataset, split into input (X) and output (y) variables is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Activation / 13 Activation
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.init as init
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
from skorch import NeuralNetClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# PyTorch classifier
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class PimaClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, activation=nn.ReLU):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer = nn.Linear(8, 12)
        self.act = activation()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(12, 1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.prob = nn.Sigmoid()
        # manually init weights
        init.kaiming_uniform_(self.layer.weight)
        init.kaiming_uniform_(self.output.weight)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x

# create model with skorch
model = NeuralNetClassifier(
    PimaClassifier,
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    criterion=nn.BCELoss,
    optimizer=optim.Adamax,
    max_epochs=100,
    batch_size=10,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'module__activation': [nn.Identity, nn.ReLU, nn.ELU, nn.ReLU6,
                           nn.GELU, nn.Softplus, nn.Softsign, nn.Tanh,
                           nn.Sigmoid, nn.Hardsigmoid]
}
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# summarize results
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---

➡️ **Next / 下一步**: File 9 of 10

---

### Dropout

# 14 — Dropout / 随机失活

**Chapter 20 — File 9 of 10 / 第20章 — 第9个文件（共10个）**

---

## Summary / 总结

This script demonstrates **load the dataset, split into input (X) and output (y) variables**.

本脚本演示 **load the dataset, split into input (X) and output (y) variables**。

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.init as init
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
from skorch import NeuralNetClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
```

---
## Step 2 — load the dataset, split into input (X) and output (y) variables

```python
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
```

---
## Step 3 — PyTorch classifier

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class PimaClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dropout_rate=0.5, weight_constraint=1.0):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer = nn.Linear(8, 12)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(12, 1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.prob = nn.Sigmoid()
        self.weight_constraint = weight_constraint
```

---
## Step 4 — manually init weights

```python
init.kaiming_uniform_(self.layer.weight)
        init.kaiming_uniform_(self.output.weight)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
```

---
## Step 5 — maxnorm weight before actual forward pass

```python
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
            norm = self.layer.weight.norm(2, dim=0, keepdim=True) \
                                    .clamp(min=self.weight_constraint / 2)
            desired = torch.clamp(norm, max=self.weight_constraint)
            self.layer.weight *= (desired / norm)
```

---
## Step 6 — actual forward pass

```python
x = self.act(self.layer(x))
        x = self.dropout(x)
        x = self.prob(self.output(x))
        return x
```

---
## Step 7 — create model with skorch

```python
model = NeuralNetClassifier(
    PimaClassifier,
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    criterion=nn.BCELoss,
    optimizer=optim.Adamax,
    max_epochs=100,
    batch_size=10,
    verbose=False
)
```

---
## Step 8 — define the grid search parameters

```python
param_grid = {
    'module__weight_constraint': [1.0, 2.0, 3.0, 4.0, 5.0],
    'module__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)
```

---
## Step 9 — summarize results

```python
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---
## Learning Notes / 学习笔记

- **概念**: load the dataset, split into input (X) and output (y) variables 是机器学习中的常用技术。  
  *load the dataset, split into input (X) and output (y) variables is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Dropout` | 随机丢弃神经元防止过拟合 | Randomly drop neurons to prevent overfitting |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

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
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.init as init
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
from skorch import NeuralNetClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# PyTorch classifier
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class PimaClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, dropout_rate=0.5, weight_constraint=1.0):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer = nn.Linear(8, 12)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(12, 1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.prob = nn.Sigmoid()
        self.weight_constraint = weight_constraint
        # manually init weights
        init.kaiming_uniform_(self.layer.weight)
        init.kaiming_uniform_(self.output.weight)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # maxnorm weight before actual forward pass
        # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
        with torch.no_grad():
            norm = self.layer.weight.norm(2, dim=0, keepdim=True) \
                                    .clamp(min=self.weight_constraint / 2)
            desired = torch.clamp(norm, max=self.weight_constraint)
            self.layer.weight *= (desired / norm)
        # actual forward pass
        x = self.act(self.layer(x))
        x = self.dropout(x)
        x = self.prob(self.output(x))
        return x

# create model with skorch
model = NeuralNetClassifier(
    PimaClassifier,
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    criterion=nn.BCELoss,
    optimizer=optim.Adamax,
    max_epochs=100,
    batch_size=10,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'module__weight_constraint': [1.0, 2.0, 3.0, 4.0, 5.0],
    'module__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# summarize results
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---

➡️ **Next / 下一步**: File 10 of 10

---

### Hidden

# 15 — Hidden / 15 Hidden

**Chapter 20 — File 10 of 10 / 第20章 — 第10个文件（共10个）**

---

## Summary / 总结

This script demonstrates **load the dataset, split into input (X) and output (y) variables**.

本脚本演示 **load the dataset, split into input (X) and output (y) variables**。

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.init as init
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
from skorch import NeuralNetClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
```

---
## Step 2 — load the dataset, split into input (X) and output (y) variables

```python
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class PimaClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, n_neurons=12):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer = nn.Linear(8, n_neurons)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.1)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(n_neurons, 1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.prob = nn.Sigmoid()
        self.weight_constraint = 2.0
```

---
## Step 3 — manually init weights

```python
init.kaiming_uniform_(self.layer.weight)
        init.kaiming_uniform_(self.output.weight)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
```

---
## Step 4 — maxnorm weight before actual forward pass

```python
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
            norm = self.layer.weight.norm(2, dim=0, keepdim=True) \
                                    .clamp(min=self.weight_constraint / 2)
            desired = torch.clamp(norm, max=self.weight_constraint)
            self.layer.weight *= (desired / norm)
```

---
## Step 5 — actual forward pass

```python
x = self.act(self.layer(x))
        x = self.dropout(x)
        x = self.prob(self.output(x))
        return x
```

---
## Step 6 — create model with skorch

```python
model = NeuralNetClassifier(
    PimaClassifier,
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    criterion=nn.BCELoss,
    optimizer=optim.Adamax,
    max_epochs=100,
    batch_size=10,
    verbose=False
)
```

---
## Step 7 — define the grid search parameters

```python
param_grid = {
    'module__n_neurons': [1, 5, 10, 15, 20, 25, 30]
}
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)
```

---
## Step 8 — summarize results

```python
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---
## Learning Notes / 学习笔记

- **概念**: load the dataset, split into input (X) and output (y) variables 是机器学习中的常用技术。  
  *load the dataset, split into input (X) and output (y) variables is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Dropout` | 随机丢弃神经元防止过拟合 | Randomly drop neurons to prevent overfitting |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `nn.Sigmoid` | 激活函数，输出0-1之间 | Activation: output between 0-1 |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hidden / 15 Hidden
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn.init as init
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.optim as optim
from skorch import NeuralNetClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class PimaClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, n_neurons=12):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer = nn.Linear(8, n_neurons)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act = nn.ReLU()
        # 随机丢弃：训练时随机关闭神经元防过拟合 / Dropout: randomly disable neurons to prevent overfitting
        self.dropout = nn.Dropout(0.1)
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(n_neurons, 1)
        # Sigmoid激活：压缩到(0,1)范围 / Sigmoid: compress to (0,1) range
        self.prob = nn.Sigmoid()
        self.weight_constraint = 2.0
        # manually init weights
        init.kaiming_uniform_(self.layer.weight)
        init.kaiming_uniform_(self.output.weight)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        # maxnorm weight before actual forward pass
        # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
        with torch.no_grad():
            norm = self.layer.weight.norm(2, dim=0, keepdim=True) \
                                    .clamp(min=self.weight_constraint / 2)
            desired = torch.clamp(norm, max=self.weight_constraint)
            self.layer.weight *= (desired / norm)
        # actual forward pass
        x = self.act(self.layer(x))
        x = self.dropout(x)
        x = self.prob(self.output(x))
        return x

# create model with skorch
model = NeuralNetClassifier(
    PimaClassifier,
    # 二元交叉熵损失：二分类任务 / BCE: binary classification loss
    criterion=nn.BCELoss,
    optimizer=optim.Adamax,
    max_epochs=100,
    batch_size=10,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'module__n_neurons': [1, 5, 10, 15, 20, 25, 30]
}
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# summarize results
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---

### Chapter Summary / 章节总结

# Chapter 20 Summary / 第20章总结

## Theme / 主题: Chapter 20 / Chapter 20

This chapter contains **10 code files** demonstrating chapter 20.

本章包含 **10 个代码文件**，演示Chapter 20。

---
## Evolution / 演化路线

  1. `04_skorch.ipynb` — Skorch
  2. `05_print.ipynb` — Print
  3. `07_minibatch.ipynb` — Minibatch
  4. `08_optimizer.ipynb` — Optimizer
  5. `10_rate.ipynb` — Rate
  6. `11_init.ipynb` — Init
  7. `12_init.ipynb` — Init
  8. `13_activation.ipynb` — Activation
  9. `14_dropout.ipynb` — Dropout
  10. `15_hidden.ipynb` — Hidden

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 20) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 20）是机器学习流水线中的基础构建块。

---
