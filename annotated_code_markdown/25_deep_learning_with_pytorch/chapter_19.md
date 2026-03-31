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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
from skorch import NeuralNetBinaryClassifier
```

---
## Step 2 — Read data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]
```

---
## Step 3 — Binary encoding of labels

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)
```

---
## Step 4 — Convert to 2D PyTorch tensors

```python
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
```

---
## Step 5 — Define the model

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class SonarClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act1 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act2 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer3 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act3 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(60, 1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
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
# 训练模型 / Train the model
model.fit(X, y)
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
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.fit` | 训练模型 | Train the model |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sonar / 03 Sonar
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
from skorch import NeuralNetBinaryClassifier

# Read data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Binary encoding of labels
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define the model
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class SonarClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act1 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act2 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer3 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act3 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(60, 1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
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
# 训练模型 / Train the model
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
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skorch import NeuralNetBinaryClassifier
```

---
## Step 2 — Read data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]
```

---
## Step 3 — Binary encoding of labels

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)
```

---
## Step 4 — Convert to 2D PyTorch tensors

```python
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
```

---
## Step 5 — Define the model

```python
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class SonarClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act1 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act2 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer3 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act3 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(60, 1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
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
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, y, cv=kfold)
# 打印输出 / Print output
print(results)
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
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Kfold / 04 Kfold
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skorch import NeuralNetBinaryClassifier

# Read data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Binary encoding of labels
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define the model
# 定义PyTorch模型类（继承nn.Module） / Define PyTorch model class (inherits nn.Module)
class SonarClassifier(nn.Module):
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        # 调用父类初始化（必须） / Call parent class init (required)
        super().__init__()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer1 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act1 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer2 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act2 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.layer3 = nn.Linear(60, 60)
        # ReLU激活：负数→0，正数不变 / ReLU activation: negative→0, positive unchanged
        self.act3 = nn.ReLU()
        # 全连接层：y = xW + b / Fully connected layer: y = xW + b
        self.output = nn.Linear(60, 1)

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
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
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, y, cv=kfold)
# 打印输出 / Print output
print(results)
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Kfold



---

### Sklearn



---

### Pytorch

# 08 — Pytorch / PyTorch

**Chapter 19 — File 5 of 7 / 第19章 — 第5个文件（共7个）**

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
import torch.nn as nn

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
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
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
import torch.nn as nn

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
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Gridsearch



---

### Skorch

# 13 — Skorch / 13 Skorch

**Chapter 19 — File 7 of 7 / 第19章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Read data**.

本脚本演示 **Read data**。

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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline, FunctionTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skorch import NeuralNetBinaryClassifier
```

---
## Step 2 — Read data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]
```

---
## Step 3 — Binary encoding of labels

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)
```

---
## Step 4 — Convert to 2D PyTorch tensors

```python
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


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

model = NeuralNetBinaryClassifier(
    SonarClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.0001,
    max_epochs=150,
    batch_size=10,
    verbose=False
)

# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipe = Pipeline([
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
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

# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid_search = GridSearchCV(pipe, param_grid, scoring='accuracy', verbose=1, cv=3)
result = grid_search.fit(X, y)
# 打印输出 / Print output
print("Best: %f using %s" % (result.best_score_, result.best_params_))
means = result.cv_results_['mean_test_score']
stds = result.cv_results_['std_test_score']
params = result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
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
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `nn.Linear` | 全连接层 y=xW+b | Fully connected layer y=xW+b |
| `nn.Module` | 所有模型的基类，需定义 __init__ 和 forward | Base class for all models |
| `nn.ReLU` | 激活函数 f(x)=max(0,x) | Activation: f(x)=max(0,x) |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `torch.tensor` | 创建张量（多维数组） | Create tensor (multi-dimensional array) |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Skorch / 13 Skorch
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch.nn as nn
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline, FunctionTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skorch import NeuralNetBinaryClassifier

# Read data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Binary encoding of labels
# 将类别标签编码为数字 / Encode categorical labels to numbers
encoder = LabelEncoder()
encoder.fit(y)
# 用已拟合的模型转换数据 / Transform data with fitted model
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
# 转换为NumPy数组 / Convert to NumPy array
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


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

model = NeuralNetBinaryClassifier(
    SonarClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.0001,
    max_epochs=150,
    batch_size=10,
    verbose=False
)

# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipe = Pipeline([
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
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

# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid_search = GridSearchCV(pipe, param_grid, scoring='accuracy', verbose=1, cv=3)
result = grid_search.fit(X, y)
# 打印输出 / Print output
print("Best: %f using %s" % (result.best_score_, result.best_params_))
means = result.cv_results_['mean_test_score']
stds = result.cv_results_['std_test_score']
params = result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
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
