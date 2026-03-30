# Python深度学习
## Chapter 13

---

### Kfold Eval

# 02 — Kfold Eval / 模型评估

**Chapter 13 — File 1 of 3 / 第13章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **MLP for Pima Indians Dataset with 10-fold cross validation via sklearn**.

本脚本演示 **MLP for Pima Indians Dataset with 10-fold cross validation via sklearn**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — MLP for Pima Indians Dataset with 10-fold cross validation via sklearn

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
```

---
## Step 2 — Function to create model, required for KerasClassifier

```python
def create_model():
```

---
## Step 3 — create model

```python
model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
```

---
## Step 4 — Compile model

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

---
## Step 5 — fix random seed for reproducibility

```python
seed = 7
np.random.seed(seed)
```

---
## Step 6 — load pima indians dataset

```python
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
```

---
## Step 7 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 8 — create model

```python
model = KerasClassifier(model=create_model, epochs=150, batch_size=10, verbose=0)
```

---
## Step 9 — evaluate using 10-fold cross validation

```python
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: MLP for Pima Indians Dataset with 10-fold cross validation via sklearn 是机器学习中的常用技术。  
  *MLP for Pima Indians Dataset with 10-fold cross validation via sklearn is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Kfold Eval / 模型评估
# Complete Code / 完整代码
# ===============================

# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(model=create_model, epochs=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Sklearn

# 03 — Sklearn / 03 Sklearn

**Chapter 13 — File 2 of 3 / 第13章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **MLP for Pima Indians Dataset with 10-fold cross validation via sklearn**.

本脚本演示 **MLP for Pima Indians Dataset with 10-fold cross validation via sklearn**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — MLP for Pima Indians Dataset with 10-fold cross validation via sklearn

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np
```

---
## Step 2 — fix random seed for reproducibility

```python
seed = 7
np.random.seed(seed)
```

---
## Step 3 — load pima indians dataset

```python
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
```

---
## Step 4 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 5 — create model

```python
model = MLPClassifier(hidden_layer_sizes=(12,8), activation='relu',
                      max_iter=150, batch_size=10, verbose=False)
```

---
## Step 6 — evaluate using 10-fold cross validation

```python
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: MLP for Pima Indians Dataset with 10-fold cross validation via sklearn 是机器学习中的常用技术。  
  *MLP for Pima Indians Dataset with 10-fold cross validation via sklearn is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sklearn / 03 Sklearn
# Complete Code / 完整代码
# ===============================

# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = MLPClassifier(hidden_layer_sizes=(12,8), activation='relu',
                      max_iter=150, batch_size=10, verbose=False)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Chapter Summary

# Chapter 13 Summary / 第13章总结

## Theme / 主题: Chapter 13 / Chapter 13

This chapter contains **3 code files** demonstrating chapter 13.

本章包含 **3 个代码文件**，演示Chapter 13。

---
## Evolution / 演化路线

  1. `02_kfold_eval.ipynb` — Kfold Eval
  2. `03_sklearn.ipynb` — Sklearn
  3. `04_gridsearch.ipynb` — Gridsearch

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 13) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 13）是机器学习流水线中的基础构建块。

---
