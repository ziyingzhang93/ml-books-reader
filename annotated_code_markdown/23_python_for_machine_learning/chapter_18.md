# Python ML
## Chapter 18

---

### Cv

# 03 — Cv / 03 Cv

**Chapter 18 — File 1 of 12 / 第18章 — 第1个文件（共12个）**

---

## Summary / 总结

This script demonstrates **evaluate a perceptron model on the dataset**.

本脚本演示 **evaluate a perceptron model on the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — evaluate a perceptron model on the dataset

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10,
                           n_redundant=0, random_state=1)
```

---
## Step 3 — define model

```python
model = Perceptron()
```

---
## Step 4 — define model evaluation method

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 5 — evaluate model

```python
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 6 — summarize result

```python
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate a perceptron model on the dataset 是机器学习中的常用技术。  
  *evaluate a perceptron model on the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cv / 03 Cv
# Complete Code / 完整代码
# ===============================

# evaluate a perceptron model on the dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10,
                           n_redundant=0, random_state=1)
# define model
model = Perceptron()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---

➡️ **Next / 下一步**: File 2 of 12

---

### Cv Keras

# 04 — Cv Keras / Keras

**Chapter 18 — File 2 of 12 / 第18章 — 第2个文件（共12个）**

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
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes
import numpy
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
	model.add(Dense(12, input_dim=8, activation='relu'))
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
numpy.random.seed(seed)
```

---
## Step 6 — load pima indians dataset

```python
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
dataset = numpy.loadtxt(URL, delimiter=",")
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
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
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
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cv Keras / Keras
# Complete Code / 完整代码
# ===============================

# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes
import numpy

# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
dataset = numpy.loadtxt(URL, delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

---

➡️ **Next / 下一步**: File 3 of 12

---

### Scope

# 07 — Scope / 07 Scope

**Chapter 18 — File 5 of 12 / 第18章 — 第5个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Scope**.

本脚本演示 **07 Scope**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
a = 1

def f(x):
    a = 2 * x
    return a

b = f(3)
print(a, b)
```

---
## Learning Notes / 学习笔记

- **概念**: Scope 是机器学习中的常用技术。  
  *Scope is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Scope / 07 Scope
# Complete Code / 完整代码
# ===============================

a = 1

def f(x):
    a = 2 * x
    return a

b = f(3)
print(a, b)
```

---

➡️ **Next / 下一步**: File 6 of 12

---

### Global

# 08 — Global / 08 Global

**Chapter 18 — File 6 of 12 / 第18章 — 第6个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Global**.

本脚本演示 **08 Global**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
a = 1

def f(x):
    global a
    a = 2 * x
    return a

b = f(3)
print(a, b)
```

---
## Learning Notes / 学习笔记

- **概念**: Global 是机器学习中的常用技术。  
  *Global is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Global / 08 Global
# Complete Code / 完整代码
# ===============================

a = 1

def f(x):
    global a
    a = 2 * x
    return a

b = f(3)
print(a, b)
```

---

➡️ **Next / 下一步**: File 7 of 12

---

### Nested Scope

# 09 — Nested Scope / 09 Nested Scope

**Chapter 18 — File 7 of 12 / 第18章 — 第7个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Nested Scope**.

本脚本演示 **09 Nested Scope**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
a = 1

def f(x):
    a = x
    def g(x):
        return a * x
    return g(3)

b = f(2)
print(b)
```

---
## Learning Notes / 学习笔记

- **概念**: Nested Scope 是机器学习中的常用技术。  
  *Nested Scope is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nested Scope / 09 Nested Scope
# Complete Code / 完整代码
# ===============================

a = 1

def f(x):
    a = x
    def g(x):
        return a * x
    return g(3)

b = f(2)
print(b)
```

---

➡️ **Next / 下一步**: File 8 of 12

---

### Random

# 12 — Random / 12 Random

**Chapter 18 — File 10 of 12 / 第18章 — 第10个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Random**.

本脚本演示 **12 Random**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import numpy as np

X = np.random.random((100,3))
print(type(X))
print(isinstance(X, np.ndarray))
```

---
## Learning Notes / 学习笔记

- **概念**: Random 是机器学习中的常用技术。  
  *Random is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Random / 12 Random
# Complete Code / 完整代码
# ===============================

import numpy as np

X = np.random.random((100,3))
print(type(X))
print(isinstance(X, np.ndarray))
```

---

➡️ **Next / 下一步**: File 11 of 12

---
