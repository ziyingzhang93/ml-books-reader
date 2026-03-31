# Python 深度学习 / Deep Learning with Python
## Chapter 08

---

### Validation Split

# 01 — Validation Split / 01 Validation Split

**Chapter 08 — File 1 of 3 / 第08章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **MLP with automatic validation set**.

本脚本演示 **MLP with automatic validation set**。

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
## Step 1 — MLP with automatic validation set

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
```

---
## Step 2 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
numpy.random.seed(7)
```

---
## Step 3 — load pima indians dataset

```python
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
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
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(8,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
```

---
## Step 6 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 7 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)
```

---
## Learning Notes / 学习笔记

- **概念**: MLP with automatic validation set 是机器学习中的常用技术。  
  *MLP with automatic validation set is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Validation Split / 01 Validation Split
# Complete Code / 完整代码
# ===============================

# MLP with automatic validation set
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(8,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# Compile model
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
# 训练模型 / Train the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Validation Data

# 02 — Validation Data / 02 Validation Data

**Chapter 08 — File 2 of 3 / 第08章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **MLP with manual validation set**.

本脚本演示 **MLP with manual validation set**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
## Step 1 — MLP with manual validation set

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
```

---
## Step 2 — fix random seed for reproducibility

```python
seed = 7
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
numpy.random.seed(seed)
```

---
## Step 3 — load pima indians dataset

```python
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
```

---
## Step 4 — split into input (X) and output (Y) variables

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 5 — split into 67% for train and 33% for test

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.33, random_state=seed)
```

---
## Step 6 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(8,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
```

---
## Step 7 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 8 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)
```

---
## Learning Notes / 学习笔记

- **概念**: MLP with manual validation set 是机器学习中的常用技术。  
  *MLP with manual validation set is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Validation Data / 02 Validation Data
# Complete Code / 完整代码
# ===============================

# MLP with manual validation set
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# fix random seed for reproducibility
seed = 7
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# split into 67% for train and 33% for test
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.33, random_state=seed)
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(8,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
# Compile model
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Scikit Learn

# 03 — Scikit Learn / 03 Scikit Learn

**Chapter 08 — File 3 of 3 / 第08章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **MLP for Pima Indians Dataset with 10-fold cross validation**.

本脚本演示 **MLP for Pima Indians Dataset with 10-fold cross validation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — MLP for Pima Indians Dataset with 10-fold cross validation

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
```

---
## Step 2 — fix random seed for reproducibility

```python
seed = 7
# 生成随机数 / Generate random numbers
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
## Step 5 — define 10-fold cross validation test harness

```python
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
```

---
## Step 6 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(8, activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, activation='sigmoid'))
```

---
## Step 7 — Compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 8 — Fit the model

```python
# 训练模型 / Train the model
model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
```

---
## Step 9 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
scores = model.evaluate(X[test], Y[test], verbose=0)
    # 打印输出 / Print output
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # 添加元素到列表末尾 / Append element to list end
    cvscores.append(scores[1] * 100)

# 计算均值 / Calculate mean
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
```

---
## Learning Notes / 学习笔记

- **概念**: MLP for Pima Indians Dataset with 10-fold cross validation 是机器学习中的常用技术。  
  *MLP for Pima Indians Dataset with 10-fold cross validation is a common technique in machine learning.*

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
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `np.mean` | 计算均值 | Calculate mean |
| `np.random` | 随机数生成 | Random number generation |
| `np.std` | 计算标准差 | Calculate standard deviation |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Scikit Learn / 03 Scikit Learn
# Complete Code / 完整代码
# ===============================

# MLP for Pima Indians Dataset with 10-fold cross validation
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# fix random seed for reproducibility
seed = 7
# 生成随机数 / Generate random numbers
np.random.seed(seed)
# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
    # create model
    # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
    model = Sequential()
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(8, activation='relu'))
    # 向模型添加一层 / Add a layer to the model
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    # 训练模型 / Train the model
    model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
    # evaluate the model
    # 评估模型在测试集上的表现 / Evaluate model on test set
    scores = model.evaluate(X[test], Y[test], verbose=0)
    # 打印输出 / Print output
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # 添加元素到列表末尾 / Append element to list end
    cvscores.append(scores[1] * 100)

# 计算均值 / Calculate mean
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
```

---

### Chapter Summary / 章节总结

# Chapter 08 Summary / 第08章总结

## Theme / 主题: Chapter 08 / Chapter 08

This chapter contains **3 code files** demonstrating chapter 08.

本章包含 **3 个代码文件**，演示Chapter 08。

---
## Evolution / 演化路线

  1. `01_validation_split.ipynb` — Validation Split
  2. `02_validation_data.ipynb` — Validation Data
  3. `03_scikit_learn.ipynb` — Scikit Learn

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 08) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 08）是机器学习流水线中的基础构建块。

---
