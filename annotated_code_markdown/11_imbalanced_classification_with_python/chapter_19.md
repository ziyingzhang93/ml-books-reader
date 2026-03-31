# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 19

---

### Dataset

# 01 — Dataset / 01 Dataset

**Chapter 19 — File 1 of 3 / 第19章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Generate and plot a synthetic imbalanced classification dataset**.

本脚本演示 **Generate and plot a synthetic imbalanced classification dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — Generate and plot a synthetic imbalanced classification dataset

```python
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=4)
```

---
## Step 3 — summarize class distribution

```python
counter = Counter(y)
# 打印输出 / Print output
print(counter)
```

---
## Step 4 — scatter plot of examples by class label

```python
# 获取字典的键值对 / Get dict key-value pairs
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Generate and plot a synthetic imbalanced classification dataset 是机器学习中的常用技术。  
  *Generate and plot a synthetic imbalanced classification dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dataset / 01 Dataset
# Complete Code / 完整代码
# ===============================

# Generate and plot a synthetic imbalanced classification dataset
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=4)
# summarize class distribution
counter = Counter(y)
# 打印输出 / Print output
print(counter)
# scatter plot of examples by class label
# 获取字典的键值对 / Get dict key-value pairs
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Neural Net

# 02 — Neural Net / 神经网络

**Chapter 19 — File 2 of 3 / 第19章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **standard neural network on an imbalanced classification dataset**.

本脚本演示 **standard neural network on an imbalanced classification dataset**。

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
## Step 1 — standard neural network on an imbalanced classification dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import roc_auc_score
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
```

---
## Step 2 — prepare train and test dataset

```python
def prepare_data():
```

---
## Step 3 — generate 2d classification dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=4)
```

---
## Step 4 — split into train and test

```python
n_train = 5000
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy
```

---
## Step 5 — define the neural network model

```python
def define_model(n_input):
```

---
## Step 6 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 7 — define first hidden layer and visible layer

```python
# 向模型添加一层 / Add a layer to the model
model.add(Dense(10, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
```

---
## Step 8 — define output layer

```python
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
```

---
## Step 9 — define loss and optimizer

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='sgd')
	return model
```

---
## Step 10 — prepare dataset

```python
trainX, trainy, testX, testy = prepare_data()
```

---
## Step 11 — define the model

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_input = trainX.shape[1]
model = define_model(n_input)
```

---
## Step 12 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainy, epochs=100, verbose=0)
```

---
## Step 13 — make predictions on the test dataset

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(testX)
```

---
## Step 14 — evaluate the ROC AUC of the predictions

```python
# 计算ROC-AUC分数（分类器好坏） / ROC-AUC score (classifier quality)
score = roc_auc_score(testy, yhat)
# 打印输出 / Print output
print('ROC AUC: %.3f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: standard neural network on an imbalanced classification dataset 是机器学习中的常用技术。  
  *standard neural network on an imbalanced classification dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Neural Net / 神经网络
# Complete Code / 完整代码
# ===============================

# standard neural network on an imbalanced classification dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import roc_auc_score
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential

# prepare train and test dataset
def prepare_data():
	# generate 2d classification dataset
	X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=4)
	# split into train and test
	n_train = 5000
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy

# define the neural network model
def define_model(n_input):
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# define first hidden layer and visible layer
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(10, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
	# define output layer
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# define loss and optimizer
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer='sgd')
	return model

# prepare dataset
trainX, trainy, testX, testy = prepare_data()
# define the model
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_input = trainX.shape[1]
model = define_model(n_input)
# fit model
# 训练模型 / Train the model
model.fit(trainX, trainy, epochs=100, verbose=0)
# make predictions on the test dataset
# 用模型做预测 / Make predictions with model
yhat = model.predict(testX)
# evaluate the ROC AUC of the predictions
# 计算ROC-AUC分数（分类器好坏） / ROC-AUC score (classifier quality)
score = roc_auc_score(testy, yhat)
# 打印输出 / Print output
print('ROC AUC: %.3f' % score)
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Balanced Neural Net

# 03 — Balanced Neural Net / 神经网络

**Chapter 19 — File 3 of 3 / 第19章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **class weighted neural network on an imbalanced classification dataset**.

本脚本演示 **class weighted neural network on an imbalanced classification dataset**。

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
## Step 1 — class weighted neural network on an imbalanced classification dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import roc_auc_score
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
```

---
## Step 2 — prepare train and test dataset

```python
def prepare_data():
```

---
## Step 3 — generate 2d classification dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=4)
```

---
## Step 4 — split into train and test

```python
n_train = 5000
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy
```

---
## Step 5 — define the neural network model

```python
def define_model(n_input):
```

---
## Step 6 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 7 — define first hidden layer and visible layer

```python
# 向模型添加一层 / Add a layer to the model
model.add(Dense(10, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
```

---
## Step 8 — define output layer

```python
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
```

---
## Step 9 — define loss and optimizer

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='sgd')
	return model
```

---
## Step 10 — prepare dataset

```python
trainX, trainy, testX, testy = prepare_data()
```

---
## Step 11 — get the model

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_input = trainX.shape[1]
model = define_model(n_input)
```

---
## Step 12 — fit model

```python
weights = {0:1, 1:100}
# 训练模型 / Train the model
history = model.fit(trainX, trainy, class_weight=weights, epochs=100, verbose=0)
```

---
## Step 13 — evaluate model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(testX)
# 计算ROC-AUC分数（分类器好坏） / ROC-AUC score (classifier quality)
score = roc_auc_score(testy, yhat)
# 打印输出 / Print output
print('ROC AUC: %.3f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: class weighted neural network on an imbalanced classification dataset 是机器学习中的常用技术。  
  *class weighted neural network on an imbalanced classification dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Balanced Neural Net / 神经网络
# Complete Code / 完整代码
# ===============================

# class weighted neural network on an imbalanced classification dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import roc_auc_score
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential

# prepare train and test dataset
def prepare_data():
	# generate 2d classification dataset
	X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=4)
	# split into train and test
	n_train = 5000
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy

# define the neural network model
def define_model(n_input):
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# define first hidden layer and visible layer
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(10, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
	# define output layer
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# define loss and optimizer
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer='sgd')
	return model

# prepare dataset
trainX, trainy, testX, testy = prepare_data()
# get the model
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_input = trainX.shape[1]
model = define_model(n_input)
# fit model
weights = {0:1, 1:100}
# 训练模型 / Train the model
history = model.fit(trainX, trainy, class_weight=weights, epochs=100, verbose=0)
# evaluate model
# 用模型做预测 / Make predictions with model
yhat = model.predict(testX)
# 计算ROC-AUC分数（分类器好坏） / ROC-AUC score (classifier quality)
score = roc_auc_score(testy, yhat)
# 打印输出 / Print output
print('ROC AUC: %.3f' % score)
```

---

### Chapter Summary / 章节总结

# Chapter 19 Summary / 第19章总结

## Theme / 主题: Chapter 19 / Chapter 19

This chapter contains **3 code files** demonstrating chapter 19.

本章包含 **3 个代码文件**，演示Chapter 19。

---
## Evolution / 演化路线

  1. `01_dataset.ipynb` — Dataset
  2. `02_neural_net.ipynb` — Neural Net
  3. `03_balanced_neural_net.ipynb` — Balanced Neural Net

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 19) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 19）是机器学习流水线中的基础构建块。

---
