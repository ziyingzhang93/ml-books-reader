# 优化深度学习 / Better Deep Learning
## Chapter 20

---

### Problem

# 01 — Problem / 01 Problem

**Chapter 20 — File 1 of 5 / 第20章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **scatter plot of blobs dataset**.

本脚本演示 **scatter plot of blobs dataset**。

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
## Step 1 — scatter plot of blobs dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
```

---
## Step 2 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=500, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 3 — scatter plot for each class value

```python
# 生成整数序列 / Generate integer sequence
for class_value in range(3):
```

---
## Step 4 — select indices of points with the class label

```python
row_ix = where(y == class_value)
```

---
## Step 5 — scatter plot for points with a different color

```python
pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
```

---
## Step 6 — show plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: scatter plot of blobs dataset 是机器学习中的常用技术。  
  *scatter plot of blobs dataset is a common technique in machine learning.*

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
# Problem / 01 Problem
# Complete Code / 完整代码
# ===============================

# scatter plot of blobs dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# generate 2d classification dataset
X, y = make_blobs(n_samples=500, centers=3, n_features=2, cluster_std=2, random_state=2)
# scatter plot for each class value
# 生成整数序列 / Generate integer sequence
for class_value in range(3):
	# select indices of points with the class label
	row_ix = where(y == class_value)
	# scatter plot for points with a different color
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Mlp

# 02 — Mlp / 02 Mlp

**Chapter 20 — File 2 of 5 / 第20章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **fit high variance mlp on blobs classification problem**.

本脚本演示 **fit high variance mlp on blobs classification problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — fit high variance mlp on blobs classification problem

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=500, centers=3, n_features=2, cluster_std=2, random_state=2)
y = to_categorical(y)
```

---
## Step 3 — split into train and test

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_train = int(0.3 * X.shape[0])
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 4 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(15, input_dim=2, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(3, activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 5 — fit model

```python
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0)
```

---
## Step 6 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
_, test_acc = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
```

---
## Step 7 — plot loss learning curves

```python
pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
```

---
## Step 8 — plot accuracy learning curves

```python
pyplot.subplot(212)
pyplot.title('Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: fit high variance mlp on blobs classification problem 是机器学习中的常用技术。  
  *fit high variance mlp on blobs classification problem is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mlp / 02 Mlp
# Complete Code / 完整代码
# ===============================

# fit high variance mlp on blobs classification problem
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate 2d classification dataset
X, y = make_blobs(n_samples=500, centers=3, n_features=2, cluster_std=2, random_state=2)
y = to_categorical(y)
# split into train and test
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_train = int(0.3 * X.shape[0])
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(15, input_dim=2, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(3, activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0)
# evaluate the model
# 评估模型在测试集上的表现 / Evaluate model on test set
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
_, test_acc = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss learning curves
pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy learning curves
pyplot.subplot(212)
pyplot.title('Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Mlp Repeated Eval

# 03 — Mlp Repeated Eval / 模型评估

**Chapter 20 — File 3 of 5 / 第20章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **demonstrate high variance of mlp model on blobs classification problem**.

本脚本演示 **demonstrate high variance of mlp model on blobs classification problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — demonstrate high variance of mlp model on blobs classification problem

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — fit and evaluate a neural net model on the dataset

```python
def evaluate_model(trainX, trainy, testX, testy):
```

---
## Step 3 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(15, input_dim=2, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 4 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainy, epochs=200, verbose=0)
```

---
## Step 5 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, test_acc = model.evaluate(testX, testy, verbose=0)
	return test_acc
```

---
## Step 6 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=500, centers=3, n_features=2, cluster_std=2, random_state=2)
y = to_categorical(y)
```

---
## Step 7 — split into train and test

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_train = int(0.3 * X.shape[0])
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 8 — repeated evaluation

```python
n_repeats = 30
scores = list()
# 生成整数序列 / Generate integer sequence
for _ in range(n_repeats):
	score = evaluate_model(trainX, trainy, testX, testy)
 # 打印输出 / Print output
	print('> %.3f' % score)
 # 添加元素到列表末尾 / Append element to list end
	scores.append(score)
```

---
## Step 9 — summarize the distribution of scores

```python
# 打印输出 / Print output
print('Scores Mean: %.3f, Standard Deviation: %.3f' % (mean(scores), std(scores)))
```

---
## Step 10 — histogram of distribution

```python
pyplot.hist(scores, bins=10)
pyplot.show()
```

---
## Step 11 — boxplot of distribution

```python
pyplot.boxplot(scores)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: demonstrate high variance of mlp model on blobs classification problem 是机器学习中的常用技术。  
  *demonstrate high variance of mlp model on blobs classification problem is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mlp Repeated Eval / 模型评估
# Complete Code / 完整代码
# ===============================

# demonstrate high variance of mlp model on blobs classification problem
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# fit and evaluate a neural net model on the dataset
def evaluate_model(trainX, trainy, testX, testy):
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(15, input_dim=2, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
 # 训练模型 / Train the model
	model.fit(trainX, trainy, epochs=200, verbose=0)
	# evaluate the model
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	return test_acc

# generate 2d classification dataset
X, y = make_blobs(n_samples=500, centers=3, n_features=2, cluster_std=2, random_state=2)
y = to_categorical(y)
# split into train and test
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_train = int(0.3 * X.shape[0])
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# repeated evaluation
n_repeats = 30
scores = list()
# 生成整数序列 / Generate integer sequence
for _ in range(n_repeats):
	score = evaluate_model(trainX, trainy, testX, testy)
 # 打印输出 / Print output
	print('> %.3f' % score)
 # 添加元素到列表末尾 / Append element to list end
	scores.append(score)
# summarize the distribution of scores
# 打印输出 / Print output
print('Scores Mean: %.3f, Standard Deviation: %.3f' % (mean(scores), std(scores)))
# histogram of distribution
pyplot.hist(scores, bins=10)
pyplot.show()
# boxplot of distribution
pyplot.boxplot(scores)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Mlp Ensemble Study

# 04 — Mlp Ensemble Study / 集成方法

**Chapter 20 — File 4 of 5 / 第20章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **model averaging ensemble and a study of ensemble size on test accuracy**.

本脚本演示 **model averaging ensemble and a study of ensemble size on test accuracy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — model averaging ensemble and a study of ensemble size on test accuracy

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
```

---
## Step 2 — fit model on dataset

```python
def fit_model(trainX, trainy):
```

---
## Step 3 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(15, input_dim=2, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 4 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainy, epochs=200, verbose=0)
	return model
```

---
## Step 5 — make an ensemble prediction for multi-class classification

```python
def ensemble_predictions(members, testX):
```

---
## Step 6 — make predictions

```python
# 用模型做预测 / Make predictions with model
yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
```

---
## Step 7 — sum across ensemble members

```python
summed = numpy.sum(yhats, axis=0)
```

---
## Step 8 — argmax across classes

```python
result = argmax(summed, axis=1)
	return result
```

---
## Step 9 — evaluate a specific number of members in an ensemble

```python
def evaluate_n_members(members, n_members, testX, testy):
```

---
## Step 10 — select a subset of members

```python
subset = members[:n_members]
 # 打印输出 / Print output
	print(len(subset))
```

---
## Step 11 — make prediction

```python
yhat = ensemble_predictions(subset, testX)
```

---
## Step 12 — calculate accuracy

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
return accuracy_score(testy, yhat)
```

---
## Step 13 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=500, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 14 — split into train and test

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_train = int(0.3 * X.shape[0])
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
trainy = to_categorical(trainy)
```

---
## Step 15 — fit all models

```python
n_members = 20
# 生成整数序列 / Generate integer sequence
members = [fit_model(trainX, trainy) for _ in range(n_members)]
```

---
## Step 16 — evaluate different numbers of ensembles

```python
scores = list()
# 生成整数序列 / Generate integer sequence
for i in range(1, n_members+1):
	score = evaluate_n_members(members, i, testX, testy)
 # 打印输出 / Print output
	print('> %.3f' % score)
 # 添加元素到列表末尾 / Append element to list end
	scores.append(score)
```

---
## Step 17 — plot score vs number of ensemble members

```python
# 生成整数序列 / Generate integer sequence
x_axis = [i for i in range(1, n_members+1)]
pyplot.plot(x_axis, scores)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: model averaging ensemble and a study of ensemble size on test accuracy 是机器学习中的常用技术。  
  *model averaging ensemble and a study of ensemble size on test accuracy is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mlp Ensemble Study / 集成方法
# Complete Code / 完整代码
# ===============================

# model averaging ensemble and a study of ensemble size on test accuracy
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy

# fit model on dataset
def fit_model(trainX, trainy):
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(15, input_dim=2, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
 # 训练模型 / Train the model
	model.fit(trainX, trainy, epochs=200, verbose=0)
	return model

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
	# make predictions
 # 用模型做预测 / Make predictions with model
	yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
	# sum across ensemble members
	summed = numpy.sum(yhats, axis=0)
	# argmax across classes
	result = argmax(summed, axis=1)
	return result

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
	# select a subset of members
	subset = members[:n_members]
 # 打印输出 / Print output
	print(len(subset))
	# make prediction
	yhat = ensemble_predictions(subset, testX)
	# calculate accuracy
 # 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
	return accuracy_score(testy, yhat)

# generate 2d classification dataset
X, y = make_blobs(n_samples=500, centers=3, n_features=2, cluster_std=2, random_state=2)
# split into train and test
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_train = int(0.3 * X.shape[0])
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
trainy = to_categorical(trainy)
# fit all models
n_members = 20
# 生成整数序列 / Generate integer sequence
members = [fit_model(trainX, trainy) for _ in range(n_members)]
# evaluate different numbers of ensembles
scores = list()
# 生成整数序列 / Generate integer sequence
for i in range(1, n_members+1):
	score = evaluate_n_members(members, i, testX, testy)
 # 打印输出 / Print output
	print('> %.3f' % score)
 # 添加元素到列表末尾 / Append element to list end
	scores.append(score)
# plot score vs number of ensemble members
# 生成整数序列 / Generate integer sequence
x_axis = [i for i in range(1, n_members+1)]
pyplot.plot(x_axis, scores)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Mlp Ensemble

# 05 — Mlp Ensemble / 集成方法

**Chapter 20 — File 5 of 5 / 第20章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **repeated evaluation of model averaging ensemble on blobs dataset**.

本脚本演示 **repeated evaluation of model averaging ensemble on blobs dataset**。

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
## Step 1 — repeated evaluation of model averaging ensemble on blobs dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
```

---
## Step 2 — fit model on dataset

```python
def fit_model(trainX, trainy):
```

---
## Step 3 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(15, input_dim=2, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 4 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainy, epochs=200, verbose=0)
	return model
```

---
## Step 5 — make an ensemble prediction for multi-class classification

```python
def ensemble_predictions(members, testX):
```

---
## Step 6 — make predictions

```python
# 用模型做预测 / Make predictions with model
yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
```

---
## Step 7 — sum across ensemble members

```python
summed = numpy.sum(yhats, axis=0)
```

---
## Step 8 — argmax across classes

```python
result = argmax(summed, axis=1)
	return result
```

---
## Step 9 — evaluate ensemble model

```python
def evaluate_members(members, testX, testy):
```

---
## Step 10 — make prediction

```python
yhat = ensemble_predictions(members, testX)
```

---
## Step 11 — calculate accuracy

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
return accuracy_score(testy, yhat)
```

---
## Step 12 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=500, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 13 — split into train and test

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_train = int(0.3 * X.shape[0])
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
trainy = to_categorical(trainy)
```

---
## Step 14 — repeated evaluation

```python
n_repeats = 30
n_members = 5
scores = list()
# 生成整数序列 / Generate integer sequence
for _ in range(n_repeats):
```

---
## Step 15 — fit all models

```python
# 生成整数序列 / Generate integer sequence
members = [fit_model(trainX, trainy) for _ in range(n_members)]
```

---
## Step 16 — evaluate ensemble

```python
score = evaluate_members(members, testX, testy)
 # 打印输出 / Print output
	print('> %.3f' % score)
 # 添加元素到列表末尾 / Append element to list end
	scores.append(score)
```

---
## Step 17 — summarize the distribution of scores

```python
# 打印输出 / Print output
print('Scores Mean: %.3f, Standard Deviation: %.3f' % (mean(scores), std(scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: repeated evaluation of model averaging ensemble on blobs dataset 是机器学习中的常用技术。  
  *repeated evaluation of model averaging ensemble on blobs dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mlp Ensemble / 集成方法
# Complete Code / 完整代码
# ===============================

# repeated evaluation of model averaging ensemble on blobs dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy

# fit model on dataset
def fit_model(trainX, trainy):
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(15, input_dim=2, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
 # 训练模型 / Train the model
	model.fit(trainX, trainy, epochs=200, verbose=0)
	return model

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
	# make predictions
 # 用模型做预测 / Make predictions with model
	yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
	# sum across ensemble members
	summed = numpy.sum(yhats, axis=0)
	# argmax across classes
	result = argmax(summed, axis=1)
	return result

# evaluate ensemble model
def evaluate_members(members, testX, testy):
	# make prediction
	yhat = ensemble_predictions(members, testX)
	# calculate accuracy
 # 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
	return accuracy_score(testy, yhat)

# generate 2d classification dataset
X, y = make_blobs(n_samples=500, centers=3, n_features=2, cluster_std=2, random_state=2)
# split into train and test
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_train = int(0.3 * X.shape[0])
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
trainy = to_categorical(trainy)
# repeated evaluation
n_repeats = 30
n_members = 5
scores = list()
# 生成整数序列 / Generate integer sequence
for _ in range(n_repeats):
	# fit all models
 # 生成整数序列 / Generate integer sequence
	members = [fit_model(trainX, trainy) for _ in range(n_members)]
	# evaluate ensemble
	score = evaluate_members(members, testX, testy)
 # 打印输出 / Print output
	print('> %.3f' % score)
 # 添加元素到列表末尾 / Append element to list end
	scores.append(score)
# summarize the distribution of scores
# 打印输出 / Print output
print('Scores Mean: %.3f, Standard Deviation: %.3f' % (mean(scores), std(scores)))
```

---

### Chapter Summary / 章节总结

# Chapter 20 Summary / 第20章总结

## Theme / 主题: Chapter 20 / Chapter 20

This chapter contains **5 code files** demonstrating chapter 20.

本章包含 **5 个代码文件**，演示Chapter 20。

---
## Evolution / 演化路线

  1. `01_problem.ipynb` — Problem
  2. `02_mlp.ipynb` — Mlp
  3. `03_mlp_repeated_eval.ipynb` — Mlp Repeated Eval
  4. `04_mlp_ensemble_study.ipynb` — Mlp Ensemble Study
  5. `05_mlp_ensemble.ipynb` — Mlp Ensemble

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 20) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 20）是机器学习流水线中的基础构建块。

---
