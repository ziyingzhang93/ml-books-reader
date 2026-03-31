# 优化深度学习 / Better Deep Learning
## Chapter 21

---

### Problem

# 01 — Problem / 01 Problem

**Chapter 21 — File 1 of 5 / 第21章 — 第1个文件（共5个）**

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
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
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
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
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

**Chapter 21 — File 2 of 5 / 第21章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **develop an mlp for blobs dataset**.

本脚本演示 **develop an mlp for blobs dataset**。

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
## Step 1 — develop an mlp for blobs dataset

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
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 3 — one hot encode output variable

```python
y = to_categorical(y)
```

---
## Step 4 — split into train and test

```python
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 5 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(25, input_dim=2, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(3, activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 6 — fit model

```python
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0)
```

---
## Step 7 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
_, test_acc = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
```

---
## Step 8 — plot loss learning curves

```python
pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
```

---
## Step 9 — plot accuracy learning curves

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

- **概念**: develop an mlp for blobs dataset 是机器学习中的常用技术。  
  *develop an mlp for blobs dataset is a common technique in machine learning.*

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

# develop an mlp for blobs dataset
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
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# one hot encode output variable
y = to_categorical(y)
# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(25, input_dim=2, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(3, activation='softmax'))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0)
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

### Model Avg Ensemble

# 03 — Model Avg Ensemble / 集成方法

**Chapter 21 — File 3 of 5 / 第21章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **model averaging ensemble for the blobs dataset**.

本脚本演示 **model averaging ensemble for the blobs dataset**。

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
## Step 1 — model averaging ensemble for the blobs dataset

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
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax
```

---
## Step 2 — fit model on dataset

```python
def fit_model(trainX, trainy):
	trainy_enc = to_categorical(trainy)
```

---
## Step 3 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(25, input_dim=2, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 4 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainy_enc, epochs=500, verbose=0)
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
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 14 — split into train and test

```python
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 15 — fit all models

```python
n_members = 10
# 生成整数序列 / Generate integer sequence
members = [fit_model(trainX, trainy) for _ in range(n_members)]
```

---
## Step 16 — evaluate different numbers of ensembles on hold out set

```python
single_scores, ensemble_scores = list(), list()
# 获取长度 / Get length
for i in range(1, len(members)+1):
```

---
## Step 17 — evaluate model with i members

```python
ensemble_score = evaluate_n_members(members, i, testX, testy)
```

---
## Step 18 — evaluate the i'th model standalone

```python
testy_enc = to_categorical(testy)
	_, single_score = members[i-1].evaluate(testX, testy_enc, verbose=0)
```

---
## Step 19 — summarize this step

```python
# 打印输出 / Print output
print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
 # 添加元素到列表末尾 / Append element to list end
	ensemble_scores.append(ensemble_score)
 # 添加元素到列表末尾 / Append element to list end
	single_scores.append(single_score)
```

---
## Step 20 — summarize average accuracy of a single final model

```python
# 打印输出 / Print output
print('Accuracy %.3f (%.3f)' % (mean(single_scores), std(single_scores)))
```

---
## Step 21 — plot score vs number of ensemble members

```python
# 获取长度 / Get length
x_axis = [i for i in range(1, len(members)+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: model averaging ensemble for the blobs dataset 是机器学习中的常用技术。  
  *model averaging ensemble for the blobs dataset is a common technique in machine learning.*

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
# Model Avg Ensemble / 集成方法
# Complete Code / 完整代码
# ===============================

# model averaging ensemble for the blobs dataset
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
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax

# fit model on dataset
def fit_model(trainX, trainy):
	trainy_enc = to_categorical(trainy)
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(25, input_dim=2, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
 # 训练模型 / Train the model
	model.fit(trainX, trainy_enc, epochs=500, verbose=0)
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
	# make prediction
	yhat = ensemble_predictions(subset, testX)
	# calculate accuracy
 # 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
	return accuracy_score(testy, yhat)

# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# fit all models
n_members = 10
# 生成整数序列 / Generate integer sequence
members = [fit_model(trainX, trainy) for _ in range(n_members)]
# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores = list(), list()
# 获取长度 / Get length
for i in range(1, len(members)+1):
	# evaluate model with i members
	ensemble_score = evaluate_n_members(members, i, testX, testy)
	# evaluate the i'th model standalone
	testy_enc = to_categorical(testy)
	_, single_score = members[i-1].evaluate(testX, testy_enc, verbose=0)
	# summarize this step
 # 打印输出 / Print output
	print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
 # 添加元素到列表末尾 / Append element to list end
	ensemble_scores.append(ensemble_score)
 # 添加元素到列表末尾 / Append element to list end
	single_scores.append(single_score)
# summarize average accuracy of a single final model
# 打印输出 / Print output
print('Accuracy %.3f (%.3f)' % (mean(single_scores), std(single_scores)))
# plot score vs number of ensemble members
# 获取长度 / Get length
x_axis = [i for i in range(1, len(members)+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Weighted Avg Ensemble

# 04 — Weighted Avg Ensemble / 集成方法

**Chapter 21 — File 4 of 5 / 第21章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **grid search for coefficients in a weighted average ensemble for the blobs problem**.

本脚本演示 **grid search for coefficients in a weighted average ensemble for the blobs problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
## Step 1 — grid search for coefficients in a weighted average ensemble for the blobs problem

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
from numpy import tensordot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import norm
from itertools import product
```

---
## Step 2 — fit model on dataset

```python
def fit_model(trainX, trainy):
	trainy_enc = to_categorical(trainy)
```

---
## Step 3 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(25, input_dim=2, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 4 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainy_enc, epochs=500, verbose=0)
	return model
```

---
## Step 5 — make an ensemble prediction for multi-class classification

```python
def ensemble_predictions(members, weights, testX):
```

---
## Step 6 — make predictions

```python
# 用模型做预测 / Make predictions with model
yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
```

---
## Step 7 — weighted sum across ensemble members

```python
summed = tensordot(yhats, weights, axes=((0),(0)))
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
def evaluate_ensemble(members, weights, testX, testy):
```

---
## Step 10 — make prediction

```python
yhat = ensemble_predictions(members, weights, testX)
```

---
## Step 11 — calculate accuracy

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
return accuracy_score(testy, yhat)
```

---
## Step 12 — normalize a vector to have unit norm

```python
def normalize(weights):
```

---
## Step 13 — calculate l1 vector norm

```python
result = norm(weights, 1)
```

---
## Step 14 — check for a vector of all zeros

```python
if result == 0.0:
		return weights
```

---
## Step 15 — return normalized vector (unit norm)

```python
return weights / result
```

---
## Step 16 — grid search weights

```python
def grid_search(members, testX, testy):
```

---
## Step 17 — define weights to consider

```python
w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	best_score, best_weights = 0.0, None
```

---
## Step 18 — iterate all possible combinations (cartesian product)

```python
# 获取长度 / Get length
for weights in product(w, repeat=len(members)):
```

---
## Step 19 — skip if all weights are equal

```python
# 获取长度 / Get length
if len(set(weights)) == 1:
			continue
```

---
## Step 20 — hack, normalize weight vector

```python
weights = normalize(weights)
```

---
## Step 21 — evaluate weights

```python
score = evaluate_ensemble(members, weights, testX, testy)
		if score > best_score:
			best_score, best_weights = score, weights
   # 打印输出 / Print output
			print('>%s %.3f' % (best_weights, best_score))
	return list(best_weights)
```

---
## Step 22 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 23 — split into train and test

```python
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 24 — fit all models

```python
n_members = 5
# 生成整数序列 / Generate integer sequence
members = [fit_model(trainX, trainy) for _ in range(n_members)]
```

---
## Step 25 — evaluate each single model on the test set

```python
testy_enc = to_categorical(testy)
# 生成整数序列 / Generate integer sequence
for i in range(n_members):
	_, test_acc = members[i].evaluate(testX, testy_enc, verbose=0)
 # 打印输出 / Print output
	print('Model %d: %.3f' % (i+1, test_acc))
```

---
## Step 26 — evaluate averaging ensemble (equal weights)

```python
# 生成整数序列 / Generate integer sequence
weights = [1.0/n_members for _ in range(n_members)]
score = evaluate_ensemble(members, weights, testX, testy)
# 打印输出 / Print output
print('Equal Weights Score: %.3f' % score)
```

---
## Step 27 — grid search weights

```python
weights = grid_search(members, testX, testy)
score = evaluate_ensemble(members, weights, testX, testy)
# 打印输出 / Print output
print('Grid Search Weights: %s, Score: %.3f' % (weights, score))
```

---
## Learning Notes / 学习笔记

- **概念**: grid search for coefficients in a weighted average ensemble for the blobs problem 是机器学习中的常用技术。  
  *grid search for coefficients in a weighted average ensemble for the blobs problem is a common technique in machine learning.*

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
# Weighted Avg Ensemble / 集成方法
# Complete Code / 完整代码
# ===============================

# grid search for coefficients in a weighted average ensemble for the blobs problem
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
from numpy import tensordot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import norm
from itertools import product

# fit model on dataset
def fit_model(trainX, trainy):
	trainy_enc = to_categorical(trainy)
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(25, input_dim=2, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
 # 训练模型 / Train the model
	model.fit(trainX, trainy_enc, epochs=500, verbose=0)
	return model

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, weights, testX):
	# make predictions
 # 用模型做预测 / Make predictions with model
	yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
	# weighted sum across ensemble members
	summed = tensordot(yhats, weights, axes=((0),(0)))
	# argmax across classes
	result = argmax(summed, axis=1)
	return result

# evaluate a specific number of members in an ensemble
def evaluate_ensemble(members, weights, testX, testy):
	# make prediction
	yhat = ensemble_predictions(members, weights, testX)
	# calculate accuracy
 # 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
	return accuracy_score(testy, yhat)

# normalize a vector to have unit norm
def normalize(weights):
	# calculate l1 vector norm
	result = norm(weights, 1)
	# check for a vector of all zeros
	if result == 0.0:
		return weights
	# return normalized vector (unit norm)
	return weights / result

# grid search weights
def grid_search(members, testX, testy):
	# define weights to consider
	w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	best_score, best_weights = 0.0, None
	# iterate all possible combinations (cartesian product)
 # 获取长度 / Get length
	for weights in product(w, repeat=len(members)):
		# skip if all weights are equal
  # 获取长度 / Get length
		if len(set(weights)) == 1:
			continue
		# hack, normalize weight vector
		weights = normalize(weights)
		# evaluate weights
		score = evaluate_ensemble(members, weights, testX, testy)
		if score > best_score:
			best_score, best_weights = score, weights
   # 打印输出 / Print output
			print('>%s %.3f' % (best_weights, best_score))
	return list(best_weights)

# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# fit all models
n_members = 5
# 生成整数序列 / Generate integer sequence
members = [fit_model(trainX, trainy) for _ in range(n_members)]
# evaluate each single model on the test set
testy_enc = to_categorical(testy)
# 生成整数序列 / Generate integer sequence
for i in range(n_members):
	_, test_acc = members[i].evaluate(testX, testy_enc, verbose=0)
 # 打印输出 / Print output
	print('Model %d: %.3f' % (i+1, test_acc))
# evaluate averaging ensemble (equal weights)
# 生成整数序列 / Generate integer sequence
weights = [1.0/n_members for _ in range(n_members)]
score = evaluate_ensemble(members, weights, testX, testy)
# 打印输出 / Print output
print('Equal Weights Score: %.3f' % score)
# grid search weights
weights = grid_search(members, testX, testy)
score = evaluate_ensemble(members, weights, testX, testy)
# 打印输出 / Print output
print('Grid Search Weights: %s, Score: %.3f' % (weights, score))
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Global Search Weights

# 05 — Global Search Weights / 05 Global Search Weights

**Chapter 21 — File 5 of 5 / 第21章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **global optimization to find coefficients for weighted ensemble on blobs problem**.

本脚本演示 **global optimization to find coefficients for weighted ensemble on blobs problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
## Step 1 — global optimization to find coefficients for weighted ensemble on blobs problem

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
from numpy import tensordot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import norm
from scipy.optimize import differential_evolution
```

---
## Step 2 — fit model on dataset

```python
def fit_model(trainX, trainy):
	trainy_enc = to_categorical(trainy)
```

---
## Step 3 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(25, input_dim=2, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 4 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainy_enc, epochs=500, verbose=0)
	return model
```

---
## Step 5 — make an ensemble prediction for multi-class classification

```python
def ensemble_predictions(members, weights, testX):
```

---
## Step 6 — make predictions

```python
# 用模型做预测 / Make predictions with model
yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
```

---
## Step 7 — weighted sum across ensemble members

```python
summed = tensordot(yhats, weights, axes=((0),(0)))
```

---
## Step 8 — argmax across classes

```python
result = argmax(summed, axis=1)
	return result
```

---
## Step 9 — # evaluate a specific number of members in an ensemble

```python
def evaluate_ensemble(members, weights, testX, testy):
```

---
## Step 10 — make prediction

```python
yhat = ensemble_predictions(members, weights, testX)
```

---
## Step 11 — calculate accuracy

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
return accuracy_score(testy, yhat)
```

---
## Step 12 — normalize a vector to have unit norm

```python
def normalize(weights):
```

---
## Step 13 — calculate l1 vector norm

```python
result = norm(weights, 1)
```

---
## Step 14 — check for a vector of all zeros

```python
if result == 0.0:
		return weights
```

---
## Step 15 — return normalized vector (unit norm)

```python
return weights / result
```

---
## Step 16 — loss function for optimization process, designed to be minimized

```python
def loss_function(weights, members, testX, testy):
```

---
## Step 17 — normalize weights

```python
normalized = normalize(weights)
```

---
## Step 18 — calculate error rate

```python
return 1.0 - evaluate_ensemble(members, normalized, testX, testy)
```

---
## Step 19 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 20 — split into train and test

```python
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 21 — fit all models

```python
n_members = 5
# 生成整数序列 / Generate integer sequence
members = [fit_model(trainX, trainy) for _ in range(n_members)]
```

---
## Step 22 — evaluate each single model on the test set

```python
testy_enc = to_categorical(testy)
# 生成整数序列 / Generate integer sequence
for i in range(n_members):
	_, test_acc = members[i].evaluate(testX, testy_enc, verbose=0)
 # 打印输出 / Print output
	print('Model %d: %.3f' % (i+1, test_acc))
```

---
## Step 23 — evaluate averaging ensemble (equal weights)

```python
# 生成整数序列 / Generate integer sequence
weights = [1.0/n_members for _ in range(n_members)]
score = evaluate_ensemble(members, weights, testX, testy)
# 打印输出 / Print output
print('Equal Weights Score: %.3f' % score)
```

---
## Step 24 — define bounds on each weight

```python
# 生成整数序列 / Generate integer sequence
bound_w = [(0.0, 1.0)  for _ in range(n_members)]
```

---
## Step 25 — arguments to the loss function

```python
search_arg = (members, testX, testy)
```

---
## Step 26 — global optimization of ensemble weights

```python
result = differential_evolution(loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)
```

---
## Step 27 — get the chosen weights

```python
weights = normalize(result['x'])
# 打印输出 / Print output
print('Optimized Weights: %s' % weights)
```

---
## Step 28 — evaluate chosen weights

```python
score = evaluate_ensemble(members, weights, testX, testy)
# 打印输出 / Print output
print('Optimized Weights Score: %.3f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: global optimization to find coefficients for weighted ensemble on blobs problem 是机器学习中的常用技术。  
  *global optimization to find coefficients for weighted ensemble on blobs problem is a common technique in machine learning.*

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
# Global Search Weights / 05 Global Search Weights
# Complete Code / 完整代码
# ===============================

# global optimization to find coefficients for weighted ensemble on blobs problem
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
from numpy import tensordot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import norm
from scipy.optimize import differential_evolution

# fit model on dataset
def fit_model(trainX, trainy):
	trainy_enc = to_categorical(trainy)
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(25, input_dim=2, activation='relu'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
 # 训练模型 / Train the model
	model.fit(trainX, trainy_enc, epochs=500, verbose=0)
	return model

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, weights, testX):
	# make predictions
 # 用模型做预测 / Make predictions with model
	yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
	# weighted sum across ensemble members
	summed = tensordot(yhats, weights, axes=((0),(0)))
	# argmax across classes
	result = argmax(summed, axis=1)
	return result

# # evaluate a specific number of members in an ensemble
def evaluate_ensemble(members, weights, testX, testy):
	# make prediction
	yhat = ensemble_predictions(members, weights, testX)
	# calculate accuracy
 # 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
	return accuracy_score(testy, yhat)

# normalize a vector to have unit norm
def normalize(weights):
	# calculate l1 vector norm
	result = norm(weights, 1)
	# check for a vector of all zeros
	if result == 0.0:
		return weights
	# return normalized vector (unit norm)
	return weights / result

# loss function for optimization process, designed to be minimized
def loss_function(weights, members, testX, testy):
	# normalize weights
	normalized = normalize(weights)
	# calculate error rate
	return 1.0 - evaluate_ensemble(members, normalized, testX, testy)

# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# fit all models
n_members = 5
# 生成整数序列 / Generate integer sequence
members = [fit_model(trainX, trainy) for _ in range(n_members)]
# evaluate each single model on the test set
testy_enc = to_categorical(testy)
# 生成整数序列 / Generate integer sequence
for i in range(n_members):
	_, test_acc = members[i].evaluate(testX, testy_enc, verbose=0)
 # 打印输出 / Print output
	print('Model %d: %.3f' % (i+1, test_acc))
# evaluate averaging ensemble (equal weights)
# 生成整数序列 / Generate integer sequence
weights = [1.0/n_members for _ in range(n_members)]
score = evaluate_ensemble(members, weights, testX, testy)
# 打印输出 / Print output
print('Equal Weights Score: %.3f' % score)
# define bounds on each weight
# 生成整数序列 / Generate integer sequence
bound_w = [(0.0, 1.0)  for _ in range(n_members)]
# arguments to the loss function
search_arg = (members, testX, testy)
# global optimization of ensemble weights
result = differential_evolution(loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)
# get the chosen weights
weights = normalize(result['x'])
# 打印输出 / Print output
print('Optimized Weights: %s' % weights)
# evaluate chosen weights
score = evaluate_ensemble(members, weights, testX, testy)
# 打印输出 / Print output
print('Optimized Weights Score: %.3f' % score)
```

---

### Chapter Summary / 章节总结

# Chapter 21 Summary / 第21章总结

## Theme / 主题: Chapter 21 / Chapter 21

This chapter contains **5 code files** demonstrating chapter 21.

本章包含 **5 个代码文件**，演示Chapter 21。

---
## Evolution / 演化路线

  1. `01_problem.ipynb` — Problem
  2. `02_mlp.ipynb` — Mlp
  3. `03_model_avg_ensemble.ipynb` — Model Avg Ensemble
  4. `04_weighted_avg_ensemble.ipynb` — Weighted Avg Ensemble
  5. `05_global_search_weights.ipynb` — Global Search Weights

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 21) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 21）是机器学习流水线中的基础构建块。

---
