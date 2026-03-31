# 优化深度学习 / Better Deep Learning
## Chapter 23

---

### Problem

# 01 — Problem / 01 Problem

**Chapter 23 — File 1 of 4 / 第23章 — 第1个文件（共4个）**

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

➡️ **Next / 下一步**: File 2 of 4

---

### Mlp

# 02 — Mlp / 02 Mlp

**Chapter 23 — File 2 of 4 / 第23章 — 第2个文件（共4个）**

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
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=1000, verbose=0)
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
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=1000, verbose=0)
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

➡️ **Next / 下一步**: File 3 of 4

---

### Save Members

# 03 — Save Members / 保存/加载模型

**Chapter 23 — File 3 of 4 / 第23章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **save horizontal voting ensemble members during training**.

本脚本演示 **save horizontal voting ensemble members during training**。

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
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — save horizontal voting ensemble members during training

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
from os import makedirs
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
## Step 6 — create directory for models

```python
makedirs('models')
```

---
## Step 7 — fit model

```python
n_epochs, n_save_after = 1000, 950
# 生成整数序列 / Generate integer sequence
for i in range(n_epochs):
```

---
## Step 8 — fit model for a single epoch

```python
# 训练模型 / Train the model
model.fit(trainX, trainy, epochs=1, verbose=0)
```

---
## Step 9 — check if we should save the model

```python
if i >= n_save_after:
  # 保存模型到文件 / Save model to file
		model.save('models/model_' + str(i) + '.h5')
```

---
## Learning Notes / 学习笔记

- **概念**: save horizontal voting ensemble members during training 是机器学习中的常用技术。  
  *save horizontal voting ensemble members during training is a common technique in machine learning.*

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
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Save Members / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# save horizontal voting ensemble members during training
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
from os import makedirs
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
# create directory for models
makedirs('models')
# fit model
n_epochs, n_save_after = 1000, 950
# 生成整数序列 / Generate integer sequence
for i in range(n_epochs):
	# fit model for a single epoch
 # 训练模型 / Train the model
	model.fit(trainX, trainy, epochs=1, verbose=0)
	# check if we should save the model
	if i >= n_save_after:
  # 保存模型到文件 / Save model to file
		model.save('models/model_' + str(i) + '.h5')
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Horizontal Ensemble

# 04 — Horizontal Ensemble / 集成方法

**Chapter 23 — File 4 of 4 / 第23章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **load models and make predictions using a horizontal voting ensemble**.

本脚本演示 **load models and make predictions using a horizontal voting ensemble**。

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
## Step 1 — load models and make predictions using a horizontal voting ensemble

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
```

---
## Step 2 — load models from file

```python
def load_all_models(n_start, n_end):
	all_models = list()
 # 生成整数序列 / Generate integer sequence
	for epoch in range(n_start, n_end):
```

---
## Step 3 — define filename for this ensemble

```python
filename = 'models/model_' + str(epoch) + '.h5'
```

---
## Step 4 — load model from file

```python
# 从文件加载模型 / Load model from file
model = load_model(filename)
```

---
## Step 5 — add to list of members

```python
# 添加元素到列表末尾 / Append element to list end
all_models.append(model)
  # 打印输出 / Print output
		print('>loaded %s' % filename)
	return all_models
```

---
## Step 6 — make an ensemble prediction for multi-class classification

```python
def ensemble_predictions(members, testX):
```

---
## Step 7 — make predictions

```python
# 用模型做预测 / Make predictions with model
yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
```

---
## Step 8 — sum across ensemble members

```python
summed = numpy.sum(yhats, axis=0)
```

---
## Step 9 — argmax across classes

```python
result = argmax(summed, axis=1)
	return result
```

---
## Step 10 — evaluate a specific number of members in an ensemble

```python
def evaluate_n_members(members, n_members, testX, testy):
```

---
## Step 11 — select a subset of members

```python
subset = members[:n_members]
```

---
## Step 12 — make prediction

```python
yhat = ensemble_predictions(subset, testX)
```

---
## Step 13 — calculate accuracy

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
return accuracy_score(testy, yhat)
```

---
## Step 14 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 15 — split into train and test

```python
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 16 — load models in order

```python
members = load_all_models(950, 1000)
# 打印输出 / Print output
print('Loaded %d models' % len(members))
```

---
## Step 17 — reverse loaded models so we build the ensemble with the last models first

```python
members = list(reversed(members))
```

---
## Step 18 — evaluate different numbers of ensembles on hold out set

```python
single_scores, ensemble_scores = list(), list()
# 获取长度 / Get length
for i in range(1, len(members)+1):
```

---
## Step 19 — evaluate model with i members

```python
ensemble_score = evaluate_n_members(members, i, testX, testy)
```

---
## Step 20 — evaluate the i'th model standalone

```python
testy_enc = to_categorical(testy)
	_, single_score = members[i-1].evaluate(testX, testy_enc, verbose=0)
```

---
## Step 21 — summarize this step

```python
# 打印输出 / Print output
print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
 # 添加元素到列表末尾 / Append element to list end
	ensemble_scores.append(ensemble_score)
 # 添加元素到列表末尾 / Append element to list end
	single_scores.append(single_score)
```

---
## Step 22 — summarize average accuracy of a single final model

```python
# 打印输出 / Print output
print('Accuracy %.3f (%.3f)' % (mean(single_scores), std(single_scores)))
```

---
## Step 23 — plot score vs number of ensemble members

```python
# 获取长度 / Get length
x_axis = [i for i in range(1, len(members)+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load models and make predictions using a horizontal voting ensemble 是机器学习中的常用技术。  
  *load models and make predictions using a horizontal voting ensemble is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `matplotlib` | 绑图库 | Plotting library |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Horizontal Ensemble / 集成方法
# Complete Code / 完整代码
# ===============================

# load models and make predictions using a horizontal voting ensemble
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy

# load models from file
def load_all_models(n_start, n_end):
	all_models = list()
 # 生成整数序列 / Generate integer sequence
	for epoch in range(n_start, n_end):
		# define filename for this ensemble
		filename = 'models/model_' + str(epoch) + '.h5'
		# load model from file
  # 从文件加载模型 / Load model from file
		model = load_model(filename)
		# add to list of members
  # 添加元素到列表末尾 / Append element to list end
		all_models.append(model)
  # 打印输出 / Print output
		print('>loaded %s' % filename)
	return all_models

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
# load models in order
members = load_all_models(950, 1000)
# 打印输出 / Print output
print('Loaded %d models' % len(members))
# reverse loaded models so we build the ensemble with the last models first
members = list(reversed(members))
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

### Chapter Summary / 章节总结

# Chapter 23 Summary / 第23章总结

## Theme / 主题: Chapter 23 / Chapter 23

This chapter contains **4 code files** demonstrating chapter 23.

本章包含 **4 个代码文件**，演示Chapter 23。

---
## Evolution / 演化路线

  1. `01_problem.ipynb` — Problem
  2. `02_mlp.ipynb` — Mlp
  3. `03_save_members.ipynb` — Save Members
  4. `04_horizontal_ensemble.ipynb` — Horizontal Ensemble

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 23) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 23）是机器学习流水线中的基础构建块。

---
