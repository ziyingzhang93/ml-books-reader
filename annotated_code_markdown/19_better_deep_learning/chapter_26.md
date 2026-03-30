# 优化深度学习
## Chapter 26

---

### Problem

# 01 — Problem / 01 Problem

**Chapter 26 — File 1 of 7 / 第26章 — 第1个文件（共7个）**

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
## Step 1 — scatter plot of blobs dataset

```python
from sklearn.datasets import make_blobs
from matplotlib import pyplot
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
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from numpy import where
# generate 2d classification dataset
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
# scatter plot for each class value
for class_value in range(3):
	# select indices of points with the class label
	row_ix = where(y == class_value)
	# scatter plot for points with a different color
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Mlp

# 02 — Mlp / 02 Mlp

**Chapter 26 — File 2 of 7 / 第26章 — 第2个文件（共7个）**

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
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
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
model = Sequential()
model.add(Dense(25, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
```

---
## Step 6 — fit model

```python
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0)
```

---
## Step 7 — evaluate the model

```python
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
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
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
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
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
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
model = Sequential()
model.add(Dense(25, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0)
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
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

➡️ **Next / 下一步**: File 3 of 7

---

### Save Models

# 03 — Save Models / 保存/加载模型

**Chapter 26 — File 3 of 7 / 第26章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **save models to file toward the end of a training run**.

本脚本演示 **save models to file toward the end of a training run**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Step 1 — save models to file toward the end of a training run

```python
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
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
model = Sequential()
model.add(Dense(25, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 6 — fit model

```python
n_epochs, n_save_after = 500, 490
for i in range(n_epochs):
```

---
## Step 7 — fit model for a single epoch

```python
model.fit(trainX, trainy, epochs=1, verbose=0)
```

---
## Step 8 — check if we should save the model

```python
if i >= n_save_after:
		model.save('model_' + str(i) + '.h5')
```

---
## Learning Notes / 学习笔记

- **概念**: save models to file toward the end of a training run 是机器学习中的常用技术。  
  *save models to file toward the end of a training run is a common technique in machine learning.*

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
# Save Models / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# save models to file toward the end of a training run
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# one hot encode output variable
y = to_categorical(y)
# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(25, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
n_epochs, n_save_after = 500, 490
for i in range(n_epochs):
	# fit model for a single epoch
	model.fit(trainX, trainy, epochs=1, verbose=0)
	# check if we should save the model
	if i >= n_save_after:
		model.save('model_' + str(i) + '.h5')
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Evaluate Avg Model Weight Ensemble

# 05 — Evaluate Avg Model Weight Ensemble / 模型评估

**Chapter 26 — File 5 of 7 / 第26章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **average of model weights on blobs problem**.

本脚本演示 **average of model weights on blobs problem**。

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
## Step 1 — average of model weights on blobs problem

```python
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import clone_model
from matplotlib import pyplot
from numpy import average
from numpy import array
```

---
## Step 2 — load models from file

```python
def load_all_models(n_start, n_end):
	all_models = list()
	for epoch in range(n_start, n_end):
```

---
## Step 3 — define filename for this ensemble

```python
filename = 'model_' + str(epoch) + '.h5'
```

---
## Step 4 — load model from file

```python
model = load_model(filename)
```

---
## Step 5 — add to list of members

```python
all_models.append(model)
		print('>loaded %s' % filename)
	return all_models
```

---
## Step 6 — # create a model from the weights of multiple models

```python
def model_weight_ensemble(members, weights):
```

---
## Step 7 — determine how many layers need to be averaged

```python
n_layers = len(members[0].get_weights())
```

---
## Step 8 — create an set of average model weights

```python
avg_model_weights = list()
	for layer in range(n_layers):
```

---
## Step 9 — collect this layer from each model

```python
layer_weights = array([model.get_weights()[layer] for model in members])
```

---
## Step 10 — weighted average of weights for this layer

```python
avg_layer_weights = average(layer_weights, axis=0, weights=weights)
```

---
## Step 11 — store average layer weights

```python
avg_model_weights.append(avg_layer_weights)
```

---
## Step 12 — create a new model with the same structure

```python
model = clone_model(members[0])
```

---
## Step 13 — set the weights in the new

```python
model.set_weights(avg_model_weights)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

---
## Step 14 — evaluate a specific number of members in an ensemble

```python
def evaluate_n_members(members, n_members, testX, testy):
```

---
## Step 15 — select a subset of members

```python
subset = members[:n_members]
```

---
## Step 16 — prepare an array of equal weights

```python
weights = [1.0/n_members for i in range(1, n_members+1)]
```

---
## Step 17 — create a new model with the weighted average of all model weights

```python
model = model_weight_ensemble(subset, weights)
```

---
## Step 18 — make predictions and evaluate accuracy

```python
_, test_acc = model.evaluate(testX, testy, verbose=0)
	return test_acc
```

---
## Step 19 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 20 — one hot encode output variable

```python
y = to_categorical(y)
```

---
## Step 21 — split into train and test

```python
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 22 — load models in order

```python
members = load_all_models(490, 500)
print('Loaded %d models' % len(members))
```

---
## Step 23 — reverse loaded models so we build the ensemble with the last models first

```python
members = list(reversed(members))
```

---
## Step 24 — evaluate different numbers of ensembles on hold out set

```python
single_scores, ensemble_scores = list(), list()
for i in range(1, len(members)+1):
```

---
## Step 25 — evaluate model with i members

```python
ensemble_score = evaluate_n_members(members, i, testX, testy)
```

---
## Step 26 — evaluate the i'th model standalone

```python
_, single_score = members[i-1].evaluate(testX, testy, verbose=0)
```

---
## Step 27 — summarize this step

```python
print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
	ensemble_scores.append(ensemble_score)
	single_scores.append(single_score)
```

---
## Step 28 — plot score vs number of ensemble members

```python
x_axis = [i for i in range(1, len(members)+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: average of model weights on blobs problem 是机器学习中的常用技术。  
  *average of model weights on blobs problem is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Avg Model Weight Ensemble / 模型评估
# Complete Code / 完整代码
# ===============================

# average of model weights on blobs problem
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import clone_model
from matplotlib import pyplot
from numpy import average
from numpy import array

# load models from file
def load_all_models(n_start, n_end):
	all_models = list()
	for epoch in range(n_start, n_end):
		# define filename for this ensemble
		filename = 'model_' + str(epoch) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

# # create a model from the weights of multiple models
def model_weight_ensemble(members, weights):
	# determine how many layers need to be averaged
	n_layers = len(members[0].get_weights())
	# create an set of average model weights
	avg_model_weights = list()
	for layer in range(n_layers):
		# collect this layer from each model
		layer_weights = array([model.get_weights()[layer] for model in members])
		# weighted average of weights for this layer
		avg_layer_weights = average(layer_weights, axis=0, weights=weights)
		# store average layer weights
		avg_model_weights.append(avg_layer_weights)
	# create a new model with the same structure
	model = clone_model(members[0])
	# set the weights in the new
	model.set_weights(avg_model_weights)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
	# select a subset of members
	subset = members[:n_members]
	# prepare an array of equal weights
	weights = [1.0/n_members for i in range(1, n_members+1)]
	# create a new model with the weighted average of all model weights
	model = model_weight_ensemble(subset, weights)
	# make predictions and evaluate accuracy
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	return test_acc

# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# one hot encode output variable
y = to_categorical(y)
# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# load models in order
members = load_all_models(490, 500)
print('Loaded %d models' % len(members))
# reverse loaded models so we build the ensemble with the last models first
members = list(reversed(members))
# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores = list(), list()
for i in range(1, len(members)+1):
	# evaluate model with i members
	ensemble_score = evaluate_n_members(members, i, testX, testy)
	# evaluate the i'th model standalone
	_, single_score = members[i-1].evaluate(testX, testy, verbose=0)
	# summarize this step
	print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
	ensemble_scores.append(ensemble_score)
	single_scores.append(single_score)
# plot score vs number of ensemble members
x_axis = [i for i in range(1, len(members)+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Linear Weighted Ensemble

# 06 — Linear Weighted Ensemble / 线性模型

**Chapter 26 — File 6 of 7 / 第26章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **linearly decreasing weighted average of models on blobs problem**.

本脚本演示 **linearly decreasing weighted average of models on blobs problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
## Step 1 — linearly decreasing weighted average of models on blobs problem

```python
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import clone_model
from matplotlib import pyplot
from numpy import average
from numpy import array
```

---
## Step 2 — load models from file

```python
def load_all_models(n_start, n_end):
	all_models = list()
	for epoch in range(n_start, n_end):
```

---
## Step 3 — define filename for this ensemble

```python
filename = 'model_' + str(epoch) + '.h5'
```

---
## Step 4 — load model from file

```python
model = load_model(filename)
```

---
## Step 5 — add to list of members

```python
all_models.append(model)
		print('>loaded %s' % filename)
	return all_models
```

---
## Step 6 — create a model from the weights of multiple models

```python
def model_weight_ensemble(members, weights):
```

---
## Step 7 — determine how many layers need to be averaged

```python
n_layers = len(members[0].get_weights())
```

---
## Step 8 — create an set of average model weights

```python
avg_model_weights = list()
	for layer in range(n_layers):
```

---
## Step 9 — collect this layer from each model

```python
layer_weights = array([model.get_weights()[layer] for model in members])
```

---
## Step 10 — weighted average of weights for this layer

```python
avg_layer_weights = average(layer_weights, axis=0, weights=weights)
```

---
## Step 11 — store average layer weights

```python
avg_model_weights.append(avg_layer_weights)
```

---
## Step 12 — create a new model with the same structure

```python
model = clone_model(members[0])
```

---
## Step 13 — set the weights in the new

```python
model.set_weights(avg_model_weights)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

---
## Step 14 — evaluate a specific number of members in an ensemble

```python
def evaluate_n_members(members, n_members, testX, testy):
```

---
## Step 15 — select a subset of members

```python
subset = members[:n_members]
```

---
## Step 16 — prepare an array of linearly decreasing weights

```python
weights = [i/n_members for i in range(n_members, 0, -1)]
```

---
## Step 17 — create a new model with the weighted average of all model weights

```python
model = model_weight_ensemble(subset, weights)
```

---
## Step 18 — make predictions and evaluate accuracy

```python
_, test_acc = model.evaluate(testX, testy, verbose=0)
	return test_acc
```

---
## Step 19 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 20 — one hot encode output variable

```python
y = to_categorical(y)
```

---
## Step 21 — split into train and test

```python
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 22 — load models in order

```python
members = load_all_models(490, 500)
print('Loaded %d models' % len(members))
```

---
## Step 23 — reverse loaded models so we build the ensemble with the last models first

```python
members = list(reversed(members))
```

---
## Step 24 — evaluate different numbers of ensembles on hold out set

```python
single_scores, ensemble_scores = list(), list()
for i in range(1, len(members)+1):
```

---
## Step 25 — evaluate model with i members

```python
ensemble_score = evaluate_n_members(members, i, testX, testy)
```

---
## Step 26 — evaluate the i'th model standalone

```python
_, single_score = members[i-1].evaluate(testX, testy, verbose=0)
```

---
## Step 27 — summarize this step

```python
print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
	ensemble_scores.append(ensemble_score)
	single_scores.append(single_score)
```

---
## Step 28 — plot score vs number of ensemble members

```python
x_axis = [i for i in range(1, len(members)+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: linearly decreasing weighted average of models on blobs problem 是机器学习中的常用技术。  
  *linearly decreasing weighted average of models on blobs problem is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Linear Weighted Ensemble / 线性模型
# Complete Code / 完整代码
# ===============================

# linearly decreasing weighted average of models on blobs problem
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import clone_model
from matplotlib import pyplot
from numpy import average
from numpy import array

# load models from file
def load_all_models(n_start, n_end):
	all_models = list()
	for epoch in range(n_start, n_end):
		# define filename for this ensemble
		filename = 'model_' + str(epoch) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

# create a model from the weights of multiple models
def model_weight_ensemble(members, weights):
	# determine how many layers need to be averaged
	n_layers = len(members[0].get_weights())
	# create an set of average model weights
	avg_model_weights = list()
	for layer in range(n_layers):
		# collect this layer from each model
		layer_weights = array([model.get_weights()[layer] for model in members])
		# weighted average of weights for this layer
		avg_layer_weights = average(layer_weights, axis=0, weights=weights)
		# store average layer weights
		avg_model_weights.append(avg_layer_weights)
	# create a new model with the same structure
	model = clone_model(members[0])
	# set the weights in the new
	model.set_weights(avg_model_weights)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
	# select a subset of members
	subset = members[:n_members]
	# prepare an array of linearly decreasing weights
	weights = [i/n_members for i in range(n_members, 0, -1)]
	# create a new model with the weighted average of all model weights
	model = model_weight_ensemble(subset, weights)
	# make predictions and evaluate accuracy
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	return test_acc

# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# one hot encode output variable
y = to_categorical(y)
# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# load models in order
members = load_all_models(490, 500)
print('Loaded %d models' % len(members))
# reverse loaded models so we build the ensemble with the last models first
members = list(reversed(members))
# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores = list(), list()
for i in range(1, len(members)+1):
	# evaluate model with i members
	ensemble_score = evaluate_n_members(members, i, testX, testy)
	# evaluate the i'th model standalone
	_, single_score = members[i-1].evaluate(testX, testy, verbose=0)
	# summarize this step
	print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
	ensemble_scores.append(ensemble_score)
	single_scores.append(single_score)
# plot score vs number of ensemble members
x_axis = [i for i in range(1, len(members)+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Exp Weighted Ensemble

# 07 — Exp Weighted Ensemble / 集成方法

**Chapter 26 — File 7 of 7 / 第26章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **exponentially decreasing weighted average of models on blobs problem**.

本脚本演示 **exponentially decreasing weighted average of models on blobs problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
## Step 1 — exponentially decreasing weighted average of models on blobs problem

```python
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import clone_model
from matplotlib import pyplot
from numpy import average
from numpy import array
from math import exp
```

---
## Step 2 — load models from file

```python
def load_all_models(n_start, n_end):
	all_models = list()
	for epoch in range(n_start, n_end):
```

---
## Step 3 — define filename for this ensemble

```python
filename = 'model_' + str(epoch) + '.h5'
```

---
## Step 4 — load model from file

```python
model = load_model(filename)
```

---
## Step 5 — add to list of members

```python
all_models.append(model)
		print('>loaded %s' % filename)
	return all_models
```

---
## Step 6 — create a model from the weights of multiple models

```python
def model_weight_ensemble(members, weights):
```

---
## Step 7 — determine how many layers need to be averaged

```python
n_layers = len(members[0].get_weights())
```

---
## Step 8 — create an set of average model weights

```python
avg_model_weights = list()
	for layer in range(n_layers):
```

---
## Step 9 — collect this layer from each model

```python
layer_weights = array([model.get_weights()[layer] for model in members])
```

---
## Step 10 — weighted average of weights for this layer

```python
avg_layer_weights = average(layer_weights, axis=0, weights=weights)
```

---
## Step 11 — store average layer weights

```python
avg_model_weights.append(avg_layer_weights)
```

---
## Step 12 — create a new model with the same structure

```python
model = clone_model(members[0])
```

---
## Step 13 — set the weights in the new

```python
model.set_weights(avg_model_weights)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

---
## Step 14 — evaluate a specific number of members in an ensemble

```python
def evaluate_n_members(members, n_members, testX, testy):
```

---
## Step 15 — select a subset of members

```python
subset = members[:n_members]
```

---
## Step 16 — prepare an array of exponentially decreasing weights

```python
alpha = 2.0
	weights = [exp(-i/alpha) for i in range(1, n_members+1)]
```

---
## Step 17 — create a new model with the weighted average of all model weights

```python
model = model_weight_ensemble(subset, weights)
```

---
## Step 18 — make predictions and evaluate accuracy

```python
_, test_acc = model.evaluate(testX, testy, verbose=0)
	return test_acc
```

---
## Step 19 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 20 — one hot encode output variable

```python
y = to_categorical(y)
```

---
## Step 21 — split into train and test

```python
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 22 — load models in order

```python
members = load_all_models(490, 500)
print('Loaded %d models' % len(members))
```

---
## Step 23 — reverse loaded models so we build the ensemble with the last models first

```python
members = list(reversed(members))
```

---
## Step 24 — evaluate different numbers of ensembles on hold out set

```python
single_scores, ensemble_scores = list(), list()
for i in range(1, len(members)+1):
```

---
## Step 25 — evaluate model with i members

```python
ensemble_score = evaluate_n_members(members, i, testX, testy)
```

---
## Step 26 — evaluate the i'th model standalone

```python
_, single_score = members[i-1].evaluate(testX, testy, verbose=0)
```

---
## Step 27 — summarize this step

```python
print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
	ensemble_scores.append(ensemble_score)
	single_scores.append(single_score)
```

---
## Step 28 — plot score vs number of ensemble members

```python
x_axis = [i for i in range(1, len(members)+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: exponentially decreasing weighted average of models on blobs problem 是机器学习中的常用技术。  
  *exponentially decreasing weighted average of models on blobs problem is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Exp Weighted Ensemble / 集成方法
# Complete Code / 完整代码
# ===============================

# exponentially decreasing weighted average of models on blobs problem
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import clone_model
from matplotlib import pyplot
from numpy import average
from numpy import array
from math import exp

# load models from file
def load_all_models(n_start, n_end):
	all_models = list()
	for epoch in range(n_start, n_end):
		# define filename for this ensemble
		filename = 'model_' + str(epoch) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

# create a model from the weights of multiple models
def model_weight_ensemble(members, weights):
	# determine how many layers need to be averaged
	n_layers = len(members[0].get_weights())
	# create an set of average model weights
	avg_model_weights = list()
	for layer in range(n_layers):
		# collect this layer from each model
		layer_weights = array([model.get_weights()[layer] for model in members])
		# weighted average of weights for this layer
		avg_layer_weights = average(layer_weights, axis=0, weights=weights)
		# store average layer weights
		avg_model_weights.append(avg_layer_weights)
	# create a new model with the same structure
	model = clone_model(members[0])
	# set the weights in the new
	model.set_weights(avg_model_weights)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
	# select a subset of members
	subset = members[:n_members]
	# prepare an array of exponentially decreasing weights
	alpha = 2.0
	weights = [exp(-i/alpha) for i in range(1, n_members+1)]
	# create a new model with the weighted average of all model weights
	model = model_weight_ensemble(subset, weights)
	# make predictions and evaluate accuracy
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	return test_acc

# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# one hot encode output variable
y = to_categorical(y)
# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# load models in order
members = load_all_models(490, 500)
print('Loaded %d models' % len(members))
# reverse loaded models so we build the ensemble with the last models first
members = list(reversed(members))
# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores = list(), list()
for i in range(1, len(members)+1):
	# evaluate model with i members
	ensemble_score = evaluate_n_members(members, i, testX, testy)
	# evaluate the i'th model standalone
	_, single_score = members[i-1].evaluate(testX, testy, verbose=0)
	# summarize this step
	print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
	ensemble_scores.append(ensemble_score)
	single_scores.append(single_score)
# plot score vs number of ensemble members
x_axis = [i for i in range(1, len(members)+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
pyplot.show()
```

---

### Chapter Summary

# Chapter 26 Summary / 第26章总结

## Theme / 主题: Chapter 26 / Chapter 26

This chapter contains **7 code files** demonstrating chapter 26.

本章包含 **7 个代码文件**，演示Chapter 26。

---
## Evolution / 演化路线

  1. `01_problem.ipynb` — Problem
  2. `02_mlp.ipynb` — Mlp
  3. `03_save_models.ipynb` — Save Models
  4. `04_avg_model_weights.ipynb` — Avg Model Weights
  5. `05_evaluate_avg_model_weight_ensemble.ipynb` — Evaluate Avg Model Weight Ensemble
  6. `06_linear_weighted_ensemble.ipynb` — Linear Weighted Ensemble
  7. `07_exp_weighted_ensemble.ipynb` — Exp Weighted Ensemble

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 26) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 26）是机器学习流水线中的基础构建块。

---
