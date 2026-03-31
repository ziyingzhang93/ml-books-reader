# 优化深度学习 / Better Deep Learning
## Chapter 10

---

### Problem



---

### Mlp Supervised Pretrain

# 02 — Mlp Supervised Pretrain / 预训练模型

**Chapter 10 — File 2 of 3 / 第10章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **supervised greedy layer-wise pretraining for blobs classification problem**.

本脚本演示 **supervised greedy layer-wise pretraining for blobs classification problem**。

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
## Step 1 — supervised greedy layer-wise pretraining for blobs classification problem

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — prepare the dataset

```python
def prepare_data():
```

---
## Step 3 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 4 — one hot encode output variable

```python
y = to_categorical(y)
```

---
## Step 5 — split into train and test

```python
n_train = 500
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, testX, trainy, testy
```

---
## Step 6 — define and fit the base model

```python
def get_base_model(trainX, trainy):
```

---
## Step 7 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(10, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
```

---
## Step 8 — compile model

```python
opt = SGD(lr=0.01, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
```

---
## Step 9 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainy, epochs=100, verbose=0)
	return model
```

---
## Step 10 — evaluate a fit model

```python
def evaluate_model(model, trainX, testX, trainy, testy):
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, train_acc = model.evaluate(trainX, trainy, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	return train_acc, test_acc
```

---
## Step 11 — add one new layer and re-train only the new layer

```python
def add_layer(model, trainX, trainy):
```

---
## Step 12 — remember the current output layer

```python
output_layer = model.layers[-1]
```

---
## Step 13 — remove the output layer

```python
model.pop()
```

---
## Step 14 — mark all remaining layers as non-trainable

```python
for layer in model.layers:
		layer.trainable = False
```

---
## Step 15 — add a new hidden layer

```python
# 向模型添加一层 / Add a layer to the model
model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
```

---
## Step 16 — re-add the output layer

```python
# 向模型添加一层 / Add a layer to the model
model.add(output_layer)
```

---
## Step 17 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainy, epochs=100, verbose=0)
```

---
## Step 18 — prepare data

```python
trainX, testX, trainy, testy = prepare_data()
```

---
## Step 19 — get the base model

```python
model = get_base_model(trainX, trainy)
```

---
## Step 20 — evaluate the base model

```python
scores = dict()
train_acc, test_acc = evaluate_model(model, trainX, testX, trainy, testy)
# 打印输出 / Print output
print('> layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
# 获取长度 / Get length
scores[len(model.layers)] = (train_acc, test_acc)
```

---
## Step 21 — add layers and evaluate the updated model

```python
n_layers = 10
# 生成整数序列 / Generate integer sequence
for i in range(n_layers):
```

---
## Step 22 — add layer

```python
add_layer(model, trainX, trainy)
```

---
## Step 23 — evaluate model

```python
train_acc, test_acc = evaluate_model(model, trainX, testX, trainy, testy)
 # 打印输出 / Print output
	print('> layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
```

---
## Step 24 — store scores for plotting

```python
# 获取长度 / Get length
scores[len(model.layers)] = (train_acc, test_acc)
```

---
## Step 25 — plot number of added layers vs accuracy

```python
# 获取字典的所有键 / Get all dict keys
pyplot.plot(list(scores.keys()), [scores[k][0] for k in scores.keys()], label='train', marker='.')
# 获取字典的所有键 / Get all dict keys
pyplot.plot(list(scores.keys()), [scores[k][1] for k in scores.keys()], label='test', marker='.')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: supervised greedy layer-wise pretraining for blobs classification problem 是机器学习中的常用技术。  
  *supervised greedy layer-wise pretraining for blobs classification problem is a common technique in machine learning.*

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
# Mlp Supervised Pretrain / 预训练模型
# Complete Code / 完整代码
# ===============================

# supervised greedy layer-wise pretraining for blobs classification problem
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# prepare the dataset
def prepare_data():
	# generate 2d classification dataset
	X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
	# one hot encode output variable
	y = to_categorical(y)
	# split into train and test
	n_train = 500
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, testX, trainy, testy

# define and fit the base model
def get_base_model(trainX, trainy):
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(10, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	# fit model
 # 训练模型 / Train the model
	model.fit(trainX, trainy, epochs=100, verbose=0)
	return model

# evaluate a fit model
def evaluate_model(model, trainX, testX, trainy, testy):
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, train_acc = model.evaluate(trainX, trainy, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	return train_acc, test_acc

# add one new layer and re-train only the new layer
def add_layer(model, trainX, trainy):
	# remember the current output layer
	output_layer = model.layers[-1]
	# remove the output layer
	model.pop()
	# mark all remaining layers as non-trainable
	for layer in model.layers:
		layer.trainable = False
	# add a new hidden layer
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
	# re-add the output layer
 # 向模型添加一层 / Add a layer to the model
	model.add(output_layer)
	# fit model
 # 训练模型 / Train the model
	model.fit(trainX, trainy, epochs=100, verbose=0)

# prepare data
trainX, testX, trainy, testy = prepare_data()
# get the base model
model = get_base_model(trainX, trainy)
# evaluate the base model
scores = dict()
train_acc, test_acc = evaluate_model(model, trainX, testX, trainy, testy)
# 打印输出 / Print output
print('> layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
# 获取长度 / Get length
scores[len(model.layers)] = (train_acc, test_acc)
# add layers and evaluate the updated model
n_layers = 10
# 生成整数序列 / Generate integer sequence
for i in range(n_layers):
	# add layer
	add_layer(model, trainX, trainy)
	# evaluate model
	train_acc, test_acc = evaluate_model(model, trainX, testX, trainy, testy)
 # 打印输出 / Print output
	print('> layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
	# store scores for plotting
 # 获取长度 / Get length
	scores[len(model.layers)] = (train_acc, test_acc)
# plot number of added layers vs accuracy
# 获取字典的所有键 / Get all dict keys
pyplot.plot(list(scores.keys()), [scores[k][0] for k in scores.keys()], label='train', marker='.')
# 获取字典的所有键 / Get all dict keys
pyplot.plot(list(scores.keys()), [scores[k][1] for k in scores.keys()], label='test', marker='.')
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Mlp Unsupervised Pretrain

# 03 — Mlp Unsupervised Pretrain / 预训练模型

**Chapter 10 — File 3 of 3 / 第10章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **unsupervised greedy layer-wise pretraining for blobs classification problem**.

本脚本演示 **unsupervised greedy layer-wise pretraining for blobs classification problem**。

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
## Step 1 — unsupervised greedy layer-wise pretraining for blobs classification problem

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — prepare the dataset

```python
def prepare_data():
```

---
## Step 3 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 4 — one hot encode output variable

```python
y = to_categorical(y)
```

---
## Step 5 — split into train and test

```python
n_train = 500
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, testX, trainy, testy
```

---
## Step 6 — define, fit and evaluate the base autoencoder

```python
def base_autoencoder(trainX, testX):
```

---
## Step 7 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(10, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(2, activation='linear'))
```

---
## Step 8 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9))
```

---
## Step 9 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainX, epochs=100, verbose=0)
```

---
## Step 10 — evaluate reconstruction loss

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
train_mse = model.evaluate(trainX, trainX, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	test_mse = model.evaluate(testX, testX, verbose=0)
 # 打印输出 / Print output
	print('> reconstruction error train=%.3f, test=%.3f' % (train_mse, test_mse))
	return model
```

---
## Step 11 — evaluate the autoencoder as a classifier

```python
def evaluate_autoencoder_as_classifier(model, trainX, trainy, testX, testy):
```

---
## Step 12 — remember the current output layer

```python
output_layer = model.layers[-1]
```

---
## Step 13 — remove the output layer

```python
model.pop()
```

---
## Step 14 — mark all remaining layers as non-trainable

```python
for layer in model.layers:
		layer.trainable = False
```

---
## Step 15 — add new output layer

```python
# 向模型添加一层 / Add a layer to the model
model.add(Dense(3, activation='softmax'))
```

---
## Step 16 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])
```

---
## Step 17 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainy, epochs=100, verbose=0)
```

---
## Step 18 — evaluate model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, test_acc = model.evaluate(testX, testy, verbose=0)
```

---
## Step 19 — put the model back together

```python
model.pop()
 # 向模型添加一层 / Add a layer to the model
	model.add(output_layer)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9))
	return train_acc, test_acc
```

---
## Step 20 — add one new layer and re-train only the new layer

```python
def add_layer_to_autoencoder(model, trainX, testX):
```

---
## Step 21 — remember the current output layer

```python
output_layer = model.layers[-1]
```

---
## Step 22 — remove the output layer

```python
model.pop()
```

---
## Step 23 — mark all remaining layers as non-trainable

```python
for layer in model.layers:
		layer.trainable = False
```

---
## Step 24 — add a new hidden layer

```python
# 向模型添加一层 / Add a layer to the model
model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
```

---
## Step 25 — re-add the output layer

```python
# 向模型添加一层 / Add a layer to the model
model.add(output_layer)
```

---
## Step 26 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainX, epochs=100, verbose=0)
```

---
## Step 27 — evaluate reconstruction loss

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
train_mse = model.evaluate(trainX, trainX, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	test_mse = model.evaluate(testX, testX, verbose=0)
 # 打印输出 / Print output
	print('> reconstruction error train=%.3f, test=%.3f' % (train_mse, test_mse))
```

---
## Step 28 — prepare data

```python
trainX, testX, trainy, testy = prepare_data()
```

---
## Step 29 — get the base autoencoder

```python
model = base_autoencoder(trainX, testX)
```

---
## Step 30 — evaluate the base model

```python
scores = dict()
train_acc, test_acc = evaluate_autoencoder_as_classifier(model, trainX, trainy, testX, testy)
# 打印输出 / Print output
print('> classifier accuracy layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
# 获取长度 / Get length
scores[len(model.layers)] = (train_acc, test_acc)
```

---
## Step 31 — add layers and evaluate the updated model

```python
n_layers = 5
# 生成整数序列 / Generate integer sequence
for _ in range(n_layers):
```

---
## Step 32 — add layer

```python
add_layer_to_autoencoder(model, trainX, testX)
```

---
## Step 33 — evaluate model

```python
train_acc, test_acc = evaluate_autoencoder_as_classifier(model, trainX, trainy, testX, testy)
 # 打印输出 / Print output
	print('> classifier accuracy layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
```

---
## Step 34 — store scores for plotting

```python
# 获取长度 / Get length
scores[len(model.layers)] = (train_acc, test_acc)
```

---
## Step 35 — plot number of added layers vs accuracy

```python
# 获取字典的所有键 / Get all dict keys
keys = list(scores.keys())
pyplot.plot(keys, [scores[k][0] for k in keys], label='train', marker='.')
pyplot.plot(keys, [scores[k][1] for k in keys], label='test', marker='.')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: unsupervised greedy layer-wise pretraining for blobs classification problem 是机器学习中的常用技术。  
  *unsupervised greedy layer-wise pretraining for blobs classification problem is a common technique in machine learning.*

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
# Mlp Unsupervised Pretrain / 预训练模型
# Complete Code / 完整代码
# ===============================

# unsupervised greedy layer-wise pretraining for blobs classification problem
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# prepare the dataset
def prepare_data():
	# generate 2d classification dataset
	X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
	# one hot encode output variable
	y = to_categorical(y)
	# split into train and test
	n_train = 500
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, testX, trainy, testy

# define, fit and evaluate the base autoencoder
def base_autoencoder(trainX, testX):
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(10, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(2, activation='linear'))
	# compile model
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9))
	# fit model
 # 训练模型 / Train the model
	model.fit(trainX, trainX, epochs=100, verbose=0)
	# evaluate reconstruction loss
 # 评估模型在测试集上的表现 / Evaluate model on test set
	train_mse = model.evaluate(trainX, trainX, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	test_mse = model.evaluate(testX, testX, verbose=0)
 # 打印输出 / Print output
	print('> reconstruction error train=%.3f, test=%.3f' % (train_mse, test_mse))
	return model

# evaluate the autoencoder as a classifier
def evaluate_autoencoder_as_classifier(model, trainX, trainy, testX, testy):
	# remember the current output layer
	output_layer = model.layers[-1]
	# remove the output layer
	model.pop()
	# mark all remaining layers as non-trainable
	for layer in model.layers:
		layer.trainable = False
	# add new output layer
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
	# compile model
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])
	# fit model
 # 训练模型 / Train the model
	model.fit(trainX, trainy, epochs=100, verbose=0)
	# evaluate model
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, train_acc = model.evaluate(trainX, trainy, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	# put the model back together
	model.pop()
 # 向模型添加一层 / Add a layer to the model
	model.add(output_layer)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9))
	return train_acc, test_acc

# add one new layer and re-train only the new layer
def add_layer_to_autoencoder(model, trainX, testX):
	# remember the current output layer
	output_layer = model.layers[-1]
	# remove the output layer
	model.pop()
	# mark all remaining layers as non-trainable
	for layer in model.layers:
		layer.trainable = False
	# add a new hidden layer
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
	# re-add the output layer
 # 向模型添加一层 / Add a layer to the model
	model.add(output_layer)
	# fit model
 # 训练模型 / Train the model
	model.fit(trainX, trainX, epochs=100, verbose=0)
	# evaluate reconstruction loss
 # 评估模型在测试集上的表现 / Evaluate model on test set
	train_mse = model.evaluate(trainX, trainX, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	test_mse = model.evaluate(testX, testX, verbose=0)
 # 打印输出 / Print output
	print('> reconstruction error train=%.3f, test=%.3f' % (train_mse, test_mse))

# prepare data
trainX, testX, trainy, testy = prepare_data()
# get the base autoencoder
model = base_autoencoder(trainX, testX)
# evaluate the base model
scores = dict()
train_acc, test_acc = evaluate_autoencoder_as_classifier(model, trainX, trainy, testX, testy)
# 打印输出 / Print output
print('> classifier accuracy layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
# 获取长度 / Get length
scores[len(model.layers)] = (train_acc, test_acc)
# add layers and evaluate the updated model
n_layers = 5
# 生成整数序列 / Generate integer sequence
for _ in range(n_layers):
	# add layer
	add_layer_to_autoencoder(model, trainX, testX)
	# evaluate model
	train_acc, test_acc = evaluate_autoencoder_as_classifier(model, trainX, trainy, testX, testy)
 # 打印输出 / Print output
	print('> classifier accuracy layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
	# store scores for plotting
 # 获取长度 / Get length
	scores[len(model.layers)] = (train_acc, test_acc)
# plot number of added layers vs accuracy
# 获取字典的所有键 / Get all dict keys
keys = list(scores.keys())
pyplot.plot(keys, [scores[k][0] for k in keys], label='train', marker='.')
pyplot.plot(keys, [scores[k][1] for k in keys], label='test', marker='.')
pyplot.legend()
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 10 Summary / 第10章总结

## Theme / 主题: Chapter 10 / Chapter 10

This chapter contains **3 code files** demonstrating chapter 10.

本章包含 **3 个代码文件**，演示Chapter 10。

---
## Evolution / 演化路线

  1. `01_problem.ipynb` — Problem
  2. `02_mlp_supervised_pretrain.ipynb` — Mlp Supervised Pretrain
  3. `03_mlp_unsupervised_pretrain.ipynb` — Mlp Unsupervised Pretrain

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 10) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 10）是机器学习流水线中的基础构建块。

---
