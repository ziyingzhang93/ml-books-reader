# 优化深度学习 / Better Deep Learning
## Chapter 08

---

### Problem

# 01 — Problem / 01 Problem

**Chapter 08 — File 1 of 4 / 第08章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **regression predictive modeling problem**.

本脚本演示 **regression predictive modeling problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — regression predictive modeling problem

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — generate regression dataset

```python
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
```

---
## Step 3 — histogram of target variable

```python
pyplot.subplot(121)
pyplot.hist(y)
```

---
## Step 4 — boxplot of target variable

```python
pyplot.subplot(122)
pyplot.boxplot(y)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: regression predictive modeling problem 是机器学习中的常用技术。  
  *regression predictive modeling problem is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Problem / 01 Problem
# Complete Code / 完整代码
# ===============================

# regression predictive modeling problem
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
# histogram of target variable
pyplot.subplot(121)
pyplot.hist(y)
# boxplot of target variable
pyplot.subplot(122)
pyplot.boxplot(y)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Mlp Exploding

# 02 — Mlp Exploding / 02 Mlp Exploding

**Chapter 08 — File 2 of 4 / 第08章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **mlp with unscaled data for the regression problem**.

本脚本演示 **mlp with unscaled data for the regression problem**。

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
## Step 1 — mlp with unscaled data for the regression problem

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — generate regression dataset

```python
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
```

---
## Step 3 — split into train and test

```python
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 4 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='linear'))
```

---
## Step 5 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9))
```

---
## Step 6 — fit model

```python
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
```

---
## Step 7 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
train_mse = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
test_mse = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
```

---
## Step 8 — plot loss during training

```python
pyplot.title('Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: mlp with unscaled data for the regression problem 是机器学习中的常用技术。  
  *mlp with unscaled data for the regression problem is a common technique in machine learning.*

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
# Mlp Exploding / 02 Mlp Exploding
# Complete Code / 完整代码
# ===============================

# mlp with unscaled data for the regression problem
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='linear'))
# compile model
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9))
# fit model
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
# evaluate the model
# 评估模型在测试集上的表现 / Evaluate model on test set
train_mse = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
test_mse = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
# plot loss during training
pyplot.title('Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Mlp Gradient Norm Scaling

# 03 — Mlp Gradient Norm Scaling / 梯度方法

**Chapter 08 — File 3 of 4 / 第08章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **mlp with unscaled data for the regression problem with gradient norm scaling**.

本脚本演示 **mlp with unscaled data for the regression problem with gradient norm scaling**。

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
## Step 1 — mlp with unscaled data for the regression problem with gradient norm scaling

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — generate regression dataset

```python
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
```

---
## Step 3 — split into train and test

```python
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 4 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='linear'))
```

---
## Step 5 — compile model

```python
opt = SGD(lr=0.01, momentum=0.9, clipnorm=1.0)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer=opt)
```

---
## Step 6 — fit model

```python
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
```

---
## Step 7 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
train_mse = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
test_mse = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
```

---
## Step 8 — plot loss during training

```python
pyplot.title('Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: mlp with unscaled data for the regression problem with gradient norm scaling 是机器学习中的常用技术。  
  *mlp with unscaled data for the regression problem with gradient norm scaling is a common technique in machine learning.*

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
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
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
# Mlp Gradient Norm Scaling / 梯度方法
# Complete Code / 完整代码
# ===============================

# mlp with unscaled data for the regression problem with gradient norm scaling
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='linear'))
# compile model
opt = SGD(lr=0.01, momentum=0.9, clipnorm=1.0)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer=opt)
# fit model
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
# evaluate the model
# 评估模型在测试集上的表现 / Evaluate model on test set
train_mse = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
test_mse = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
# plot loss during training
pyplot.title('Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Mlp Gradient Value Clipping

# 04 — Mlp Gradient Value Clipping / 梯度方法

**Chapter 08 — File 4 of 4 / 第08章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **mlp with unscaled data for the regression problem with gradient clipping**.

本脚本演示 **mlp with unscaled data for the regression problem with gradient clipping**。

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
## Step 1 — mlp with unscaled data for the regression problem with gradient clipping

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — generate regression dataset

```python
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
```

---
## Step 3 — split into train and test

```python
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 4 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='linear'))
```

---
## Step 5 — compile model

```python
opt = SGD(lr=0.01, momentum=0.9, clipvalue=5.0)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer=opt)
```

---
## Step 6 — fit model

```python
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
```

---
## Step 7 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
train_mse = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
test_mse = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
```

---
## Step 8 — plot loss during training

```python
pyplot.title('Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: mlp with unscaled data for the regression problem with gradient clipping 是机器学习中的常用技术。  
  *mlp with unscaled data for the regression problem with gradient clipping is a common technique in machine learning.*

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
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
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
# Mlp Gradient Value Clipping / 梯度方法
# Complete Code / 完整代码
# ===============================

# mlp with unscaled data for the regression problem with gradient clipping
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='linear'))
# compile model
opt = SGD(lr=0.01, momentum=0.9, clipvalue=5.0)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer=opt)
# fit model
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
# evaluate the model
# 评估模型在测试集上的表现 / Evaluate model on test set
train_mse = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
test_mse = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
# plot loss during training
pyplot.title('Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 08 Summary / 第08章总结

## Theme / 主题: Chapter 08 / Chapter 08

This chapter contains **4 code files** demonstrating chapter 08.

本章包含 **4 个代码文件**，演示Chapter 08。

---
## Evolution / 演化路线

  1. `01_problem.ipynb` — Problem
  2. `02_mlp_exploding.ipynb` — Mlp Exploding
  3. `03_mlp_gradient_norm_scaling.ipynb` — Mlp Gradient Norm Scaling
  4. `04_mlp_gradient_value_clipping.ipynb` — Mlp Gradient Value Clipping

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 08) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 08）是机器学习流水线中的基础构建块。

---
