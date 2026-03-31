# 优化深度学习 / Better Deep Learning
## Chapter 04

---

### Mlp Mse Loss

# 01 — Mlp Mse Loss / 损失函数

**Chapter 04 — File 1 of 11 / 第04章 — 第1个文件（共11个）**

---

## Summary / 总结

This script demonstrates **mlp for regression with mse loss function**.

本脚本演示 **mlp for regression with mse loss function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
## Step 1 — mlp for regression with mse loss function

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
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
## Step 3 — standardize dataset

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
X = StandardScaler().fit_transform(X)
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]
```

---
## Step 4 — split into train and test

```python
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 5 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='linear'))
opt = SGD(lr=0.01, momentum=0.9)
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
pyplot.title('Mean Squared Error Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: mlp for regression with mse loss function 是机器学习中的常用技术。  
  *mlp for regression with mse loss function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
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
# Mlp Mse Loss / 损失函数
# Complete Code / 完整代码
# ===============================

# mlp for regression with mse loss function
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
# standardize dataset
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
X = StandardScaler().fit_transform(X)
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]
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
opt = SGD(lr=0.01, momentum=0.9)
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
pyplot.title('Mean Squared Error Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 11

---

### Mlp Msle Loss

# 02 — Mlp Msle Loss / 损失函数

**Chapter 04 — File 2 of 11 / 第04章 — 第2个文件（共11个）**

---

## Summary / 总结

This script demonstrates **mlp for regression with msle loss function**.

本脚本演示 **mlp for regression with msle loss function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
## Step 1 — mlp for regression with msle loss function

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
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
## Step 3 — standardize dataset

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
X = StandardScaler().fit_transform(X)
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]
```

---
## Step 4 — split into train and test

```python
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 5 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='linear'))
opt = SGD(lr=0.01, momentum=0.9)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_logarithmic_error', optimizer=opt, metrics=['mse'])
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
_, train_mse = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
_, test_mse = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
```

---
## Step 8 — plot loss during training

```python
pyplot.subplot(211)
pyplot.title('Mean Squared Logarithmic Error Loss', pad=-20)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
```

---
## Step 9 — plot mse during training

```python
pyplot.subplot(212)
pyplot.title('Mean Squared Error', pad=-20)
pyplot.plot(history.history['mse'], label='train')
pyplot.plot(history.history['val_mse'], label='test')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: mlp for regression with msle loss function 是机器学习中的常用技术。  
  *mlp for regression with msle loss function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
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
# Mlp Msle Loss / 损失函数
# Complete Code / 完整代码
# ===============================

# mlp for regression with msle loss function
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
# standardize dataset
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
X = StandardScaler().fit_transform(X)
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]
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
opt = SGD(lr=0.01, momentum=0.9)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_logarithmic_error', optimizer=opt, metrics=['mse'])
# fit model
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
# evaluate the model
# 评估模型在测试集上的表现 / Evaluate model on test set
_, train_mse = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
_, test_mse = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
# plot loss during training
pyplot.subplot(211)
pyplot.title('Mean Squared Logarithmic Error Loss', pad=-20)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot mse during training
pyplot.subplot(212)
pyplot.title('Mean Squared Error', pad=-20)
pyplot.plot(history.history['mse'], label='train')
pyplot.plot(history.history['val_mse'], label='test')
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 11

---

### Mlp Mae Loss

# 03 — Mlp Mae Loss / 损失函数

**Chapter 04 — File 3 of 11 / 第04章 — 第3个文件（共11个）**

---

## Summary / 总结

This script demonstrates **mlp for regression with mae loss function**.

本脚本演示 **mlp for regression with mae loss function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
## Step 1 — mlp for regression with mae loss function

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
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
## Step 3 — standardize dataset

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
X = StandardScaler().fit_transform(X)
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]
```

---
## Step 4 — split into train and test

```python
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 5 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='linear'))
opt = SGD(lr=0.01, momentum=0.9)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mse'])
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
_, train_mse = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
_, test_mse = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
```

---
## Step 8 — plot loss during training

```python
pyplot.subplot(211)
pyplot.title('Mean Absolute Error Loss', pad=-20)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
```

---
## Step 9 — plot mse during training

```python
pyplot.subplot(212)
pyplot.title('Mean Squared Error', pad=-20)
pyplot.plot(history.history['mse'], label='train')
pyplot.plot(history.history['val_mse'], label='test')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: mlp for regression with mae loss function 是机器学习中的常用技术。  
  *mlp for regression with mae loss function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
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
# Mlp Mae Loss / 损失函数
# Complete Code / 完整代码
# ===============================

# mlp for regression with mae loss function
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
# standardize dataset
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
X = StandardScaler().fit_transform(X)
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]
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
opt = SGD(lr=0.01, momentum=0.9)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mse'])
# fit model
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
# evaluate the model
# 评估模型在测试集上的表现 / Evaluate model on test set
_, train_mse = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
_, test_mse = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
# plot loss during training
pyplot.subplot(211)
pyplot.title('Mean Absolute Error Loss', pad=-20)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot mse during training
pyplot.subplot(212)
pyplot.title('Mean Squared Error', pad=-20)
pyplot.plot(history.history['mse'], label='train')
pyplot.plot(history.history['val_mse'], label='test')
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 11

---

### Two Circles Problem

# 04 — Two Circles Problem / 04 Two Circles Problem

**Chapter 04 — File 4 of 11 / 第04章 — 第4个文件（共11个）**

---

## Summary / 总结

This script demonstrates **scatter plot of the circles dataset with points colored by class**.

本脚本演示 **scatter plot of the circles dataset with points colored by class**。

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
## Step 1 — scatter plot of the circles dataset with points colored by class

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — generate circles

```python
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
```

---
## Step 3 — select indices of points with each class label

```python
# 生成整数序列 / Generate integer sequence
for i in range(2):
	samples_ix = where(y == i)
	pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1], label=str(i))
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: scatter plot of the circles dataset with points colored by class 是机器学习中的常用技术。  
  *scatter plot of the circles dataset with points colored by class is a common technique in machine learning.*

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
# Two Circles Problem / 04 Two Circles Problem
# Complete Code / 完整代码
# ===============================

# scatter plot of the circles dataset with points colored by class
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate circles
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# select indices of points with each class label
# 生成整数序列 / Generate integer sequence
for i in range(2):
	samples_ix = where(y == i)
	pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1], label=str(i))
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 11

---

### Mlp Binary Ce Loss

# 05 — Mlp Binary Ce Loss / 损失函数

**Chapter 04 — File 5 of 11 / 第04章 — 第5个文件（共11个）**

---

## Summary / 总结

This script demonstrates **mlp for the circles problem with cross-entropy loss**.

本脚本演示 **mlp for the circles problem with cross-entropy loss**。

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
## Step 1 — mlp for the circles problem with cross-entropy loss

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — generate 2d classification dataset

```python
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
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
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
opt = SGD(lr=0.01, momentum=0.9)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
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
## Step 7 — plot loss during training

```python
pyplot.subplot(211)
pyplot.title('Binary Cross-Entropy Loss', pad=-20)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
```

---
## Step 8 — plot accuracy during training

```python
pyplot.subplot(212)
pyplot.title('Classification Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: mlp for the circles problem with cross-entropy loss 是机器学习中的常用技术。  
  *mlp for the circles problem with cross-entropy loss is a common technique in machine learning.*

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
# Mlp Binary Ce Loss / 损失函数
# Complete Code / 完整代码
# ===============================

# mlp for the circles problem with cross-entropy loss
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate 2d classification dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='sigmoid'))
opt = SGD(lr=0.01, momentum=0.9)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
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
# plot loss during training
pyplot.subplot(211)
pyplot.title('Binary Cross-Entropy Loss', pad=-20)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Classification Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 6 of 11

---

### Mlp Hinge Loss

# 06 — Mlp Hinge Loss / 损失函数

**Chapter 04 — File 6 of 11 / 第04章 — 第6个文件（共11个）**

---

## Summary / 总结

This script demonstrates **mlp for the circles problem with hinge loss**.

本脚本演示 **mlp for the circles problem with hinge loss**。

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
## Step 1 — mlp for the circles problem with hinge loss

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
```

---
## Step 2 — generate 2d classification dataset

```python
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
```

---
## Step 3 — change y from {0,1} to {-1,1}

```python
y[where(y == 0)] = -1
```

---
## Step 4 — split into train and test

```python
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 5 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='tanh'))
opt = SGD(lr=0.01, momentum=0.9)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='hinge', optimizer=opt, metrics=['accuracy'])
```

---
## Step 6 — fit model

```python
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0)
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
## Step 8 — plot loss during training

```python
pyplot.subplot(211)
pyplot.title('Hinge Loss', pad=-20)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
```

---
## Step 9 — plot accuracy during training

```python
pyplot.subplot(212)
pyplot.title('Classification Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: mlp for the circles problem with hinge loss 是机器学习中的常用技术。  
  *mlp for the circles problem with hinge loss is a common technique in machine learning.*

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
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mlp Hinge Loss / 损失函数
# Complete Code / 完整代码
# ===============================

# mlp for the circles problem with hinge loss
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# generate 2d classification dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# change y from {0,1} to {-1,1}
y[where(y == 0)] = -1
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='tanh'))
opt = SGD(lr=0.01, momentum=0.9)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='hinge', optimizer=opt, metrics=['accuracy'])
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
# plot loss during training
pyplot.subplot(211)
pyplot.title('Hinge Loss', pad=-20)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Classification Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 7 of 11

---

### Mlp Squared Hinge Loss

# 07 — Mlp Squared Hinge Loss / 损失函数

**Chapter 04 — File 7 of 11 / 第04章 — 第7个文件（共11个）**

---

## Summary / 总结

This script demonstrates **mlp for the circles problem with squared hinge loss**.

本脚本演示 **mlp for the circles problem with squared hinge loss**。

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
## Step 1 — mlp for the circles problem with squared hinge loss

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
```

---
## Step 2 — generate 2d classification dataset

```python
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
```

---
## Step 3 — change y from {0,1} to {-1,1}

```python
y[where(y == 0)] = -1
```

---
## Step 4 — split into train and test

```python
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 5 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='tanh'))
opt = SGD(lr=0.01, momentum=0.9)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='squared_hinge', optimizer=opt, metrics=['accuracy'])
```

---
## Step 6 — fit model

```python
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0)
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
## Step 8 — plot loss during training

```python
pyplot.subplot(211)
pyplot.title('Squared Hinge Loss', pad=-20)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
```

---
## Step 9 — plot accuracy during training

```python
pyplot.subplot(212)
pyplot.title('Classification Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: mlp for the circles problem with squared hinge loss 是机器学习中的常用技术。  
  *mlp for the circles problem with squared hinge loss is a common technique in machine learning.*

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
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mlp Squared Hinge Loss / 损失函数
# Complete Code / 完整代码
# ===============================

# mlp for the circles problem with squared hinge loss
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# generate 2d classification dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# change y from {0,1} to {-1,1}
y[where(y == 0)] = -1
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1, activation='tanh'))
opt = SGD(lr=0.01, momentum=0.9)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='squared_hinge', optimizer=opt, metrics=['accuracy'])
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
# plot loss during training
pyplot.subplot(211)
pyplot.title('Squared Hinge Loss', pad=-20)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Classification Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 8 of 11

---

### Blobs Problem

# 08 — Blobs Problem / 08 Blobs Problem

**Chapter 04 — File 8 of 11 / 第04章 — 第8个文件（共11个）**

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — generate dataset

```python
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 3 — select indices of points with each class label

```python
# 生成整数序列 / Generate integer sequence
for i in range(3):
	samples_ix = where(y == i)
	pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1])
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
# Blobs Problem / 08 Blobs Problem
# Complete Code / 完整代码
# ===============================

# scatter plot of blobs dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate dataset
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
# select indices of points with each class label
# 生成整数序列 / Generate integer sequence
for i in range(3):
	samples_ix = where(y == i)
	pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1])
pyplot.show()
```

---

➡️ **Next / 下一步**: File 9 of 11

---

### Mlp Categorical Ce Loss



---

### Mlp Sparse Categorical Ce Loss



---

### Mlp Kld Loss

# 11 — Mlp Kld Loss / 损失函数

**Chapter 04 — File 11 of 11 / 第04章 — 第11个文件（共11个）**

---

## Summary / 总结

This script demonstrates **mlp for the blobs multi-class classification problem with kl divergence loss**.

本脚本演示 **mlp for the blobs multi-class classification problem with kl divergence loss**。

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
## Step 1 — mlp for the blobs multi-class classification problem with kl divergence loss

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
## Step 2 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 3 — one hot encode output variable

```python
y = to_categorical(y)
```

---
## Step 4 — split into train and test

```python
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 5 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(3, activation='softmax'))
```

---
## Step 6 — compile model

```python
opt = SGD(lr=0.01, momentum=0.9)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='kullback_leibler_divergence', optimizer=opt, metrics=['accuracy'])
```

---
## Step 7 — fit model

```python
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
```

---
## Step 8 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
_, test_acc = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
```

---
## Step 9 — plot loss during training

```python
pyplot.subplot(211)
pyplot.title('Kullback Leibler Divergence Loss', pad=-20)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
```

---
## Step 10 — plot accuracy during training

```python
pyplot.subplot(212)
pyplot.title('Classification Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: mlp for the blobs multi-class classification problem with kl divergence loss 是机器学习中的常用技术。  
  *mlp for the blobs multi-class classification problem with kl divergence loss is a common technique in machine learning.*

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
# Mlp Kld Loss / 损失函数
# Complete Code / 完整代码
# ===============================

# mlp for the blobs multi-class classification problem with kl divergence loss
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
# generate 2d classification dataset
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
# one hot encode output variable
y = to_categorical(y)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(3, activation='softmax'))
# compile model
opt = SGD(lr=0.01, momentum=0.9)
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='kullback_leibler_divergence', optimizer=opt, metrics=['accuracy'])
# fit model
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
# evaluate the model
# 评估模型在测试集上的表现 / Evaluate model on test set
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
# 评估模型在测试集上的表现 / Evaluate model on test set
_, test_acc = model.evaluate(testX, testy, verbose=0)
# 打印输出 / Print output
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
pyplot.subplot(211)
pyplot.title('Kullback Leibler Divergence Loss', pad=-20)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Classification Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 04 Summary / 第04章总结

## Theme / 主题: Chapter 04 / Chapter 04

This chapter contains **11 code files** demonstrating chapter 04.

本章包含 **11 个代码文件**，演示Chapter 04。

---
## Evolution / 演化路线

  1. `01_mlp_mse_loss.ipynb` — Mlp Mse Loss
  2. `02_mlp_msle_loss.ipynb` — Mlp Msle Loss
  3. `03_mlp_mae_loss.ipynb` — Mlp Mae Loss
  4. `04_two_circles_problem.ipynb` — Two Circles Problem
  5. `05_mlp_binary_ce_loss.ipynb` — Mlp Binary Ce Loss
  6. `06_mlp_hinge_loss.ipynb` — Mlp Hinge Loss
  7. `07_mlp_squared_hinge_loss.ipynb` — Mlp Squared Hinge Loss
  8. `08_blobs_problem.ipynb` — Blobs Problem
  9. `09_mlp_categorical_ce_loss.ipynb` — Mlp Categorical Ce Loss
  10. `10_mlp_sparse_categorical_ce_loss.ipynb` — Mlp Sparse Categorical Ce Loss
  11. `11_mlp_kld_loss.ipynb` — Mlp Kld Loss

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 04) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 04）是机器学习流水线中的基础构建块。

---
