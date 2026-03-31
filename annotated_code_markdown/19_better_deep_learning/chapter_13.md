# 优化深度学习 / Better Deep Learning
## Chapter 13

---

### Problem



---

### Mlp Overfit



---

### Mlp Weight Reg



---

### Grid Search

# 04 — Grid Search / 04 Grid Search

**Chapter 13 — File 4 of 4 / 第13章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **grid search regularization values for moons dataset**.

本脚本演示 **grid search regularization values for moons dataset**。

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
## Step 1 — grid search regularization values for moons dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_moons
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.regularizers import l2
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — generate 2d classification dataset

```python
X, y = make_moons(n_samples=100, noise=0.2, random_state=1)
```

---
## Step 3 — split into train and test

```python
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 4 — grid search values

```python
values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
all_train, all_test = list(), list()
for param in values:
```

---
## Step 5 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(500, input_dim=2, activation='relu', kernel_regularizer=l2(param)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Step 6 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainy, epochs=4000, verbose=0)
```

---
## Step 7 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, test_acc = model.evaluate(testX, testy, verbose=0)
 # 打印输出 / Print output
	print('Param: %f, Train: %.3f, Test: %.3f' % (param, train_acc, test_acc))
 # 添加元素到列表末尾 / Append element to list end
	all_train.append(train_acc)
 # 添加元素到列表末尾 / Append element to list end
	all_test.append(test_acc)
```

---
## Step 8 — plot train and test means

```python
pyplot.semilogx(values, all_train, label='train', marker='o')
pyplot.semilogx(values, all_test, label='test', marker='o')
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: grid search regularization values for moons dataset 是机器学习中的常用技术。  
  *grid search regularization values for moons dataset is a common technique in machine learning.*

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
| `regularization` | 正则化：防止过拟合 | Regularization: prevent overfitting |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grid Search / 04 Grid Search
# Complete Code / 完整代码
# ===============================

# grid search regularization values for moons dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_moons
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.regularizers import l2
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate 2d classification dataset
X, y = make_moons(n_samples=100, noise=0.2, random_state=1)
# split into train and test
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# grid search values
values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
all_train, all_test = list(), list()
for param in values:
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(500, input_dim=2, activation='relu', kernel_regularizer=l2(param)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
 # 训练模型 / Train the model
	model.fit(trainX, trainy, epochs=4000, verbose=0)
	# evaluate the model
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, train_acc = model.evaluate(trainX, trainy, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, test_acc = model.evaluate(testX, testy, verbose=0)
 # 打印输出 / Print output
	print('Param: %f, Train: %.3f, Test: %.3f' % (param, train_acc, test_acc))
 # 添加元素到列表末尾 / Append element to list end
	all_train.append(train_acc)
 # 添加元素到列表末尾 / Append element to list end
	all_test.append(test_acc)
# plot train and test means
pyplot.semilogx(values, all_train, label='train', marker='o')
pyplot.semilogx(values, all_test, label='test', marker='o')
pyplot.legend()
pyplot.show()
```

---

### Chapter Summary / 章节总结



---
