# 机器学习微积分 / Calculus for Machine Learning
## Chapter 31

---

### Mlp

# 08 — Mlp / 08 Mlp

**Chapter 31 — File 1 of 1 / 第31章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Find a small float to avoid division by zero**.

本脚本演示 **Find a small float to avoid division by zero**。

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
## Step 1 — Step 1

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 生成随机数 / Generate random numbers
np.random.seed(0)
```

---
## Step 2 — Find a small float to avoid division by zero

```python
epsilon = np.finfo(float).eps
```

---
## Step 3 — Sigmoid function and its differentiation

```python
def sigmoid(z):
    return 1/(1+np.exp(-z.clip(-500, 500)))
def dsigmoid(z):
    s = sigmoid(z)
    return 2 * s * (1-s)
```

---
## Step 4 — ReLU function and its differentiation

```python
def relu(z):
    return np.maximum(0, z)
def drelu(z):
    # 转换数据类型 / Convert data type
    return (z > 0).astype(float)
```

---
## Step 5 — Loss function L(y, yhat) and its differentiation

```python
def cross_entropy(y, yhat):
    """Binary cross entropy function
        L = - y log yhat - (1-y) log (1-yhat)

    Args:
        y, yhat (np.array): nx1 matrices which n are the number of data instances
    Returns:
        average cross entropy value of shape 1x1, averaging over the n instances
    """
    return ( -(y.T @ np.log(yhat.clip(epsilon)) +
               (1-y.T) @ np.log((1-yhat).clip(epsilon))
              # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
              ) / y.shape[1] )

def d_cross_entropy(y, yhat):
    """ dL/dyhat """
    return ( - np.divide(y, yhat.clip(epsilon))
             + np.divide(1-y, (1-yhat).clip(epsilon)) )

class mlp:
    '''Multilayer perceptron using numpy
    '''
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, layersizes, activations, derivatives, lossderiv):
        """remember config, then initialize array to hold NN parameters
        without init"""
```

---
## Step 6 — hold NN config

```python
self.layersizes = tuple(layersizes)
        self.activations = tuple(activations)
        self.derivatives = tuple(derivatives)
        self.lossderiv = lossderiv
        # 获取长度 / Get length
        assert len(self.layersizes)-1 == len(self.activations), \
            "number of layers and the number of activation functions do not match"
        # 获取长度 / Get length
        assert len(self.activations) == len(self.derivatives), \
            "number of activation functions and number of derivatives do not match"
        assert all(isinstance(n, int) and n >= 1 for n in layersizes), \
            "Only positive integral number of perceptons is allowed in each layer"
```

---
## Step 7 — parameters, each is a 2D numpy array

```python
# 获取长度 / Get length
L = len(self.layersizes)
        self.z = [None] * L
        self.W = [None] * L
        self.b = [None] * L
        self.a = [None] * L
        self.dz = [None] * L
        self.dW = [None] * L
        self.db = [None] * L
        self.da = [None] * L

    def initialize(self, seed=42):
        """initialize the value of weight matrices and bias vectors with small
        random numbers."""
        # 生成随机数 / Generate random numbers
        np.random.seed(seed)
        sigma = 0.1
        # 同时获取索引和值 / Get both index and value
        for l, (n_in, n_out) in enumerate(zip(self.layersizes, self.layersizes[1:]), 1):
            # 生成随机数 / Generate random numbers
            self.W[l] = np.random.randn(n_in, n_out) * sigma
            # 生成随机数 / Generate random numbers
            self.b[l] = np.random.randn(1, n_out) * sigma

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        """Feed forward using existing `W` and `b`, and overwrite the result
        variables `a` and `z`

        Args:
            x (numpy.ndarray): Input data to feed forward
        """
        self.a[0] = x
        # 同时获取索引和值 / Get both index and value
        for l, func in enumerate(self.activations, 1):
```

---
## Step 8 — z = W a + b, with `a` as output from previous layer
`W` is of size rxs and `a` the size sxn with n the number of data
instances, `z` the size rxn, `b` is rx1 and broadcast to each
column of `z`

```python
self.z[l] = (self.a[l-1] @ self.W[l]) + self.b[l]
```

---
## Step 9 — a = g(z), with `a` as output of this layer, of size rxn

```python
self.a[l] = func(self.z[l])
        return self.a[-1]

    def backward(self, y, yhat):
        """back propagation using NN output yhat and the reference output y,
        generates dW, dz, db, da
        """
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        assert y.shape[1] == self.layersizes[-1], \
            "Output size doesn't match network output size"
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        assert y.shape == yhat.shape, \
            "Output size doesn't match reference"
```

---
## Step 10 — first `da`, at the output

```python
self.da[-1] = self.lossderiv(y, yhat)
        # 同时获取索引和值 / Get both index and value
        for l, func in reversed(list(enumerate(self.derivatives, 1))):
```

---
## Step 11 — compute the differentials at this layer

```python
self.dz[l] = self.da[l] * func(self.z[l])
            self.dW[l] = self.a[l-1].T @ self.dz[l]
            # 计算均值 / Calculate mean
            self.db[l] = np.mean(self.dz[l], axis=0, keepdims=True)
            self.da[l-1] = self.dz[l] @ self.W[l].T
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            assert self.z[l].shape == self.dz[l].shape
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            assert self.W[l].shape == self.dW[l].shape
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            assert self.b[l].shape == self.db[l].shape
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            assert self.a[l].shape == self.da[l].shape

    def update(self, eta):
        """Updates W and b

        Args:
            eta (float): Learning rate
        """
        # 获取长度 / Get length
        for l in range(1, len(self.W)):
            self.W[l] -= eta * self.dW[l]
            self.b[l] -= eta * self.db[l]
```

---
## Step 12 — Make data: Two circles on x-y plane as a classification problem

```python
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = y.reshape(-1,1) # our model expects a 2D array of (n_sample, n_dim)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(y.shape)
```

---
## Step 13 — Build a model

```python
model = mlp(layersizes=[2, 4, 3, 1],
            activations=[relu, relu, sigmoid],
            derivatives=[drelu, drelu, dsigmoid],
            lossderiv=d_cross_entropy)
model.initialize()
yhat = model.forward(X)
loss = cross_entropy(y, yhat)
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y, (yhat > 0.5))
# 打印输出 / Print output
print(f"Before training - loss value {loss} accuracy {score}")
```

---
## Step 14 — train for each epoch

```python
n_epochs = 150
learning_rate = 0.005
# 生成整数序列 / Generate integer sequence
for n in range(n_epochs):
    model.forward(X)
    yhat = model.a[-1]
    # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
    model.backward(y, yhat)
    model.update(learning_rate)
    loss = cross_entropy(y, yhat)
    # 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
    score = accuracy_score(y, (yhat > 0.5))
    # 打印输出 / Print output
    print(f"Iteration {n} - loss value {loss} accuracy {score}")
```

---
## Learning Notes / 学习笔记

- **概念**: Find a small float to avoid division by zero 是机器学习中的常用技术。  
  *Find a small float to avoid division by zero is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.mean` | 计算均值 | Calculate mean |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mlp / 08 Mlp
# Complete Code / 完整代码
# ===============================

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 生成随机数 / Generate random numbers
np.random.seed(0)

# Find a small float to avoid division by zero
epsilon = np.finfo(float).eps

# Sigmoid function and its differentiation
def sigmoid(z):
    return 1/(1+np.exp(-z.clip(-500, 500)))
def dsigmoid(z):
    s = sigmoid(z)
    return 2 * s * (1-s)

# ReLU function and its differentiation
def relu(z):
    return np.maximum(0, z)
def drelu(z):
    # 转换数据类型 / Convert data type
    return (z > 0).astype(float)

# Loss function L(y, yhat) and its differentiation
def cross_entropy(y, yhat):
    """Binary cross entropy function
        L = - y log yhat - (1-y) log (1-yhat)

    Args:
        y, yhat (np.array): nx1 matrices which n are the number of data instances
    Returns:
        average cross entropy value of shape 1x1, averaging over the n instances
    """
    return ( -(y.T @ np.log(yhat.clip(epsilon)) +
               (1-y.T) @ np.log((1-yhat).clip(epsilon))
              # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
              ) / y.shape[1] )

def d_cross_entropy(y, yhat):
    """ dL/dyhat """
    return ( - np.divide(y, yhat.clip(epsilon))
             + np.divide(1-y, (1-yhat).clip(epsilon)) )

class mlp:
    '''Multilayer perceptron using numpy
    '''
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, layersizes, activations, derivatives, lossderiv):
        """remember config, then initialize array to hold NN parameters
        without init"""
        # hold NN config
        self.layersizes = tuple(layersizes)
        self.activations = tuple(activations)
        self.derivatives = tuple(derivatives)
        self.lossderiv = lossderiv
        # 获取长度 / Get length
        assert len(self.layersizes)-1 == len(self.activations), \
            "number of layers and the number of activation functions do not match"
        # 获取长度 / Get length
        assert len(self.activations) == len(self.derivatives), \
            "number of activation functions and number of derivatives do not match"
        assert all(isinstance(n, int) and n >= 1 for n in layersizes), \
            "Only positive integral number of perceptons is allowed in each layer"
        # parameters, each is a 2D numpy array
        # 获取长度 / Get length
        L = len(self.layersizes)
        self.z = [None] * L
        self.W = [None] * L
        self.b = [None] * L
        self.a = [None] * L
        self.dz = [None] * L
        self.dW = [None] * L
        self.db = [None] * L
        self.da = [None] * L

    def initialize(self, seed=42):
        """initialize the value of weight matrices and bias vectors with small
        random numbers."""
        # 生成随机数 / Generate random numbers
        np.random.seed(seed)
        sigma = 0.1
        # 同时获取索引和值 / Get both index and value
        for l, (n_in, n_out) in enumerate(zip(self.layersizes, self.layersizes[1:]), 1):
            # 生成随机数 / Generate random numbers
            self.W[l] = np.random.randn(n_in, n_out) * sigma
            # 生成随机数 / Generate random numbers
            self.b[l] = np.random.randn(1, n_out) * sigma

    # 前向传播：定义数据如何流过模型 / Forward pass: define data flow through model
    def forward(self, x):
        """Feed forward using existing `W` and `b`, and overwrite the result
        variables `a` and `z`

        Args:
            x (numpy.ndarray): Input data to feed forward
        """
        self.a[0] = x
        # 同时获取索引和值 / Get both index and value
        for l, func in enumerate(self.activations, 1):
            # z = W a + b, with `a` as output from previous layer
            # `W` is of size rxs and `a` the size sxn with n the number of data
            # instances, `z` the size rxn, `b` is rx1 and broadcast to each
            # column of `z`
            self.z[l] = (self.a[l-1] @ self.W[l]) + self.b[l]
            # a = g(z), with `a` as output of this layer, of size rxn
            self.a[l] = func(self.z[l])
        return self.a[-1]

    def backward(self, y, yhat):
        """back propagation using NN output yhat and the reference output y,
        generates dW, dz, db, da
        """
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        assert y.shape[1] == self.layersizes[-1], \
            "Output size doesn't match network output size"
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        assert y.shape == yhat.shape, \
            "Output size doesn't match reference"
        # first `da`, at the output
        self.da[-1] = self.lossderiv(y, yhat)
        # 同时获取索引和值 / Get both index and value
        for l, func in reversed(list(enumerate(self.derivatives, 1))):
            # compute the differentials at this layer
            self.dz[l] = self.da[l] * func(self.z[l])
            self.dW[l] = self.a[l-1].T @ self.dz[l]
            # 计算均值 / Calculate mean
            self.db[l] = np.mean(self.dz[l], axis=0, keepdims=True)
            self.da[l-1] = self.dz[l] @ self.W[l].T
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            assert self.z[l].shape == self.dz[l].shape
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            assert self.W[l].shape == self.dW[l].shape
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            assert self.b[l].shape == self.db[l].shape
            # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
            assert self.a[l].shape == self.da[l].shape

    def update(self, eta):
        """Updates W and b

        Args:
            eta (float): Learning rate
        """
        # 获取长度 / Get length
        for l in range(1, len(self.W)):
            self.W[l] -= eta * self.dW[l]
            self.b[l] -= eta * self.db[l]

# Make data: Two circles on x-y plane as a classification problem
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
y = y.reshape(-1,1) # our model expects a 2D array of (n_sample, n_dim)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(y.shape)

# Build a model
model = mlp(layersizes=[2, 4, 3, 1],
            activations=[relu, relu, sigmoid],
            derivatives=[drelu, drelu, dsigmoid],
            lossderiv=d_cross_entropy)
model.initialize()
yhat = model.forward(X)
loss = cross_entropy(y, yhat)
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y, (yhat > 0.5))
# 打印输出 / Print output
print(f"Before training - loss value {loss} accuracy {score}")

# train for each epoch
n_epochs = 150
learning_rate = 0.005
# 生成整数序列 / Generate integer sequence
for n in range(n_epochs):
    model.forward(X)
    yhat = model.a[-1]
    # 反向传播：计算所有参数的梯度 / Backprop: compute gradients for all parameters
    model.backward(y, yhat)
    model.update(learning_rate)
    loss = cross_entropy(y, yhat)
    # 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
    score = accuracy_score(y, (yhat > 0.5))
    # 打印输出 / Print output
    print(f"Iteration {n} - loss value {loss} accuracy {score}")
```

---

### Chapter Summary / 章节总结

# Chapter 31 Summary / 第31章总结

## Theme / 主题: Chapter 31 / Chapter 31

This chapter contains **1 code files** demonstrating chapter 31.

本章包含 **1 个代码文件**，演示Chapter 31。

---
## Evolution / 演化路线

  1. `08_mlp.ipynb` — Mlp

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 31) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 31）是机器学习流水线中的基础构建块。

---
