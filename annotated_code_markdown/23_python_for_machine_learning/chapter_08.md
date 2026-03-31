# Python 机器学习 / Python for Machine Learning
## Chapter 08

---

### Append

# 01 — Append / 01 Append

**Chapter 08 — File 1 of 6 / 第08章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Append**.

本脚本演示 **01 Append**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
longstr = ""
# 生成整数序列 / Generate integer sequence
for x in range(1000):
  longstr += str(x)
```

---
## Learning Notes / 学习笔记

- **概念**: Append 是机器学习中的常用技术。  
  *Append is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Append / 01 Append
# Complete Code / 完整代码
# ===============================

longstr = ""
# 生成整数序列 / Generate integer sequence
for x in range(1000):
  longstr += str(x)
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Join

# 02 — Join / 02 Join

**Chapter 08 — File 2 of 6 / 第08章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Join**.

本脚本演示 **02 Join**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 生成整数序列 / Generate integer sequence
longstr = "".join([str(x) for x in range(1000)])
```

---
## Learning Notes / 学习笔记

- **概念**: Join 是机器学习中的常用技术。  
  *Join is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Join / 02 Join
# Complete Code / 完整代码
# ===============================

# 生成整数序列 / Generate integer sequence
longstr = "".join([str(x) for x in range(1000)])
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Profiling

# 08 — Profiling / 08 Profiling

**Chapter 08 — File 3 of 6 / 第08章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Profiling**.

本脚本演示 **08 Profiling**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import timeit
# 生成整数序列 / Generate integer sequence
measurements = timeit.repeat('[x**0.5 for x in range(1000)]', number=10000)
# 打印输出 / Print output
print(measurements)
```

---
## Learning Notes / 学习笔记

- **概念**: Profiling 是机器学习中的常用技术。  
  *Profiling is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Profiling / 08 Profiling
# Complete Code / 完整代码
# ===============================

import timeit
# 生成整数序列 / Generate integer sequence
measurements = timeit.repeat('[x**0.5 for x in range(1000)]', number=10000)
# 打印输出 / Print output
print(measurements)
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Hillclimb

# 09 — Hillclimb / 09 Hillclimb

**Chapter 08 — File 4 of 6 / 第08章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **manually search perceptron hyperparameters for binary classification**.

本脚本演示 **manually search perceptron hyperparameters for binary classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — manually search perceptron hyperparameters for binary classification

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Perceptron
```

---
## Step 2 — objective function

```python
def objective(X, y, cfg):
```

---
## Step 3 — unpack config

```python
eta, alpha = cfg
```

---
## Step 4 — define model

```python
model = Perceptron(penalty='elasticnet', alpha=alpha, eta0=eta)
```

---
## Step 5 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 6 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 7 — calculate mean accuracy

```python
result = mean(scores)
	return result
```

---
## Step 8 — take a step in the search space

```python
def step(cfg, step_size):
```

---
## Step 9 — unpack the configuration

```python
eta, alpha = cfg
```

---
## Step 10 — step eta

```python
new_eta = eta + randn() * step_size
```

---
## Step 11 — check the bounds of eta

```python
if new_eta <= 0.0:
		new_eta = 1e-8
	if new_eta > 1.0:
		new_eta = 1.0
```

---
## Step 12 — step alpha

```python
new_alpha = alpha + randn() * step_size
```

---
## Step 13 — check the bounds of alpha

```python
if new_alpha < 0.0:
		new_alpha = 0.0
```

---
## Step 14 — return the new configuration

```python
return [new_eta, new_alpha]
```

---
## Step 15 — hill climbing local search algorithm

```python
def hillclimbing(X, y, objective, n_iter, step_size):
```

---
## Step 16 — starting point for the search

```python
solution = [rand(), rand()]
```

---
## Step 17 — evaluate the initial point

```python
solution_eval = objective(X, y, solution)
```

---
## Step 18 — run the hill climb

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_iter):
```

---
## Step 19 — take a step

```python
candidate = step(solution, step_size)
```

---
## Step 20 — evaluate candidate point

```python
candidate_eval = objective(X, y, candidate)
```

---
## Step 21 — check if we should keep the new point

```python
if candidate_eval >= solution_eval:
```

---
## Step 22 — store the new point

```python
solution, solution_eval = candidate, candidate_eval
```

---
## Step 23 — report progress

```python
# 打印输出 / Print output
print('>%d, cfg=%s %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]
```

---
## Step 24 — define dataset

```python
X, y = make_classification(n_samples=1000,
                           n_features=5, n_informative=2, n_redundant=1, random_state=1)
```

---
## Step 25 — define the total iterations

```python
n_iter = 100
```

---
## Step 26 — step size in the search space

```python
step_size = 0.1
```

---
## Step 27 — perform the hill climbing search

```python
cfg, score = hillclimbing(X, y, objective, n_iter, step_size)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('cfg=%s: Mean Accuracy: %f' % (cfg, score))
```

---
## Learning Notes / 学习笔记

- **概念**: manually search perceptron hyperparameters for binary classification 是机器学习中的常用技术。  
  *manually search perceptron hyperparameters for binary classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hillclimb / 09 Hillclimb
# Complete Code / 完整代码
# ===============================

# manually search perceptron hyperparameters for binary classification
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Perceptron

# objective function
def objective(X, y, cfg):
	# unpack config
	eta, alpha = cfg
	# define model
	model = Perceptron(penalty='elasticnet', alpha=alpha, eta0=eta)
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# calculate mean accuracy
	result = mean(scores)
	return result

# take a step in the search space
def step(cfg, step_size):
	# unpack the configuration
	eta, alpha = cfg
	# step eta
	new_eta = eta + randn() * step_size
	# check the bounds of eta
	if new_eta <= 0.0:
		new_eta = 1e-8
	if new_eta > 1.0:
		new_eta = 1.0
	# step alpha
	new_alpha = alpha + randn() * step_size
	# check the bounds of alpha
	if new_alpha < 0.0:
		new_alpha = 0.0
	# return the new configuration
	return [new_eta, new_alpha]

# hill climbing local search algorithm
def hillclimbing(X, y, objective, n_iter, step_size):
	# starting point for the search
	solution = [rand(), rand()]
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
 # 生成整数序列 / Generate integer sequence
	for i in range(n_iter):
		# take a step
		candidate = step(solution, step_size)
		# evaluate candidate point
		candidate_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidate_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidate_eval
			# report progress
   # 打印输出 / Print output
			print('>%d, cfg=%s %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_classification(n_samples=1000,
                           n_features=5, n_informative=2, n_redundant=1, random_state=1)
# define the total iterations
n_iter = 100
# step size in the search space
step_size = 0.1
# perform the hill climbing search
cfg, score = hillclimbing(X, y, objective, n_iter, step_size)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('cfg=%s: Mean Accuracy: %f' % (cfg, score))
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Hillclimb

# 14 — Hillclimb / 14 Hillclimb

**Chapter 08 — File 5 of 6 / 第08章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **manually search perceptron hyperparameters for binary classification**.

本脚本演示 **manually search perceptron hyperparameters for binary classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — manually search perceptron hyperparameters for binary classification

```python
import cProfile as profile
import pstats
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Perceptron
```

---
## Step 2 — objective function

```python
def objective(X, y, cfg):
```

---
## Step 3 — unpack config

```python
eta, alpha = cfg
```

---
## Step 4 — define model

```python
model = Perceptron(penalty='elasticnet', alpha=alpha, eta0=eta)
```

---
## Step 5 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 6 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 7 — calculate mean accuracy

```python
result = mean(scores)
    return result
```

---
## Step 8 — take a step in the search space

```python
def step(cfg, step_size):
```

---
## Step 9 — unpack the configuration

```python
eta, alpha = cfg
```

---
## Step 10 — step eta

```python
new_eta = eta + randn() * step_size
```

---
## Step 11 — check the bounds of eta

```python
if new_eta <= 0.0:
        new_eta = 1e-8
    if new_eta > 1.0:
        new_eta = 1.0
```

---
## Step 12 — step alpha

```python
new_alpha = alpha + randn() * step_size
```

---
## Step 13 — check the bounds of alpha

```python
if new_alpha < 0.0:
        new_alpha = 0.0
```

---
## Step 14 — return the new configuration

```python
return [new_eta, new_alpha]
```

---
## Step 15 — hill climbing local search algorithm

```python
def hillclimbing(X, y, objective, n_iter, step_size):
```

---
## Step 16 — starting point for the search

```python
solution = [rand(), rand()]
```

---
## Step 17 — evaluate the initial point

```python
solution_eval = objective(X, y, solution)
```

---
## Step 18 — run the hill climb

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_iter):
```

---
## Step 19 — take a step

```python
candidate = step(solution, step_size)
```

---
## Step 20 — evaluate candidate point

```python
candidate_eval = objective(X, y, candidate)
```

---
## Step 21 — check if we should keep the new point

```python
if candidate_eval >= solution_eval:
```

---
## Step 22 — store the new point

```python
solution, solution_eval = candidate, candidate_eval
```

---
## Step 23 — report progress

```python
# 打印输出 / Print output
print('>%d, cfg=%s %.5f' % (i, solution, solution_eval))
    return [solution, solution_eval]
```

---
## Step 24 — define dataset

```python
X, y = make_classification(n_samples=1000,
                           n_features=5, n_informative=2, n_redundant=1, random_state=1)
```

---
## Step 25 — define the total iterations

```python
n_iter = 100
```

---
## Step 26 — step size in the search space

```python
step_size = 0.1
```

---
## Step 27 — perform the hill climbing search with profiling

```python
prof = profile.Profile()
prof.enable()
cfg, score = hillclimbing(X, y, objective, n_iter, step_size)
prof.disable()
```

---
## Step 28 — print program output

```python
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('cfg=%s: Mean Accuracy: %f' % (cfg, score))
```

---
## Step 29 — print profiling output

```python
stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
stats.print_stats(10) # top 10 rows
```

---
## Learning Notes / 学习笔记

- **概念**: manually search perceptron hyperparameters for binary classification 是机器学习中的常用技术。  
  *manually search perceptron hyperparameters for binary classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hillclimb / 14 Hillclimb
# Complete Code / 完整代码
# ===============================

# manually search perceptron hyperparameters for binary classification
import cProfile as profile
import pstats
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Perceptron

# objective function
def objective(X, y, cfg):
    # unpack config
    eta, alpha = cfg
    # define model
    model = Perceptron(penalty='elasticnet', alpha=alpha, eta0=eta)
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # calculate mean accuracy
    result = mean(scores)
    return result

# take a step in the search space
def step(cfg, step_size):
    # unpack the configuration
    eta, alpha = cfg
    # step eta
    new_eta = eta + randn() * step_size
    # check the bounds of eta
    if new_eta <= 0.0:
        new_eta = 1e-8
    if new_eta > 1.0:
        new_eta = 1.0
    # step alpha
    new_alpha = alpha + randn() * step_size
    # check the bounds of alpha
    if new_alpha < 0.0:
        new_alpha = 0.0
    # return the new configuration
    return [new_eta, new_alpha]

# hill climbing local search algorithm
def hillclimbing(X, y, objective, n_iter, step_size):
    # starting point for the search
    solution = [rand(), rand()]
    # evaluate the initial point
    solution_eval = objective(X, y, solution)
    # run the hill climb
    # 生成整数序列 / Generate integer sequence
    for i in range(n_iter):
        # take a step
        candidate = step(solution, step_size)
        # evaluate candidate point
        candidate_eval = objective(X, y, candidate)
        # check if we should keep the new point
        if candidate_eval >= solution_eval:
            # store the new point
            solution, solution_eval = candidate, candidate_eval
            # report progress
            # 打印输出 / Print output
            print('>%d, cfg=%s %.5f' % (i, solution, solution_eval))
    return [solution, solution_eval]

# define dataset
X, y = make_classification(n_samples=1000,
                           n_features=5, n_informative=2, n_redundant=1, random_state=1)
# define the total iterations
n_iter = 100
# step size in the search space
step_size = 0.1
# perform the hill climbing search with profiling
prof = profile.Profile()
prof.enable()
cfg, score = hillclimbing(X, y, objective, n_iter, step_size)
prof.disable()
# print program output
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('cfg=%s: Mean Accuracy: %f' % (cfg, score))
# print profiling output
stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
stats.print_stats(10) # top 10 rows
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Lenet5

# 15 — Lenet5 / 15 Lenet5

**Chapter 08 — File 6 of 6 / 第08章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Load and reshape data to shape of (n_sample, height, width, n_channel)**.

本脚本演示 **Load and reshape data to shape of (n_sample, height, width, n_channel)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
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
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import EarlyStopping
```

---
## Step 2 — Load and reshape data to shape of (n_sample, height, width, n_channel)

```python
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 转换数据类型 / Convert data type
X_train = np.expand_dims(X_train, axis=3).astype('float32')
# 转换数据类型 / Convert data type
X_test = np.expand_dims(X_test, axis=3).astype('float32')
```

---
## Step 3 — One-hot encode the output

```python
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

---
## Step 4 — LeNet5 model

```python
model = Sequential([
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(6, (5,5), input_shape=(28,28,1), padding="same", activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(16, (5,5), activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(120, (5,5), activation="tanh"),
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    Flatten(),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(84, activation="tanh"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(10, activation="softmax")
])
model.summary(line_length=100)
```

---
## Step 5 — Training

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# 早停：验证集不再提升时自动停止训练 / EarlyStopping: stop when validation stops improving
earlystopping = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=20, batch_size=32, callbacks=[earlystopping])
```

---
## Step 6 — Evaluate

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
print(model.evaluate(X_test, y_test, verbose=0))
```

---
## Learning Notes / 学习笔记

- **概念**: Load and reshape data to shape of (n_sample, height, width, n_channel) 是机器学习中的常用技术。  
  *Load and reshape data to shape of (n_sample, height, width, n_channel) is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `EarlyStopping` | 早停：验证集不再提升时停止训练 | Stop when validation stops improving |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
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
# Lenet5 / 15 Lenet5
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.utils import to_categorical
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.callbacks import EarlyStopping

# Load and reshape data to shape of (n_sample, height, width, n_channel)
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 转换数据类型 / Convert data type
X_train = np.expand_dims(X_train, axis=3).astype('float32')
# 转换数据类型 / Convert data type
X_test = np.expand_dims(X_test, axis=3).astype('float32')

# One-hot encode the output
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# LeNet5 model
model = Sequential([
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(6, (5,5), input_shape=(28,28,1), padding="same", activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(16, (5,5), activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    # 二维卷积层（Keras） / 2D convolution layer (Keras)
    Conv2D(120, (5,5), activation="tanh"),
    # 展平层：多维→一维 / Flatten: multi-dim → 1D
    Flatten(),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(84, activation="tanh"),
    # 全连接层（Keras） / Fully connected layer (Keras)
    Dense(10, activation="softmax")
])
model.summary(line_length=100)

# Training
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# 早停：验证集不再提升时自动停止训练 / EarlyStopping: stop when validation stops improving
earlystopping = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
# 训练模型 / Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=20, batch_size=32, callbacks=[earlystopping])

# Evaluate
# 评估模型在测试集上的表现 / Evaluate model on test set
print(model.evaluate(X_test, y_test, verbose=0))
```

---

### Chapter Summary / 章节总结

# Chapter 08 Summary / 第08章总结

## Theme / 主题: Chapter 08 / Chapter 08

This chapter contains **6 code files** demonstrating chapter 08.

本章包含 **6 个代码文件**，演示Chapter 08。

---
## Evolution / 演化路线

  1. `01_append.ipynb` — Append
  2. `02_join.ipynb` — Join
  3. `08_profiling.ipynb` — Profiling
  4. `09_hillclimb.ipynb` — Hillclimb
  5. `14_hillclimb.ipynb` — Hillclimb
  6. `15_lenet5.ipynb` — Lenet5

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 08) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 08）是机器学习流水线中的基础构建块。

---
