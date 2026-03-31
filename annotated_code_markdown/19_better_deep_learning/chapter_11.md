# 优化深度学习 / Better Deep Learning
## Chapter 11

---

### Problem



---

### Mlp Problem1

# 02 — Mlp Problem1 / 02 Mlp Problem1

**Chapter 11 — File 2 of 5 / 第11章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **fit mlp model on problem 1 and save model to file**.

本脚本演示 **fit mlp model on problem 1 and save model to file**。

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
## Step 1 — fit mlp model on problem 1 and save model to file

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — prepare a blobs examples with a given random seed

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
def samples_for_seed(seed):
```

---
## Step 3 — generate samples

```python
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=seed)
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
	return trainX, trainy, testX, testy
```

---
## Step 6 — define and fit model on a training dataset

```python
def fit_model(trainX, trainy, testX, testy):
```

---
## Step 7 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
```

---
## Step 8 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

---
## Step 9 — fit model

```python
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
	return model, history
```

---
## Step 10 — summarize the performance of the fit model

```python
def summarize_model(model, history, trainX, trainy, testX, testy):
```

---
## Step 11 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, test_acc = model.evaluate(testX, testy, verbose=0)
 # 打印输出 / Print output
	print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
```

---
## Step 12 — plot loss during training

```python
pyplot.subplot(211)
	pyplot.title('Loss')
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
```

---
## Step 13 — plot accuracy during training

```python
pyplot.subplot(212)
	pyplot.title('Accuracy')
	pyplot.plot(history.history['accuracy'], label='train')
	pyplot.plot(history.history['val_accuracy'], label='test')
	pyplot.legend()
	pyplot.show()
```

---
## Step 14 — prepare data

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
trainX, trainy, testX, testy = samples_for_seed(1)
```

---
## Step 15 — fit model on train dataset

```python
model, history = fit_model(trainX, trainy, testX, testy)
```

---
## Step 16 — evaluate model behavior

```python
summarize_model(model, history, trainX, trainy, testX, testy)
```

---
## Step 17 — save model to file

```python
# 保存模型到文件 / Save model to file
model.save('model.h5')
```

---
## Learning Notes / 学习笔记

- **概念**: fit mlp model on problem 1 and save model to file 是机器学习中的常用技术。  
  *fit mlp model on problem 1 and save model to file is a common technique in machine learning.*

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
# Mlp Problem1 / 02 Mlp Problem1
# Complete Code / 完整代码
# ===============================

# fit mlp model on problem 1 and save model to file
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# prepare a blobs examples with a given random seed
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
def samples_for_seed(seed):
	# generate samples
	X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=seed)
	# one hot encode output variable
	y = to_categorical(y)
	# split into train and test
	n_train = 500
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy

# define and fit model on a training dataset
def fit_model(trainX, trainy, testX, testy):
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
	# compile model
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	# fit model
 # 训练模型 / Train the model
	history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
	return model, history

# summarize the performance of the fit model
def summarize_model(model, history, trainX, trainy, testX, testy):
	# evaluate the model
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, train_acc = model.evaluate(trainX, trainy, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, test_acc = model.evaluate(testX, testy, verbose=0)
 # 打印输出 / Print output
	print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
	# plot loss during training
	pyplot.subplot(211)
	pyplot.title('Loss')
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	# plot accuracy during training
	pyplot.subplot(212)
	pyplot.title('Accuracy')
	pyplot.plot(history.history['accuracy'], label='train')
	pyplot.plot(history.history['val_accuracy'], label='test')
	pyplot.legend()
	pyplot.show()

# prepare data
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
trainX, trainy, testX, testy = samples_for_seed(1)
# fit model on train dataset
model, history = fit_model(trainX, trainy, testX, testy)
# evaluate model behavior
summarize_model(model, history, trainX, trainy, testX, testy)
# save model to file
# 保存模型到文件 / Save model to file
model.save('model.h5')
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Mlp Problem2

# 03 — Mlp Problem2 / 03 Mlp Problem2

**Chapter 11 — File 3 of 5 / 第11章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **fit mlp model on problem 2 and save model to file**.

本脚本演示 **fit mlp model on problem 2 and save model to file**。

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
## Step 1 — fit mlp model on problem 2 and save model to file

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — prepare a blobs examples with a given random seed

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
def samples_for_seed(seed):
```

---
## Step 3 — generate samples

```python
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=seed)
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
	return trainX, trainy, testX, testy
```

---
## Step 6 — define and fit model on a training dataset

```python
def fit_model(trainX, trainy, testX, testy):
```

---
## Step 7 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
```

---
## Step 8 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

---
## Step 9 — fit model

```python
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
	return model, history
```

---
## Step 10 — summarize the performance of the fit model

```python
def summarize_model(model, history, trainX, trainy, testX, testy):
```

---
## Step 11 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, test_acc = model.evaluate(testX, testy, verbose=0)
 # 打印输出 / Print output
	print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
```

---
## Step 12 — plot loss during training

```python
pyplot.subplot(211)
	pyplot.title('Loss')
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
```

---
## Step 13 — plot accuracy during training

```python
pyplot.subplot(212)
	pyplot.title('Accuracy')
	pyplot.plot(history.history['accuracy'], label='train')
	pyplot.plot(history.history['val_accuracy'], label='test')
	pyplot.legend()
	pyplot.show()
```

---
## Step 14 — prepare data

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
trainX, trainy, testX, testy = samples_for_seed(2)
```

---
## Step 15 — fit model on train dataset

```python
model, history = fit_model(trainX, trainy, testX, testy)
```

---
## Step 16 — evaluate model behavior

```python
summarize_model(model, history, trainX, trainy, testX, testy)
```

---
## Learning Notes / 学习笔记

- **概念**: fit mlp model on problem 2 and save model to file 是机器学习中的常用技术。  
  *fit mlp model on problem 2 and save model to file is a common technique in machine learning.*

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
# Mlp Problem2 / 03 Mlp Problem2
# Complete Code / 完整代码
# ===============================

# fit mlp model on problem 2 and save model to file
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# prepare a blobs examples with a given random seed
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
def samples_for_seed(seed):
	# generate samples
	X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=seed)
	# one hot encode output variable
	y = to_categorical(y)
	# split into train and test
	n_train = 500
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy

# define and fit model on a training dataset
def fit_model(trainX, trainy, testX, testy):
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
	# compile model
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	# fit model
 # 训练模型 / Train the model
	history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
	return model, history

# summarize the performance of the fit model
def summarize_model(model, history, trainX, trainy, testX, testy):
	# evaluate the model
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, train_acc = model.evaluate(trainX, trainy, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, test_acc = model.evaluate(testX, testy, verbose=0)
 # 打印输出 / Print output
	print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
	# plot loss during training
	pyplot.subplot(211)
	pyplot.title('Loss')
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	# plot accuracy during training
	pyplot.subplot(212)
	pyplot.title('Accuracy')
	pyplot.plot(history.history['accuracy'], label='train')
	pyplot.plot(history.history['val_accuracy'], label='test')
	pyplot.legend()
	pyplot.show()

# prepare data
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
trainX, trainy, testX, testy = samples_for_seed(2)
# fit model on train dataset
model, history = fit_model(trainX, trainy, testX, testy)
# evaluate model behavior
summarize_model(model, history, trainX, trainy, testX, testy)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Mlp Transfer Learning Problem2

# 04 — Mlp Transfer Learning Problem2 / 迁移学习

**Chapter 11 — File 4 of 5 / 第11章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **transfer learning with mlp model on problem 2**.

本脚本演示 **transfer learning with mlp model on problem 2**。

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
## Step 1 — transfer learning with mlp model on problem 2

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — prepare a blobs examples with a given random seed

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
def samples_for_seed(seed):
```

---
## Step 3 — generate samples

```python
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=seed)
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
	return trainX, trainy, testX, testy
```

---
## Step 6 — load and re-fit model on a training dataset

```python
def fit_model(trainX, trainy, testX, testy):
```

---
## Step 7 — load model

```python
# 从文件加载模型 / Load model from file
model = load_model('model.h5')
```

---
## Step 8 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

---
## Step 9 — re-fit model

```python
# 训练模型 / Train the model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
	return model, history
```

---
## Step 10 — summarize the performance of the fit model

```python
def summarize_model(model, history, trainX, trainy, testX, testy):
```

---
## Step 11 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, test_acc = model.evaluate(testX, testy, verbose=0)
 # 打印输出 / Print output
	print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
```

---
## Step 12 — plot loss during training

```python
pyplot.subplot(211)
	pyplot.title('Loss')
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
```

---
## Step 13 — plot accuracy during training

```python
pyplot.subplot(212)
	pyplot.title('Accuracy')
	pyplot.plot(history.history['accuracy'], label='train')
	pyplot.plot(history.history['val_accuracy'], label='test')
	pyplot.legend()
	pyplot.show()
```

---
## Step 14 — prepare data

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
trainX, trainy, testX, testy = samples_for_seed(2)
```

---
## Step 15 — fit model on train dataset

```python
model, history = fit_model(trainX, trainy, testX, testy)
```

---
## Step 16 — evaluate model behavior

```python
summarize_model(model, history, trainX, trainy, testX, testy)
```

---
## Learning Notes / 学习笔记

- **概念**: transfer learning with mlp model on problem 2 是机器学习中的常用技术。  
  *transfer learning with mlp model on problem 2 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
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
# Mlp Transfer Learning Problem2 / 迁移学习
# Complete Code / 完整代码
# ===============================

# transfer learning with mlp model on problem 2
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# prepare a blobs examples with a given random seed
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
def samples_for_seed(seed):
	# generate samples
	X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=seed)
	# one hot encode output variable
	y = to_categorical(y)
	# split into train and test
	n_train = 500
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy

# load and re-fit model on a training dataset
def fit_model(trainX, trainy, testX, testy):
	# load model
 # 从文件加载模型 / Load model from file
	model = load_model('model.h5')
	# compile model
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	# re-fit model
 # 训练模型 / Train the model
	history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
	return model, history

# summarize the performance of the fit model
def summarize_model(model, history, trainX, trainy, testX, testy):
	# evaluate the model
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, train_acc = model.evaluate(trainX, trainy, verbose=0)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, test_acc = model.evaluate(testX, testy, verbose=0)
 # 打印输出 / Print output
	print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
	# plot loss during training
	pyplot.subplot(211)
	pyplot.title('Loss')
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	# plot accuracy during training
	pyplot.subplot(212)
	pyplot.title('Accuracy')
	pyplot.plot(history.history['accuracy'], label='train')
	pyplot.plot(history.history['val_accuracy'], label='test')
	pyplot.legend()
	pyplot.show()

# prepare data
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
trainX, trainy, testX, testy = samples_for_seed(2)
# fit model on train dataset
model, history = fit_model(trainX, trainy, testX, testy)
# evaluate model behavior
summarize_model(model, history, trainX, trainy, testX, testy)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Mlp Evaluate Transfer Learning

# 05 — Mlp Evaluate Transfer Learning / 模型评估

**Chapter 11 — File 5 of 5 / 第11章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **compare standalone mlp model performance to transfer learning**.

本脚本演示 **compare standalone mlp model performance to transfer learning**。

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
## Step 1 — compare standalone mlp model performance to transfer learning

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
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
```

---
## Step 2 — prepare a blobs examples with a given random seed

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
def samples_for_seed(seed):
```

---
## Step 3 — generate samples

```python
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=seed)
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
	return trainX, trainy, testX, testy
```

---
## Step 6 — define and fit model on a training dataset

```python
def fit_model(trainX, trainy):
```

---
## Step 7 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
```

---
## Step 8 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

---
## Step 9 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainy, epochs=100, verbose=0)
	return model
```

---
## Step 10 — repeated evaluation of a standalone model

```python
def eval_standalone_model(trainX, trainy, testX, testy, n_repeats):
	scores = list()
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_repeats):
```

---
## Step 11 — define and fit a new model on the train dataset

```python
model = fit_model(trainX, trainy)
```

---
## Step 12 — evaluate model on test dataset

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, test_acc = model.evaluate(testX, testy, verbose=0)
  # 添加元素到列表末尾 / Append element to list end
		scores.append(test_acc)
	return scores
```

---
## Step 13 — repeated evaluation of a model with transfer learning

```python
def eval_transfer_model(trainX, trainy, testX, testy, n_fixed, n_repeats):
	scores = list()
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_repeats):
```

---
## Step 14 — load model

```python
# 从文件加载模型 / Load model from file
model = load_model('model.h5')
```

---
## Step 15 — mark layer weights as fixed or not trainable

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_fixed):
			model.layers[i].trainable = False
```

---
## Step 16 — re-compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

---
## Step 17 — fit model on train dataset

```python
# 训练模型 / Train the model
model.fit(trainX, trainy, epochs=100, verbose=0)
```

---
## Step 18 — evaluate model on test dataset

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, test_acc = model.evaluate(testX, testy, verbose=0)
  # 添加元素到列表末尾 / Append element to list end
		scores.append(test_acc)
	return scores
```

---
## Step 19 — prepare data for problem 2

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
trainX, trainy, testX, testy = samples_for_seed(2)
n_repeats = 30
dists, dist_labels = list(), list()
```

---
## Step 20 — repeated evaluation of standalone model

```python
standalone_scores = eval_standalone_model(trainX, trainy, testX, testy, n_repeats)
# 打印输出 / Print output
print('Standalone %.3f (%.3f)' % (mean(standalone_scores), std(standalone_scores)))
# 添加元素到列表末尾 / Append element to list end
dists.append(standalone_scores)
# 添加元素到列表末尾 / Append element to list end
dist_labels.append('standalone')
```

---
## Step 21 — repeated evaluation of transfer learning model, vary fixed layers

```python
n_fixed = 3
# 生成整数序列 / Generate integer sequence
for i in range(n_fixed):
	scores = eval_transfer_model(trainX, trainy, testX, testy, i, n_repeats)
 # 打印输出 / Print output
	print('Transfer (fixed=%d) %.3f (%.3f)' % (i, mean(scores), std(scores)))
 # 添加元素到列表末尾 / Append element to list end
	dists.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	dist_labels.append('transfer f='+str(i))
```

---
## Step 22 — box and whisker plot of score distributions

```python
pyplot.boxplot(dists, labels=dist_labels)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: compare standalone mlp model performance to transfer learning 是机器学习中的常用技术。  
  *compare standalone mlp model performance to transfer learning is a common technique in machine learning.*

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
# Mlp Evaluate Transfer Learning / 模型评估
# Complete Code / 完整代码
# ===============================

# compare standalone mlp model performance to transfer learning
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
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

# prepare a blobs examples with a given random seed
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
def samples_for_seed(seed):
	# generate samples
	X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=seed)
	# one hot encode output variable
	y = to_categorical(y)
	# split into train and test
	n_train = 500
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy

# define and fit model on a training dataset
def fit_model(trainX, trainy):
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(3, activation='softmax'))
	# compile model
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	# fit model
 # 训练模型 / Train the model
	model.fit(trainX, trainy, epochs=100, verbose=0)
	return model

# repeated evaluation of a standalone model
def eval_standalone_model(trainX, trainy, testX, testy, n_repeats):
	scores = list()
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_repeats):
		# define and fit a new model on the train dataset
		model = fit_model(trainX, trainy)
		# evaluate model on test dataset
  # 评估模型在测试集上的表现 / Evaluate model on test set
		_, test_acc = model.evaluate(testX, testy, verbose=0)
  # 添加元素到列表末尾 / Append element to list end
		scores.append(test_acc)
	return scores

# repeated evaluation of a model with transfer learning
def eval_transfer_model(trainX, trainy, testX, testy, n_fixed, n_repeats):
	scores = list()
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_repeats):
		# load model
  # 从文件加载模型 / Load model from file
		model = load_model('model.h5')
		# mark layer weights as fixed or not trainable
  # 生成整数序列 / Generate integer sequence
		for i in range(n_fixed):
			model.layers[i].trainable = False
		# re-compile model
  # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
		model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
		# fit model on train dataset
  # 训练模型 / Train the model
		model.fit(trainX, trainy, epochs=100, verbose=0)
		# evaluate model on test dataset
  # 评估模型在测试集上的表现 / Evaluate model on test set
		_, test_acc = model.evaluate(testX, testy, verbose=0)
  # 添加元素到列表末尾 / Append element to list end
		scores.append(test_acc)
	return scores

# prepare data for problem 2
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
trainX, trainy, testX, testy = samples_for_seed(2)
n_repeats = 30
dists, dist_labels = list(), list()

# repeated evaluation of standalone model
standalone_scores = eval_standalone_model(trainX, trainy, testX, testy, n_repeats)
# 打印输出 / Print output
print('Standalone %.3f (%.3f)' % (mean(standalone_scores), std(standalone_scores)))
# 添加元素到列表末尾 / Append element to list end
dists.append(standalone_scores)
# 添加元素到列表末尾 / Append element to list end
dist_labels.append('standalone')

# repeated evaluation of transfer learning model, vary fixed layers
n_fixed = 3
# 生成整数序列 / Generate integer sequence
for i in range(n_fixed):
	scores = eval_transfer_model(trainX, trainy, testX, testy, i, n_repeats)
 # 打印输出 / Print output
	print('Transfer (fixed=%d) %.3f (%.3f)' % (i, mean(scores), std(scores)))
 # 添加元素到列表末尾 / Append element to list end
	dists.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	dist_labels.append('transfer f='+str(i))

# box and whisker plot of score distributions
pyplot.boxplot(dists, labels=dist_labels)
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **5 code files** demonstrating chapter 11.

本章包含 **5 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `01_problem.ipynb` — Problem
  2. `02_mlp_problem1.ipynb` — Mlp Problem1
  3. `03_mlp_problem2.ipynb` — Mlp Problem2
  4. `04_mlp_transfer_learning_problem2.ipynb` — Mlp Transfer Learning Problem2
  5. `05_mlp_evaluate_transfer_learning.ipynb` — Mlp Evaluate Transfer Learning

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---
