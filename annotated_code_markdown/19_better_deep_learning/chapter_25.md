# 优化深度学习 / Better Deep Learning
## Chapter 25

---

### Problem

# 01 — Problem / 01 Problem

**Chapter 25 — File 1 of 5 / 第25章 — 第1个文件（共5个）**

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

**Chapter 25 — File 2 of 5 / 第25章 — 第2个文件（共5个）**

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

### Mlp Save Members

# 03 — Mlp Save Members / 保存/加载模型

**Chapter 25 — File 3 of 5 / 第25章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of saving sub-models for later use in a stacking ensemble**.

本脚本演示 **example of saving sub-models for later use in a stacking ensemble**。

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
## Step 1 — example of saving sub-models for later use in a stacking ensemble

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
## Step 2 — fit model on dataset

```python
def fit_model(trainX, trainy):
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
model.fit(trainX, trainy, epochs=500, verbose=0)
	return model
```

---
## Step 5 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 6 — one hot encode output variable

```python
y = to_categorical(y)
```

---
## Step 7 — split into train and test

```python
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 8 — create directory for models

```python
makedirs('models')
```

---
## Step 9 — fit and save models

```python
n_members = 5
# 生成整数序列 / Generate integer sequence
for i in range(n_members):
```

---
## Step 10 — fit model

```python
model = fit_model(trainX, trainy)
```

---
## Step 11 — save model

```python
filename = 'models/model_' + str(i + 1) + '.h5'
 # 保存模型到文件 / Save model to file
	model.save(filename)
 # 打印输出 / Print output
	print('>Saved %s' % filename)
```

---
## Learning Notes / 学习笔记

- **概念**: example of saving sub-models for later use in a stacking ensemble 是机器学习中的常用技术。  
  *example of saving sub-models for later use in a stacking ensemble is a common technique in machine learning.*

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
# Mlp Save Members / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# example of saving sub-models for later use in a stacking ensemble
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
from os import makedirs

# fit model on dataset
def fit_model(trainX, trainy):
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
	model.fit(trainX, trainy, epochs=500, verbose=0)
	return model

# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# one hot encode output variable
y = to_categorical(y)
# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# create directory for models
makedirs('models')
# fit and save models
n_members = 5
# 生成整数序列 / Generate integer sequence
for i in range(n_members):
	# fit model
	model = fit_model(trainX, trainy)
	# save model
	filename = 'models/model_' + str(i + 1) + '.h5'
 # 保存模型到文件 / Save model to file
	model.save(filename)
 # 打印输出 / Print output
	print('>Saved %s' % filename)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Mlp Logistic Regression Stacking

# 04 — Mlp Logistic Regression Stacking / 回归

**Chapter 25 — File 4 of 5 / 第25章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **stacked generalization with linear meta model on blobs dataset**.

本脚本演示 **stacked generalization with linear meta model on blobs dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
## Step 1 — stacked generalization with linear meta model on blobs dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import dstack
```

---
## Step 2 — load models from file

```python
def load_all_models(n_models):
	all_models = list()
 # 生成整数序列 / Generate integer sequence
	for i in range(n_models):
```

---
## Step 3 — define filename for this ensemble

```python
filename = 'models/model_' + str(i + 1) + '.h5'
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
## Step 6 — create stacked model input dataset as outputs from the ensemble

```python
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
```

---
## Step 7 — make prediction

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(inputX, verbose=0)
```

---
## Step 8 — stack predictions into [rows, members, probabilities]

```python
if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))
```

---
## Step 9 — flatten predictions to [rows, members x probabilities]

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX
```

---
## Step 10 — fit a model based on the outputs from the ensemble members

```python
def fit_stacked_model(members, inputX, inputy):
```

---
## Step 11 — create dataset using ensemble

```python
stackedX = stacked_dataset(members, inputX)
```

---
## Step 12 — fit standalone model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
 # 训练模型 / Train the model
	model.fit(stackedX, inputy)
	return model
```

---
## Step 13 — make a prediction with the stacked model

```python
def stacked_prediction(members, model, inputX):
```

---
## Step 14 — create dataset using ensemble

```python
stackedX = stacked_dataset(members, inputX)
```

---
## Step 15 — make a prediction

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(stackedX)
	return yhat
```

---
## Step 16 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 17 — split into train and test

```python
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 18 — load all models

```python
n_members = 5
members = load_all_models(n_members)
# 打印输出 / Print output
print('Loaded %d models' % len(members))
```

---
## Step 19 — evaluate standalone models on test dataset

```python
for model in members:
	testy_enc = to_categorical(testy)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, acc = model.evaluate(testX, testy_enc, verbose=0)
 # 打印输出 / Print output
	print('Model Accuracy: %.3f' % acc)
```

---
## Step 20 — fit stacked model using the ensemble

```python
model = fit_stacked_model(members, testX, testy)
```

---
## Step 21 — evaluate model on test set

```python
yhat = stacked_prediction(members, model, testX)
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
acc = accuracy_score(testy, yhat)
# 打印输出 / Print output
print('Stacked Test Accuracy: %.3f' % acc)
```

---
## Learning Notes / 学习笔记

- **概念**: stacked generalization with linear meta model on blobs dataset 是机器学习中的常用技术。  
  *stacked generalization with linear meta model on blobs dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mlp Logistic Regression Stacking / 回归
# Complete Code / 完整代码
# ===============================

# stacked generalization with linear meta model on blobs dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import dstack

# load models from file
def load_all_models(n_models):
	all_models = list()
 # 生成整数序列 / Generate integer sequence
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'models/model_' + str(i + 1) + '.h5'
		# load model from file
  # 从文件加载模型 / Load model from file
		model = load_model(filename)
		# add to list of members
  # 添加元素到列表末尾 / Append element to list end
		all_models.append(model)
  # 打印输出 / Print output
		print('>loaded %s' % filename)
	return all_models

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
  # 用模型做预测 / Make predictions with model
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# fit standalone model
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
 # 训练模型 / Train the model
	model.fit(stackedX, inputy)
	return model

# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
 # 用模型做预测 / Make predictions with model
	yhat = model.predict(stackedX)
	return yhat

# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# load all models
n_members = 5
members = load_all_models(n_members)
# 打印输出 / Print output
print('Loaded %d models' % len(members))
# evaluate standalone models on test dataset
for model in members:
	testy_enc = to_categorical(testy)
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, acc = model.evaluate(testX, testy_enc, verbose=0)
 # 打印输出 / Print output
	print('Model Accuracy: %.3f' % acc)
# fit stacked model using the ensemble
model = fit_stacked_model(members, testX, testy)
# evaluate model on test set
yhat = stacked_prediction(members, model, testX)
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
acc = accuracy_score(testy, yhat)
# 打印输出 / Print output
print('Stacked Test Accuracy: %.3f' % acc)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Mlp Stacking

# 05 — Mlp Stacking / 堆叠方法

**Chapter 25 — File 5 of 5 / 第25章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **stacked generalization with neural net meta model on blobs dataset**.

本脚本演示 **stacked generalization with neural net meta model on blobs dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
## Step 1 — stacked generalization with neural net meta model on blobs dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import plot_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.merge import concatenate
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax
```

---
## Step 2 — load models from file

```python
def load_all_models(n_models):
	all_models = list()
 # 生成整数序列 / Generate integer sequence
	for i in range(n_models):
```

---
## Step 3 — define filename for this ensemble

```python
filename = 'models/model_' + str(i + 1) + '.h5'
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
## Step 6 — define stacked model from multiple member input models

```python
def define_stacked_model(members):
```

---
## Step 7 — update all layers in all models to not be trainable

```python
# 获取长度 / Get length
for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
```

---
## Step 8 — make not trainable

```python
layer.trainable = False
```

---
## Step 9 — rename to avoid 'unique layer name' issue

```python
layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
```

---
## Step 10 — define multi-headed input

```python
ensemble_visible = [model.input for model in members]
```

---
## Step 11 — concatenate merge output from each model

```python
ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
 # 全连接层（Keras） / Fully connected layer (Keras)
	hidden = Dense(10, activation='relu')(merge)
 # 全连接层（Keras） / Fully connected layer (Keras)
	output = Dense(3, activation='softmax')(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)
```

---
## Step 12 — plot graph of ensemble

```python
plot_model(model, show_shapes=True, to_file='model_graph.png')
```

---
## Step 13 — compile

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

---
## Step 14 — fit a stacked model

```python
def fit_stacked_model(model, inputX, inputy):
```

---
## Step 15 — prepare input data

```python
# 获取长度 / Get length
X = [inputX for _ in range(len(model.input))]
```

---
## Step 16 — encode output data

```python
inputy_enc = to_categorical(inputy)
```

---
## Step 17 — fit model

```python
# 训练模型 / Train the model
model.fit(X, inputy_enc, epochs=300, verbose=0)
```

---
## Step 18 — make a prediction with a stacked model

```python
def predict_stacked_model(model, inputX):
```

---
## Step 19 — prepare input data

```python
# 获取长度 / Get length
X = [inputX for _ in range(len(model.input))]
```

---
## Step 20 — make prediction

```python
# 用模型做预测 / Make predictions with model
return model.predict(X, verbose=0)
```

---
## Step 21 — generate 2d classification dataset

```python
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
```

---
## Step 22 — split into train and test

```python
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
```

---
## Step 23 — load all models

```python
n_members = 5
members = load_all_models(n_members)
# 打印输出 / Print output
print('Loaded %d models' % len(members))
```

---
## Step 24 — define ensemble model

```python
stacked_model = define_stacked_model(members)
```

---
## Step 25 — fit stacked model on test dataset

```python
fit_stacked_model(stacked_model, testX, testy)
```

---
## Step 26 — make predictions and evaluate

```python
yhat = predict_stacked_model(stacked_model, testX)
yhat = argmax(yhat, axis=1)
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
acc = accuracy_score(testy, yhat)
# 打印输出 / Print output
print('Stacked Test Accuracy: %.3f' % acc)
```

---
## Learning Notes / 学习笔记

- **概念**: stacked generalization with neural net meta model on blobs dataset 是机器学习中的常用技术。  
  *stacked generalization with neural net meta model on blobs dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
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
# Mlp Stacking / 堆叠方法
# Complete Code / 完整代码
# ===============================

# stacked generalization with neural net meta model on blobs dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import plot_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.merge import concatenate
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax

# load models from file
def load_all_models(n_models):
	all_models = list()
 # 生成整数序列 / Generate integer sequence
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'models/model_' + str(i + 1) + '.h5'
		# load model from file
  # 从文件加载模型 / Load model from file
		model = load_model(filename)
		# add to list of members
  # 添加元素到列表末尾 / Append element to list end
		all_models.append(model)
  # 打印输出 / Print output
		print('>loaded %s' % filename)
	return all_models

# define stacked model from multiple member input models
def define_stacked_model(members):
	# update all layers in all models to not be trainable
 # 获取长度 / Get length
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
			# rename to avoid 'unique layer name' issue
			layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
 # 全连接层（Keras） / Fully connected layer (Keras)
	hidden = Dense(10, activation='relu')(merge)
 # 全连接层（Keras） / Fully connected layer (Keras)
	output = Dense(3, activation='softmax')(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)
	# plot graph of ensemble
	plot_model(model, show_shapes=True, to_file='model_graph.png')
	# compile
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
	# prepare input data
 # 获取长度 / Get length
	X = [inputX for _ in range(len(model.input))]
	# encode output data
	inputy_enc = to_categorical(inputy)
	# fit model
 # 训练模型 / Train the model
	model.fit(X, inputy_enc, epochs=300, verbose=0)

# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
	# prepare input data
 # 获取长度 / Get length
	X = [inputX for _ in range(len(model.input))]
	# make prediction
 # 用模型做预测 / Make predictions with model
	return model.predict(X, verbose=0)

# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# load all models
n_members = 5
members = load_all_models(n_members)
# 打印输出 / Print output
print('Loaded %d models' % len(members))
# define ensemble model
stacked_model = define_stacked_model(members)
# fit stacked model on test dataset
fit_stacked_model(stacked_model, testX, testy)
# make predictions and evaluate
yhat = predict_stacked_model(stacked_model, testX)
yhat = argmax(yhat, axis=1)
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
acc = accuracy_score(testy, yhat)
# 打印输出 / Print output
print('Stacked Test Accuracy: %.3f' % acc)
```

---

### Chapter Summary / 章节总结

# Chapter 25 Summary / 第25章总结

## Theme / 主题: Chapter 25 / Chapter 25

This chapter contains **5 code files** demonstrating chapter 25.

本章包含 **5 个代码文件**，演示Chapter 25。

---
## Evolution / 演化路线

  1. `01_problem.ipynb` — Problem
  2. `02_mlp.ipynb` — Mlp
  3. `03_mlp_save_members.ipynb` — Mlp Save Members
  4. `04_mlp_logistic_regression_stacking.ipynb` — Mlp Logistic Regression Stacking
  5. `05_mlp_stacking.ipynb` — Mlp Stacking

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 25) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 25）是机器学习流水线中的基础构建块。

---
