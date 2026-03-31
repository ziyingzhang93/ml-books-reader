# 计算机视觉深度学习 / Deep Learning for Computer Vision
## Chapter 19

---

### Load Dataset



---

### Baseline Model

# 02 — Baseline Model / 02 Baseline Model

**Chapter 19 — File 2 of 7 / 第19章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **baseline cnn model for fashion mnist**.

本脚本演示 **baseline cnn model for fashion mnist**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — baseline cnn model for fashion mnist

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets import fashion_mnist
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
```

---
## Step 2 — load train and test dataset

```python
def load_dataset():
```

---
## Step 3 — load dataset

```python
# 加载数据集 / Load dataset
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
```

---
## Step 4 — reshape dataset to have a single channel

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
```

---
## Step 5 — one hot encode target values

```python
trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
```

---
## Step 6 — scale pixels

```python
def prep_pixels(train, test):
```

---
## Step 7 — convert from integers to floats

```python
# 转换数据类型 / Convert data type
train_norm = train.astype('float32')
 # 转换数据类型 / Convert data type
	test_norm = test.astype('float32')
```

---
## Step 8 — normalize to range 0-1

```python
train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
```

---
## Step 9 — return normalized images

```python
return train_norm, test_norm
```

---
## Step 10 — define cnn model

```python
def define_model():
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(10, activation='softmax'))
```

---
## Step 11 — compile model

```python
opt = SGD(lr=0.01, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
```

---
## Step 12 — evaluate a model using k-fold cross-validation

```python
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
```

---
## Step 13 — prepare cross validation

```python
kfold = KFold(n_folds, shuffle=True, random_state=1)
```

---
## Step 14 — enumerate splits

```python
for train_ix, test_ix in kfold.split(dataX):
```

---
## Step 15 — define model

```python
model = define_model()
```

---
## Step 16 — select rows for train and test

```python
trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
```

---
## Step 17 — fit model

```python
# 训练模型 / Train the model
history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
```

---
## Step 18 — evaluate model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, acc = model.evaluate(testX, testY, verbose=0)
  # 打印输出 / Print output
		print('> %.3f' % (acc * 100.0))
```

---
## Step 19 — append scores

```python
# 添加元素到列表末尾 / Append element to list end
scores.append(acc)
  # 添加元素到列表末尾 / Append element to list end
		histories.append(history)
	return scores, histories
```

---
## Step 20 — plot diagnostic learning curves

```python
def summarize_diagnostics(histories):
 # 获取长度 / Get length
	for i in range(len(histories)):
```

---
## Step 21 — plot loss

```python
pyplot.subplot(211)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
```

---
## Step 22 — plot accuracy

```python
pyplot.subplot(212)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	pyplot.show()
```

---
## Step 23 — summarize model performance

```python
def summarize_performance(scores):
```

---
## Step 24 — print summary

```python
# 打印输出 / Print output
print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
```

---
## Step 25 — box and whisker plots of results

```python
pyplot.boxplot(scores)
	pyplot.show()
```

---
## Step 26 — run the test harness for evaluating a model

```python
def run_test_harness():
```

---
## Step 27 — load dataset

```python
trainX, trainY, testX, testY = load_dataset()
```

---
## Step 28 — prepare pixel data

```python
trainX, testX = prep_pixels(trainX, testX)
```

---
## Step 29 — evaluate model

```python
scores, histories = evaluate_model(trainX, trainY)
```

---
## Step 30 — learning curves

```python
summarize_diagnostics(histories)
```

---
## Step 31 — summarize estimated performance

```python
summarize_performance(scores)
```

---
## Step 32 — entry point, run the test harness

```python
run_test_harness()
```

---
## Learning Notes / 学习笔记

- **概念**: baseline cnn model for fashion mnist 是机器学习中的常用技术。  
  *baseline cnn model for fashion mnist is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
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
# Baseline Model / 02 Baseline Model
# Complete Code / 完整代码
# ===============================

# baseline cnn model for fashion mnist
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets import fashion_mnist
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD

# load train and test dataset
def load_dataset():
	# load dataset
 # 加载数据集 / Load dataset
	(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
	# reshape dataset to have a single channel
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
 # 转换数据类型 / Convert data type
	train_norm = train.astype('float32')
 # 转换数据类型 / Convert data type
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
  # 训练模型 / Train the model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
  # 评估模型在测试集上的表现 / Evaluate model on test set
		_, acc = model.evaluate(testX, testY, verbose=0)
  # 打印输出 / Print output
		print('> %.3f' % (acc * 100.0))
		# append scores
  # 添加元素到列表末尾 / Append element to list end
		scores.append(acc)
  # 添加元素到列表末尾 / Append element to list end
		histories.append(history)
	return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
 # 获取长度 / Get length
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(211)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(212)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	pyplot.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
 # 打印输出 / Print output
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)

# entry point, run the test harness
run_test_harness()
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Baseline With Padding



---

### Baseline With Padding More Filters



---

### Save Final Model

# 05 — Save Final Model / 保存/加载模型

**Chapter 19 — File 5 of 7 / 第19章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **save the final model to file**.

本脚本演示 **save the final model to file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
```

---
## Step 1 — save the final model to file

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets import fashion_mnist
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
```

---
## Step 2 — load train and test dataset

```python
def load_dataset():
```

---
## Step 3 — load dataset

```python
# 加载数据集 / Load dataset
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
```

---
## Step 4 — reshape dataset to have a single channel

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
```

---
## Step 5 — one hot encode target values

```python
trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
```

---
## Step 6 — scale pixels

```python
def prep_pixels(train, test):
```

---
## Step 7 — convert from integers to floats

```python
# 转换数据类型 / Convert data type
train_norm = train.astype('float32')
 # 转换数据类型 / Convert data type
	test_norm = test.astype('float32')
```

---
## Step 8 — normalize to range 0-1

```python
train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
```

---
## Step 9 — return normalized images

```python
return train_norm, test_norm
```

---
## Step 10 — define cnn model

```python
def define_model():
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(10, activation='softmax'))
```

---
## Step 11 — compile model

```python
opt = SGD(lr=0.01, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
```

---
## Step 12 — run the test harness for evaluating a model

```python
def run_test_harness():
```

---
## Step 13 — load dataset

```python
trainX, trainY, testX, testY = load_dataset()
```

---
## Step 14 — prepare pixel data

```python
trainX, testX = prep_pixels(trainX, testX)
```

---
## Step 15 — define model

```python
model = define_model()
```

---
## Step 16 — fit model

```python
# 训练模型 / Train the model
model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
```

---
## Step 17 — save model

```python
# 保存模型到文件 / Save model to file
model.save('final_model.h5')
```

---
## Step 18 — entry point, run the test harness

```python
run_test_harness()
```

---
## Learning Notes / 学习笔记

- **概念**: save the final model to file 是机器学习中的常用技术。  
  *save the final model to file is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
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
# Save Final Model / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# save the final model to file
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets import fashion_mnist
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD

# load train and test dataset
def load_dataset():
	# load dataset
 # 加载数据集 / Load dataset
	(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
	# reshape dataset to have a single channel
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
 # 转换数据类型 / Convert data type
	train_norm = train.astype('float32')
 # 转换数据类型 / Convert data type
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model()
	# fit model
 # 训练模型 / Train the model
	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
	# save model
 # 保存模型到文件 / Save model to file
	model.save('final_model.h5')

# entry point, run the test harness
run_test_harness()
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Evaluate Final Model

# 06 — Evaluate Final Model / 模型评估

**Chapter 19 — File 6 of 7 / 第19章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **evaluate the deep model on the test dataset**.

本脚本演示 **evaluate the deep model on the test dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — evaluate the deep model on the test dataset

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets import fashion_mnist
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical
```

---
## Step 2 — load train and test dataset

```python
def load_dataset():
```

---
## Step 3 — load dataset

```python
# 加载数据集 / Load dataset
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
```

---
## Step 4 — reshape dataset to have a single channel

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
```

---
## Step 5 — one hot encode target values

```python
trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
```

---
## Step 6 — scale pixels

```python
def prep_pixels(train, test):
```

---
## Step 7 — convert from integers to floats

```python
# 转换数据类型 / Convert data type
train_norm = train.astype('float32')
 # 转换数据类型 / Convert data type
	test_norm = test.astype('float32')
```

---
## Step 8 — normalize to range 0-1

```python
train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
```

---
## Step 9 — return normalized images

```python
return train_norm, test_norm
```

---
## Step 10 — run the test harness for evaluating a model

```python
def run_test_harness():
```

---
## Step 11 — load dataset

```python
trainX, trainY, testX, testY = load_dataset()
```

---
## Step 12 — prepare pixel data

```python
trainX, testX = prep_pixels(trainX, testX)
```

---
## Step 13 — load model

```python
# 从文件加载模型 / Load model from file
model = load_model('final_model.h5')
```

---
## Step 14 — evaluate model on test dataset

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, acc = model.evaluate(testX, testY, verbose=0)
 # 打印输出 / Print output
	print('> %.3f' % (acc * 100.0))
```

---
## Step 15 — entry point, run the test harness

```python
run_test_harness()
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate the deep model on the test dataset 是机器学习中的常用技术。  
  *evaluate the deep model on the test dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.evaluate` | 评估模型 | Evaluate the model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Final Model / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate the deep model on the test dataset
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets import fashion_mnist
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import to_categorical

# load train and test dataset
def load_dataset():
	# load dataset
 # 加载数据集 / Load dataset
	(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
	# reshape dataset to have a single channel
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
 # 转换数据类型 / Convert data type
	train_norm = train.astype('float32')
 # 转换数据类型 / Convert data type
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# load model
 # 从文件加载模型 / Load model from file
	model = load_model('final_model.h5')
	# evaluate model on test dataset
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, acc = model.evaluate(testX, testY, verbose=0)
 # 打印输出 / Print output
	print('> %.3f' % (acc * 100.0))

# entry point, run the test harness
run_test_harness()
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Predict Final Model

# 07 — Predict Final Model / 07 Predict Final Model

**Chapter 19 — File 7 of 7 / 第19章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **make a prediction for a new image.**.

本脚本演示 **make a prediction for a new image.**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — make a prediction for a new image.

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
```

---
## Step 2 — load and prepare the image

```python
def load_image(filename):
```

---
## Step 3 — load the image

```python
img = load_img(filename, grayscale=True, target_size=(28, 28))
```

---
## Step 4 — convert to array

```python
img = img_to_array(img)
```

---
## Step 5 — reshape into a single sample with 1 channel

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
img = img.reshape(1, 28, 28, 1)
```

---
## Step 6 — prepare pixel data

```python
# 转换数据类型 / Convert data type
img = img.astype('float32')
	img = img / 255.0
	return img
```

---
## Step 7 — load an image and predict the class

```python
def run_example():
```

---
## Step 8 — load the image

```python
img = load_image('sample_image.png')
```

---
## Step 9 — load model

```python
# 从文件加载模型 / Load model from file
model = load_model('final_model.h5')
```

---
## Step 10 — predict the class

```python
result = model.predict_classes(img)
 # 打印输出 / Print output
	print(result[0])
```

---
## Step 11 — entry point, run the example

```python
run_example()
```

---
## Learning Notes / 学习笔记

- **概念**: make a prediction for a new image. 是机器学习中的常用技术。  
  *make a prediction for a new image. is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Predict Final Model / 07 Predict Final Model
# Complete Code / 完整代码
# ===============================

# make a prediction for a new image.
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
 # 转换数据类型 / Convert data type
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class
def run_example():
	# load the image
	img = load_image('sample_image.png')
	# load model
 # 从文件加载模型 / Load model from file
	model = load_model('final_model.h5')
	# predict the class
	result = model.predict_classes(img)
 # 打印输出 / Print output
	print(result[0])

# entry point, run the example
run_example()
```

---

### Chapter Summary / 章节总结



---
