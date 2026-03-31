# 计算机视觉深度学习 / Deep Learning for Computer Vision
## Chapter 07

---

### Mnist Dataset

# 01 — Mnist Dataset / 01 Mnist Dataset

**Chapter 07 — File 1 of 4 / 第07章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **load and summarize the MNIST dataset**.

本脚本演示 **load and summarize the MNIST dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — load and summarize the MNIST dataset

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets import mnist
```

---
## Step 2 — load dataset

```python
# 加载数据集 / Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

---
## Step 3 — summarize dataset shape

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train', train_images.shape, train_labels.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Test', (test_images.shape, test_labels.shape))
```

---
## Step 4 — summarize pixel values

```python
# 打印输出 / Print output
print('Train', train_images.min(), train_images.max(), train_images.mean(), train_images.std())
# 打印输出 / Print output
print('Test', test_images.min(), test_images.max(), test_images.mean(), test_images.std())
```

---
## Learning Notes / 学习笔记

- **概念**: load and summarize the MNIST dataset 是机器学习中的常用技术。  
  *load and summarize the MNIST dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mnist Dataset / 01 Mnist Dataset
# Complete Code / 完整代码
# ===============================

# load and summarize the MNIST dataset
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets import mnist
# load dataset
# 加载数据集 / Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# summarize dataset shape
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train', train_images.shape, train_labels.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Test', (test_images.shape, test_labels.shape))
# summarize pixel values
# 打印输出 / Print output
print('Train', train_images.min(), train_images.max(), train_images.mean(), train_images.std())
# 打印输出 / Print output
print('Test', test_images.min(), test_images.max(), test_images.mean(), test_images.std())
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Normalization

# 02 — Normalization / 02 Normalization

**Chapter 07 — File 2 of 4 / 第07章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of normalizing a image dataset**.

本脚本演示 **example of normalizing a image dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
```

---
## Step 1 — example of normalizing a image dataset

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets import mnist
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
```

---
## Step 2 — load dataset

```python
# 加载数据集 / Load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
```

---
## Step 3 — reshape dataset to have a single channel

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
width, height, channels = trainX.shape[1], trainX.shape[2], 1
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = testX.reshape((testX.shape[0], width, height, channels))
```

---
## Step 4 — confirm scale of pixels

```python
# 打印输出 / Print output
print('Train min=%.3f, max=%.3f' % (trainX.min(), trainX.max()))
# 打印输出 / Print output
print('Test min=%.3f, max=%.3f' % (testX.min(), testX.max()))
```

---
## Step 5 — create generator (1.0/255.0 = 0.003921568627451)

```python
datagen = ImageDataGenerator(rescale=1.0/255.0)
```

---
## Step 6 — Note: there is no need to fit the generator in this case
prepare a iterators to scale images

```python
train_iterator = datagen.flow(trainX, trainY, batch_size=64)
test_iterator = datagen.flow(testX, testY, batch_size=64)
# 打印输出 / Print output
print('Batches train=%d, test=%d' % (len(train_iterator), len(test_iterator)))
```

---
## Step 7 — confirm the scaling works

```python
batchX, batchy = train_iterator.next()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
```

---
## Learning Notes / 学习笔记

- **概念**: example of normalizing a image dataset 是机器学习中的常用技术。  
  *example of normalizing a image dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Normalization / 02 Normalization
# Complete Code / 完整代码
# ===============================

# example of normalizing a image dataset
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets import mnist
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# load dataset
# 加载数据集 / Load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
# reshape dataset to have a single channel
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
width, height, channels = trainX.shape[1], trainX.shape[2], 1
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = testX.reshape((testX.shape[0], width, height, channels))
# confirm scale of pixels
# 打印输出 / Print output
print('Train min=%.3f, max=%.3f' % (trainX.min(), trainX.max()))
# 打印输出 / Print output
print('Test min=%.3f, max=%.3f' % (testX.min(), testX.max()))
# create generator (1.0/255.0 = 0.003921568627451)
datagen = ImageDataGenerator(rescale=1.0/255.0)
# Note: there is no need to fit the generator in this case
# prepare a iterators to scale images
train_iterator = datagen.flow(trainX, trainY, batch_size=64)
test_iterator = datagen.flow(testX, testY, batch_size=64)
# 打印输出 / Print output
print('Batches train=%d, test=%d' % (len(train_iterator), len(test_iterator)))
# confirm the scaling works
batchX, batchy = train_iterator.next()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Centering

# 03 — Centering / 03 Centering

**Chapter 07 — File 3 of 4 / 第07章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of centering a image dataset**.

本脚本演示 **example of centering a image dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 训练模型 / Train the model

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
## Step 1 — example of centering a image dataset

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets import mnist
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
```

---
## Step 2 — load dataset

```python
# 加载数据集 / Load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
```

---
## Step 3 — reshape dataset to have a single channel

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
width, height, channels = trainX.shape[1], trainX.shape[2], 1
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = testX.reshape((testX.shape[0], width, height, channels))
```

---
## Step 4 — report per-image mean

```python
# 打印输出 / Print output
print('Means train=%.3f, test=%.3f' % (trainX.mean(), testX.mean()))
```

---
## Step 5 — create generator that centers pixel values

```python
datagen = ImageDataGenerator(featurewise_center=True)
```

---
## Step 6 — calculate the mean on the training dataset

```python
datagen.fit(trainX)
# 打印输出 / Print output
print('Data Generator Mean: %.3f' % datagen.mean)
```

---
## Step 7 — demonstrate effect on a single batch of samples

```python
iterator = datagen.flow(trainX, trainy, batch_size=64)
```

---
## Step 8 — get a batch

```python
batchX, batchy = iterator.next()
```

---
## Step 9 — mean pixel value in the batch

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(batchX.shape, batchX.mean())
```

---
## Step 10 — demonstrate effect on entire training dataset

```python
# 获取长度 / Get length
iterator = datagen.flow(trainX, trainy, batch_size=len(trainX), shuffle=False)
```

---
## Step 11 — get a batch

```python
batchX, batchy = iterator.next()
```

---
## Step 12 — mean pixel value in the batch

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(batchX.shape, batchX.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: example of centering a image dataset 是机器学习中的常用技术。  
  *example of centering a image dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Centering / 03 Centering
# Complete Code / 完整代码
# ===============================

# example of centering a image dataset
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets import mnist
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# load dataset
# 加载数据集 / Load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# reshape dataset to have a single channel
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
width, height, channels = trainX.shape[1], trainX.shape[2], 1
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = testX.reshape((testX.shape[0], width, height, channels))
# report per-image mean
# 打印输出 / Print output
print('Means train=%.3f, test=%.3f' % (trainX.mean(), testX.mean()))
# create generator that centers pixel values
datagen = ImageDataGenerator(featurewise_center=True)
# calculate the mean on the training dataset
datagen.fit(trainX)
# 打印输出 / Print output
print('Data Generator Mean: %.3f' % datagen.mean)
# demonstrate effect on a single batch of samples
iterator = datagen.flow(trainX, trainy, batch_size=64)
# get a batch
batchX, batchy = iterator.next()
# mean pixel value in the batch
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(batchX.shape, batchX.mean())
# demonstrate effect on entire training dataset
# 获取长度 / Get length
iterator = datagen.flow(trainX, trainy, batch_size=len(trainX), shuffle=False)
# get a batch
batchX, batchy = iterator.next()
# mean pixel value in the batch
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(batchX.shape, batchX.mean())
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Standardization Keras

# 04 — Standardization Keras / Keras

**Chapter 07 — File 4 of 4 / 第07章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of standardizing a image dataset**.

本脚本演示 **example of standardizing a image dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 训练模型 / Train the model

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
## Step 1 — example of standardizing a image dataset

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets import mnist
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
```

---
## Step 2 — load dataset

```python
# 加载数据集 / Load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
```

---
## Step 3 — reshape dataset to have a single channel

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
width, height, channels = trainX.shape[1], trainX.shape[2], 1
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = testX.reshape((testX.shape[0], width, height, channels))
```

---
## Step 4 — report pixel means and standard deviations

```python
# 打印输出 / Print output
print('Statistics train=%.3f (%.3f), test=%.3f (%.3f)' % (trainX.mean(), trainX.std(), testX.mean(), testX.std()))
```

---
## Step 5 — create generator that centers pixel values

```python
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
```

---
## Step 6 — calculate the mean on the training dataset

```python
datagen.fit(trainX)
# 打印输出 / Print output
print('Data Generator mean=%.3f, std=%.3f' % (datagen.mean, datagen.std))
```

---
## Step 7 — demonstrate effect on a single batch of samples

```python
iterator = datagen.flow(trainX, trainy, batch_size=64)
```

---
## Step 8 — get a batch

```python
batchX, batchy = iterator.next()
```

---
## Step 9 — pixel stats in the batch

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(batchX.shape, batchX.mean(), batchX.std())
```

---
## Step 10 — demonstrate effect on entire training dataset

```python
# 获取长度 / Get length
iterator = datagen.flow(trainX, trainy, batch_size=len(trainX), shuffle=False)
```

---
## Step 11 — get a batch

```python
batchX, batchy = iterator.next()
```

---
## Step 12 — pixel stats in the batch

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(batchX.shape, batchX.mean(), batchX.std())
```

---
## Learning Notes / 学习笔记

- **概念**: example of standardizing a image dataset 是机器学习中的常用技术。  
  *example of standardizing a image dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Standardization Keras / Keras
# Complete Code / 完整代码
# ===============================

# example of standardizing a image dataset
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets import mnist
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# load dataset
# 加载数据集 / Load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# reshape dataset to have a single channel
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
width, height, channels = trainX.shape[1], trainX.shape[2], 1
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = testX.reshape((testX.shape[0], width, height, channels))
# report pixel means and standard deviations
# 打印输出 / Print output
print('Statistics train=%.3f (%.3f), test=%.3f (%.3f)' % (trainX.mean(), trainX.std(), testX.mean(), testX.std()))
# create generator that centers pixel values
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# calculate the mean on the training dataset
datagen.fit(trainX)
# 打印输出 / Print output
print('Data Generator mean=%.3f, std=%.3f' % (datagen.mean, datagen.std))
# demonstrate effect on a single batch of samples
iterator = datagen.flow(trainX, trainy, batch_size=64)
# get a batch
batchX, batchy = iterator.next()
# pixel stats in the batch
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(batchX.shape, batchX.mean(), batchX.std())
# demonstrate effect on entire training dataset
# 获取长度 / Get length
iterator = datagen.flow(trainX, trainy, batch_size=len(trainX), shuffle=False)
# get a batch
batchX, batchy = iterator.next()
# pixel stats in the batch
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(batchX.shape, batchX.mean(), batchX.std())
```

---

### Chapter Summary / 章节总结

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **4 code files** demonstrating chapter 07.

本章包含 **4 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `01_mnist_dataset.ipynb` — Mnist Dataset
  2. `02_normalization.ipynb` — Normalization
  3. `03_centering.ipynb` — Centering
  4. `04_standardization_keras.ipynb` — Standardization Keras

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---
