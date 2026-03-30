# OpenCV ML
## Chapter 13

---

### Read Data

# 01 — Read Data / 01 Read Data

**Chapter 13 — File 1 of 8 / 第13章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Function to load the dataset**.

本脚本演示 **Function to load the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import csv
import numpy as np
from sklearn import model_selection as ms
```

---
## Step 2 — Function to load the dataset

```python
def load_csv(filename):
    with open(filename, "r") as file:
        lines = csv.reader(file)
        dataset = list(lines)
    return dataset
```

---
## Step 3 — Function to convert a string column to float

```python
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = np.float32(row[column].strip())
```

---
## Step 4 — Load the dataset from text file

```python
data = load_csv('Data/data_banknote_authentication.txt')
```

---
## Step 5 — Convert the dataset string numbers to float

```python
for i in range(len(data[0])):
    str_column_to_float(data, i)
```

---
## Step 6 — Convert list to array

```python
data = np.array(data)
```

---
## Step 7 — Separate the dataset samples from the groundtruth

```python
samples = data[:, :4]
target = data[:, -1, np.newaxis].astype(np.int32)
```

---
## Step 8 — Split the data into training and test sets

```python
x_train, x_test, y_train, y_test = \
        ms.train_test_split(samples, target, test_size=0.2, random_state=10)
```

---
## Learning Notes / 学习笔记

- **概念**: Function to load the dataset 是机器学习中的常用技术。  
  *Function to load the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Read Data / 01 Read Data
# Complete Code / 完整代码
# ===============================

import csv
import numpy as np
from sklearn import model_selection as ms

# Function to load the dataset
def load_csv(filename):
    with open(filename, "r") as file:
        lines = csv.reader(file)
        dataset = list(lines)
    return dataset

# Function to convert a string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = np.float32(row[column].strip())

# Load the dataset from text file
data = load_csv('Data/data_banknote_authentication.txt')

# Convert the dataset string numbers to float
for i in range(len(data[0])):
    str_column_to_float(data, i)

# Convert list to array
data = np.array(data)

# Separate the dataset samples from the groundtruth
samples = data[:, :4]
target = data[:, -1, np.newaxis].astype(np.int32)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
        ms.train_test_split(samples, target, test_size=0.2, random_state=10)
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Create Forest

# 02 — Create Forest / 随机森林

**Chapter 13 — File 2 of 8 / 第13章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Create an empty random forest**.

本脚本演示 **Create an empty random forest**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import cv2
```

---
## Step 2 — Create an empty random forest

```python
rtrees = cv2.ml.RTrees_create()
```

---
## Learning Notes / 学习笔记

- **概念**: Create an empty random forest 是机器学习中的常用技术。  
  *Create an empty random forest is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Create Forest / 随机森林
# Complete Code / 完整代码
# ===============================

import cv2

# Create an empty random forest
rtrees = cv2.ml.RTrees_create()
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Train Forest

# 04 — Train Forest / 随机森林

**Chapter 13 — File 3 of 8 / 第13章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Function to load the dataset**.

本脚本演示 **Function to load the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
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
import csv
import cv2
import numpy as np
from sklearn import model_selection as ms
```

---
## Step 2 — Function to load the dataset

```python
def load_csv(filename):
    with open(filename, "r") as file:
        lines = csv.reader(file)
        dataset = list(lines)
    return dataset
```

---
## Step 3 — Function to convert a string column to float

```python
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = np.float32(row[column].strip())
```

---
## Step 4 — Load the dataset from text file

```python
data = load_csv('Data/data_banknote_authentication.txt')
```

---
## Step 5 — Convert the dataset string numbers to float

```python
for i in range(len(data[0])):
    str_column_to_float(data, i)
```

---
## Step 6 — Convert list to array

```python
data = np.array(data)
```

---
## Step 7 — Separate the dataset samples from the groundtruth

```python
samples = data[:, :4]
target = data[:, -1, np.newaxis].astype(np.int32)
```

---
## Step 8 — Split the data into training and test sets

```python
x_train, x_test, y_train, y_test = \
        ms.train_test_split(samples, target, test_size=0.2, random_state=10)
```

---
## Step 9 — Create an empty random forest

```python
rtrees = cv2.ml.RTrees_create()
```

---
## Step 10 — Train the random forest

```python
rtrees.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
```

---
## Step 11 — Predict the target labels of the testing data

```python
_, y_pred = rtrees.predict(x_test)
```

---
## Step 12 — Compute and print the achieved accuracy

```python
accuracy = (np.sum(y_pred.astype(np.int32) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, '%')
```

---
## Learning Notes / 学习笔记

- **概念**: Function to load the dataset 是机器学习中的常用技术。  
  *Function to load the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Forest / 随机森林
# Complete Code / 完整代码
# ===============================

import csv
import cv2
import numpy as np
from sklearn import model_selection as ms

# Function to load the dataset
def load_csv(filename):
    with open(filename, "r") as file:
        lines = csv.reader(file)
        dataset = list(lines)
    return dataset

# Function to convert a string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = np.float32(row[column].strip())

# Load the dataset from text file
data = load_csv('Data/data_banknote_authentication.txt')

# Convert the dataset string numbers to float
for i in range(len(data[0])):
    str_column_to_float(data, i)

# Convert list to array
data = np.array(data)

# Separate the dataset samples from the groundtruth
samples = data[:, :4]
target = data[:, -1, np.newaxis].astype(np.int32)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
        ms.train_test_split(samples, target, test_size=0.2, random_state=10)

# Create an empty random forest
rtrees = cv2.ml.RTrees_create()

# Train the random forest
rtrees.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

# Predict the target labels of the testing data
_, y_pred = rtrees.predict(x_test)

# Compute and print the achieved accuracy
accuracy = (np.sum(y_pred.astype(np.int32) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, '%')
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Digits

# 05 — Digits / 05 Digits

**Chapter 13 — File 4 of 8 / 第13章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Load the digits image**.

本脚本演示 **Load the digits image**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
## Step 1 — Step 1

```python
import cv2
import numpy as np
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors
```

---
## Step 2 — Load the digits image

```python
img, sub_imgs = split_images('Images/digits.png', 20)
```

---
## Step 3 — Obtain training and testing datasets from the digits image

```python
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = \
        split_data(20, sub_imgs, 0.8)
```

---
## Step 4 — Convert the image data into HOG descriptors

```python
digits_train_hog = hog_descriptors(digits_train_imgs)
digits_test_hog = hog_descriptors(digits_test_imgs)
```

---
## Step 5 — Create an empty random forest

```python
rtrees_digits = cv2.ml.RTrees_create()
```

---
## Step 6 — Train the random forest

```python
rtrees_digits.train(digits_train_hog, cv2.ml.ROW_SAMPLE, digits_train_labels)
```

---
## Step 7 — Predict the target labels of the testing data

```python
_, digits_test_pred = rtrees_digits.predict(digits_test_hog)
```

---
## Step 8 — Compute and print the achieved accuracy

```python
accuracy_digits = (np.sum(digits_test_pred.astype(int) == digits_test_labels)
                    / digits_test_labels.size) * 100
print('Accuracy:', accuracy_digits, '%')
```

---
## Learning Notes / 学习笔记

- **概念**: Load the digits image 是机器学习中的常用技术。  
  *Load the digits image is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Digits / 05 Digits
# Complete Code / 完整代码
# ===============================

import cv2
import numpy as np
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = \
        split_data(20, sub_imgs, 0.8)

# Convert the image data into HOG descriptors
digits_train_hog = hog_descriptors(digits_train_imgs)
digits_test_hog = hog_descriptors(digits_test_imgs)

# Create an empty random forest
rtrees_digits = cv2.ml.RTrees_create()

# Train the random forest
rtrees_digits.train(digits_train_hog, cv2.ml.ROW_SAMPLE, digits_train_labels)

# Predict the target labels of the testing data
_, digits_test_pred = rtrees_digits.predict(digits_test_hog)

# Compute and print the achieved accuracy
accuracy_digits = (np.sum(digits_test_pred.astype(int) == digits_test_labels)
                    / digits_test_labels.size) * 100
print('Accuracy:', accuracy_digits, '%')
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Tree Depth

# 06 — Tree Depth / 决策树

**Chapter 13 — File 5 of 8 / 第13章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Tree Depth**.

本脚本演示 **决策树**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import cv2

rtrees_digits = cv2.ml.RTrees_create()
print('Default tree depth:', rtrees_digits.getMaxDepth())
print('Default termination criteria:', rtrees_digits.getTermCriteria())
```

---
## Learning Notes / 学习笔记

- **概念**: Tree Depth 是机器学习中的常用技术。  
  *Tree Depth is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Tree Depth / 决策树
# Complete Code / 完整代码
# ===============================

import cv2

rtrees_digits = cv2.ml.RTrees_create()
print('Default tree depth:', rtrees_digits.getMaxDepth())
print('Default termination criteria:', rtrees_digits.getTermCriteria())
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Chapter Summary

# Chapter 13 Summary / 第13章总结

## Theme / 主题: Chapter 13 / Chapter 13

This chapter contains **8 code files** demonstrating chapter 13.

本章包含 **8 个代码文件**，演示Chapter 13。

---
## Evolution / 演化路线

  1. `01_read_data.ipynb` — Read Data
  2. `02_create_forest.ipynb` — Create Forest
  3. `04_train_forest.ipynb` — Train Forest
  4. `05_digits.ipynb` — Digits
  5. `06_tree_depth.ipynb` — Tree Depth
  6. `07_deeper_tree.ipynb` — Deeper Tree
  7. `digits_dataset.ipynb` — Digits Dataset
  8. `feature_extraction.ipynb` — Feature Extraction

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 13) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 13）是机器学习流水线中的基础构建块。

---

### Digits Dataset

# 01 — Digits Dataset / Digits Dataset

**Chapter 13 — File 7 of 8 / 第13章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Load the full image from the specified file**.

本脚本演示 **Load the full image from the specified file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import cv2
import numpy as np

def split_images(img_name, img_size):
```

---
## Step 2 — Load the full image from the specified file

```python
img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
```

---
## Step 3 — Find the number of sub-images on each row and column according to their size

```python
num_rows = img.shape[0] / img_size
    num_cols = img.shape[1] / img_size
```

---
## Step 4 — Split the full image horizontally and vertically into sub-images

```python
sub_imgs = [np.hsplit(row, num_cols) for row in np.vsplit(img, num_rows)]

    return img, np.array(sub_imgs)

def split_data(img_size, sub_imgs, ratio):
```

---
## Step 5 — Compute the partition between the training and testing data

```python
partition = int(sub_imgs.shape[1] * ratio)
```

---
## Step 6 — Split dataset into training and test sets

```python
train = sub_imgs[:, :partition, :, :]
    test = sub_imgs[:, partition:sub_imgs.shape[1], :, :]
```

---
## Step 7 — Flatten each image into a one-dimensional vector

```python
train_imgs = train.reshape(-1, img_size ** 2)
    test_imgs = test.reshape(-1, img_size ** 2)
```

---
## Step 8 — Create the groundtruth labels

```python
labels = np.arange(10)
    train_labels = np.repeat(labels, train_imgs.shape[0] / labels.shape[0])[:, np.newaxis]
    test_labels = np.repeat(labels, test_imgs.shape[0] / labels.shape[0])[:, np.newaxis]

    return train_imgs, train_labels, test_imgs, test_labels
```

---
## Learning Notes / 学习笔记

- **概念**: Load the full image from the specified file 是机器学习中的常用技术。  
  *Load the full image from the specified file is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Digits Dataset / Digits Dataset
# Complete Code / 完整代码
# ===============================

import cv2
import numpy as np

def split_images(img_name, img_size):
    # Load the full image from the specified file
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    # Find the number of sub-images on each row and column according to their size
    num_rows = img.shape[0] / img_size
    num_cols = img.shape[1] / img_size

    # Split the full image horizontally and vertically into sub-images
    sub_imgs = [np.hsplit(row, num_cols) for row in np.vsplit(img, num_rows)]

    return img, np.array(sub_imgs)

def split_data(img_size, sub_imgs, ratio):
    # Compute the partition between the training and testing data
    partition = int(sub_imgs.shape[1] * ratio)

    # Split dataset into training and test sets
    train = sub_imgs[:, :partition, :, :]
    test = sub_imgs[:, partition:sub_imgs.shape[1], :, :]

    # Flatten each image into a one-dimensional vector
    train_imgs = train.reshape(-1, img_size ** 2)
    test_imgs = test.reshape(-1, img_size ** 2)

    # Create the groundtruth labels
    labels = np.arange(10)
    train_labels = np.repeat(labels, train_imgs.shape[0] / labels.shape[0])[:, np.newaxis]
    test_labels = np.repeat(labels, test_imgs.shape[0] / labels.shape[0])[:, np.newaxis]

    return train_imgs, train_labels, test_imgs, test_labels
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Feature Extraction

# 01 — Feature Extraction / 特征工程

**Chapter 13 — File 8 of 8 / 第13章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Create a list to store the HOG feature vectors**.

本脚本演示 **Create a list to store the HOG feature vectors**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import cv2
import numpy as np


def hog_descriptors(imgs):
```

---
## Step 2 — Create a list to store the HOG feature vectors

```python
hog_features = []
```

---
## Step 3 — Set parameter values for the HOG descriptor based on the image data in use

```python
winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9
```

---
## Step 4 — Set the remaining parameters to their default values

```python
derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = False
    nlevels = 64
```

---
## Step 5 — Create a HOG descriptor

```python
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                            derivAperture, winSigma, histogramNormType, L2HysThreshold,
                            gammaCorrection, nlevels)
```

---
## Step 6 — Compute HOG for the input images and append the feature vectors to the list

```python
for img in imgs:
        hist = hog.compute(img.reshape(20, 20).astype(np.uint8))
        hog_features.append(hist)

    return np.array(hog_features)


def bow_descriptors(imgs):
```

---
## Step 7 — Create a SIFT descriptor

```python
sift = cv2.SIFT_create()
```

---
## Step 8 — Create a BoW descriptor. The number of clusters (50, analogous to
the vocabulary size) has been chosen empirically

```python
bow_trainer = cv2.BOWKMeansTrainer(50)
    bow_extractor = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))

    for img in imgs:
```

---
## Step 9 — Reshape each RGB image and convert it to grayscale

```python
img = np.reshape(img, (32, 32, 3), 'F')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).transpose()
```

---
## Step 10 — Extract the SIFT descriptors

```python
_, descriptors = sift.detectAndCompute(img, None)
```

---
## Step 11 — Add the SIFT descriptors to the BoW vocabulary trainer

```python
if descriptors is not None:
            bow_trainer.add(descriptors)
```

---
## Step 12 — Perform k-means clustering and return the vocabulary

```python
voc = bow_trainer.cluster()
```

---
## Step 13 — Assign the vocabulary to the BoW descriptor extractor

```python
bow_extractor.setVocabulary(voc)
```

---
## Step 14 — Create a list to store the BoW feature vectors

```python
bow_features = []

    for img in imgs:
```

---
## Step 15 — Reshape each RGB image and convert it to grayscale

```python
img = np.reshape(img, (32, 32, 3), 'F')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).transpose()
```

---
## Step 16 — Compute the BoW feature vector

```python
hist = bow_extractor.compute(img, sift.detect(img))
```

---
## Step 17 — Append the feature vectors to the list

```python
if hist is not None:
            bow_features.append(hist[0])

    return np.array(bow_features)
```

---
## Learning Notes / 学习笔记

- **概念**: Create a list to store the HOG feature vectors 是机器学习中的常用技术。  
  *Create a list to store the HOG feature vectors is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `KMeans` | K均值聚类 | K-Means clustering |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Feature Extraction / 特征工程
# Complete Code / 完整代码
# ===============================

import cv2
import numpy as np


def hog_descriptors(imgs):
    # Create a list to store the HOG feature vectors
    hog_features = []

    # Set parameter values for the HOG descriptor based on the image data in use
    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9

    # Set the remaining parameters to their default values
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = False
    nlevels = 64

    # Create a HOG descriptor
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                            derivAperture, winSigma, histogramNormType, L2HysThreshold,
                            gammaCorrection, nlevels)

    # Compute HOG for the input images and append the feature vectors to the list
    for img in imgs:
        hist = hog.compute(img.reshape(20, 20).astype(np.uint8))
        hog_features.append(hist)

    return np.array(hog_features)


def bow_descriptors(imgs):
    # Create a SIFT descriptor
    sift = cv2.SIFT_create()

    # Create a BoW descriptor. The number of clusters (50, analogous to
    # the vocabulary size) has been chosen empirically
    bow_trainer = cv2.BOWKMeansTrainer(50)
    bow_extractor = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))

    for img in imgs:
        # Reshape each RGB image and convert it to grayscale
        img = np.reshape(img, (32, 32, 3), 'F')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).transpose()

        # Extract the SIFT descriptors
        _, descriptors = sift.detectAndCompute(img, None)

        # Add the SIFT descriptors to the BoW vocabulary trainer
        if descriptors is not None:
            bow_trainer.add(descriptors)

    # Perform k-means clustering and return the vocabulary
    voc = bow_trainer.cluster()

    # Assign the vocabulary to the BoW descriptor extractor
    bow_extractor.setVocabulary(voc)

    # Create a list to store the BoW feature vectors
    bow_features = []

    for img in imgs:
        # Reshape each RGB image and convert it to grayscale
        img = np.reshape(img, (32, 32, 3), 'F')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).transpose()

        # Compute the BoW feature vector
        hist = bow_extractor.compute(img, sift.detect(img))

        # Append the feature vectors to the list
        if hist is not None:
            bow_features.append(hist[0])

    return np.array(bow_features)
```

---
