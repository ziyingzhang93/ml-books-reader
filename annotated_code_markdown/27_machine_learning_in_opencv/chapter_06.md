# OpenCV ML
## Chapter 06

---

### Feature Vectors

# 05 — Feature Vectors / 特征工程

**Chapter 06 — File 1 of 3 / 第06章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Load the digits image**.

本脚本演示 **Load the digits image**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import cv2
import numpy as np
from digits_dataset import split_images, split_data
from cifar_dataset import load_images
```

---
## Step 2 — Load the digits image

```python
img, sub_imgs = split_images('Images/digits.png', 20)
```

---
## Step 3 — Obtain a dataset from the digits image

```python
digits_imgs, _, _, _ = split_data(20, sub_imgs, 0.8)
```

---
## Step 4 — Load a batch of images from the CIFAR dataset

```python
cifar_imgs, _, _, _ = load_images('Images/cifar-10-batches-py/')
```

---
## Step 5 — Consider only a subset of images

```python
digits_subset = digits_imgs[0:100]
cifar_subset = cifar_imgs[0:100]

def hog_descriptors(imgs):
```

---
## Step 6 — Create a list to store the HOG feature vectors

```python
hog_features = []
```

---
## Step 7 — Set parameter values for the HOG descriptor based on the image data in use

```python
winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9
```

---
## Step 8 — Set the remaining parameters to their default values

```python
derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = False
    nlevels = 64
```

---
## Step 9 — Create a HOG descriptor

```python
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                           derivAperture, winSigma, histogramNormType, L2HysThreshold,
                           gammaCorrection, nlevels)
```

---
## Step 10 — Compute HOG for the input images and append the feature vectors to the list

```python
for img in imgs:
        hist = hog.compute(img.reshape(20, 20).astype(np.uint8))
        hog_features.append(hist)

    return np.array(hog_features)


def bow_descriptors(imgs):
```

---
## Step 11 — Create a SIFT descriptor

```python
sift = cv2.SIFT_create()
```

---
## Step 12 — Create a BoW descriptor. The number of clusters (50, analogous to
the vocabulary size) has been chosen empirically

```python
bow_trainer = cv2.BOWKMeansTrainer(50)
    bow_extractor = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))

    for img in imgs:
```

---
## Step 13 — Reshape each RGB image and convert it to grayscale

```python
img = np.reshape(img, (32, 32, 3), 'F')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).transpose()
```

---
## Step 14 — Extract the SIFT descriptors

```python
_, descriptors = sift.detectAndCompute(img, None)
```

---
## Step 15 — Add the SIFT descriptors to the BoW vocabulary trainer

```python
if descriptors is not None:
            bow_trainer.add(descriptors)
```

---
## Step 16 — Perform k-means clustering and return the vocabulary

```python
voc = bow_trainer.cluster()
```

---
## Step 17 — Assign the vocabulary to the BoW descriptor extractor

```python
bow_extractor.setVocabulary(voc)
```

---
## Step 18 — Create a list to store the BoW feature vectors

```python
bow_features = []

    for img in imgs:
```

---
## Step 19 — Reshape each RGB image and convert it to grayscale

```python
img = np.reshape(img, (32, 32, 3), 'F')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).transpose()
```

---
## Step 20 — Compute the BoW feature vector

```python
hist = bow_extractor.compute(img, sift.detect(img))
```

---
## Step 21 — Append the feature vectors to the list

```python
if hist is not None:
            bow_features.append(hist[0])

    return np.array(bow_features)


digits_hog = hog_descriptors(digits_subset)
print('Size of HOG feature vectors:', digits_hog.shape)

cifar_bow = bow_descriptors(cifar_subset)
print('Size of BoW feature vectors:', cifar_bow.shape)
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
| `KMeans` | K均值聚类 | K-Means clustering |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Feature Vectors / 特征工程
# Complete Code / 完整代码
# ===============================

import cv2
import numpy as np
from digits_dataset import split_images, split_data
from cifar_dataset import load_images

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain a dataset from the digits image
digits_imgs, _, _, _ = split_data(20, sub_imgs, 0.8)

# Load a batch of images from the CIFAR dataset
cifar_imgs, _, _, _ = load_images('Images/cifar-10-batches-py/')

# Consider only a subset of images
digits_subset = digits_imgs[0:100]
cifar_subset = cifar_imgs[0:100]

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


digits_hog = hog_descriptors(digits_subset)
print('Size of HOG feature vectors:', digits_hog.shape)

cifar_bow = bow_descriptors(cifar_subset)
print('Size of BoW feature vectors:', cifar_bow.shape)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Chapter Summary

# Chapter 06 Summary / 第06章总结

## Theme / 主题: Chapter 06 / Chapter 06

This chapter contains **3 code files** demonstrating chapter 06.

本章包含 **3 个代码文件**，演示Chapter 06。

---
## Evolution / 演化路线

  1. `05_feature_vectors.ipynb` — Feature Vectors
  2. `cifar_dataset.ipynb` — Cifar Dataset
  3. `digits_dataset.ipynb` — Digits Dataset

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 06) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 06）是机器学习流水线中的基础构建块。

---

### Cifar Dataset

# 01 — Cifar Dataset / Cifar Dataset

**Chapter 06 — File 2 of 3 / 第06章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Create empty lists to store the images and labels**.

本脚本演示 **Create empty lists to store the images and labels**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import os
import pickle
import numpy as np

def load_images(path):
```

---
## Step 2 — Create empty lists to store the images and labels

```python
imgs = []
    labels = []
```

---
## Step 3 — Iterate over the dataset's files

```python
for batch in range(5):
```

---
## Step 4 — Specify the path to the training data

```python
train_path_batch = os.path.join(path, 'data_batch_' + str(batch + 1))
```

---
## Step 5 — Extract the training images and labels from the dataset files

```python
train_imgs_batch, train_labels_batch = extract_data(train_path_batch)
```

---
## Step 6 — Store the training images

```python
imgs.append(train_imgs_batch)
        train_imgs = np.array(imgs).reshape(-1, 3072)
```

---
## Step 7 — Store the training labels

```python
labels.append(train_labels_batch)
        train_labels = np.array(labels).reshape(-1, 1)
```

---
## Step 8 — Specify the path to the testing data

```python
test_path_batch = path + 'test_batch'
```

---
## Step 9 — Extract the testing images and labels from the dataset files

```python
test_imgs, test_labels = extract_data(test_path_batch)
    test_labels = np.array(test_labels)[:, np.newaxis]

    return train_imgs, train_labels, test_imgs, test_labels

def extract_data(path):
```

---
## Step 10 — Open pickle file and return a dictionary

```python
with open(path, 'rb') as fo:
        loaded_dict = pickle.load(fo, encoding='bytes')
```

---
## Step 11 — Extract the dictionary values

```python
dict_values = list(loaded_dict.values())
```

---
## Step 12 — Extract the images and labels

```python
imgs = dict_values[2]
    labels = dict_values[1]

    return imgs, labels
```

---
## Learning Notes / 学习笔记

- **概念**: Create empty lists to store the images and labels 是机器学习中的常用技术。  
  *Create empty lists to store the images and labels is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cifar Dataset / Cifar Dataset
# Complete Code / 完整代码
# ===============================

import os
import pickle
import numpy as np

def load_images(path):
    # Create empty lists to store the images and labels
    imgs = []
    labels = []

    # Iterate over the dataset's files
    for batch in range(5):
        # Specify the path to the training data
        train_path_batch = os.path.join(path, 'data_batch_' + str(batch + 1))

        # Extract the training images and labels from the dataset files
        train_imgs_batch, train_labels_batch = extract_data(train_path_batch)

        # Store the training images
        imgs.append(train_imgs_batch)
        train_imgs = np.array(imgs).reshape(-1, 3072)

        # Store the training labels
        labels.append(train_labels_batch)
        train_labels = np.array(labels).reshape(-1, 1)

    # Specify the path to the testing data
    test_path_batch = path + 'test_batch'

    # Extract the testing images and labels from the dataset files
    test_imgs, test_labels = extract_data(test_path_batch)
    test_labels = np.array(test_labels)[:, np.newaxis]

    return train_imgs, train_labels, test_imgs, test_labels

def extract_data(path):
    # Open pickle file and return a dictionary
    with open(path, 'rb') as fo:
        loaded_dict = pickle.load(fo, encoding='bytes')

    # Extract the dictionary values
    dict_values = list(loaded_dict.values())

    # Extract the images and labels
    imgs = dict_values[2]
    labels = dict_values[1]

    return imgs, labels
```

---

➡️ **Next / 下一步**: File 3 of 3

---
