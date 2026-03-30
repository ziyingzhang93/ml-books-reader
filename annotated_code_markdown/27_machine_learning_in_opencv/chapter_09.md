# OpenCV 机器学习 / Machine Learning in OpenCV
## Chapter 09

---

### Hog

# 01 — Hog / 01 Hog

**Chapter 09 — File 1 of 4 / 第09章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load the full training image**.

本脚本演示 **Load the full training image**。

---
## Step 1 — Step 1

```python
import cv2
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors
```

---
## Step 2 — Load the full training image

```python
img, sub_imgs = split_images('Images/digits.png', 20)
```

---
## Step 3 — Show entire image to check that the correct image has been loaded

```python
cv2.imshow('Training image', img)
cv2.waitKey(0)
```

---
## Step 4 — Show one sample to check that the sub-images have been correctly split

```python
cv2.imshow('Sub-image', sub_imgs[0, 0, :, :].reshape(20, 20))
cv2.waitKey(0)
```

---
## Step 5 — Split the dataset into training and testing

```python
train_imgs, train_labels, test_imgs, test_labels = split_data(20, sub_imgs, 0.5)
```

---
## Step 6 — Convert the training and testing images into feature vectors using the HOG technique

```python
train_hog = hog_descriptors(train_imgs)
test_hog = hog_descriptors(test_imgs)
```

---
## Learning Notes / 学习笔记

- **概念**: Load the full training image 是机器学习中的常用技术。  
  *Load the full training image is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hog / 01 Hog
# Complete Code / 完整代码
# ===============================

import cv2
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors

# Load the full training image
img, sub_imgs = split_images('Images/digits.png', 20)

# Show entire image to check that the correct image has been loaded
cv2.imshow('Training image', img)
cv2.waitKey(0)

# Show one sample to check that the sub-images have been correctly split
cv2.imshow('Sub-image', sub_imgs[0, 0, :, :].reshape(20, 20))
cv2.waitKey(0)

# Split the dataset into training and testing
train_imgs, train_labels, test_imgs, test_labels = split_data(20, sub_imgs, 0.5)

# Convert the training and testing images into feature vectors using the HOG technique
train_hog = hog_descriptors(train_imgs)
test_hog = hog_descriptors(test_imgs)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Knn

# 06 — Knn / 06 Knn

**Chapter 09 — File 2 of 4 / 第09章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load the full training image**.

本脚本演示 **Load the full training image**。

---
## Step 1 — Step 1

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors
```

---
## Step 2 — Load the full training image

```python
img, sub_imgs = split_images('Images/digits.png', 20)
```

---
## Step 3 — Check that the correct image has been loaded

```python
cv2.imshow('Training image', img)
cv2.waitKey(0)
```

---
## Step 4 — Check that the sub-images have been correctly split

```python
cv2.imshow('Sub-image', sub_imgs[0, 0, :, :].reshape(20, 20))
cv2.waitKey(0)
```

---
## Step 5 — Define different training-testing splits

```python
ratio = [0.5, 0.7, 0.9]

for i in ratio:
```

---
## Step 6 — Split the dataset into training and testing

```python
train_imgs, train_labels, test_imgs, test_labels = split_data(20, sub_imgs, i)
```

---
## Step 7 — Convert the training and testing images into feature vectors using the HOG technique

```python
train_hog = hog_descriptors(train_imgs)
    test_hog = hog_descriptors(test_imgs)
```

---
## Step 8 — Initiate a kNN classifier and train it on the training data

```python
knn = cv2.ml.KNearest_create()
    knn.train(train_hog, cv2.ml.ROW_SAMPLE, train_labels)
```

---
## Step 9 — Initiate a dictionary to hold the ratio and accuracy values

```python
accuracy_dict = {}
```

---
## Step 10 — Populate the dictionary with the keys corresponding to the values of 'k'

```python
keys = range(3, 16)

    for k in keys:
```

---
## Step 11 — Test the kNN classifier on the testing data

```python
ret, result, neighbours, dist = knn.findNearest(test_hog, k)
```

---
## Step 12 — Compute the accuracy and print it

```python
accuracy = (np.sum(result == test_labels) / test_labels.size) * 100
        print("Accuracy: {0:.2f}%, Training: {1:.0f}%, k: {2}".format(accuracy, i*100, k))
```

---
## Step 13 — Populate the dictionary with the values corresponding to the accuracy

```python
accuracy_dict[k] = accuracy
```

---
## Step 14 — Plot the accuracy values against the value of 'k'

```python
plt.plot(accuracy_dict.keys(), accuracy_dict.values(),
             marker='o', label=str(i*100)+'%')
    plt.title('Accuracy of the $k-nearest neighbors model')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')

plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load the full training image 是机器学习中的常用技术。  
  *Load the full training image is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Knn / 06 Knn
# Complete Code / 完整代码
# ===============================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors

# Load the full training image
img, sub_imgs = split_images('Images/digits.png', 20)

# Check that the correct image has been loaded
cv2.imshow('Training image', img)
cv2.waitKey(0)

# Check that the sub-images have been correctly split
cv2.imshow('Sub-image', sub_imgs[0, 0, :, :].reshape(20, 20))
cv2.waitKey(0)

# Define different training-testing splits
ratio = [0.5, 0.7, 0.9]

for i in ratio:
    # Split the dataset into training and testing
    train_imgs, train_labels, test_imgs, test_labels = split_data(20, sub_imgs, i)

    # Convert the training and testing images into feature vectors using the HOG technique
    train_hog = hog_descriptors(train_imgs)
    test_hog = hog_descriptors(test_imgs)

    # Initiate a kNN classifier and train it on the training data
    knn = cv2.ml.KNearest_create()
    knn.train(train_hog, cv2.ml.ROW_SAMPLE, train_labels)

    # Initiate a dictionary to hold the ratio and accuracy values
    accuracy_dict = {}

    # Populate the dictionary with the keys corresponding to the values of 'k'
    keys = range(3, 16)

    for k in keys:
        # Test the kNN classifier on the testing data
        ret, result, neighbours, dist = knn.findNearest(test_hog, k)

        # Compute the accuracy and print it
        accuracy = (np.sum(result == test_labels) / test_labels.size) * 100
        print("Accuracy: {0:.2f}%, Training: {1:.0f}%, k: {2}".format(accuracy, i*100, k))

        # Populate the dictionary with the values corresponding to the accuracy
        accuracy_dict[k] = accuracy

    # Plot the accuracy values against the value of 'k'
    plt.plot(accuracy_dict.keys(), accuracy_dict.values(),
             marker='o', label=str(i*100)+'%')
    plt.title('Accuracy of the $k-nearest neighbors model')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')

plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Chapter Summary / 章节总结

# Chapter 09 Summary / 第09章总结

## Theme / 主题: Chapter 09 / Chapter 09

This chapter contains **4 code files** demonstrating chapter 09.

本章包含 **4 个代码文件**，演示Chapter 09。

---
## Evolution / 演化路线

  1. `01_hog.ipynb` — Hog
  2. `06_knn.ipynb` — Knn
  3. `digits_dataset.ipynb` — Digits Dataset
  4. `feature_extraction.ipynb` — Feature Extraction

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 09) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 09）是机器学习流水线中的基础构建块。

---

### Digits Dataset

# 01 — Digits Dataset / Digits Dataset

**Chapter 09 — File 3 of 4 / 第09章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load the full image from the specified file**.

本脚本演示 **Load the full image from the specified file**。

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

➡️ **Next / 下一步**: File 4 of 4

---

### Feature Extraction

# 01 — Feature Extraction / 特征工程

**Chapter 09 — File 4 of 4 / 第09章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Create a list to store the HOG feature vectors**.

本脚本演示 **Create a list to store the HOG feature vectors**。

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
