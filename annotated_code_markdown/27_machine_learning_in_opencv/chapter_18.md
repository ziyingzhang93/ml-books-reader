# OpenCV ML
## Chapter 18

---

### Svc

# 05 — Svc / 05 Svc

**Chapter 18 — File 1 of 5 / 第18章 — 第1个文件（共5个）**

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
## Step 4 — Create a new SVM

```python
svm_digits = cv2.ml.SVM_create()
```

---
## Step 5 — Set the SVM kernel to RBF

```python
svm_digits.setKernel(cv2.ml.SVM_RBF)
svm_digits.setType(cv2.ml.SVM_C_SVC)
svm_digits.setGamma(0.5)
svm_digits.setC(12)
svm_digits.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 100, 1e-6))
```

---
## Step 6 — Converting the image data into HOG descriptors

```python
digits_train_hog = hog_descriptors(digits_train_imgs)
digits_test_hog = hog_descriptors(digits_test_imgs)
```

---
## Step 7 — Train the SVM on the set of training data

```python
svm_digits.train(digits_train_hog.astype(np.float32), cv2.ml.ROW_SAMPLE,
                 digits_train_labels)
```

---
## Step 8 — Predict labels for the testing data

```python
_, digits_test_pred = svm_digits.predict(digits_test_hog.astype(np.float32))
```

---
## Step 9 — Compute and print the achieved accuracy

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
| `SVM` | 支持向量机 | Support Vector Machine |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Svc / 05 Svc
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

# Create a new SVM
svm_digits = cv2.ml.SVM_create()

# Set the SVM kernel to RBF
svm_digits.setKernel(cv2.ml.SVM_RBF)
svm_digits.setType(cv2.ml.SVM_C_SVC)
svm_digits.setGamma(0.5)
svm_digits.setC(12)
svm_digits.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 100, 1e-6))

# Converting the image data into HOG descriptors
digits_train_hog = hog_descriptors(digits_train_imgs)
digits_test_hog = hog_descriptors(digits_test_imgs)

# Train the SVM on the set of training data
svm_digits.train(digits_train_hog.astype(np.float32), cv2.ml.ROW_SAMPLE,
                 digits_train_labels)

# Predict labels for the testing data
_, digits_test_pred = svm_digits.predict(digits_test_hog.astype(np.float32))

# Compute and print the achieved accuracy
accuracy_digits = (np.sum(digits_test_pred.astype(int) == digits_test_labels)
                    / digits_test_labels.size) * 100
print('Accuracy:', accuracy_digits, '%')
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Test Img

# 06 — Test Img / 06 Test Img

**Chapter 18 — File 2 of 5 / 第18章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Load the digits image**.

本脚本演示 **Load the digits image**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
import random
import numpy as np
import matplotlib.pyplot as plt
from digits_dataset import split_images, split_data
```

---
## Step 2 — Load the digits image

```python
img, sub_imgs = split_images('Images/digits.png', 20)
```

---
## Step 3 — Obtain training and testing datasets from the digits image

```python
digits_train_imgs, _, digits_test_imgs, _ = split_data(20, sub_imgs, 0.8)
```

---
## Step 4 — Create an empty list to store the random numbers

```python
rand_nums = []
```

---
## Step 5 — Seed the random number generator for repeatability

```python
random.seed(10)
```

---
## Step 6 — Choose 25 random digits from the testing dataset

```python
for i in range(0, digits_test_imgs.shape[0], int(digits_test_imgs.shape[0] / 25)):
```

---
## Step 7 — Generate a random integer

```python
rand = random.randint(i, int(digits_test_imgs.shape[0] / 25) + i - 1)
```

---
## Step 8 — Append it to the list

```python
rand_nums.append(rand)
```

---
## Step 9 — Shuffle the order of the generated random integers

```python
random.shuffle(rand_nums)
```

---
## Step 10 — Read the image data corresponding to the random integers

```python
rand_test_imgs = digits_test_imgs[rand_nums, :]
```

---
## Step 11 — Initialize an array to hold the test image

```python
test_img = np.zeros((100, 100), dtype=np.uint8)
```

---
## Step 12 — Start a sub-image counter

```python
img_count = 0
```

---
## Step 13 — Iterate over the test image

```python
for i in range(0, test_img.shape[0], 20):
    for j in range(0, test_img.shape[1], 20):
```

---
## Step 14 — Populate the test image with the chosen digits

```python
test_img[i:i + 20, j:j + 20] = rand_test_imgs[img_count].reshape(20, 20)
```

---
## Step 15 — Increment the sub-image counter

```python
img_count += 1
```

---
## Step 16 — Display the test image

```python
plt.imshow(test_img, cmap='gray')
plt.show()
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
| `matplotlib` | 绑图库 | Plotting library |
| `np.zeros` | 全零数组 | Array filled with zeros |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Test Img / 06 Test Img
# Complete Code / 完整代码
# ===============================

import random
import numpy as np
import matplotlib.pyplot as plt
from digits_dataset import split_images, split_data

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, _, digits_test_imgs, _ = split_data(20, sub_imgs, 0.8)

# Create an empty list to store the random numbers
rand_nums = []

# Seed the random number generator for repeatability
random.seed(10)

# Choose 25 random digits from the testing dataset
for i in range(0, digits_test_imgs.shape[0], int(digits_test_imgs.shape[0] / 25)):
    # Generate a random integer
    rand = random.randint(i, int(digits_test_imgs.shape[0] / 25) + i - 1)
    # Append it to the list
    rand_nums.append(rand)

# Shuffle the order of the generated random integers
random.shuffle(rand_nums)

# Read the image data corresponding to the random integers
rand_test_imgs = digits_test_imgs[rand_nums, :]

# Initialize an array to hold the test image
test_img = np.zeros((100, 100), dtype=np.uint8)

# Start a sub-image counter
img_count = 0

# Iterate over the test image
for i in range(0, test_img.shape[0], 20):
    for j in range(0, test_img.shape[1], 20):
        # Populate the test image with the chosen digits
        test_img[i:i + 20, j:j + 20] = rand_test_imgs[img_count].reshape(20, 20)
        # Increment the sub-image counter
        img_count += 1

# Display the test image
plt.imshow(test_img, cmap='gray')
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Digits Dataset

# 01 — Digits Dataset / Digits Dataset

**Chapter 18 — File 4 of 5 / 第18章 — 第4个文件（共5个）**

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

➡️ **Next / 下一步**: File 5 of 5

---

### Feature Extraction

# 01 — Feature Extraction / 特征工程

**Chapter 18 — File 5 of 5 / 第18章 — 第5个文件（共5个）**

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
