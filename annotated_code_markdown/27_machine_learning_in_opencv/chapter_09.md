# OpenCV 机器学习 / Machine Learning in OpenCV
## Chapter 09

---

### Hog



---

### Knn

# 06 — Knn / 06 Knn

**Chapter 09 — File 2 of 4 / 第09章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load the full training image**.

本脚本演示 **Load the full training image**。

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
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
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
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
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
# 生成整数序列 / Generate integer sequence
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
# 求和 / Calculate sum
accuracy = (np.sum(result == test_labels) / test_labels.size) * 100
        # 打印输出 / Print output
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
# 转换为NumPy数组 / Convert to NumPy array
plt.plot(accuracy_dict.keys(), accuracy_dict.values(),
             marker='o', label=str(i*100)+'%')
    # 设置图表标题 / Set chart title
    plt.title('Accuracy of the $k-nearest neighbors model')
    # 设置X轴标签 / Set X-axis label
    plt.xlabel('k')
    # 设置Y轴标签 / Set Y-axis label
    plt.ylabel('Accuracy')
    # 显示图例 / Show legend
    plt.legend(loc='upper right')

# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load the full training image 是机器学习中的常用技术。  
  *Load the full training image is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Knn / 06 Knn
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors

# Load the full training image
img, sub_imgs = split_images('Images/digits.png', 20)

# Check that the correct image has been loaded
cv2.imshow('Training image', img)
cv2.waitKey(0)

# Check that the sub-images have been correctly split
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
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
    # 生成整数序列 / Generate integer sequence
    keys = range(3, 16)

    for k in keys:
        # Test the kNN classifier on the testing data
        ret, result, neighbours, dist = knn.findNearest(test_hog, k)

        # Compute the accuracy and print it
        # 求和 / Calculate sum
        accuracy = (np.sum(result == test_labels) / test_labels.size) * 100
        # 打印输出 / Print output
        print("Accuracy: {0:.2f}%, Training: {1:.0f}%, k: {2}".format(accuracy, i*100, k))

        # Populate the dictionary with the values corresponding to the accuracy
        accuracy_dict[k] = accuracy

    # Plot the accuracy values against the value of 'k'
    # 转换为NumPy数组 / Convert to NumPy array
    plt.plot(accuracy_dict.keys(), accuracy_dict.values(),
             marker='o', label=str(i*100)+'%')
    # 设置图表标题 / Set chart title
    plt.title('Accuracy of the $k-nearest neighbors model')
    # 设置X轴标签 / Set X-axis label
    plt.xlabel('k')
    # 设置Y轴标签 / Set Y-axis label
    plt.ylabel('Accuracy')
    # 显示图例 / Show legend
    plt.legend(loc='upper right')

# 显示图表 / Display the plot
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



---

### Feature Extraction

# 01 — Feature Extraction / 特征工程

**Chapter 09 — File 4 of 4 / 第09章 — 第4个文件（共4个）**

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
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
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
        # 转换数据类型 / Convert data type
        hist = hog.compute(img.reshape(20, 20).astype(np.uint8))
        # 添加元素到列表末尾 / Append element to list end
        hog_features.append(hist)

    # 创建NumPy数组 / Create NumPy array
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
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
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
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
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
            # 添加元素到列表末尾 / Append element to list end
            bow_features.append(hist[0])

    # 创建NumPy数组 / Create NumPy array
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

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
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
        # 转换数据类型 / Convert data type
        hist = hog.compute(img.reshape(20, 20).astype(np.uint8))
        # 添加元素到列表末尾 / Append element to list end
        hog_features.append(hist)

    # 创建NumPy数组 / Create NumPy array
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
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
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
        # 改变数组形状（不改变数据） / Reshape array (data unchanged)
        img = np.reshape(img, (32, 32, 3), 'F')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).transpose()

        # Compute the BoW feature vector
        hist = bow_extractor.compute(img, sift.detect(img))

        # Append the feature vectors to the list
        if hist is not None:
            # 添加元素到列表末尾 / Append element to list end
            bow_features.append(hist[0])

    # 创建NumPy数组 / Create NumPy array
    return np.array(bow_features)
```

---
