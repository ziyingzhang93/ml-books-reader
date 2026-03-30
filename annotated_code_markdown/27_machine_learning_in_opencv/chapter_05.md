# OpenCV 机器学习 / Machine Learning in OpenCV
## Chapter 05

---

### Split Images

# 01 — Split Images / 图像处理

**Chapter 05 — File 1 of 6 / 第05章 — 第1个文件（共6个）**

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
# Split Images / 图像处理
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
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Split Data

# 02 — Split Data / 02 Split Data

**Chapter 05 — File 2 of 6 / 第05章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Compute the partition between the training and testing data**.

本脚本演示 **Compute the partition between the training and testing data**。

---
## Step 1 — Step 1

```python
import numpy as np

def split_data(img_size, sub_imgs, ratio):
```

---
## Step 2 — Compute the partition between the training and testing data

```python
partition = int(sub_imgs.shape[1] * ratio)
```

---
## Step 3 — Split dataset into training and test sets

```python
train = sub_imgs[:, :partition, :, :]
    test = sub_imgs[:, partition:sub_imgs.shape[1], :, :]
```

---
## Step 4 — Flatten each image into a one-dimensional vector

```python
train_imgs = train.reshape(-1, img_size ** 2)
    test_imgs = test.reshape(-1, img_size ** 2)
```

---
## Step 5 — Create the groundtruth labels

```python
labels = np.arange(10)
    train_labels = np.repeat(labels, train_imgs.shape[0] / labels.shape[0])[:, np.newaxis]
    test_labels = np.repeat(labels, test_imgs.shape[0] / labels.shape[0])[:, np.newaxis]

    return train_imgs, train_labels, test_imgs, test_labels
```

---
## Learning Notes / 学习笔记

- **概念**: Compute the partition between the training and testing data 是机器学习中的常用技术。  
  *Compute the partition between the training and testing data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Split Data / 02 Split Data
# Complete Code / 完整代码
# ===============================

import numpy as np

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

➡️ **Next / 下一步**: File 3 of 6

---

### Cifar Dataset

# 03 — Cifar Dataset / 03 Cifar Dataset

**Chapter 05 — File 3 of 6 / 第05章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Create empty lists to store the images and labels**.

本脚本演示 **Create empty lists to store the images and labels**。

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

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cifar Dataset / 03 Cifar Dataset
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

➡️ **Next / 下一步**: File 4 of 6

---

### Load

# 04 — Load / 04 Load

**Chapter 05 — File 4 of 6 / 第05章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Load the digits image**.

本脚本演示 **Load the digits image**。

---
## Step 1 — Step 1

```python
from digits_dataset import split_images, split_data
from cifar_dataset import load_images
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
## Step 4 — Obtain training and testing datasets from the CIFAR-10 dataset

```python
cifar_train_imgs, cifar_train_labels, cifar_test_imgs, cifar_test_labels = \
    load_images('Images/cifar-10-batches-py/')
```

---
## Learning Notes / 学习笔记

- **概念**: Load the digits image 是机器学习中的常用技术。  
  *Load the digits image is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load / 04 Load
# Complete Code / 完整代码
# ===============================

from digits_dataset import split_images, split_data
from cifar_dataset import load_images

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = \
    split_data(20, sub_imgs, 0.8)

# Obtain training and testing datasets from the CIFAR-10 dataset
cifar_train_imgs, cifar_train_labels, cifar_test_imgs, cifar_test_labels = \
    load_images('Images/cifar-10-batches-py/')
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Chapter Summary / 章节总结

# Chapter 05 Summary / 第05章总结

## Theme / 主题: Chapter 05 / Chapter 05

This chapter contains **6 code files** demonstrating chapter 05.

本章包含 **6 个代码文件**，演示Chapter 05。

---
## Evolution / 演化路线

  1. `01_split_images.ipynb` — Split Images
  2. `02_split_data.ipynb` — Split Data
  3. `03_cifar_dataset.ipynb` — Cifar Dataset
  4. `04_load.ipynb` — Load
  5. `cifar_dataset.ipynb` — Cifar Dataset
  6. `digits_dataset.ipynb` — Digits Dataset

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 05) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 05）是机器学习流水线中的基础构建块。

---

### Cifar Dataset

# 01 — Cifar Dataset / Cifar Dataset

**Chapter 05 — File 5 of 6 / 第05章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Create empty lists to store the images and labels**.

本脚本演示 **Create empty lists to store the images and labels**。

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

➡️ **Next / 下一步**: File 6 of 6

---

### Digits Dataset

# 01 — Digits Dataset / Digits Dataset

**Chapter 05 — File 6 of 6 / 第05章 — 第6个文件（共6个）**

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
