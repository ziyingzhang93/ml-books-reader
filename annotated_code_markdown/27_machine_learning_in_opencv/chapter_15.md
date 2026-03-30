# OpenCV 机器学习 / Machine Learning in OpenCV
## Chapter 15

---

### Train

# 06 — Train / 06 Train

**Chapter 15 — File 1 of 4 / 第15章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load the digits image**.

本脚本演示 **Load the digits image**。

---
## Step 1 — Step 1

```python
import cv2
import numpy as np
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
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = \
    split_data(20, sub_imgs, 0.8)
```

---
## Step 4 — Create an empty logistic regression model

```python
lr_digits = cv2.ml.LogisticRegression_create()
```

---
## Step 5 — Check the default training method

```python
print('Training Method:', lr_digits.getTrainMethod())
```

---
## Step 6 — Set the training method to mini-batch gradient descent and the size of the mini-batch

```python
lr_digits.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr_digits.setMiniBatchSize(400)
```

---
## Step 7 — Set the number of iterations

```python
lr_digits.setIterations(10)
```

---
## Step 8 — Train the logistic regressor on the set of training data

```python
lr_digits.train(digits_train_imgs.astype(np.float32),
                cv2.ml.ROW_SAMPLE,
                digits_train_labels.astype(np.float32))
```

---
## Step 9 — Print the number of learned coefficients, and the number of input features

```python
print('Number of coefficients:', len(lr_digits.get_learnt_thetas()[0]))
print('Number of input features:', len(digits_train_imgs[0, :]))
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
# Train / 06 Train
# Complete Code / 完整代码
# ===============================

import cv2
import numpy as np
from digits_dataset import split_images, split_data

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = \
    split_data(20, sub_imgs, 0.8)

# Create an empty logistic regression model
lr_digits = cv2.ml.LogisticRegression_create()

# Check the default training method
print('Training Method:', lr_digits.getTrainMethod())

# Set the training method to mini-batch gradient descent and the size of the mini-batch
lr_digits.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr_digits.setMiniBatchSize(400)

# Set the number of iterations
lr_digits.setIterations(10)

# Train the logistic regressor on the set of training data
lr_digits.train(digits_train_imgs.astype(np.float32),
                cv2.ml.ROW_SAMPLE,
                digits_train_labels.astype(np.float32))

# Print the number of learned coefficients, and the number of input features
print('Number of coefficients:', len(lr_digits.get_learnt_thetas()[0]))
print('Number of input features:', len(digits_train_imgs[0, :]))
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Predict

# 07 — Predict / 07 Predict

**Chapter 15 — File 2 of 4 / 第15章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load the digits image**.

本脚本演示 **Load the digits image**。

---
## Step 1 — Step 1

```python
import cv2
import numpy as np
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
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = \
    split_data(20, sub_imgs, 0.8)
```

---
## Step 4 — Create an empty logistic regression model

```python
lr_digits = cv2.ml.LogisticRegression_create()
```

---
## Step 5 — Check the default training method

```python
print('Training Method:', lr_digits.getTrainMethod())
```

---
## Step 6 — Set the training method to mini-batch gradient descent and the size of the mini-batch

```python
lr_digits.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr_digits.setMiniBatchSize(400)
```

---
## Step 7 — Set the number of iterations

```python
lr_digits.setIterations(10)
```

---
## Step 8 — Train the logistic regressor on the set of training data

```python
lr_digits.train(digits_train_imgs.astype(np.float32),
                cv2.ml.ROW_SAMPLE,
                digits_train_labels.astype(np.float32))
```

---
## Step 9 — Predict the target labels of the testing data

```python
_, y_pred = lr_digits.predict(digits_test_imgs.astype(np.float32))
```

---
## Step 10 — Compute and print the achieved accuracy

```python
accuracy = (np.sum(y_pred[:, 0] == digits_test_labels[:, 0]) / digits_test_labels.size) * 100
print('Accuracy:', accuracy, '%')
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
# Predict / 07 Predict
# Complete Code / 完整代码
# ===============================

import cv2
import numpy as np
from digits_dataset import split_images, split_data

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = \
    split_data(20, sub_imgs, 0.8)

# Create an empty logistic regression model
lr_digits = cv2.ml.LogisticRegression_create()

# Check the default training method
print('Training Method:', lr_digits.getTrainMethod())

# Set the training method to mini-batch gradient descent and the size of the mini-batch
lr_digits.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr_digits.setMiniBatchSize(400)

# Set the number of iterations
lr_digits.setIterations(10)

# Train the logistic regressor on the set of training data
lr_digits.train(digits_train_imgs.astype(np.float32),
                cv2.ml.ROW_SAMPLE,
                digits_train_labels.astype(np.float32))

# Predict the target labels of the testing data
_, y_pred = lr_digits.predict(digits_test_imgs.astype(np.float32))

# Compute and print the achieved accuracy
accuracy = (np.sum(y_pred[:, 0] == digits_test_labels[:, 0]) / digits_test_labels.size) * 100
print('Accuracy:', accuracy, '%')
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Multiclass

# 09 — Multiclass / 09 Multiclass

**Chapter 15 — File 3 of 4 / 第15章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load the digits image**.

本脚本演示 **Load the digits image**。

---
## Step 1 — Step 1

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = \
    split_data(20, sub_imgs, 0.8)
```

---
## Step 4 — Create an empty logistic regression model

```python
lr_digits = cv2.ml.LogisticRegression_create()
```

---
## Step 5 — Check the default training method

```python
print('Training Method:', lr_digits.getTrainMethod())
```

---
## Step 6 — Set the training method to mini-batch gradient descent and the size of the mini-batch

```python
lr_digits.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr_digits.setMiniBatchSize(400)
```

---
## Step 7 — Set the number of iterations

```python
lr_digits.setIterations(10)
```

---
## Step 8 — Train the logistic regressor on the set of training data

```python
lr_digits.train(digits_train_imgs.astype(np.float32),
                cv2.ml.ROW_SAMPLE,
                digits_train_labels.astype(np.float32))
```

---
## Step 9 — Print the number of learned coefficients, and the number of input features

```python
print('Number of coefficients:', len(lr_digits.get_learnt_thetas()[0]))
print('Number of input features:', len(digits_train_imgs[0, :]))
```

---
## Step 10 — Predict the target labels of the testing data

```python
_, y_pred = lr_digits.predict(digits_test_imgs.astype(np.float32))
```

---
## Step 11 — Compute and print the achieved accuracy

```python
accuracy = np.sum(y_pred[:, 0] == digits_test_labels[:, 0]) / digits_test_labels.size
print('Accuracy:', accuracy*100, '%')
```

---
## Step 12 — Generate and plot confusion matrix

```python
cm = confusion_matrix(digits_test_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
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
# Multiclass / 09 Multiclass
# Complete Code / 完整代码
# ===============================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from digits_dataset import split_images, split_data

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = \
    split_data(20, sub_imgs, 0.8)

# Create an empty logistic regression model
lr_digits = cv2.ml.LogisticRegression_create()

# Check the default training method
print('Training Method:', lr_digits.getTrainMethod())

# Set the training method to mini-batch gradient descent and the size of the mini-batch
lr_digits.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr_digits.setMiniBatchSize(400)

# Set the number of iterations
lr_digits.setIterations(10)

# Train the logistic regressor on the set of training data
lr_digits.train(digits_train_imgs.astype(np.float32),
                cv2.ml.ROW_SAMPLE,
                digits_train_labels.astype(np.float32))

# Print the number of learned coefficients, and the number of input features
print('Number of coefficients:', len(lr_digits.get_learnt_thetas()[0]))
print('Number of input features:', len(digits_train_imgs[0, :]))

# Predict the target labels of the testing data
_, y_pred = lr_digits.predict(digits_test_imgs.astype(np.float32))

# Compute and print the achieved accuracy
accuracy = np.sum(y_pred[:, 0] == digits_test_labels[:, 0]) / digits_test_labels.size
print('Accuracy:', accuracy*100, '%')

# Generate and plot confusion matrix
cm = confusion_matrix(digits_test_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Chapter Summary / 章节总结

# Chapter 15 Summary / 第15章总结

## Theme / 主题: Chapter 15 / Chapter 15

This chapter contains **4 code files** demonstrating chapter 15.

本章包含 **4 个代码文件**，演示Chapter 15。

---
## Evolution / 演化路线

  1. `06_train.ipynb` — Train
  2. `07_predict.ipynb` — Predict
  3. `09_multiclass.ipynb` — Multiclass
  4. `digits_dataset.ipynb` — Digits Dataset

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 15) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 15）是机器学习流水线中的基础构建块。

---

### Digits Dataset

# 01 — Digits Dataset / Digits Dataset

**Chapter 15 — File 4 of 4 / 第15章 — 第4个文件（共4个）**

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
