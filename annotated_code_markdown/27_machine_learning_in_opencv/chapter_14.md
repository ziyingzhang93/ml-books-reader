# OpenCV 机器学习 / Machine Learning in OpenCV
## Chapter 14

---

### Create Data

# 01 — Create Data / 01 Create Data

**Chapter 14 — File 1 of 6 / 第14章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Generate a dataset of 2D data points and their groundtruth labels**.

本脚本演示 **Generate a dataset of 2D data points and their groundtruth labels**。

---
## Step 1 — Step 1

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
```

---
## Step 2 — Generate a dataset of 2D data points and their groundtruth labels

```python
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)
```

---
## Step 3 — Plot the dataset

```python
plt.scatter(x[:, 0], x[:, 1], c=y_true)
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Generate a dataset of 2D data points and their groundtruth labels 是机器学习中的常用技术。  
  *Generate a dataset of 2D data points and their groundtruth labels is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Create Data / 01 Create Data
# Complete Code / 完整代码
# ===============================

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)

# Plot the dataset
plt.scatter(x[:, 0], x[:, 1], c=y_true)
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Split

# 02 — Split / 02 Split

**Chapter 14 — File 2 of 6 / 第14章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Generate a dataset of 2D data points and their groundtruth labels**.

本脚本演示 **Generate a dataset of 2D data points and their groundtruth labels**。

---
## Step 1 — Step 1

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms
```

---
## Step 2 — Generate a dataset of 2D data points and their groundtruth labels

```python
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)
```

---
## Step 3 — Split the data into training and test sets

```python
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)
```

---
## Step 4 — Plot the training and testing datasets

```python
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
ax1.set_title('Training data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax2.set_title('Testing data')
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Generate a dataset of 2D data points and their groundtruth labels 是机器学习中的常用技术。  
  *Generate a dataset of 2D data points and their groundtruth labels is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Split / 02 Split
# Complete Code / 完整代码
# ===============================

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Plot the training and testing datasets
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
ax1.set_title('Training data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax2.set_title('Testing data')
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Training Method

# 04 — Training Method / 04 Training Method

**Chapter 14 — File 3 of 6 / 第14章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Training Method**.

本脚本演示 **04 Training Method**。

---
## Step 1 — Step 1

```python
import cv2

lr = cv2.ml.LogisticRegression_create()
print('Training Method:', lr.getTrainMethod())
```

---
## Learning Notes / 学习笔记

- **概念**: Training Method 是机器学习中的常用技术。  
  *Training Method is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Training Method / 04 Training Method
# Complete Code / 完整代码
# ===============================

import cv2

lr = cv2.ml.LogisticRegression_create()
print('Training Method:', lr.getTrainMethod())
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Train

# 08 — Train / 08 Train

**Chapter 14 — File 4 of 6 / 第14章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Generate a dataset of 2D data points and their groundtruth labels**.

本脚本演示 **Generate a dataset of 2D data points and their groundtruth labels**。

---
## Step 1 — Step 1

```python
import cv2
import numpy as np
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms
```

---
## Step 2 — Generate a dataset of 2D data points and their groundtruth labels

```python
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)
```

---
## Step 3 — Split the data into training and test sets

```python
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)
```

---
## Step 4 — Create an empty logistic regression model

```python
lr = cv2.ml.LogisticRegression_create()
```

---
## Step 5 — Set the training method to mini-batch gradient descent and the size of the mini-batch

```python
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(5)
```

---
## Step 6 — Set the number of iterations

```python
lr.setIterations(10)
```

---
## Step 7 — Train the logistic regressor on the set of training data

```python
lr.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))
```

---
## Step 8 — Print the learned coefficients

```python
print(lr.get_learnt_thetas())
```

---
## Learning Notes / 学习笔记

- **概念**: Generate a dataset of 2D data points and their groundtruth labels 是机器学习中的常用技术。  
  *Generate a dataset of 2D data points and their groundtruth labels is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train / 08 Train
# Complete Code / 完整代码
# ===============================

import cv2
import numpy as np
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Create an empty logistic regression model
lr = cv2.ml.LogisticRegression_create()

# Set the training method to mini-batch gradient descent and the size of the mini-batch
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(5)

# Set the number of iterations
lr.setIterations(10)

# Train the logistic regressor on the set of training data
lr.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))

# Print the learned coefficients
print(lr.get_learnt_thetas())
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Accuracy

# 09 — Accuracy / 09 Accuracy

**Chapter 14 — File 5 of 6 / 第14章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Generate a dataset of 2D data points and their groundtruth labels**.

本脚本演示 **Generate a dataset of 2D data points and their groundtruth labels**。

---
## Step 1 — Step 1

```python
import cv2
import numpy as np
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms
```

---
## Step 2 — Generate a dataset of 2D data points and their groundtruth labels

```python
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)
```

---
## Step 3 — Split the data into training and test sets

```python
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)
```

---
## Step 4 — Create an empty logistic regression model

```python
lr = cv2.ml.LogisticRegression_create()
```

---
## Step 5 — Check the default training method

```python
print('Training Method:', lr.getTrainMethod())
```

---
## Step 6 — Set the training method to mini-batch gradient descent and the size of the mini-batch

```python
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(5)
```

---
## Step 7 — Set the number of iterations

```python
lr.setIterations(10)
```

---
## Step 8 — Train the logistic regressor on the set of training data

```python
lr.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))
```

---
## Step 9 — Predict the target labels of the testing data

```python
_, y_pred = lr.predict(x_test.astype(np.float32))
```

---
## Step 10 — Compute and print the achieved accuracy

```python
accuracy = (np.sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, '%')
```

---
## Learning Notes / 学习笔记

- **概念**: Generate a dataset of 2D data points and their groundtruth labels 是机器学习中的常用技术。  
  *Generate a dataset of 2D data points and their groundtruth labels is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Accuracy / 09 Accuracy
# Complete Code / 完整代码
# ===============================

import cv2
import numpy as np
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Create an empty logistic regression model
lr = cv2.ml.LogisticRegression_create()

# Check the default training method
print('Training Method:', lr.getTrainMethod())

# Set the training method to mini-batch gradient descent and the size of the mini-batch
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(5)

# Set the number of iterations
lr.setIterations(10)

# Train the logistic regressor on the set of training data
lr.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))

# Predict the target labels of the testing data
_, y_pred = lr.predict(x_test.astype(np.float32))

# Compute and print the achieved accuracy
accuracy = (np.sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, '%')
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Logistic

# 11 — Logistic / 11 Logistic

**Chapter 14 — File 6 of 6 / 第14章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Generate a dataset of 2D data points and their groundtruth labels**.

本脚本演示 **Generate a dataset of 2D data points and their groundtruth labels**。

---
## Step 1 — Step 1

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms
```

---
## Step 2 — Generate a dataset of 2D data points and their groundtruth labels

```python
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)
```

---
## Step 3 — Plot the dataset

```python
plt.scatter(x[:, 0], x[:, 1], c=y_true)
plt.show()
```

---
## Step 4 — Split the data into training and test sets

```python
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)
```

---
## Step 5 — Plot the training and testing datasets

```python
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
ax1.set_title('Training data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax2.set_title('Testing data')
plt.show()
```

---
## Step 6 — Create an empty logistic regression model

```python
lr = cv2.ml.LogisticRegression_create()
```

---
## Step 7 — Check the default training method

```python
print('Training Method:', lr.getTrainMethod())
```

---
## Step 8 — Set the training method to mini-batch gradient descent and the size of the mini-batch

```python
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(5)
```

---
## Step 9 — Set the number of iterations

```python
lr.setIterations(10)
```

---
## Step 10 — Train the logistic regressor on the set of training data

```python
lr.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))
```

---
## Step 11 — Print the learned coefficients

```python
print(lr.get_learnt_thetas())
```

---
## Step 12 — Predict the target labels of the testing data

```python
_, y_pred = lr.predict(x_test.astype(np.float32))
```

---
## Step 13 — Compute and print the achieved accuracy

```python
accuracy = (np.sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, '%')
```

---
## Step 14 — Plot the groundtruth and predicted class labels

```python
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax1.set_title('Groundtruth testing data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_pred)
ax2.set_title('Predicted testing data')
plt.show()
```

---
## Step 15 — Print the groundtruth and predicted class labels of the testing data

```python
print('Groundtruth class labels:', y_test)
print('Predicted class labels:  ', y_pred[:, 0].astype(int))
```

---
## Learning Notes / 学习笔记

- **概念**: Generate a dataset of 2D data points and their groundtruth labels 是机器学习中的常用技术。  
  *Generate a dataset of 2D data points and their groundtruth labels is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Logistic / 11 Logistic
# Complete Code / 完整代码
# ===============================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)

# Plot the dataset
plt.scatter(x[:, 0], x[:, 1], c=y_true)
plt.show()

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Plot the training and testing datasets
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
ax1.set_title('Training data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax2.set_title('Testing data')
plt.show()

# Create an empty logistic regression model
lr = cv2.ml.LogisticRegression_create()

# Check the default training method
print('Training Method:', lr.getTrainMethod())

# Set the training method to mini-batch gradient descent and the size of the mini-batch
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(5)

# Set the number of iterations
lr.setIterations(10)

# Train the logistic regressor on the set of training data
lr.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))

# Print the learned coefficients
print(lr.get_learnt_thetas())

# Predict the target labels of the testing data
_, y_pred = lr.predict(x_test.astype(np.float32))

# Compute and print the achieved accuracy
accuracy = (np.sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, '%')

# Plot the groundtruth and predicted class labels
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax1.set_title('Groundtruth testing data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_pred)
ax2.set_title('Predicted testing data')
plt.show()

# Print the groundtruth and predicted class labels of the testing data
print('Groundtruth class labels:', y_test)
print('Predicted class labels:  ', y_pred[:, 0].astype(int))
```

---

### Chapter Summary / 章节总结

# Chapter 14 Summary / 第14章总结

## Theme / 主题: Chapter 14 / Chapter 14

This chapter contains **6 code files** demonstrating chapter 14.

本章包含 **6 个代码文件**，演示Chapter 14。

---
## Evolution / 演化路线

  1. `01_create_data.ipynb` — Create Data
  2. `02_split.ipynb` — Split
  3. `04_training_method.ipynb` — Training Method
  4. `08_train.ipynb` — Train
  5. `09_accuracy.ipynb` — Accuracy
  6. `11_logistic.ipynb` — Logistic

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 14) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 14）是机器学习流水线中的基础构建块。

---
