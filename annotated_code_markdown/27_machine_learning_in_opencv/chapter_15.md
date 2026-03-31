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
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏋️ 训练模型 / Train Model
```

---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
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
# 打印输出 / Print output
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
# 转换数据类型 / Convert data type
lr_digits.train(digits_train_imgs.astype(np.float32),
                cv2.ml.ROW_SAMPLE,
                # 转换数据类型 / Convert data type
                digits_train_labels.astype(np.float32))
```

---
## Step 9 — Print the number of learned coefficients, and the number of input features

```python
# 打印输出 / Print output
print('Number of coefficients:', len(lr_digits.get_learnt_thetas()[0]))
# 打印输出 / Print output
print('Number of input features:', len(digits_train_imgs[0, :]))
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
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train / 06 Train
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
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
# 打印输出 / Print output
print('Training Method:', lr_digits.getTrainMethod())

# Set the training method to mini-batch gradient descent and the size of the mini-batch
lr_digits.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr_digits.setMiniBatchSize(400)

# Set the number of iterations
lr_digits.setIterations(10)

# Train the logistic regressor on the set of training data
# 转换数据类型 / Convert data type
lr_digits.train(digits_train_imgs.astype(np.float32),
                cv2.ml.ROW_SAMPLE,
                # 转换数据类型 / Convert data type
                digits_train_labels.astype(np.float32))

# Print the number of learned coefficients, and the number of input features
# 打印输出 / Print output
print('Number of coefficients:', len(lr_digits.get_learnt_thetas()[0]))
# 打印输出 / Print output
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
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
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
# 打印输出 / Print output
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
# 转换数据类型 / Convert data type
lr_digits.train(digits_train_imgs.astype(np.float32),
                cv2.ml.ROW_SAMPLE,
                # 转换数据类型 / Convert data type
                digits_train_labels.astype(np.float32))
```

---
## Step 9 — Predict the target labels of the testing data

```python
# 转换数据类型 / Convert data type
_, y_pred = lr_digits.predict(digits_test_imgs.astype(np.float32))
```

---
## Step 10 — Compute and print the achieved accuracy

```python
# 求和 / Calculate sum
accuracy = (np.sum(y_pred[:, 0] == digits_test_labels[:, 0]) / digits_test_labels.size) * 100
# 打印输出 / Print output
print('Accuracy:', accuracy, '%')
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
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Predict / 07 Predict
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
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
# 打印输出 / Print output
print('Training Method:', lr_digits.getTrainMethod())

# Set the training method to mini-batch gradient descent and the size of the mini-batch
lr_digits.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr_digits.setMiniBatchSize(400)

# Set the number of iterations
lr_digits.setIterations(10)

# Train the logistic regressor on the set of training data
# 转换数据类型 / Convert data type
lr_digits.train(digits_train_imgs.astype(np.float32),
                cv2.ml.ROW_SAMPLE,
                # 转换数据类型 / Convert data type
                digits_train_labels.astype(np.float32))

# Predict the target labels of the testing data
# 转换数据类型 / Convert data type
_, y_pred = lr_digits.predict(digits_test_imgs.astype(np.float32))

# Compute and print the achieved accuracy
# 求和 / Calculate sum
accuracy = (np.sum(y_pred[:, 0] == digits_test_labels[:, 0]) / digits_test_labels.size) * 100
# 打印输出 / Print output
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
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
# 打印输出 / Print output
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
# 转换数据类型 / Convert data type
lr_digits.train(digits_train_imgs.astype(np.float32),
                cv2.ml.ROW_SAMPLE,
                # 转换数据类型 / Convert data type
                digits_train_labels.astype(np.float32))
```

---
## Step 9 — Print the number of learned coefficients, and the number of input features

```python
# 打印输出 / Print output
print('Number of coefficients:', len(lr_digits.get_learnt_thetas()[0]))
# 打印输出 / Print output
print('Number of input features:', len(digits_train_imgs[0, :]))
```

---
## Step 10 — Predict the target labels of the testing data

```python
# 转换数据类型 / Convert data type
_, y_pred = lr_digits.predict(digits_test_imgs.astype(np.float32))
```

---
## Step 11 — Compute and print the achieved accuracy

```python
# 求和 / Calculate sum
accuracy = np.sum(y_pred[:, 0] == digits_test_labels[:, 0]) / digits_test_labels.size
# 打印输出 / Print output
print('Accuracy:', accuracy*100, '%')
```

---
## Step 12 — Generate and plot confusion matrix

```python
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
cm = confusion_matrix(digits_test_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
# 显示图表 / Display the plot
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
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `confusion_matrix` | 混淆矩阵：展示预测对错分布 | Confusion matrix: prediction error distribution |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Multiclass / 09 Multiclass
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
# 打印输出 / Print output
print('Training Method:', lr_digits.getTrainMethod())

# Set the training method to mini-batch gradient descent and the size of the mini-batch
lr_digits.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr_digits.setMiniBatchSize(400)

# Set the number of iterations
lr_digits.setIterations(10)

# Train the logistic regressor on the set of training data
# 转换数据类型 / Convert data type
lr_digits.train(digits_train_imgs.astype(np.float32),
                cv2.ml.ROW_SAMPLE,
                # 转换数据类型 / Convert data type
                digits_train_labels.astype(np.float32))

# Print the number of learned coefficients, and the number of input features
# 打印输出 / Print output
print('Number of coefficients:', len(lr_digits.get_learnt_thetas()[0]))
# 打印输出 / Print output
print('Number of input features:', len(digits_train_imgs[0, :]))

# Predict the target labels of the testing data
# 转换数据类型 / Convert data type
_, y_pred = lr_digits.predict(digits_test_imgs.astype(np.float32))

# Compute and print the achieved accuracy
# 求和 / Calculate sum
accuracy = np.sum(y_pred[:, 0] == digits_test_labels[:, 0]) / digits_test_labels.size
# 打印输出 / Print output
print('Accuracy:', accuracy*100, '%')

# Generate and plot confusion matrix
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
cm = confusion_matrix(digits_test_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
# 显示图表 / Display the plot
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
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
num_rows = img.shape[0] / img_size
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    num_cols = img.shape[1] / img_size
```

---
## Step 4 — Split the full image horizontally and vertically into sub-images

```python
sub_imgs = [np.hsplit(row, num_cols) for row in np.vsplit(img, num_rows)]

    # 创建NumPy数组 / Create NumPy array
    return img, np.array(sub_imgs)

def split_data(img_size, sub_imgs, ratio):
```

---
## Step 5 — Compute the partition between the training and testing data

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
partition = int(sub_imgs.shape[1] * ratio)
```

---
## Step 6 — Split dataset into training and test sets

```python
train = sub_imgs[:, :partition, :, :]
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    test = sub_imgs[:, partition:sub_imgs.shape[1], :, :]
```

---
## Step 7 — Flatten each image into a one-dimensional vector

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
train_imgs = train.reshape(-1, img_size ** 2)
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    test_imgs = test.reshape(-1, img_size ** 2)
```

---
## Step 8 — Create the groundtruth labels

```python
# 生成等差数组 / Generate array with step
labels = np.arange(10)
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    train_labels = np.repeat(labels, train_imgs.shape[0] / labels.shape[0])[:, np.newaxis]
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
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

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

def split_images(img_name, img_size):
    # Load the full image from the specified file
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    # Find the number of sub-images on each row and column according to their size
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    num_rows = img.shape[0] / img_size
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    num_cols = img.shape[1] / img_size

    # Split the full image horizontally and vertically into sub-images
    sub_imgs = [np.hsplit(row, num_cols) for row in np.vsplit(img, num_rows)]

    # 创建NumPy数组 / Create NumPy array
    return img, np.array(sub_imgs)

def split_data(img_size, sub_imgs, ratio):
    # Compute the partition between the training and testing data
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    partition = int(sub_imgs.shape[1] * ratio)

    # Split dataset into training and test sets
    train = sub_imgs[:, :partition, :, :]
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    test = sub_imgs[:, partition:sub_imgs.shape[1], :, :]

    # Flatten each image into a one-dimensional vector
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    train_imgs = train.reshape(-1, img_size ** 2)
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    test_imgs = test.reshape(-1, img_size ** 2)

    # Create the groundtruth labels
    # 生成等差数组 / Generate array with step
    labels = np.arange(10)
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    train_labels = np.repeat(labels, train_imgs.shape[0] / labels.shape[0])[:, np.newaxis]
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    test_labels = np.repeat(labels, test_imgs.shape[0] / labels.shape[0])[:, np.newaxis]

    return train_imgs, train_labels, test_imgs, test_labels
```

---
