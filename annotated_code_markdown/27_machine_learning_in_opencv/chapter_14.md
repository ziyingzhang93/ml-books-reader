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
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
# 绘制散点图 / Draw scatter plot
plt.scatter(x[:, 0], x[:, 1], c=y_true)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Generate a dataset of 2D data points and their groundtruth labels 是机器学习中的常用技术。  
  *Generate a dataset of 2D data points and their groundtruth labels is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Create Data / 01 Create Data
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)

# Plot the dataset
# 绘制散点图 / Draw scatter plot
plt.scatter(x[:, 0], x[:, 1], c=y_true)
# 显示图表 / Display the plot
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
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
    # 划分训练集和测试集 / Split into train and test sets
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
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Generate a dataset of 2D data points and their groundtruth labels 是机器学习中的常用技术。  
  *Generate a dataset of 2D data points and their groundtruth labels is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Split / 02 Split
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Plot the training and testing datasets
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
ax1.set_title('Training data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax2.set_title('Testing data')
# 显示图表 / Display the plot
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
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

lr = cv2.ml.LogisticRegression_create()
# 打印输出 / Print output
print('Training Method:', lr.getTrainMethod())
```

---
## Learning Notes / 学习笔记

- **概念**: Training Method 是机器学习中的常用技术。  
  *Training Method is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Training Method / 04 Training Method
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

lr = cv2.ml.LogisticRegression_create()
# 打印输出 / Print output
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
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model

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
```

---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
    # 划分训练集和测试集 / Split into train and test sets
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
# 转换数据类型 / Convert data type
lr.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))
```

---
## Step 8 — Print the learned coefficients

```python
# 打印输出 / Print output
print(lr.get_learnt_thetas())
```

---
## Learning Notes / 学习笔记

- **概念**: Generate a dataset of 2D data points and their groundtruth labels 是机器学习中的常用技术。  
  *Generate a dataset of 2D data points and their groundtruth labels is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `numpy` | 数值计算库 | Numerical computing library |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train / 08 Train
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Create an empty logistic regression model
lr = cv2.ml.LogisticRegression_create()

# Set the training method to mini-batch gradient descent and the size of the mini-batch
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(5)

# Set the number of iterations
lr.setIterations(10)

# Train the logistic regressor on the set of training data
# 转换数据类型 / Convert data type
lr.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))

# Print the learned coefficients
# 打印输出 / Print output
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
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
    # 划分训练集和测试集 / Split into train and test sets
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
# 打印输出 / Print output
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
# 转换数据类型 / Convert data type
lr.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))
```

---
## Step 9 — Predict the target labels of the testing data

```python
# 转换数据类型 / Convert data type
_, y_pred = lr.predict(x_test.astype(np.float32))
```

---
## Step 10 — Compute and print the achieved accuracy

```python
# 转换数据类型 / Convert data type
accuracy = (np.sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
# 打印输出 / Print output
print('Accuracy:', accuracy, '%')
```

---
## Learning Notes / 学习笔记

- **概念**: Generate a dataset of 2D data points and their groundtruth labels 是机器学习中的常用技术。  
  *Generate a dataset of 2D data points and their groundtruth labels is a common technique in machine learning.*

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
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Accuracy / 09 Accuracy
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Create an empty logistic regression model
lr = cv2.ml.LogisticRegression_create()

# Check the default training method
# 打印输出 / Print output
print('Training Method:', lr.getTrainMethod())

# Set the training method to mini-batch gradient descent and the size of the mini-batch
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(5)

# Set the number of iterations
lr.setIterations(10)

# Train the logistic regressor on the set of training data
# 转换数据类型 / Convert data type
lr.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))

# Predict the target labels of the testing data
# 转换数据类型 / Convert data type
_, y_pred = lr.predict(x_test.astype(np.float32))

# Compute and print the achieved accuracy
# 转换数据类型 / Convert data type
accuracy = (np.sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
# 打印输出 / Print output
print('Accuracy:', accuracy, '%')
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Logistic



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
