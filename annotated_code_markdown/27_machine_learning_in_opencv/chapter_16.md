# OpenCV 机器学习 / Machine Learning in OpenCV
## Chapter 16

---

### Two Classes

# 01 — Two Classes / 01 Two Classes

**Chapter 16 — File 1 of 8 / 第16章 — 第1个文件（共8个）**

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
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)
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
# Two Classes / 01 Two Classes
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)

# Plot the dataset
# 绘制散点图 / Draw scatter plot
plt.scatter(x[:, 0], x[:, 1], c=y_true)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Split

# 02 — Split / 02 Split

**Chapter 16 — File 2 of 8 / 第16章 — 第2个文件（共8个）**

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
from sklearn import model_selection as ms
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
```

---
## Step 2 — Generate a dataset of 2D data points and their groundtruth labels

```python
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)
```

---
## Step 3 — Split the data into training and test sets

```python
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)
```

---
## Step 4 — Plot the training and test datasets

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
from sklearn import model_selection as ms
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Plot the training and test datasets
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
ax1.set_title('Training data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax2.set_title('Testing data')
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Svm

# 05 — Svm / 支持向量机

**Chapter 16 — File 3 of 8 / 第16章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Generate a dataset of 2D data points and their groundtruth labels**.

本脚本演示 **Generate a dataset of 2D data points and their groundtruth labels**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn import model_selection as ms
```

---
## Step 2 — Generate a dataset of 2D data points and their groundtruth labels

```python
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)
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
## Step 4 — Split the data into training and test sets

```python
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)
```

---
## Step 5 — Plot the training and test datasets

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
## Step 6 — Create a new SVM

```python
svm = cv2.ml.SVM_create()
```

---
## Step 7 — Set the SVM kernel to linear

```python
svm.setKernel(cv2.ml.SVM_LINEAR)
```

---
## Step 8 — Train the SVM on the set of training data

```python
# 转换数据类型 / Convert data type
svm.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)
```

---
## Step 9 — Predict the target labels of the testing data

```python
# 转换数据类型 / Convert data type
_, y_pred = svm.predict(x_test.astype(np.float32))
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
| `SVM` | 支持向量机 | Support Vector Machine |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Svm / 支持向量机
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)

# Plot the dataset
# 绘制散点图 / Draw scatter plot
plt.scatter(x[:, 0], x[:, 1], c=y_true)
# 显示图表 / Display the plot
plt.show()

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Plot the training and test datasets
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
ax1.set_title('Training data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax2.set_title('Testing data')
# 显示图表 / Display the plot
plt.show()

# Create a new SVM
svm = cv2.ml.SVM_create()

# Set the SVM kernel to linear
svm.setKernel(cv2.ml.SVM_LINEAR)

# Train the SVM on the set of training data
# 转换数据类型 / Convert data type
svm.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)

# Predict the target labels of the testing data
# 转换数据类型 / Convert data type
_, y_pred = svm.predict(x_test.astype(np.float32))

# Compute and print the achieved accuracy
# 转换数据类型 / Convert data type
accuracy = (np.sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
# 打印输出 / Print output
print('Accuracy:', accuracy, '%')
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Boundary

# 08 — Boundary / 08 Boundary

**Chapter 16 — File 4 of 8 / 第16章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Generate a dataset of 2D data points and their groundtruth labels**.

本脚本演示 **Generate a dataset of 2D data points and their groundtruth labels**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn import model_selection as ms
```

---
## Step 2 — Generate a dataset of 2D data points and their groundtruth labels

```python
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)
```

---
## Step 3 — Split the data into training and test sets

```python
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)
```

---
## Step 4 — Create a new SVM

```python
svm = cv2.ml.SVM_create()
```

---
## Step 5 — Set the SVM kernel to linear

```python
svm.setKernel(cv2.ml.SVM_LINEAR)
```

---
## Step 6 — Train the SVM on the set of training data

```python
# 转换数据类型 / Convert data type
svm.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)

# 生成等差数组 / Generate array with step
x_bound, y_bound = np.meshgrid(np.arange(x_test[:, 0].min() - 1, x_test[:, 0].max() + 1, 0.05),
                               # 生成等差数组 / Generate array with step
                               np.arange(x_test[:, 1].min() - 1, x_test[:, 1].max() + 1, 0.05))

# 改变数组形状（不改变数据） / Reshape array (data unchanged)
bound_points = np.column_stack((x_bound.reshape(-1, 1), y_bound.reshape(-1, 1)))
# 转换数据类型 / Convert data type
_, bound_pred = svm.predict(bound_points.astype(np.float32))

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
plt.contourf(x_bound, y_bound, bound_pred.reshape(x_bound.shape), cmap=plt.cm.coolwarm)
# 绘制散点图 / Draw scatter plot
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
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
| `SVM` | 支持向量机 | Support Vector Machine |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Boundary / 08 Boundary
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Create a new SVM
svm = cv2.ml.SVM_create()

# Set the SVM kernel to linear
svm.setKernel(cv2.ml.SVM_LINEAR)

# Train the SVM on the set of training data
# 转换数据类型 / Convert data type
svm.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)

# 生成等差数组 / Generate array with step
x_bound, y_bound = np.meshgrid(np.arange(x_test[:, 0].min() - 1, x_test[:, 0].max() + 1, 0.05),
                               # 生成等差数组 / Generate array with step
                               np.arange(x_test[:, 1].min() - 1, x_test[:, 1].max() + 1, 0.05))

# 改变数组形状（不改变数据） / Reshape array (data unchanged)
bound_points = np.column_stack((x_bound.reshape(-1, 1), y_bound.reshape(-1, 1)))
# 转换数据类型 / Convert data type
_, bound_pred = svm.predict(bound_points.astype(np.float32))

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
plt.contourf(x_bound, y_bound, bound_pred.reshape(x_bound.shape), cmap=plt.cm.coolwarm)
# 绘制散点图 / Draw scatter plot
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Support Vectors

# 10 — Support Vectors / 10 Support Vectors

**Chapter 16 — File 5 of 8 / 第16章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Generate a dataset of 2D data points and their groundtruth labels**.

本脚本演示 **Generate a dataset of 2D data points and their groundtruth labels**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn import model_selection as ms
```

---
## Step 2 — Generate a dataset of 2D data points and their groundtruth labels

```python
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)
```

---
## Step 3 — Split the data into training and test sets

```python
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)
```

---
## Step 4 — Create a new SVM

```python
svm = cv2.ml.SVM_create()
```

---
## Step 5 — Set the SVM kernel to linear

```python
svm.setKernel(cv2.ml.SVM_LINEAR)
```

---
## Step 6 — Train the SVM on the set of training data

```python
# 转换数据类型 / Convert data type
svm.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)

# 生成等差数组 / Generate array with step
x_bound, y_bound = np.meshgrid(np.arange(x_test[:,0].min()-1, x_test[:,0].max()+1, 0.05),
                               # 生成等差数组 / Generate array with step
                               np.arange(x_test[:,1].min()-1, x_test[:,1].max()+1, 0.05))

# 改变数组形状（不改变数据） / Reshape array (data unchanged)
bound_points = np.column_stack((x_bound.reshape(-1, 1), y_bound.reshape(-1, 1)))
# 转换数据类型 / Convert data type
_, bound_pred = svm.predict(bound_points.astype(np.float32))
```

---
## Step 7 — Plot the test set

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
plt.contourf(x_bound, y_bound, bound_pred.reshape(x_bound.shape), cmap=plt.cm.coolwarm)
# 绘制散点图 / Draw scatter plot
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
# 显示图表 / Display the plot
plt.show()

support_vect = svm.getUncompressedSupportVectors()
```

---
## Step 8 — Plot the support vectors

```python
# 绘制散点图 / Draw scatter plot
plt.scatter(x[:, 0], x[:, 1], c=y_true)
# 绘制散点图 / Draw scatter plot
plt.scatter(support_vect[:, 0], support_vect[:, 1], c='red')
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
| `SVM` | 支持向量机 | Support Vector Machine |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Support Vectors / 10 Support Vectors
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Create a new SVM
svm = cv2.ml.SVM_create()

# Set the SVM kernel to linear
svm.setKernel(cv2.ml.SVM_LINEAR)

# Train the SVM on the set of training data
# 转换数据类型 / Convert data type
svm.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)

# 生成等差数组 / Generate array with step
x_bound, y_bound = np.meshgrid(np.arange(x_test[:,0].min()-1, x_test[:,0].max()+1, 0.05),
                               # 生成等差数组 / Generate array with step
                               np.arange(x_test[:,1].min()-1, x_test[:,1].max()+1, 0.05))

# 改变数组形状（不改变数据） / Reshape array (data unchanged)
bound_points = np.column_stack((x_bound.reshape(-1, 1), y_bound.reshape(-1, 1)))
# 转换数据类型 / Convert data type
_, bound_pred = svm.predict(bound_points.astype(np.float32))

# Plot the test set
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
plt.contourf(x_bound, y_bound, bound_pred.reshape(x_bound.shape), cmap=plt.cm.coolwarm)
# 绘制散点图 / Draw scatter plot
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
# 显示图表 / Display the plot
plt.show()

support_vect = svm.getUncompressedSupportVectors()

# Plot the support vectors
# 绘制散点图 / Draw scatter plot
plt.scatter(x[:, 0], x[:, 1], c=y_true)
# 绘制散点图 / Draw scatter plot
plt.scatter(support_vect[:, 0], support_vect[:, 1], c='red')
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Nonlinear

# 11 — Nonlinear / 线性模型

**Chapter 16 — File 6 of 8 / 第16章 — 第6个文件（共8个）**

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
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=8, random_state=15)
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
## Step 4 — Split the data into training and test sets

```python
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)
```

---
## Step 5 — Plot the training and test datasets

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
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nonlinear / 线性模型
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=8, random_state=15)

# Plot the dataset
# 绘制散点图 / Draw scatter plot
plt.scatter(x[:, 0], x[:, 1], c=y_true)
# 显示图表 / Display the plot
plt.show()

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Plot the training and test datasets
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
ax1.set_title('Training data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax2.set_title('Testing data')
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Relaxed

# 12 — Relaxed / 12 Relaxed

**Chapter 16 — File 7 of 8 / 第16章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Generate a dataset of 2D data points and their groundtruth labels**.

本脚本演示 **Generate a dataset of 2D data points and their groundtruth labels**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn import model_selection as ms
```

---
## Step 2 — Generate a dataset of 2D data points and their groundtruth labels

```python
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=8, random_state=15)
```

---
## Step 3 — Split the data into training and test sets

```python
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)
```

---
## Step 4 — Create a new SVM

```python
svm = cv2.ml.SVM_create()
```

---
## Step 5 — Set the SVM kernel to linear

```python
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(10)
```

---
## Step 6 — Train the SVM on the set of training data

```python
# 转换数据类型 / Convert data type
svm.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)

# 生成等差数组 / Generate array with step
x_bound, y_bound = np.meshgrid(np.arange(x_test[:, 0].min() - 1, x_test[:, 0].max() + 1, 0.05),
                               # 生成等差数组 / Generate array with step
                               np.arange(x_test[:, 1].min() - 1, x_test[:, 1].max() + 1, 0.05))

# 改变数组形状（不改变数据） / Reshape array (data unchanged)
bound_points = np.column_stack((x_bound.reshape(-1, 1), y_bound.reshape(-1, 1)))
# 转换数据类型 / Convert data type
_, bound_pred = svm.predict(bound_points.astype(np.float32))
```

---
## Step 7 — Plot the test set

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
plt.contourf(x_bound, y_bound, bound_pred.reshape(x_bound.shape), cmap=plt.cm.coolwarm)
# 绘制散点图 / Draw scatter plot
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
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
| `SVM` | 支持向量机 | Support Vector Machine |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Relaxed / 12 Relaxed
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=8, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Create a new SVM
svm = cv2.ml.SVM_create()

# Set the SVM kernel to linear
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(10)

# Train the SVM on the set of training data
# 转换数据类型 / Convert data type
svm.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)

# 生成等差数组 / Generate array with step
x_bound, y_bound = np.meshgrid(np.arange(x_test[:, 0].min() - 1, x_test[:, 0].max() + 1, 0.05),
                               # 生成等差数组 / Generate array with step
                               np.arange(x_test[:, 1].min() - 1, x_test[:, 1].max() + 1, 0.05))

# 改变数组形状（不改变数据） / Reshape array (data unchanged)
bound_points = np.column_stack((x_bound.reshape(-1, 1), y_bound.reshape(-1, 1)))
# 转换数据类型 / Convert data type
_, bound_pred = svm.predict(bound_points.astype(np.float32))

# Plot the test set
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
plt.contourf(x_bound, y_bound, bound_pred.reshape(x_bound.shape), cmap=plt.cm.coolwarm)
# 绘制散点图 / Draw scatter plot
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Rbf

# 13 — Rbf / 13 Rbf

**Chapter 16 — File 8 of 8 / 第16章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Generate a dataset of 2D data points and their groundtruth labels**.

本脚本演示 **Generate a dataset of 2D data points and their groundtruth labels**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn import model_selection as ms
```

---
## Step 2 — Generate a dataset of 2D data points and their groundtruth labels

```python
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=8, random_state=15)
```

---
## Step 3 — Split the data into training and test sets

```python
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)
```

---
## Step 4 — Create a new SVM

```python
svm = cv2.ml.SVM_create()
```

---
## Step 5 — Set the SVM kernel to RBF

```python
svm.setKernel(cv2.ml.SVM_RBF)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(10)
svm.setGamma(0.1)
```

---
## Step 6 — Train the SVM on the set of training data

```python
# 转换数据类型 / Convert data type
svm.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)

# 生成等差数组 / Generate array with step
x_bound, y_bound = np.meshgrid(np.arange(x_test[:, 0].min() - 1, x_test[:, 0].max() + 1, 0.05),
                               # 生成等差数组 / Generate array with step
                               np.arange(x_test[:, 1].min() - 1, x_test[:, 1].max() + 1, 0.05))

# 改变数组形状（不改变数据） / Reshape array (data unchanged)
bound_points = np.column_stack((x_bound.reshape(-1, 1), y_bound.reshape(-1, 1)))
# 转换数据类型 / Convert data type
_, bound_pred = svm.predict(bound_points.astype(np.float32))
```

---
## Step 7 — Plot the test set

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
plt.contourf(x_bound, y_bound, bound_pred.reshape(x_bound.shape), cmap=plt.cm.coolwarm)
# 绘制散点图 / Draw scatter plot
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
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
| `SVM` | 支持向量机 | Support Vector Machine |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Rbf / 13 Rbf
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=8, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    # 划分训练集和测试集 / Split into train and test sets
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Create a new SVM
svm = cv2.ml.SVM_create()

# Set the SVM kernel to RBF
svm.setKernel(cv2.ml.SVM_RBF)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(10)
svm.setGamma(0.1)

# Train the SVM on the set of training data
# 转换数据类型 / Convert data type
svm.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)

# 生成等差数组 / Generate array with step
x_bound, y_bound = np.meshgrid(np.arange(x_test[:, 0].min() - 1, x_test[:, 0].max() + 1, 0.05),
                               # 生成等差数组 / Generate array with step
                               np.arange(x_test[:, 1].min() - 1, x_test[:, 1].max() + 1, 0.05))

# 改变数组形状（不改变数据） / Reshape array (data unchanged)
bound_points = np.column_stack((x_bound.reshape(-1, 1), y_bound.reshape(-1, 1)))
# 转换数据类型 / Convert data type
_, bound_pred = svm.predict(bound_points.astype(np.float32))

# Plot the test set
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
plt.contourf(x_bound, y_bound, bound_pred.reshape(x_bound.shape), cmap=plt.cm.coolwarm)
# 绘制散点图 / Draw scatter plot
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 16 Summary / 第16章总结

## Theme / 主题: Chapter 16 / Chapter 16

This chapter contains **8 code files** demonstrating chapter 16.

本章包含 **8 个代码文件**，演示Chapter 16。

---
## Evolution / 演化路线

  1. `01_two_classes.ipynb` — Two Classes
  2. `02_split.ipynb` — Split
  3. `05_svm.ipynb` — Svm
  4. `08_boundary.ipynb` — Boundary
  5. `10_support_vectors.ipynb` — Support Vectors
  6. `11_nonlinear.ipynb` — Nonlinear
  7. `12_relaxed.ipynb` — Relaxed
  8. `13_rbf.ipynb` — Rbf

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 16) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 16）是机器学习流水线中的基础构建块。

---
