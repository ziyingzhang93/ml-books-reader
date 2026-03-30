# OpenCV ML
## Chapter 12

---

### Create Data

# 01 — Create Data / 01 Create Data

**Chapter 12 — File 1 of 5 / 第12章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Generating a dataset of 2D data points and their groundtruth labels**.

本脚本演示 **Generating a dataset of 2D data points and their groundtruth labels**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
```

---
## Step 2 — Generating a dataset of 2D data points and their groundtruth labels

```python
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)
```

---
## Step 3 — Plotting the dataset

```python
plt.scatter(x[:, 0], x[:, 1], c=y_true)
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Generating a dataset of 2D data points and their groundtruth labels 是机器学习中的常用技术。  
  *Generating a dataset of 2D data points and their groundtruth labels is a common technique in machine learning.*

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

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generating a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)

# Plotting the dataset
plt.scatter(x[:, 0], x[:, 1], c=y_true)
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Bayes

# 04 — Bayes / 04 Bayes

**Chapter 12 — File 2 of 5 / 第12章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Generate a dataset of 2D data points and their groundtruth labels**.

本脚本演示 **Generate a dataset of 2D data points and their groundtruth labels**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
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
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

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
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)
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
## Step 5 — Create a new Normal Bayes Classifier

```python
norm_bayes = cv2.ml.NormalBayesClassifier_create()
```

---
## Step 6 — Train the classifier on the training data

```python
norm_bayes.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)
```

---
## Step 7 — Generate a prediction from the trained classifier

```python
ret, y_pred, y_probs = norm_bayes.predictProb(x_test.astype(np.float32))
```

---
## Step 8 — Plot the class predictions

```python
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred)
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
# Bayes / 04 Bayes
# Complete Code / 完整代码
# ===============================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)

# Plot the dataset
plt.scatter(x[:, 0], x[:, 1], c=y_true)
plt.show()

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Create a new Normal Bayes Classifier
norm_bayes = cv2.ml.NormalBayesClassifier_create()

# Train the classifier on the training data
norm_bayes.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)

# Generate a prediction from the trained classifier
ret, y_pred, y_probs = norm_bayes.predictProb(x_test.astype(np.float32))

# Plot the class predictions
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred)
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Skin

# 05 — Skin / 05 Skin

**Chapter 12 — File 3 of 5 / 第12章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Load data from text file**.

本脚本演示 **Load data from text file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
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
import cv2
import numpy as np
from matplotlib.colors import rgb_to_hsv
```

---
## Step 2 — Load data from text file

```python
data = np.loadtxt("Data/Skin_NonSkin.txt", dtype=int)
```

---
## Step 3 — Select the BGR values from the loaded data

```python
BGR = data[:, :3]
```

---
## Step 4 — Convert to RGB by swapping the array columns

```python
RGB = BGR.copy()
RGB[:, [2, 0]] = RGB[:, [0, 2]]
```

---
## Step 5 — Convert RGB values to HSV

```python
HSV = rgb_to_hsv(RGB.reshape(RGB.shape[0], -1, 3) / 255)
HSV = HSV.reshape(RGB.shape[0], 3)
```

---
## Step 6 — Select only the hue values

```python
hue = HSV[:, 0] * 360
```

---
## Step 7 — Select the labels from the loaded data

```python
labels = data[:, -1]
```

---
## Step 8 — Create a new Normal Bayes Classifier

```python
norm_bayes = cv2.ml.NormalBayesClassifier_create()
```

---
## Step 9 — Train the classifier on the hue values

```python
norm_bayes.train(hue.astype(np.float32), cv2.ml.ROW_SAMPLE, labels)
```

---
## Learning Notes / 学习笔记

- **概念**: Load data from text file 是机器学习中的常用技术。  
  *Load data from text file is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Skin / 05 Skin
# Complete Code / 完整代码
# ===============================

import cv2
import numpy as np
from matplotlib.colors import rgb_to_hsv

# Load data from text file
data = np.loadtxt("Data/Skin_NonSkin.txt", dtype=int)

# Select the BGR values from the loaded data
BGR = data[:, :3]

# Convert to RGB by swapping the array columns
RGB = BGR.copy()
RGB[:, [2, 0]] = RGB[:, [0, 2]]

# Convert RGB values to HSV
HSV = rgb_to_hsv(RGB.reshape(RGB.shape[0], -1, 3) / 255)
HSV = HSV.reshape(RGB.shape[0], 3)

# Select only the hue values
hue = HSV[:, 0] * 360

# Select the labels from the loaded data
labels = data[:, -1]

# Create a new Normal Bayes Classifier
norm_bayes = cv2.ml.NormalBayesClassifier_create()

# Train the classifier on the hue values
norm_bayes.train(hue.astype(np.float32), cv2.ml.ROW_SAMPLE, labels)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Histogram

# 06 — Histogram / 06 Histogram

**Chapter 12 — File 4 of 5 / 第12章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Load data from text file**.

本脚本演示 **Load data from text file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
```

---
## Step 2 — Load data from text file

```python
data = np.loadtxt("Data/Skin_NonSkin.txt", dtype=int)
```

---
## Step 3 — Select the BGR values from the loaded data

```python
BGR = data[:, :3]
```

---
## Step 4 — Convert to RGB by swapping the array columns

```python
RGB = BGR.copy()
RGB[:, [2, 0]] = RGB[:, [0, 2]]
```

---
## Step 5 — Convert RGB values to HSV

```python
HSV = rgb_to_hsv(RGB.reshape(RGB.shape[0], -1, 3) / 255)
HSV = HSV.reshape(RGB.shape[0], 3)
```

---
## Step 6 — Select only the hue values

```python
hue = HSV[:, 0] * 360
```

---
## Step 7 — Select the labels from the loaded data

```python
labels = data[:, -1]
```

---
## Step 8 — Choose the skin-labelled hue values

```python
skin = hue[labels == 1]
```

---
## Step 9 — Compute their histogram

```python
hist, bin_edges = np.histogram(skin, range=[0, 360], bins=360)
```

---
## Step 10 — Display the computed histogram

```python
plt.bar(bin_edges[:-1], hist, width=4)
plt.xlabel('Hue')
plt.ylabel('Frequency')
plt.title('Histogram of the hue values of skin pixels')
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load data from text file 是机器学习中的常用技术。  
  *Load data from text file is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Histogram / 06 Histogram
# Complete Code / 完整代码
# ===============================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv

# Load data from text file
data = np.loadtxt("Data/Skin_NonSkin.txt", dtype=int)

# Select the BGR values from the loaded data
BGR = data[:, :3]

# Convert to RGB by swapping the array columns
RGB = BGR.copy()
RGB[:, [2, 0]] = RGB[:, [0, 2]]

# Convert RGB values to HSV
HSV = rgb_to_hsv(RGB.reshape(RGB.shape[0], -1, 3) / 255)
HSV = HSV.reshape(RGB.shape[0], 3)

# Select only the hue values
hue = HSV[:, 0] * 360

# Select the labels from the loaded data
labels = data[:, -1]

# Choose the skin-labelled hue values
skin = hue[labels == 1]

# Compute their histogram
hist, bin_edges = np.histogram(skin, range=[0, 360], bins=360)

# Display the computed histogram
plt.bar(bin_edges[:-1], hist, width=4)
plt.xlabel('Hue')
plt.ylabel('Frequency')
plt.title('Histogram of the hue values of skin pixels')
plt.show()
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Segment

# 07 — Segment / 07 Segment

**Chapter 12 — File 5 of 5 / 第12章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Load data from text file**.

本脚本演示 **Load data from text file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
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
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
```

---
## Step 2 — Load data from text file

```python
data = np.loadtxt("Data/Skin_NonSkin.txt", dtype=int)
```

---
## Step 3 — Select the BGR values from the loaded data

```python
BGR = data[:, :3]
```

---
## Step 4 — Convert to RGB by swapping the array columns

```python
RGB = BGR.copy()
RGB[:, [2, 0]] = RGB[:, [0, 2]]
```

---
## Step 5 — Convert RGB values to HSV

```python
HSV = rgb_to_hsv(RGB.reshape(RGB.shape[0], -1, 3) / 255)
HSV = HSV.reshape(RGB.shape[0], 3)
```

---
## Step 6 — Select only the hue values

```python
hue = HSV[:, 0] * 360
```

---
## Step 7 — Select the labels from the loaded data

```python
labels = data[:, -1]
```

---
## Step 8 — Create a new Normal Bayes Classifier

```python
norm_bayes = cv2.ml.NormalBayesClassifier_create()
```

---
## Step 9 — Train the classifier on the hue values

```python
norm_bayes.train(hue.astype(np.float32), cv2.ml.ROW_SAMPLE, labels)
```

---
## Step 10 — Load a test image

```python
face_img = cv2.imread("Images/face.jpg")
```

---
## Step 11 — Reshape the image into a three-column array

```python
face_BGR = face_img.reshape(-1, 3)
```

---
## Step 12 — Convert to RGB by swapping the array columns

```python
face_RGB = face_BGR.copy()
face_RGB[:, [2, 0]] = face_RGB[:, [0, 2]]
```

---
## Step 13 — Convert from RGB to HSV

```python
face_HSV = rgb_to_hsv(face_RGB.reshape(face_RGB.shape[0], -1, 3) / 255)
face_HSV = face_HSV.reshape(face_RGB.shape[0], 3)
```

---
## Step 14 — Select only the hue values

```python
face_hue = face_HSV[:, 0] * 360
```

---
## Step 15 — Display the hue image

```python
plt.imshow(face_hue.reshape(face_img.shape[0], face_img.shape[1]))
plt.show()
```

---
## Step 16 — Generate a prediction from the trained classifier

```python
ret, labels_pred, output_probs = norm_bayes.predictProb(face_hue.astype(np.float32))
```

---
## Step 17 — Reshape array into the input image size and choose the skin-labelled pixels

```python
skin_mask = labels_pred.reshape(face_img.shape[0], face_img.shape[1], 1) == 1
```

---
## Step 18 — Display the segmented image

```python
plt.imshow(skin_mask, cmap='gray')
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load data from text file 是机器学习中的常用技术。  
  *Load data from text file is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Segment / 07 Segment
# Complete Code / 完整代码
# ===============================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv

# Load data from text file
data = np.loadtxt("Data/Skin_NonSkin.txt", dtype=int)

# Select the BGR values from the loaded data
BGR = data[:, :3]

# Convert to RGB by swapping the array columns
RGB = BGR.copy()
RGB[:, [2, 0]] = RGB[:, [0, 2]]

# Convert RGB values to HSV
HSV = rgb_to_hsv(RGB.reshape(RGB.shape[0], -1, 3) / 255)
HSV = HSV.reshape(RGB.shape[0], 3)

# Select only the hue values
hue = HSV[:, 0] * 360

# Select the labels from the loaded data
labels = data[:, -1]

# Create a new Normal Bayes Classifier
norm_bayes = cv2.ml.NormalBayesClassifier_create()

# Train the classifier on the hue values
norm_bayes.train(hue.astype(np.float32), cv2.ml.ROW_SAMPLE, labels)
# Load a test image
face_img = cv2.imread("Images/face.jpg")

# Reshape the image into a three-column array
face_BGR = face_img.reshape(-1, 3)

# Convert to RGB by swapping the array columns
face_RGB = face_BGR.copy()
face_RGB[:, [2, 0]] = face_RGB[:, [0, 2]]

# Convert from RGB to HSV
face_HSV = rgb_to_hsv(face_RGB.reshape(face_RGB.shape[0], -1, 3) / 255)
face_HSV = face_HSV.reshape(face_RGB.shape[0], 3)

# Select only the hue values
face_hue = face_HSV[:, 0] * 360

# Display the hue image
plt.imshow(face_hue.reshape(face_img.shape[0], face_img.shape[1]))
plt.show()

# Generate a prediction from the trained classifier
ret, labels_pred, output_probs = norm_bayes.predictProb(face_hue.astype(np.float32))

# Reshape array into the input image size and choose the skin-labelled pixels
skin_mask = labels_pred.reshape(face_img.shape[0], face_img.shape[1], 1) == 1

# Display the segmented image
plt.imshow(skin_mask, cmap='gray')
plt.show()
```

---

### Chapter Summary

# Chapter 12 Summary / 第12章总结

## Theme / 主题: Chapter 12 / Chapter 12

This chapter contains **5 code files** demonstrating chapter 12.

本章包含 **5 个代码文件**，演示Chapter 12。

---
## Evolution / 演化路线

  1. `01_create_data.ipynb` — Create Data
  2. `04_bayes.ipynb` — Bayes
  3. `05_skin.ipynb` — Skin
  4. `06_histogram.ipynb` — Histogram
  5. `07_segment.ipynb` — Segment

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 12) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 12）是机器学习流水线中的基础构建块。

---
