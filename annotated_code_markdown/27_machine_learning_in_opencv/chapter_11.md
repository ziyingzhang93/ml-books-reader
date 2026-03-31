# OpenCV 机器学习 / Machine Learning in OpenCV
## Chapter 11

---

### Check Shape

# 02 — Check Shape / 02 Check Shape

**Chapter 11 — File 1 of 6 / 第11章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Load the digits image and divide it into sub-images**.

本脚本演示 **Load the digits image and divide it into sub-images**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from digits_dataset import split_images, split_data
```

---
## Step 2 — Load the digits image and divide it into sub-images

```python
img, sub_imgs = split_images('Images/digits.png', 20)
```

---
## Step 3 — Create the groundtruth labels

```python
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)
```

---
## Step 4 — Check the shape of the 'imgs' array

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(imgs.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: Load the digits image and divide it into sub-images 是机器学习中的常用技术。  
  *Load the digits image and divide it into sub-images is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Check Shape / 02 Check Shape
# Complete Code / 完整代码
# ===============================

from digits_dataset import split_images, split_data

# Load the digits image and divide it into sub-images
img, sub_imgs = split_images('Images/digits.png', 20)

# Create the groundtruth labels
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)

# Check the shape of the 'imgs' array
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(imgs.shape)
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Show Cluster



---

### Classification

# 07 — Classification / 分类

**Chapter 11 — File 3 of 6 / 第11章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Load the digits image and divide it into sub-images**.

本脚本演示 **Load the digits image and divide it into sub-images**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  📈 可视化结果 / Visualize Results
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
```

---
## Step 2 — Load the digits image and divide it into sub-images

```python
img, sub_imgs = split_images('Images/digits.png', 20)
```

---
## Step 3 — Create the groundtruth labels

```python
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)
```

---
## Step 4 — Check the shape of the 'imgs' array

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(imgs.shape)
```

---
## Step 5 — Specify the algorithm's termination criteria

```python
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
```

---
## Step 6 — Run the k-means clustering algorithm on the image data

```python
# 转换数据类型 / Convert data type
_, clusters, centers = cv2.kmeans(data=imgs.astype(np.float32), K=10, bestLabels=None,
                                  criteria=criteria, attempts=10,
                                  flags=cv2.KMEANS_RANDOM_CENTERS)
```

---
## Step 7 — Reshape array into 20x20 images

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
imgs_centers = centers.reshape(-1, 20, 20)
```

---
## Step 8 — Visualize the cluster centers

```python
fig, ax = plt.subplots(2, 5)

# 将多个序列配对 / Pair multiple sequences
for i, center in zip(ax.flat, imgs_centers):
    i.imshow(center)

# 显示图表 / Display the plot
plt.show()
```

---
## Step 9 — Cluster labels

```python
# 创建NumPy数组 / Create NumPy array
labels = np.array([2, 0, 7, 5, 1, 4, 6, 9, 3, 8])

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
labels_pred = np.zeros(labels_true.shape, dtype='int')
```

---
## Step 10 — Re-order the cluster labels

```python
# 生成整数序列 / Generate integer sequence
for i in range(10):
    mask = clusters.ravel() == i
    labels_pred[mask] = labels[i]
```

---
## Step 11 — Calculate the algorithm's accuracy

```python
# 求和 / Calculate sum
accuracy = (np.sum(labels_true == labels_pred) / labels_true.size) * 100
```

---
## Step 12 — Print the accuracy

```python
# 打印输出 / Print output
print("Accuracy: {0:.2f}%".format(accuracy))
```

---
## Learning Notes / 学习笔记

- **概念**: Load the digits image and divide it into sub-images 是机器学习中的常用技术。  
  *Load the digits image and divide it into sub-images is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `KMeans` | K均值聚类 | K-Means clustering |
| `matplotlib` | 绑图库 | Plotting library |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.zeros` | 全零数组 | Array filled with zeros |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Classification / 分类
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from digits_dataset import split_images, split_data

# Load the digits image and divide it into sub-images
img, sub_imgs = split_images('Images/digits.png', 20)

# Create the groundtruth labels
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)

# Check the shape of the 'imgs' array
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(imgs.shape)

# Specify the algorithm's termination criteria
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the image data
# 转换数据类型 / Convert data type
_, clusters, centers = cv2.kmeans(data=imgs.astype(np.float32), K=10, bestLabels=None,
                                  criteria=criteria, attempts=10,
                                  flags=cv2.KMEANS_RANDOM_CENTERS)

# Reshape array into 20x20 images
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
imgs_centers = centers.reshape(-1, 20, 20)

# Visualize the cluster centers
fig, ax = plt.subplots(2, 5)

# 将多个序列配对 / Pair multiple sequences
for i, center in zip(ax.flat, imgs_centers):
    i.imshow(center)

# 显示图表 / Display the plot
plt.show()

# Cluster labels
# 创建NumPy数组 / Create NumPy array
labels = np.array([2, 0, 7, 5, 1, 4, 6, 9, 3, 8])

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
labels_pred = np.zeros(labels_true.shape, dtype='int')

# Re-order the cluster labels
# 生成整数序列 / Generate integer sequence
for i in range(10):
    mask = clusters.ravel() == i
    labels_pred[mask] = labels[i]

# Calculate the algorithm's accuracy
# 求和 / Calculate sum
accuracy = (np.sum(labels_true == labels_pred) / labels_true.size) * 100

# Print the accuracy
# 打印输出 / Print output
print("Accuracy: {0:.2f}%".format(accuracy))
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Matrix

# 08 — Matrix / 08 Matrix

**Chapter 11 — File 4 of 6 / 第11章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Load the digits image and divide it into sub-images**.

本脚本演示 **Load the digits image and divide it into sub-images**。

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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import confusion_matrix
from digits_dataset import split_images, split_data
```

---
## Step 2 — Load the digits image and divide it into sub-images

```python
img, sub_imgs = split_images('Images/digits.png', 20)
```

---
## Step 3 — Create the groundtruth labels

```python
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)
```

---
## Step 4 — Specify the algorithm's termination criteria

```python
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
```

---
## Step 5 — Run the k-means clustering algorithm on the image data

```python
# 转换数据类型 / Convert data type
_, clusters, centers = cv2.kmeans(data=imgs.astype(np.float32), K=10, bestLabels=None,
                                  criteria=criteria, attempts=10,
                                  flags=cv2.KMEANS_RANDOM_CENTERS)
```

---
## Step 6 — Reshape array into 20x20 images

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
imgs_centers = centers.reshape(-1, 20, 20)
```

---
## Step 7 — Cluster labels

```python
# 创建NumPy数组 / Create NumPy array
labels = np.array([2, 0, 7, 5, 1, 4, 6, 9, 3, 8])

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
labels_pred = np.zeros(labels_true.shape, dtype='int')
```

---
## Step 8 — Re-order the cluster labels

```python
# 生成整数序列 / Generate integer sequence
for i in range(10):
    mask = clusters.ravel() == i
    labels_pred[mask] = labels[i]
```

---
## Step 9 — Print confusion matrix

```python
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
print(confusion_matrix(labels_true, labels_pred))
```

---
## Learning Notes / 学习笔记

- **概念**: Load the digits image and divide it into sub-images 是机器学习中的常用技术。  
  *Load the digits image and divide it into sub-images is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `KMeans` | K均值聚类 | K-Means clustering |
| `confusion_matrix` | 混淆矩阵：展示预测对错分布 | Confusion matrix: prediction error distribution |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.zeros` | 全零数组 | Array filled with zeros |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Matrix / 08 Matrix
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import confusion_matrix
from digits_dataset import split_images, split_data

# Load the digits image and divide it into sub-images
img, sub_imgs = split_images('Images/digits.png', 20)

# Create the groundtruth labels
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)

# Specify the algorithm's termination criteria
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the image data
# 转换数据类型 / Convert data type
_, clusters, centers = cv2.kmeans(data=imgs.astype(np.float32), K=10, bestLabels=None,
                                  criteria=criteria, attempts=10,
                                  flags=cv2.KMEANS_RANDOM_CENTERS)

# Reshape array into 20x20 images
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
imgs_centers = centers.reshape(-1, 20, 20)

# Cluster labels
# 创建NumPy数组 / Create NumPy array
labels = np.array([2, 0, 7, 5, 1, 4, 6, 9, 3, 8])

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
labels_pred = np.zeros(labels_true.shape, dtype='int')

# Re-order the cluster labels
# 生成整数序列 / Generate integer sequence
for i in range(10):
    mask = clusters.ravel() == i
    labels_pred[mask] = labels[i]

# Print confusion matrix
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
print(confusion_matrix(labels_true, labels_pred))
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Deskew

# 09 — Deskew / 09 Deskew

**Chapter 11 — File 5 of 6 / 第11章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Calculate the image moments**.

本脚本演示 **Calculate the image moments**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  📈 可视化结果 / Visualize Results
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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import confusion_matrix


def deskew_image(img):
```

---
## Step 2 — Calculate the image moments

```python
img_moments = cv2.moments(img)
```

---
## Step 3 — Moment m02 indicates how much the pixel intensities are spread out along the
vertical axis, mu11 is the central moment or the weight average intensity

```python
if abs(img_moments['mu02']) > 1e-2:
```

---
## Step 4 — Calculate the image skew

```python
img_skew = (img_moments['mu11'] / img_moments['mu02'])
```

---
## Step 5 — Generate the transformation matrix: We are here tweaking slightly the
approximation of vertical translation due to skew by making use of a
scaling factor of 0.6, because we empirically found that this value
worked better for this application

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
m = np.float32([[1, img_skew, -0.6 * img.shape[0] * img_skew], [0, 1, 0]])
```

---
## Step 6 — Apply the transformation matrix to the image

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
img_deskew = cv2.warpAffine(src=img, M=m, dsize=img.shape,
                                    flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    else:
```

---
## Step 7 — If the vertical spread of pixel intensities is small, return a copy of the
original image

```python
img_deskew = img.copy()

    return img_deskew
```

---
## Step 8 — Load the digits image and divide it into sub-images

```python
img, sub_imgs = split_images('Images/digits.png', 20)
```

---
## Step 9 — Create the groundtruth labels

```python
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)
```

---
## Step 10 — De-skew all dataset images

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
imgs_deskewed = np.zeros(imgs.shape)

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for i in range(imgs_deskewed.shape[0]):
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    new = deskew_image(imgs[i, :].reshape(20, 20))
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    imgs_deskewed[i, :] = new.reshape(1, -1)
```

---
## Step 11 — Specify the algorithm's termination criteria

```python
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
```

---
## Step 12 — Run the k-means clustering algorithm on the de-skewed image data

```python
# 转换数据类型 / Convert data type
_, clusters, centers = cv2.kmeans(data=imgs_deskewed.astype(np.float32), K=10,
                                  bestLabels=None, criteria=criteria, attempts=10,
                                  flags=cv2.KMEANS_RANDOM_CENTERS)
```

---
## Step 13 — Reshape array into 20x20 images

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
imgs_centers = centers.reshape(-1, 20, 20)
```

---
## Step 14 — Visualize the cluster centers

```python
fig, ax = plt.subplots(2, 5)

# 将多个序列配对 / Pair multiple sequences
for i, center in zip(ax.flat, imgs_centers):
    i.imshow(center)

# 显示图表 / Display the plot
plt.show()
```

---
## Step 15 — Cluster labels

```python
# 创建NumPy数组 / Create NumPy array
labels = np.array([9, 5, 6, 4, 2, 3, 7, 8, 1, 0])

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
labels_pred = np.zeros(labels_true.shape, dtype='int')
```

---
## Step 16 — Re-order the cluster labels

```python
# 生成整数序列 / Generate integer sequence
for i in range(10):
    mask = clusters.ravel() == i
    labels_pred[mask] = labels[i]
```

---
## Step 17 — Calculate the algorithm's accuracy

```python
# 求和 / Calculate sum
accuracy = (np.sum(labels_true == labels_pred) / labels_true.size) * 100
```

---
## Step 18 — Print the accuracy

```python
# 打印输出 / Print output
print("Accuracy: {0:.2f}%".format(accuracy))
```

---
## Step 19 — Print confusion matrix

```python
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
print(confusion_matrix(labels_true, labels_pred))
```

---
## Learning Notes / 学习笔记

- **概念**: Calculate the image moments 是机器学习中的常用技术。  
  *Calculate the image moments is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `KMeans` | K均值聚类 | K-Means clustering |
| `confusion_matrix` | 混淆矩阵：展示预测对错分布 | Confusion matrix: prediction error distribution |
| `matplotlib` | 绑图库 | Plotting library |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.zeros` | 全零数组 | Array filled with zeros |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Deskew / 09 Deskew
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from digits_dataset import split_images, split_data
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import confusion_matrix


def deskew_image(img):
    # Calculate the image moments
    img_moments = cv2.moments(img)

    # Moment m02 indicates how much the pixel intensities are spread out along the
    # vertical axis, mu11 is the central moment or the weight average intensity
    if abs(img_moments['mu02']) > 1e-2:
        # Calculate the image skew
        img_skew = (img_moments['mu11'] / img_moments['mu02'])

        # Generate the transformation matrix: We are here tweaking slightly the
        # approximation of vertical translation due to skew by making use of a
        # scaling factor of 0.6, because we empirically found that this value
        # worked better for this application
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        m = np.float32([[1, img_skew, -0.6 * img.shape[0] * img_skew], [0, 1, 0]])

        # Apply the transformation matrix to the image
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        img_deskew = cv2.warpAffine(src=img, M=m, dsize=img.shape,
                                    flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    else:
        # If the vertical spread of pixel intensities is small, return a copy of the
        # original image
        img_deskew = img.copy()

    return img_deskew

# Load the digits image and divide it into sub-images
img, sub_imgs = split_images('Images/digits.png', 20)

# Create the groundtruth labels
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)

# De-skew all dataset images
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
imgs_deskewed = np.zeros(imgs.shape)

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for i in range(imgs_deskewed.shape[0]):
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    new = deskew_image(imgs[i, :].reshape(20, 20))
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    imgs_deskewed[i, :] = new.reshape(1, -1)

# Specify the algorithm's termination criteria
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the de-skewed image data
# 转换数据类型 / Convert data type
_, clusters, centers = cv2.kmeans(data=imgs_deskewed.astype(np.float32), K=10,
                                  bestLabels=None, criteria=criteria, attempts=10,
                                  flags=cv2.KMEANS_RANDOM_CENTERS)

# Reshape array into 20x20 images
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
imgs_centers = centers.reshape(-1, 20, 20)

# Visualize the cluster centers
fig, ax = plt.subplots(2, 5)

# 将多个序列配对 / Pair multiple sequences
for i, center in zip(ax.flat, imgs_centers):
    i.imshow(center)

# 显示图表 / Display the plot
plt.show()

# Cluster labels
# 创建NumPy数组 / Create NumPy array
labels = np.array([9, 5, 6, 4, 2, 3, 7, 8, 1, 0])

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
labels_pred = np.zeros(labels_true.shape, dtype='int')

# Re-order the cluster labels
# 生成整数序列 / Generate integer sequence
for i in range(10):
    mask = clusters.ravel() == i
    labels_pred[mask] = labels[i]

# Calculate the algorithm's accuracy
# 求和 / Calculate sum
accuracy = (np.sum(labels_true == labels_pred) / labels_true.size) * 100

# Print the accuracy
# 打印输出 / Print output
print("Accuracy: {0:.2f}%".format(accuracy))

# Print confusion matrix
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
print(confusion_matrix(labels_true, labels_pred))
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Chapter Summary / 章节总结

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **6 code files** demonstrating chapter 11.

本章包含 **6 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `02_check_shape.ipynb` — Check Shape
  2. `04_show_cluster.ipynb` — Show Cluster
  3. `07_classification.ipynb` — Classification
  4. `08_matrix.ipynb` — Matrix
  5. `09_deskew.ipynb` — Deskew
  6. `digits_dataset.ipynb` — Digits Dataset

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---

### Digits Dataset

# 01 — Digits Dataset / Digits Dataset

**Chapter 11 — File 6 of 6 / 第11章 — 第6个文件（共6个）**

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
