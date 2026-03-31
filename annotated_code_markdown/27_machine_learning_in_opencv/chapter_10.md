# OpenCV 机器学习 / Machine Learning in OpenCV
## Chapter 10

---

### Clusters



---

### Kmeans

# 03 — Kmeans / 03 Kmeans

**Chapter 10 — File 2 of 3 / 第10章 — 第2个文件（共3个）**

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
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
```

---
## Step 2 — Generate a dataset of 2D data points and their groundtruth labels

```python
x, y_true = make_blobs(n_samples=100, centers=5, cluster_std=1.5, random_state=10)
```

---
## Step 3 — Plot the dataset

```python
# 绘制散点图 / Draw scatter plot
plt.scatter(x[:, 0], x[:, 1])
# 显示图表 / Display the plot
plt.show()
```

---
## Step 4 — Specify the algorithm's termination criteria

```python
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
```

---
## Step 5 — Run the k-means clustering algorithm on the input data

```python
# 转换数据类型 / Convert data type
compactness, y_pred, centers = cv2.kmeans(data=x.astype(np.float32), K=5, bestLabels=None,
                                          criteria=criteria, attempts=10,
                                          flags=cv2.KMEANS_RANDOM_CENTERS)
```

---
## Step 6 — Plot the data clusters, each having a different color, together with the
corresponding cluster centers

```python
# 绘制散点图 / Draw scatter plot
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
# 绘制散点图 / Draw scatter plot
plt.scatter(centers[:, 0], centers[:, 1], c='red')
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
| `KMeans` | K均值聚类 | K-Means clustering |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Kmeans / 03 Kmeans
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

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=5, cluster_std=1.5, random_state=10)

# Plot the dataset
# 绘制散点图 / Draw scatter plot
plt.scatter(x[:, 0], x[:, 1])
# 显示图表 / Display the plot
plt.show()

# Specify the algorithm's termination criteria
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the input data
# 转换数据类型 / Convert data type
compactness, y_pred, centers = cv2.kmeans(data=x.astype(np.float32), K=5, bestLabels=None,
                                          criteria=criteria, attempts=10,
                                          flags=cv2.KMEANS_RANDOM_CENTERS)

# Plot the data clusters, each having a different color, together with the
# corresponding cluster centers
# 绘制散点图 / Draw scatter plot
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
# 绘制散点图 / Draw scatter plot
plt.scatter(centers[:, 0], centers[:, 1], c='red')
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Quantize



---

### Chapter Summary / 章节总结

# Chapter 10 Summary / 第10章总结

## Theme / 主题: Chapter 10 / Chapter 10

This chapter contains **3 code files** demonstrating chapter 10.

本章包含 **3 个代码文件**，演示Chapter 10。

---
## Evolution / 演化路线

  1. `01_clusters.ipynb` — Clusters
  2. `03_kmeans.ipynb` — Kmeans
  3. `07_quantize.ipynb` — Quantize

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 10) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 10）是机器学习流水线中的基础构建块。

---
