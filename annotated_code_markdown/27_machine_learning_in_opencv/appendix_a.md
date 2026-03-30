# OpenCV ML
## Appendix A

---

### Save

# 03 — Save / 保存/加载模型

**Chapter appendix_a — File 1 of 2 / 第appendix_a章 — 第1个文件（共2个）**

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
import cv2
from sklearn.datasets import make_blobs
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
svm.train(x_train.astype('float32'), cv2.ml.ROW_SAMPLE, y_train)
```

---
## Step 7 — Save the trained model

```python
svm.save("rbf_svm.dat")
```

---
## Step 8 — Predict the target labels of the testing data

```python
_, y_pred = svm.predict(x_test.astype('float32'))
```

---
## Step 9 — Compute and print the achieved accuracy

```python
accuracy = (sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
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
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Save / 保存/加载模型
# Complete Code / 完整代码
# ===============================

import cv2
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=8, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Create a new SVM
svm = cv2.ml.SVM_create()

# Set the SVM kernel to RBF
svm.setKernel(cv2.ml.SVM_RBF)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(10)
svm.setGamma(0.1)

# Train the SVM on the set of training data
svm.train(x_train.astype('float32'), cv2.ml.ROW_SAMPLE, y_train)

# Save the trained model
svm.save("rbf_svm.dat")

# Predict the target labels of the testing data
_, y_pred = svm.predict(x_test.astype('float32'))

# Compute and print the achieved accuracy
accuracy = (sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, '%')
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Chapter Summary

# Chapter appendix_a Summary / 第appendix_a章总结

## Theme / 主题: Appendix / 附录

This chapter contains **2 code files** demonstrating appendix.

本章包含 **2 个代码文件**，演示附录。

---
## Evolution / 演化路线

  1. `03_save.ipynb` — Save
  2. `04_load.ipynb` — Load

---
## ML Relevance / ML 关联

The techniques in this chapter (Appendix) are fundamental building blocks in machine learning pipelines.

本章技术（附录）是机器学习流水线中的基础构建块。

---
