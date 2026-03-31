# NLP 深度学习 / Deep Learning for NLP
## Chapter 23

---

### Classify Image

# 1 — Classify Image / 图像处理

**Chapter 23 — File 1 of 1 / 第23章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **load the model**.

本脚本演示 **load the model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.applications.vgg16 import preprocess_input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.applications.vgg16 import decode_predictions
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.applications.vgg16 import VGG16
```

---
## Step 2 — load the model

```python
model = VGG16()
```

---
## Step 3 — load an image from file

```python
image = load_img('mug.jpg', target_size=(224, 224))
```

---
## Step 4 — convert the image pixels to a numpy array

```python
image = img_to_array(image)
```

---
## Step 5 — reshape data for the model

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
```

---
## Step 6 — prepare the image for the VGG model

```python
image = preprocess_input(image)
```

---
## Step 7 — predict the probability across all output classes

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(image)
```

---
## Step 8 — convert the probabilities to class labels

```python
label = decode_predictions(yhat)
```

---
## Step 9 — retrieve the most likely result, e.g. highest probability

```python
label = label[0][0]
```

---
## Step 10 — print the classification

```python
# 打印输出 / Print output
print('%s (%.2f%%)' % (label[1], label[2]*100))
```

---
## Learning Notes / 学习笔记

- **概念**: load the model 是机器学习中的常用技术。  
  *load the model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Classify Image / 图像处理
# Complete Code / 完整代码
# ===============================

# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.applications.vgg16 import preprocess_input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.applications.vgg16 import decode_predictions
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.applications.vgg16 import VGG16
# load the model
model = VGG16()
# load an image from file
image = load_img('mug.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
# 用模型做预测 / Make predictions with model
yhat = model.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
# 打印输出 / Print output
print('%s (%.2f%%)' % (label[1], label[2]*100))
```

---

### Chapter Summary / 章节总结

# Chapter 23 Summary / 第23章总结

## Theme / 主题: Chapter 23 / Chapter 23

This chapter contains **1 code files** demonstrating chapter 23.

本章包含 **1 个代码文件**，演示Chapter 23。

---
## Evolution / 演化路线

  1. `1_classify_image.ipynb` — Classify Image

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 23) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 23）是机器学习流水线中的基础构建块。

---
