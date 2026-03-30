# OpenCV 机器学习 / Machine Learning in OpenCV
## Chapter 20

---

### Haar Detect

# 04 — Haar Detect / 04 Haar Detect

**Chapter 20 — File 1 of 1 / 第20章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Photo https://unsplash.com/photos/people-walking-on-sidewalk-during-daytime-GBkAx9qUeus**.

本脚本演示 **Photo https://unsplash.com/photos/people-walking-on-sidewalk-during-daytime-GBkAx9qUeus**。

---
## Step 1 — Step 1

```python
import os
import cv2
```

---
## Step 2 — Photo https://unsplash.com/photos/people-walking-on-sidewalk-during-daytime-GBkAx9qUeus

```python
filename = 'people2.jpg'
```

---
## Step 3 — Load the Haar cascade for face detection

```python
filepath = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(filepath)
```

---
## Step 4 — Read the input image

```python
img = cv2.imread(filename)
```

---
## Step 5 — Convert the image to grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---
## Step 6 — Perform face detection

```python
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4,
                                      minSize=(20, 20))
```

---
## Step 7 — Draw rectangles around the detected faces

```python
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)
```

---
## Step 8 — Display the result

```python
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---
## Learning Notes / 学习笔记

- **概念**: Photo https://unsplash.com/photos/people-walking-on-sidewalk-during-daytime-GBkAx9qUeus 是机器学习中的常用技术。  
  *Photo https://unsplash.com/photos/people-walking-on-sidewalk-during-daytime-GBkAx9qUeus is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Haar Detect / 04 Haar Detect
# Complete Code / 完整代码
# ===============================

import os
import cv2

# Photo https://unsplash.com/photos/people-walking-on-sidewalk-during-daytime-GBkAx9qUeus
filename = 'people2.jpg'

# Load the Haar cascade for face detection
filepath = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(filepath)

# Read the input image
img = cv2.imread(filename)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4,
                                      minSize=(20, 20))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)

# Display the result
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Chapter Summary / 章节总结

# Chapter 20 Summary / 第20章总结

## Theme / 主题: Chapter 20 / Chapter 20

This chapter contains **1 code files** demonstrating chapter 20.

本章包含 **1 个代码文件**，演示Chapter 20。

---
## Evolution / 演化路线

  1. `04_haar_detect.ipynb` — Haar Detect

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 20) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 20）是机器学习流水线中的基础构建块。

---
