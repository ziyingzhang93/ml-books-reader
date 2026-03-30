# OpenCV ML
## Chapter 08

---

### Sift Surf

# 01 — Sift Surf / 01 Sift Surf

**Chapter 08 — File 1 of 3 / 第08章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Load the image and convery to grayscale**.

本脚本演示 **Load the image and convery to grayscale**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import cv2
```

---
## Step 2 — Load the image and convery to grayscale

```python
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---
## Step 3 — Initialize SIFT and SURF detectors

```python
sift = cv2.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
```

---
## Step 4 — Detect key points and compute descriptors

```python
keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)
keypoints_surf, descriptors_surf = surf.detectAndCompute(img, None)
```

---
## Learning Notes / 学习笔记

- **概念**: Load the image and convery to grayscale 是机器学习中的常用技术。  
  *Load the image and convery to grayscale is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sift Surf / 01 Sift Surf
# Complete Code / 完整代码
# ===============================

import cv2

# Load the image and convery to grayscale
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize SIFT and SURF detectors
sift = cv2.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

# Detect key points and compute descriptors
keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)
keypoints_surf, descriptors_surf = surf.detectAndCompute(img, None)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Chapter Summary

# Chapter 08 Summary / 第08章总结

## Theme / 主题: Chapter 08 / Chapter 08

This chapter contains **3 code files** demonstrating chapter 08.

本章包含 **3 个代码文件**，演示Chapter 08。

---
## Evolution / 演化路线

  1. `01_sift_surf.ipynb` — Sift Surf
  2. `02_sift_keypoints.ipynb` — Sift Keypoints
  3. `03_orb_keypoints.ipynb` — Orb Keypoints

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 08) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 08）是机器学习流水线中的基础构建块。

---
