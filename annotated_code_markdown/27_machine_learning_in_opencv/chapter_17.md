# OpenCV 机器学习 / Machine Learning in OpenCV
## Chapter 17

---

### Hog Hierarchy

# 01 — Hog Hierarchy / 01 Hog Hierarchy

**Chapter 17 — File 1 of 2 / 第17章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Load the image and convert to grayscale**.

本脚本演示 **Load the image and convert to grayscale**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
```

---
## Step 2 — Load the image and convert to grayscale

```python
img = cv2.imread('people.jpg')  # 1920x1280 pixels
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---
## Step 3 — define each block as 4x4 cells of 128x128 pixels each

```python
cell_size = (128, 128)      # h x w in pixels
block_size = (4, 4)         # h x w in cells
win_size = (8, 6)           # h x w in cells

nbins = 9  # number of orientation bins
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
img_size = img.shape[:2]  # h x w in pixels
```

---
## Step 4 — create a HOG object

```python
hog = cv2.HOGDescriptor(
    _winSize=(win_size[1] * cell_size[1],
              win_size[0] * cell_size[0]),
    _blockSize=(block_size[1] * cell_size[1],
                block_size[0] * cell_size[0]),
    _blockStride=(cell_size[1], cell_size[0]),
    _cellSize=(cell_size[1], cell_size[0]),
    _nbins=nbins
)
n_cells = (img_size[0] // cell_size[0], img_size[1] // cell_size[1])
```

---
## Step 5 — find features as a 1xN vector, then reshape into spatial hierarchy

```python
hog_feats = hog.compute(img)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
hog_feats = hog_feats.reshape(
    n_cells[1] - win_size[1] + 1,
    n_cells[0] - win_size[0] + 1,
    win_size[1] - block_size[1] + 1,
    win_size[0] - block_size[0] + 1,
    block_size[1],
    block_size[0],
    nbins)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(hog_feats.shape)  # (10, 3, 3, 5, 4, 4, 9)
```

---
## Learning Notes / 学习笔记

- **概念**: Load the image and convert to grayscale 是机器学习中的常用技术。  
  *Load the image and convert to grayscale is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hog Hierarchy / 01 Hog Hierarchy
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

# Load the image and convert to grayscale
img = cv2.imread('people.jpg')  # 1920x1280 pixels
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# define each block as 4x4 cells of 128x128 pixels each
cell_size = (128, 128)      # h x w in pixels
block_size = (4, 4)         # h x w in cells
win_size = (8, 6)           # h x w in cells

nbins = 9  # number of orientation bins
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
img_size = img.shape[:2]  # h x w in pixels

# create a HOG object
hog = cv2.HOGDescriptor(
    _winSize=(win_size[1] * cell_size[1],
              win_size[0] * cell_size[0]),
    _blockSize=(block_size[1] * cell_size[1],
                block_size[0] * cell_size[0]),
    _blockStride=(cell_size[1], cell_size[0]),
    _cellSize=(cell_size[1], cell_size[0]),
    _nbins=nbins
)
n_cells = (img_size[0] // cell_size[0], img_size[1] // cell_size[1])

# find features as a 1xN vector, then reshape into spatial hierarchy
hog_feats = hog.compute(img)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
hog_feats = hog_feats.reshape(
    n_cells[1] - win_size[1] + 1,
    n_cells[0] - win_size[0] + 1,
    win_size[1] - block_size[1] + 1,
    win_size[0] - block_size[0] + 1,
    block_size[1],
    block_size[0],
    nbins)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(hog_feats.shape)  # (10, 3, 3, 5, 4, 4, 9)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### People Detector

# 03 — People Detector / 03 People Detector

**Chapter 17 — File 2 of 2 / 第17章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Load the image and convert it to grayscale**.

本脚本演示 **Load the image and convert it to grayscale**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
```

---
## Step 2 — Load the image and convert it to grayscale

```python
img = cv2.imread('people.jpg')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
```

---
## Step 3 — Detect people in the image

```python
locations, confidence = hog.detectMultiScale(img)
```

---
## Step 4 — Draw rectangles around the detected people

```python
for (x, y, w, h) in locations:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
```

---
## Step 5 — Display the image with detected people

```python
cv2.imshow('People', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---
## Learning Notes / 学习笔记

- **概念**: Load the image and convert it to grayscale 是机器学习中的常用技术。  
  *Load the image and convert it to grayscale is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `SVM` | 支持向量机 | Support Vector Machine |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# People Detector / 03 People Detector
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

# Load the image and convert it to grayscale
img = cv2.imread('people.jpg')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Detect people in the image
locations, confidence = hog.detectMultiScale(img)

# Draw rectangles around the detected people
for (x, y, w, h) in locations:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)

# Display the image with detected people
cv2.imshow('People', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Chapter Summary / 章节总结



---
