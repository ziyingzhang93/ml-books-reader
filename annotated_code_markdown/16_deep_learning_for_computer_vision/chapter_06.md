# CV深度学习
## Chapter 06

---

### Load Image

# 01 — Load Image / 图像处理

**Chapter 06 — File 1 of 3 / 第06章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **example of loading an image with the Keras API**.

本脚本演示 **example of loading an image with the Keras API**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of loading an image with the Keras API

```python
from keras.preprocessing.image import load_img
```

---
## Step 2 — load the image

```python
img = load_img('bondi_beach.jpg')
```

---
## Step 3 — report details about the image

```python
print(type(img))
print(img.format)
print(img.mode)
print(img.size)
```

---
## Step 4 — show the image

```python
img.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of loading an image with the Keras API 是机器学习中的常用技术。  
  *example of loading an image with the Keras API is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Image / 图像处理
# Complete Code / 完整代码
# ===============================

# example of loading an image with the Keras API
from keras.preprocessing.image import load_img
# load the image
img = load_img('bondi_beach.jpg')
# report details about the image
print(type(img))
print(img.format)
print(img.mode)
print(img.size)
# show the image
img.show()
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Convert Image

# 02 — Convert Image / 图像处理

**Chapter 06 — File 2 of 3 / 第06章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **example of converting an image with the Keras API**.

本脚本演示 **example of converting an image with the Keras API**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of converting an image with the Keras API

```python
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
```

---
## Step 2 — load the image

```python
img = load_img('bondi_beach.jpg')
print(type(img))
```

---
## Step 3 — convert to numpy array

```python
img_array = img_to_array(img)
print(img_array.dtype)
print(img_array.shape)
```

---
## Step 4 — convert back to image

```python
img_pil = array_to_img(img_array)
print(type(img))
```

---
## Learning Notes / 学习笔记

- **概念**: example of converting an image with the Keras API 是机器学习中的常用技术。  
  *example of converting an image with the Keras API is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Convert Image / 图像处理
# Complete Code / 完整代码
# ===============================

# example of converting an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
# load the image
img = load_img('bondi_beach.jpg')
print(type(img))
# convert to numpy array
img_array = img_to_array(img)
print(img_array.dtype)
print(img_array.shape)
# convert back to image
img_pil = array_to_img(img_array)
print(type(img))
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Save Image

# 03 — Save Image / 图像处理

**Chapter 06 — File 3 of 3 / 第06章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **example of saving an image with the Keras API**.

本脚本演示 **example of saving an image with the Keras API**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of saving an image with the Keras API

```python
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
```

---
## Step 2 — load image as as grayscale

```python
img = load_img('bondi_beach.jpg', color_mode='grayscale')
```

---
## Step 3 — convert image to a numpy array

```python
img_array = img_to_array(img)
```

---
## Step 4 — save the image with a new filename

```python
save_img('bondi_beach_grayscale.jpg', img_array)
```

---
## Step 5 — load the image to confirm it was saved correctly

```python
img = load_img('bondi_beach_grayscale.jpg')
print(type(img))
print(img.format)
print(img.mode)
print(img.size)
img.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of saving an image with the Keras API 是机器学习中的常用技术。  
  *example of saving an image with the Keras API is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Save Image / 图像处理
# Complete Code / 完整代码
# ===============================

# example of saving an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
# load image as as grayscale
img = load_img('bondi_beach.jpg', color_mode='grayscale')
# convert image to a numpy array
img_array = img_to_array(img)
# save the image with a new filename
save_img('bondi_beach_grayscale.jpg', img_array)
# load the image to confirm it was saved correctly
img = load_img('bondi_beach_grayscale.jpg')
print(type(img))
print(img.format)
print(img.mode)
print(img.size)
img.show()
```

---

### Chapter Summary

# Chapter 06 Summary / 第06章总结

## Theme / 主题: Chapter 06 / Chapter 06

This chapter contains **3 code files** demonstrating chapter 06.

本章包含 **3 个代码文件**，演示Chapter 06。

---
## Evolution / 演化路线

  1. `01_load_image.ipynb` — Load Image
  2. `02_convert_image.ipynb` — Convert Image
  3. `03_save_image.ipynb` — Save Image

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 06) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 06）是机器学习流水线中的基础构建块。

---
