# CV深度学习
## Chapter 10

---

### Add Channels

# 01 — Add Channels / 01 Add Channels

**Chapter 10 — File 1 of 4 / 第10章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of expanding dimensions**.

本脚本演示 **example of expanding dimensions**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of expanding dimensions

```python
from numpy import expand_dims
from numpy import asarray
from PIL import Image
```

---
## Step 2 — load the image

```python
img = Image.open('penguin_parade.jpg')
```

---
## Step 3 — convert the image to grayscale

```python
img = img.convert(mode='L')
```

---
## Step 4 — convert to numpy array

```python
data = asarray(img)
print(data.shape)
```

---
## Step 5 — add channels first

```python
data_first = expand_dims(data, axis=0)
print(data_first.shape)
```

---
## Step 6 — add channels last

```python
data_last = expand_dims(data, axis=2)
print(data_last.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: example of expanding dimensions 是机器学习中的常用技术。  
  *example of expanding dimensions is a common technique in machine learning.*

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
# Add Channels / 01 Add Channels
# Complete Code / 完整代码
# ===============================

# example of expanding dimensions
from numpy import expand_dims
from numpy import asarray
from PIL import Image
# load the image
img = Image.open('penguin_parade.jpg')
# convert the image to grayscale
img = img.convert(mode='L')
# convert to numpy array
data = asarray(img)
print(data.shape)
# add channels first
data_first = expand_dims(data, axis=0)
print(data_first.shape)
# add channels last
data_last = expand_dims(data, axis=2)
print(data_last.shape)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Change Channel Ordering

# 02 — Change Channel Ordering / 02 Change Channel Ordering

**Chapter 10 — File 2 of 4 / 第10章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **change image from channels last to channels first format**.

本脚本演示 **change image from channels last to channels first format**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — change image from channels last to channels first format

```python
from numpy import moveaxis
from numpy import asarray
from PIL import Image
```

---
## Step 2 — load the color image

```python
img = Image.open('penguin_parade.jpg')
```

---
## Step 3 — convert to numpy array

```python
data = asarray(img)
print(data.shape)
```

---
## Step 4 — change channels last to channels first format

```python
data = moveaxis(data, 2, 0)
print(data.shape)
```

---
## Step 5 — change channels first to channels last format

```python
data = moveaxis(data, 0, 2)
print(data.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: change image from channels last to channels first format 是机器学习中的常用技术。  
  *change image from channels last to channels first format is a common technique in machine learning.*

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
# Change Channel Ordering / 02 Change Channel Ordering
# Complete Code / 完整代码
# ===============================

# change image from channels last to channels first format
from numpy import moveaxis
from numpy import asarray
from PIL import Image
# load the color image
img = Image.open('penguin_parade.jpg')
# convert to numpy array
data = asarray(img)
print(data.shape)
# change channels last to channels first format
data = moveaxis(data, 2, 0)
print(data.shape)
# change channels first to channels last format
data = moveaxis(data, 0, 2)
print(data.shape)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Show Channel Ordering

# 03 — Show Channel Ordering / 03 Show Channel Ordering

**Chapter 10 — File 3 of 4 / 第10章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **show preferred channel order**.

本脚本演示 **show preferred channel order**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — show preferred channel order

```python
from keras import backend
print(backend.image_data_format())
```

---
## Learning Notes / 学习笔记

- **概念**: show preferred channel order 是机器学习中的常用技术。  
  *show preferred channel order is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Show Channel Ordering / 03 Show Channel Ordering
# Complete Code / 完整代码
# ===============================

# show preferred channel order
from keras import backend
print(backend.image_data_format())
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Force Channel Ordering

# 04 — Force Channel Ordering / 04 Force Channel Ordering

**Chapter 10 — File 4 of 4 / 第10章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **force a channel ordering**.

本脚本演示 **force a channel ordering**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — force a channel ordering

```python
from keras import backend
```

---
## Step 2 — force channels-first ordering

```python
backend.set_image_data_format('channels_first')
print(backend.image_data_format())
```

---
## Step 3 — force channels-last ordering

```python
backend.set_image_data_format('channels_last')
print(backend.image_data_format())
```

---
## Learning Notes / 学习笔记

- **概念**: force a channel ordering 是机器学习中的常用技术。  
  *force a channel ordering is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Force Channel Ordering / 04 Force Channel Ordering
# Complete Code / 完整代码
# ===============================

# force a channel ordering
from keras import backend
# force channels-first ordering
backend.set_image_data_format('channels_first')
print(backend.image_data_format())
# force channels-last ordering
backend.set_image_data_format('channels_last')
print(backend.image_data_format())
```

---

### Chapter Summary

# Chapter 10 Summary / 第10章总结

## Theme / 主题: Chapter 10 / Chapter 10

This chapter contains **4 code files** demonstrating chapter 10.

本章包含 **4 个代码文件**，演示Chapter 10。

---
## Evolution / 演化路线

  1. `01_add_channels.ipynb` — Add Channels
  2. `02_change_channel_ordering.ipynb` — Change Channel Ordering
  3. `03_show_channel_ordering.ipynb` — Show Channel Ordering
  4. `04_force_channel_ordering.ipynb` — Force Channel Ordering

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 10) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 10）是机器学习流水线中的基础构建块。

---
