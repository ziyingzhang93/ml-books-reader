# CV深度学习
## Chapter 05

---

### Load Show

# 01 — Load Show / 01 Load Show

**Chapter 05 — File 1 of 7 / 第05章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **load and show an image with Pillow**.

本脚本演示 **load and show an image with Pillow**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — load and show an image with Pillow

```python
from PIL import Image
```

---
## Step 2 — load the image

```python
image = Image.open('sydney_bridge.jpg')
```

---
## Step 3 — summarize some details about the image

```python
print(image.format)
print(image.mode)
print(image.size)
```

---
## Step 4 — show the image

```python
image.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load and show an image with Pillow 是机器学习中的常用技术。  
  *load and show an image with Pillow is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Show / 01 Load Show
# Complete Code / 完整代码
# ===============================

# load and show an image with Pillow
from PIL import Image
# load the image
image = Image.open('sydney_bridge.jpg')
# summarize some details about the image
print(image.format)
print(image.mode)
print(image.size)
# show the image
image.show()
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Normalize

# 02 — Normalize / 02 Normalize

**Chapter 05 — File 2 of 7 / 第05章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of pixel normalization**.

本脚本演示 **example of pixel normalization**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — example of pixel normalization

```python
from numpy import asarray
from PIL import Image
```

---
## Step 2 — load image

```python
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
```

---
## Step 3 — confirm pixel range is 0-255

```python
print('Data Type: %s' % pixels.dtype)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
```

---
## Step 4 — convert from integers to floats

```python
pixels = pixels.astype('float32')
```

---
## Step 5 — normalize to the range 0-1

```python
pixels /= 255.0
```

---
## Step 6 — confirm the normalization

```python
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
```

---
## Learning Notes / 学习笔记

- **概念**: example of pixel normalization 是机器学习中的常用技术。  
  *example of pixel normalization is a common technique in machine learning.*

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
# Normalize / 02 Normalize
# Complete Code / 完整代码
# ===============================

# example of pixel normalization
from numpy import asarray
from PIL import Image
# load image
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
# confirm pixel range is 0-255
print('Data Type: %s' % pixels.dtype)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
# convert from integers to floats
pixels = pixels.astype('float32')
# normalize to the range 0-1
pixels /= 255.0
# confirm the normalization
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Global Center

# 03 — Global Center / 03 Global Center

**Chapter 05 — File 3 of 7 / 第05章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of global centering (subtract mean)**.

本脚本演示 **example of global centering (subtract mean)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of global centering (subtract mean)

```python
from numpy import asarray
from PIL import Image
```

---
## Step 2 — load image

```python
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
```

---
## Step 3 — convert from integers to floats

```python
pixels = pixels.astype('float32')
```

---
## Step 4 — calculate global mean

```python
mean = pixels.mean()
print('Mean: %.3f' % mean)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
```

---
## Step 5 — global centering of pixels

```python
pixels = pixels - mean
```

---
## Step 6 — confirm it had the desired effect

```python
mean = pixels.mean()
print('Mean: %.3f' % mean)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
```

---
## Learning Notes / 学习笔记

- **概念**: example of global centering (subtract mean) 是机器学习中的常用技术。  
  *example of global centering (subtract mean) is a common technique in machine learning.*

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
# Global Center / 03 Global Center
# Complete Code / 完整代码
# ===============================

# example of global centering (subtract mean)
from numpy import asarray
from PIL import Image
# load image
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
# convert from integers to floats
pixels = pixels.astype('float32')
# calculate global mean
mean = pixels.mean()
print('Mean: %.3f' % mean)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
# global centering of pixels
pixels = pixels - mean
# confirm it had the desired effect
mean = pixels.mean()
print('Mean: %.3f' % mean)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Local Center

# 04 — Local Center / 04 Local Center

**Chapter 05 — File 4 of 7 / 第05章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of per-channel centering (subtract mean)**.

本脚本演示 **example of per-channel centering (subtract mean)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of per-channel centering (subtract mean)

```python
from numpy import asarray
from PIL import Image
```

---
## Step 2 — load image

```python
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
```

---
## Step 3 — convert from integers to floats

```python
pixels = pixels.astype('float32')
```

---
## Step 4 — calculate per-channel means and standard deviations

```python
means = pixels.mean(axis=(0,1), dtype='float64')
print('Means: %s' % means)
print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))
```

---
## Step 5 — per-channel centering of pixels

```python
pixels -= means
```

---
## Step 6 — confirm it had the desired effect

```python
means = pixels.mean(axis=(0,1), dtype='float64')
print('Means: %s' % means)
print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))
```

---
## Learning Notes / 学习笔记

- **概念**: example of per-channel centering (subtract mean) 是机器学习中的常用技术。  
  *example of per-channel centering (subtract mean) is a common technique in machine learning.*

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
# Local Center / 04 Local Center
# Complete Code / 完整代码
# ===============================

# example of per-channel centering (subtract mean)
from numpy import asarray
from PIL import Image
# load image
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
# convert from integers to floats
pixels = pixels.astype('float32')
# calculate per-channel means and standard deviations
means = pixels.mean(axis=(0,1), dtype='float64')
print('Means: %s' % means)
print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))
# per-channel centering of pixels
pixels -= means
# confirm it had the desired effect
means = pixels.mean(axis=(0,1), dtype='float64')
print('Means: %s' % means)
print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0,1)), pixels.max(axis=(0,1))))
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Global Standardization

# 05 — Global Standardization / 05 Global Standardization

**Chapter 05 — File 5 of 7 / 第05章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of global pixel standardization**.

本脚本演示 **example of global pixel standardization**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of global pixel standardization

```python
from numpy import asarray
from PIL import Image
```

---
## Step 2 — load image

```python
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
```

---
## Step 3 — convert from integers to floats

```python
pixels = pixels.astype('float32')
```

---
## Step 4 — calculate global mean and standard deviation

```python
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
```

---
## Step 5 — global standardization of pixels

```python
pixels = (pixels - mean) / std
```

---
## Step 6 — confirm it had the desired effect

```python
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
```

---
## Learning Notes / 学习笔记

- **概念**: example of global pixel standardization 是机器学习中的常用技术。  
  *example of global pixel standardization is a common technique in machine learning.*

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
# Global Standardization / 05 Global Standardization
# Complete Code / 完整代码
# ===============================

# example of global pixel standardization
from numpy import asarray
from PIL import Image
# load image
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
# convert from integers to floats
pixels = pixels.astype('float32')
# calculate global mean and standard deviation
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
# global standardization of pixels
pixels = (pixels - mean) / std
# confirm it had the desired effect
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Positive Global Standardization

# 06 — Positive Global Standardization / 06 Positive Global Standardization

**Chapter 05 — File 6 of 7 / 第05章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of global pixel standardization shifted to positive domain**.

本脚本演示 **example of global pixel standardization shifted to positive domain**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of global pixel standardization shifted to positive domain

```python
from numpy import asarray
from numpy import clip
from PIL import Image
```

---
## Step 2 — load image

```python
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
```

---
## Step 3 — convert from integers to floats

```python
pixels = pixels.astype('float32')
```

---
## Step 4 — calculate global mean and standard deviation

```python
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
```

---
## Step 5 — global standardization of pixels

```python
pixels = (pixels - mean) / std
```

---
## Step 6 — clip pixel values to [-1,1]

```python
pixels = clip(pixels, -1.0, 1.0)
```

---
## Step 7 — shift from [-1,1] to [0,1] with 0.5 mean

```python
pixels = (pixels + 1.0) / 2.0
```

---
## Step 8 — confirm it had the desired effect

```python
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
```

---
## Learning Notes / 学习笔记

- **概念**: example of global pixel standardization shifted to positive domain 是机器学习中的常用技术。  
  *example of global pixel standardization shifted to positive domain is a common technique in machine learning.*

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
# Positive Global Standardization / 06 Positive Global Standardization
# Complete Code / 完整代码
# ===============================

# example of global pixel standardization shifted to positive domain
from numpy import asarray
from numpy import clip
from PIL import Image
# load image
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
# convert from integers to floats
pixels = pixels.astype('float32')
# calculate global mean and standard deviation
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
# global standardization of pixels
pixels = (pixels - mean) / std
# clip pixel values to [-1,1]
pixels = clip(pixels, -1.0, 1.0)
# shift from [-1,1] to [0,1] with 0.5 mean
pixels = (pixels + 1.0) / 2.0
# confirm it had the desired effect
mean, std = pixels.mean(), pixels.std()
print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Local Standardization

# 07 — Local Standardization / 07 Local Standardization

**Chapter 05 — File 7 of 7 / 第05章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of per-channel pixel standardization**.

本脚本演示 **example of per-channel pixel standardization**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of per-channel pixel standardization

```python
from numpy import asarray
from PIL import Image
```

---
## Step 2 — load image

```python
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
```

---
## Step 3 — convert from integers to floats

```python
pixels = pixels.astype('float32')
```

---
## Step 4 — calculate per-channel means and standard deviations

```python
means = pixels.mean(axis=(0,1), dtype='float64')
stds = pixels.std(axis=(0,1), dtype='float64')
print('Means: %s, Stds: %s' % (means, stds))
```

---
## Step 5 — per-channel standardization of pixels

```python
pixels = (pixels - means) / stds
```

---
## Step 6 — confirm it had the desired effect

```python
means = pixels.mean(axis=(0,1), dtype='float64')
stds = pixels.std(axis=(0,1), dtype='float64')
print('Means: %s, Stds: %s' % (means, stds))
```

---
## Learning Notes / 学习笔记

- **概念**: example of per-channel pixel standardization 是机器学习中的常用技术。  
  *example of per-channel pixel standardization is a common technique in machine learning.*

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
# Local Standardization / 07 Local Standardization
# Complete Code / 完整代码
# ===============================

# example of per-channel pixel standardization
from numpy import asarray
from PIL import Image
# load image
image = Image.open('sydney_bridge.jpg')
pixels = asarray(image)
# convert from integers to floats
pixels = pixels.astype('float32')
# calculate per-channel means and standard deviations
means = pixels.mean(axis=(0,1), dtype='float64')
stds = pixels.std(axis=(0,1), dtype='float64')
print('Means: %s, Stds: %s' % (means, stds))
# per-channel standardization of pixels
pixels = (pixels - means) / stds
# confirm it had the desired effect
means = pixels.mean(axis=(0,1), dtype='float64')
stds = pixels.std(axis=(0,1), dtype='float64')
print('Means: %s, Stds: %s' % (means, stds))
```

---

### Chapter Summary

# Chapter 05 Summary / 第05章总结

## Theme / 主题: Chapter 05 / Chapter 05

This chapter contains **7 code files** demonstrating chapter 05.

本章包含 **7 个代码文件**，演示Chapter 05。

---
## Evolution / 演化路线

  1. `01_load_show.ipynb` — Load Show
  2. `02_normalize.ipynb` — Normalize
  3. `03_global_center.ipynb` — Global Center
  4. `04_local_center.ipynb` — Local Center
  5. `05_global_standardization.ipynb` — Global Standardization
  6. `06_positive_global_standardization.ipynb` — Positive Global Standardization
  7. `07_local_standardization.ipynb` — Local Standardization

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 05) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 05）是机器学习流水线中的基础构建块。

---
