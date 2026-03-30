# CV深度学习
## Chapter 04

---

### Version

# 01 — Version / 库版本信息

**Chapter 04 — File 1 of 12 / 第04章 — 第1个文件（共12个）**

---

## Summary / 总结

This script demonstrates **check Pillow version number**.

本脚本演示 **check Pillow version number**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — check Pillow version number

```python
import PIL
print('Pillow Version:', PIL.__version__)
```

---
## Learning Notes / 学习笔记

- **概念**: check Pillow version number 是机器学习中的常用技术。  
  *check Pillow version number is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Version / 库版本信息
# Complete Code / 完整代码
# ===============================

# check Pillow version number
import PIL
print('Pillow Version:', PIL.__version__)
```

---

➡️ **Next / 下一步**: File 2 of 12

---

### Load

# 02 — Load / 02 Load

**Chapter 04 — File 2 of 12 / 第04章 — 第2个文件（共12个）**

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
image = Image.open('opera_house.jpg')
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
# Load / 02 Load
# Complete Code / 完整代码
# ===============================

# load and show an image with Pillow
from PIL import Image
# load the image
image = Image.open('opera_house.jpg')
# summarize some details about the image
print(image.format)
print(image.mode)
print(image.size)
# show the image
image.show()
```

---

➡️ **Next / 下一步**: File 3 of 12

---

### Convert Image

# 03 — Convert Image / 图像处理

**Chapter 04 — File 3 of 12 / 第04章 — 第3个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load and display an image with Matplotlib**.

本脚本演示 **load and display an image with Matplotlib**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — load and display an image with Matplotlib

```python
from matplotlib import image
from matplotlib import pyplot
```

---
## Step 2 — load image as pixel array

```python
data = image.imread('opera_house.jpg')
```

---
## Step 3 — summarize shape of the pixel array

```python
print(data.dtype)
print(data.shape)
```

---
## Step 4 — display the array of pixels as an image

```python
pyplot.imshow(data)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load and display an image with Matplotlib 是机器学习中的常用技术。  
  *load and display an image with Matplotlib is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Convert Image / 图像处理
# Complete Code / 完整代码
# ===============================

# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
# load image as pixel array
data = image.imread('opera_house.jpg')
# summarize shape of the pixel array
print(data.dtype)
print(data.shape)
# display the array of pixels as an image
pyplot.imshow(data)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 12

---

### Convert Image Alt

# 04 — Convert Image Alt / 图像处理

**Chapter 04 — File 4 of 12 / 第04章 — 第4个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load image and convert to and from NumPy array**.

本脚本演示 **load image and convert to and from NumPy array**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — load image and convert to and from NumPy array

```python
from PIL import Image
from numpy import asarray
```

---
## Step 2 — load the image

```python
image = Image.open('opera_house.jpg')
```

---
## Step 3 — convert image to numpy array

```python
data = asarray(image)
```

---
## Step 4 — summarize shape

```python
print(data.shape)
```

---
## Step 5 — create Pillow image

```python
image2 = Image.fromarray(data)
```

---
## Step 6 — summarize image details

```python
print(image2.format)
print(image2.mode)
print(image2.size)
```

---
## Learning Notes / 学习笔记

- **概念**: load image and convert to and from NumPy array 是机器学习中的常用技术。  
  *load image and convert to and from NumPy array is a common technique in machine learning.*

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
# Convert Image Alt / 图像处理
# Complete Code / 完整代码
# ===============================

# load image and convert to and from NumPy array
from PIL import Image
from numpy import asarray
# load the image
image = Image.open('opera_house.jpg')
# convert image to numpy array
data = asarray(image)
# summarize shape
print(data.shape)
# create Pillow image
image2 = Image.fromarray(data)
# summarize image details
print(image2.format)
print(image2.mode)
print(image2.size)
```

---

➡️ **Next / 下一步**: File 5 of 12

---

### Load From Dir

# 05 — Load From Dir / 05 Load From Dir

**Chapter 04 — File 5 of 12 / 第04章 — 第5个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load all images in a directory**.

本脚本演示 **load all images in a directory**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — load all images in a directory

```python
from os import listdir
from matplotlib import image
```

---
## Step 2 — load all images in a directory

```python
loaded_images = list()
for filename in listdir('images'):
```

---
## Step 3 — load image

```python
img_data = image.imread('images/' + filename)
```

---
## Step 4 — store loaded image

```python
loaded_images.append(img_data)
	print('> loaded %s %s' % (filename, img_data.shape))
```

---
## Learning Notes / 学习笔记

- **概念**: load all images in a directory 是机器学习中的常用技术。  
  *load all images in a directory is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load From Dir / 05 Load From Dir
# Complete Code / 完整代码
# ===============================

# load all images in a directory
from os import listdir
from matplotlib import image
# load all images in a directory
loaded_images = list()
for filename in listdir('images'):
	# load image
	img_data = image.imread('images/' + filename)
	# store loaded image
	loaded_images.append(img_data)
	print('> loaded %s %s' % (filename, img_data.shape))
```

---

➡️ **Next / 下一步**: File 6 of 12

---

### Save Png

# 06 — Save Png / 保存/加载模型

**Chapter 04 — File 6 of 12 / 第04章 — 第6个文件（共12个）**

---

## Summary / 总结

This script demonstrates **example of saving an image in another format**.

本脚本演示 **example of saving an image in another format**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of saving an image in another format

```python
from PIL import Image
```

---
## Step 2 — load the image

```python
image = Image.open('opera_house.jpg')
```

---
## Step 3 — save as PNG format

```python
image.save('opera_house.png', format='PNG')
```

---
## Step 4 — load the image again and inspect the format

```python
image2 = Image.open('opera_house.png')
print(image2.format)
```

---
## Learning Notes / 学习笔记

- **概念**: example of saving an image in another format 是机器学习中的常用技术。  
  *example of saving an image in another format is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Save Png / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# example of saving an image in another format
from PIL import Image
# load the image
image = Image.open('opera_house.jpg')
# save as PNG format
image.save('opera_house.png', format='PNG')
# load the image again and inspect the format
image2 = Image.open('opera_house.png')
print(image2.format)
```

---

➡️ **Next / 下一步**: File 7 of 12

---

### Save Grayscale

# 07 — Save Grayscale / 保存/加载模型

**Chapter 04 — File 7 of 12 / 第04章 — 第7个文件（共12个）**

---

## Summary / 总结

This script demonstrates **example of saving a grayscale version of a loaded image**.

本脚本演示 **example of saving a grayscale version of a loaded image**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of saving a grayscale version of a loaded image

```python
from PIL import Image
```

---
## Step 2 — load the image

```python
image = Image.open('opera_house.jpg')
```

---
## Step 3 — convert the image to grayscale

```python
gs_image = image.convert(mode='L')
```

---
## Step 4 — save in jpeg format

```python
gs_image.save('opera_house_grayscale.jpg')
```

---
## Step 5 — load the image again and show it

```python
image2 = Image.open('opera_house_grayscale.jpg')
```

---
## Step 6 — show the image

```python
image2.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of saving a grayscale version of a loaded image 是机器学习中的常用技术。  
  *example of saving a grayscale version of a loaded image is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Save Grayscale / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# example of saving a grayscale version of a loaded image
from PIL import Image
# load the image
image = Image.open('opera_house.jpg')
# convert the image to grayscale
gs_image = image.convert(mode='L')
# save in jpeg format
gs_image.save('opera_house_grayscale.jpg')
# load the image again and show it
image2 = Image.open('opera_house_grayscale.jpg')
# show the image
image2.show()
```

---

➡️ **Next / 下一步**: File 8 of 12

---

### Resize Image

# 08 — Resize Image / 图像处理

**Chapter 04 — File 8 of 12 / 第04章 — 第8个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create a thumbnail of an image**.

本脚本演示 **create a thumbnail of an image**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — create a thumbnail of an image

```python
from PIL import Image
```

---
## Step 2 — load the image

```python
image = Image.open('opera_house.jpg')
```

---
## Step 3 — report the size of the image

```python
print(image.size)
```

---
## Step 4 — create a thumbnail and preserve aspect ratio

```python
image.thumbnail((100,100))
```

---
## Step 5 — report the size of the modified image

```python
print(image.size)
```

---
## Step 6 — show the image

```python
image.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create a thumbnail of an image 是机器学习中的常用技术。  
  *create a thumbnail of an image is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Resize Image / 图像处理
# Complete Code / 完整代码
# ===============================

# create a thumbnail of an image
from PIL import Image
# load the image
image = Image.open('opera_house.jpg')
# report the size of the image
print(image.size)
# create a thumbnail and preserve aspect ratio
image.thumbnail((100,100))
# report the size of the modified image
print(image.size)
# show the image
image.show()
```

---

➡️ **Next / 下一步**: File 9 of 12

---

### Resize Aspect Ratio

# 09 — Resize Aspect Ratio / 09 Resize Aspect Ratio

**Chapter 04 — File 9 of 12 / 第04章 — 第9个文件（共12个）**

---

## Summary / 总结

This script demonstrates **resize image and force a new shape**.

本脚本演示 **resize image and force a new shape**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — resize image and force a new shape

```python
from PIL import Image
```

---
## Step 2 — load the image

```python
image = Image.open('opera_house.jpg')
```

---
## Step 3 — report the size of the image

```python
print(image.size)
```

---
## Step 4 — resize image and ignore original aspect ratio

```python
img_resized = image.resize((200,200))
```

---
## Step 5 — report the size of the thumbnail

```python
print(img_resized.size)
```

---
## Step 6 — show the image

```python
img_resized.show()
```

---
## Learning Notes / 学习笔记

- **概念**: resize image and force a new shape 是机器学习中的常用技术。  
  *resize image and force a new shape is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Resize Aspect Ratio / 09 Resize Aspect Ratio
# Complete Code / 完整代码
# ===============================

# resize image and force a new shape
from PIL import Image
# load the image
image = Image.open('opera_house.jpg')
# report the size of the image
print(image.size)
# resize image and ignore original aspect ratio
img_resized = image.resize((200,200))
# report the size of the thumbnail
print(img_resized.size)
# show the image
img_resized.show()
```

---

➡️ **Next / 下一步**: File 10 of 12

---

### Flip Image

# 10 — Flip Image / 图像处理

**Chapter 04 — File 10 of 12 / 第04章 — 第10个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create flipped versions of an image**.

本脚本演示 **create flipped versions of an image**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — create flipped versions of an image

```python
from PIL import Image
from matplotlib import pyplot
```

---
## Step 2 — load image

```python
image = Image.open('opera_house.jpg')
```

---
## Step 3 — horizontal flip

```python
hoz_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
```

---
## Step 4 — vertical flip

```python
ver_flip = image.transpose(Image.FLIP_TOP_BOTTOM)
```

---
## Step 5 — plot all three images using matplotlib

```python
pyplot.subplot(311)
pyplot.imshow(image)
pyplot.subplot(312)
pyplot.imshow(hoz_flip)
pyplot.subplot(313)
pyplot.imshow(ver_flip)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create flipped versions of an image 是机器学习中的常用技术。  
  *create flipped versions of an image is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Flip Image / 图像处理
# Complete Code / 完整代码
# ===============================

# create flipped versions of an image
from PIL import Image
from matplotlib import pyplot
# load image
image = Image.open('opera_house.jpg')
# horizontal flip
hoz_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
# vertical flip
ver_flip = image.transpose(Image.FLIP_TOP_BOTTOM)
# plot all three images using matplotlib
pyplot.subplot(311)
pyplot.imshow(image)
pyplot.subplot(312)
pyplot.imshow(hoz_flip)
pyplot.subplot(313)
pyplot.imshow(ver_flip)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 11 of 12

---

### Rotated Image

# 11 — Rotated Image / 图像处理

**Chapter 04 — File 11 of 12 / 第04章 — 第11个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create rotated versions of an image**.

本脚本演示 **create rotated versions of an image**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — create rotated versions of an image

```python
from PIL import Image
from matplotlib import pyplot
```

---
## Step 2 — load image

```python
image = Image.open('opera_house.jpg')
```

---
## Step 3 — plot original image

```python
pyplot.subplot(311)
pyplot.imshow(image)
```

---
## Step 4 — rotate 45 degrees

```python
pyplot.subplot(312)
pyplot.imshow(image.rotate(45))
```

---
## Step 5 — rotate 90 degrees

```python
pyplot.subplot(313)
pyplot.imshow(image.rotate(90))
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create rotated versions of an image 是机器学习中的常用技术。  
  *create rotated versions of an image is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Rotated Image / 图像处理
# Complete Code / 完整代码
# ===============================

# create rotated versions of an image
from PIL import Image
from matplotlib import pyplot
# load image
image = Image.open('opera_house.jpg')
# plot original image
pyplot.subplot(311)
pyplot.imshow(image)
# rotate 45 degrees
pyplot.subplot(312)
pyplot.imshow(image.rotate(45))
# rotate 90 degrees
pyplot.subplot(313)
pyplot.imshow(image.rotate(90))
pyplot.show()
```

---

➡️ **Next / 下一步**: File 12 of 12

---

### Cropped Image

# 12 — Cropped Image / 图像处理

**Chapter 04 — File 12 of 12 / 第04章 — 第12个文件（共12个）**

---

## Summary / 总结

This script demonstrates **example of cropping an image**.

本脚本演示 **example of cropping an image**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of cropping an image

```python
from PIL import Image
```

---
## Step 2 — load image

```python
image = Image.open('opera_house.jpg')
```

---
## Step 3 — create a cropped image

```python
cropped = image.crop((100, 100, 200, 200))
```

---
## Step 4 — show cropped image

```python
cropped.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of cropping an image 是机器学习中的常用技术。  
  *example of cropping an image is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cropped Image / 图像处理
# Complete Code / 完整代码
# ===============================

# example of cropping an image
from PIL import Image
# load image
image = Image.open('opera_house.jpg')
# create a cropped image
cropped = image.crop((100, 100, 200, 200))
# show cropped image
cropped.show()
```

---

### Chapter Summary

# Chapter 04 Summary / 第04章总结

## Theme / 主题: Chapter 04 / Chapter 04

This chapter contains **12 code files** demonstrating chapter 04.

本章包含 **12 个代码文件**，演示Chapter 04。

---
## Evolution / 演化路线

  1. `01_version.ipynb` — Version
  2. `02_load.ipynb` — Load
  3. `03_convert_image.ipynb` — Convert Image
  4. `04_convert_image_alt.ipynb` — Convert Image Alt
  5. `05_load_from_dir.ipynb` — Load From Dir
  6. `06_save_png.ipynb` — Save Png
  7. `07_save_grayscale.ipynb` — Save Grayscale
  8. `08_resize_image.ipynb` — Resize Image
  9. `09_resize_aspect_ratio.ipynb` — Resize Aspect Ratio
  10. `10_flip_image.ipynb` — Flip Image
  11. `11_rotated_image.ipynb` — Rotated Image
  12. `12_cropped_image.ipynb` — Cropped Image

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 04) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 04）是机器学习流水线中的基础构建块。

---
