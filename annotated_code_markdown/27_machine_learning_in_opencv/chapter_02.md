# OpenCV 机器学习 / Machine Learning in OpenCV
## Chapter 02

---

### Imread

# 03 — Imread / 03 Imread

**Chapter 02 — File 1 of 10 / 第02章 — 第1个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Imread**.

本脚本演示 **03 Imread**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

img = cv2.imread('Images/Dog.jpg')
# 打印输出 / Print output
print('Datatype:', img.dtype)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Dimensions:', img.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: Imread 是机器学习中的常用技术。  
  *Imread is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Imread / 03 Imread
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

img = cv2.imread('Images/Dog.jpg')
# 打印输出 / Print output
print('Datatype:', img.dtype)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Dimensions:', img.shape)
```

---

➡️ **Next / 下一步**: File 2 of 10

---

### Pixel

# 04 — Pixel / 04 Pixel

**Chapter 02 — File 2 of 10 / 第02章 — 第2个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Pixel**.

本脚本演示 **04 Pixel**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

img = cv2.imread('Images/Dog.jpg')
# 打印输出 / Print output
print(img[0, 0])
```

---
## Learning Notes / 学习笔记

- **概念**: Pixel 是机器学习中的常用技术。  
  *Pixel is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pixel / 04 Pixel
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

img = cv2.imread('Images/Dog.jpg')
# 打印输出 / Print output
print(img[0, 0])
```

---

➡️ **Next / 下一步**: File 3 of 10

---

### Matplotlib Show

# 05 — Matplotlib Show / 05 Matplotlib Show

**Chapter 02 — File 3 of 10 / 第02章 — 第3个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Matplotlib Show**.

本脚本演示 **05 Matplotlib Show**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

img = cv2.imread('Images/Dog.jpg')
# 显示图像 / Display image
plt.imshow(img)
# 设置图表标题 / Set chart title
plt.title('Displaying image using Matplotlib')
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Matplotlib Show 是机器学习中的常用技术。  
  *Matplotlib Show is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Matplotlib Show / 05 Matplotlib Show
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

img = cv2.imread('Images/Dog.jpg')
# 显示图像 / Display image
plt.imshow(img)
# 设置图表标题 / Set chart title
plt.title('Displaying image using Matplotlib')
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 4 of 10

---

### Imshow



---

### Write



---

### Show Rgb

# 09 — Show Rgb / 09 Show Rgb

**Chapter 02 — File 6 of 10 / 第02章 — 第6个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Show Rgb**.

本脚本演示 **09 Show Rgb**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

img = cv2.imread('Images/Dog.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 显示图像 / Display image
plt.imshow(img_rgb)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Show Rgb 是机器学习中的常用技术。  
  *Show Rgb is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Show Rgb / 09 Show Rgb
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

img = cv2.imread('Images/Dog.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 显示图像 / Display image
plt.imshow(img_rgb)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 7 of 10

---

### Rgb Pixel

# 10 — Rgb Pixel / 10 Rgb Pixel

**Chapter 02 — File 7 of 10 / 第02章 — 第7个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Rgb Pixel**.

本脚本演示 **10 Rgb Pixel**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

img = cv2.imread('Images/Dog.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 打印输出 / Print output
print(img_rgb[0, 0])
```

---
## Learning Notes / 学习笔记

- **概念**: Rgb Pixel 是机器学习中的常用技术。  
  *Rgb Pixel is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Rgb Pixel / 10 Rgb Pixel
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

img = cv2.imread('Images/Dog.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 打印输出 / Print output
print(img_rgb[0, 0])
```

---

➡️ **Next / 下一步**: File 8 of 10

---

### Grayscale

# 11 — Grayscale / 数据缩放

**Chapter 02 — File 8 of 10 / 第02章 — 第8个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Grayscale**.

本脚本演示 **数据缩放**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

img = cv2.imread('Images/Dog.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

cv2.imshow('Grayscale Image', img_gray)
cv2.waitKey(0)
```

---
## Learning Notes / 学习笔记

- **概念**: Grayscale 是机器学习中的常用技术。  
  *Grayscale is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grayscale / 数据缩放
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

img = cv2.imread('Images/Dog.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

cv2.imshow('Grayscale Image', img_gray)
cv2.waitKey(0)
```

---

➡️ **Next / 下一步**: File 9 of 10

---

### Show Gray Pixel

# 12 — Show Gray Pixel / 12 Show Gray Pixel

**Chapter 02 — File 9 of 10 / 第02章 — 第9个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Show Gray Pixel**.

本脚本演示 **12 Show Gray Pixel**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

img = cv2.imread('Images/Dog.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# 打印输出 / Print output
print(img_gray[0, 0])
```

---
## Learning Notes / 学习笔记

- **概念**: Show Gray Pixel 是机器学习中的常用技术。  
  *Show Gray Pixel is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Show Gray Pixel / 12 Show Gray Pixel
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

img = cv2.imread('Images/Dog.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# 打印输出 / Print output
print(img_gray[0, 0])
```

---

➡️ **Next / 下一步**: File 10 of 10

---

### Grayscale

# 14 — Grayscale / 数据缩放

**Chapter 02 — File 10 of 10 / 第02章 — 第10个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Grayscale**.

本脚本演示 **数据缩放**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

img_gray = cv2.imread('Images/Dog.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Grayscale Image', img_gray)
cv2.waitKey(0)
```

---
## Learning Notes / 学习笔记

- **概念**: Grayscale 是机器学习中的常用技术。  
  *Grayscale is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grayscale / 数据缩放
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

img_gray = cv2.imread('Images/Dog.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Grayscale Image', img_gray)
cv2.waitKey(0)
```

---

### Chapter Summary / 章节总结

# Chapter 02 Summary / 第02章总结

## Theme / 主题: Chapter 02 / Chapter 02

This chapter contains **10 code files** demonstrating chapter 02.

本章包含 **10 个代码文件**，演示Chapter 02。

---
## Evolution / 演化路线

  1. `03_imread.ipynb` — Imread
  2. `04_pixel.ipynb` — Pixel
  3. `05_matplotlib_show.ipynb` — Matplotlib Show
  4. `06_imshow.ipynb` — Imshow
  5. `07_write.ipynb` — Write
  6. `09_show_rgb.ipynb` — Show Rgb
  7. `10_rgb_pixel.ipynb` — Rgb Pixel
  8. `11_grayscale.ipynb` — Grayscale
  9. `12_show_gray_pixel.ipynb` — Show Gray Pixel
  10. `14_grayscale.ipynb` — Grayscale

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 02) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 02）是机器学习流水线中的基础构建块。

---
