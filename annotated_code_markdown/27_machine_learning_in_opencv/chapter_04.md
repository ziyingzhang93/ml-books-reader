# OpenCV 机器学习 / Machine Learning in OpenCV
## Chapter 04

---

### Show

# 01 — Show / 01 Show

**Chapter 04 — File 1 of 3 / 第04章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Show**.

本脚本演示 **01 Show**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

imgfile = "Hooded_mountain_tanager_(Buthraupis_montana_cucullata)_Caldas.jpg"

img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
cv2.imshow("bird", img)
cv2.waitKey(0)
```

---
## Learning Notes / 学习笔记

- **概念**: Show 是机器学习中的常用技术。  
  *Show is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Show / 01 Show
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

imgfile = "Hooded_mountain_tanager_(Buthraupis_montana_cucullata)_Caldas.jpg"

img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
cv2.imshow("bird", img)
cv2.waitKey(0)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Kenburns



---

### Check Fourcc

# 12 — Check Fourcc / 12 Check Fourcc

**Chapter 04 — File 3 of 3 / 第04章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Check Fourcc**.

本脚本演示 **12 Check Fourcc**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

try:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter('temp.mkv', fourcc, 30, (640, 480))
    assert writer.isOpened()
    # 打印输出 / Print output
    print("Supported")
except:
    # 打印输出 / Print output
    print("Not supported")
```

---
## Learning Notes / 学习笔记

- **概念**: Check Fourcc 是机器学习中的常用技术。  
  *Check Fourcc is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Check Fourcc / 12 Check Fourcc
# Complete Code / 完整代码
# ===============================

# 导入OpenCV计算机视觉库 / Import OpenCV computer vision library
import cv2

try:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter('temp.mkv', fourcc, 30, (640, 480))
    assert writer.isOpened()
    # 打印输出 / Print output
    print("Supported")
except:
    # 打印输出 / Print output
    print("Not supported")
```

---

### Chapter Summary / 章节总结

# Chapter 04 Summary / 第04章总结

## Theme / 主题: Chapter 04 / Chapter 04

This chapter contains **3 code files** demonstrating chapter 04.

本章包含 **3 个代码文件**，演示Chapter 04。

---
## Evolution / 演化路线

  1. `01_show.ipynb` — Show
  2. `09_kenburns.ipynb` — Kenburns
  3. `12_check_fourcc.ipynb` — Check Fourcc

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 04) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 04）是机器学习流水线中的基础构建块。

---
