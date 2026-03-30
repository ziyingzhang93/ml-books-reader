# CV深度学习
## Chapter 28

---

### Confirm Opencv

# 01 — Confirm Opencv / 01 Confirm Opencv

**Chapter 28 — File 1 of 10 / 第28章 — 第1个文件（共10个）**

---

## Summary / 总结

This script demonstrates **check opencv version**.

本脚本演示 **check opencv version**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — check opencv version

```python
import cv2
```

---
## Step 2 — print version number

```python
print(cv2.__version__)
```

---
## Learning Notes / 学习笔记

- **概念**: check opencv version 是机器学习中的常用技术。  
  *check opencv version is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Confirm Opencv / 01 Confirm Opencv
# Complete Code / 完整代码
# ===============================

# check opencv version
import cv2
# print version number
print(cv2.__version__)
```

---

➡️ **Next / 下一步**: File 2 of 10

---

### Opencv Face Detector

# 02 — Opencv Face Detector / 人脸识别

**Chapter 28 — File 2 of 10 / 第28章 — 第2个文件（共10个）**

---

## Summary / 总结

This script demonstrates **example of face detection with opencv cascade classifier**.

本脚本演示 **example of face detection with opencv cascade classifier**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of face detection with opencv cascade classifier

```python
from cv2 import imread
from cv2 import CascadeClassifier
```

---
## Step 2 — load the photograph

```python
pixels = imread('test1.jpg')
```

---
## Step 3 — load the pre-trained model

```python
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
```

---
## Step 4 — perform face detection

```python
bboxes = classifier.detectMultiScale(pixels)
```

---
## Step 5 — print bounding box for each detected face

```python
for box in bboxes:
	print(box)
```

---
## Learning Notes / 学习笔记

- **概念**: example of face detection with opencv cascade classifier 是机器学习中的常用技术。  
  *example of face detection with opencv cascade classifier is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Opencv Face Detector / 人脸识别
# Complete Code / 完整代码
# ===============================

# example of face detection with opencv cascade classifier
from cv2 import imread
from cv2 import CascadeClassifier
# load the photograph
pixels = imread('test1.jpg')
# load the pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
# perform face detection
bboxes = classifier.detectMultiScale(pixels)
# print bounding box for each detected face
for box in bboxes:
	print(box)
```

---

➡️ **Next / 下一步**: File 3 of 10

---

### Opencv Face Detector Plot1

# 03 — Opencv Face Detector Plot1 / 人脸识别

**Chapter 28 — File 3 of 10 / 第28章 — 第3个文件（共10个）**

---

## Summary / 总结

This script demonstrates **plot photo with detected faces using opencv cascade classifier**.

本脚本演示 **plot photo with detected faces using opencv cascade classifier**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — plot photo with detected faces using opencv cascade classifier

```python
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
```

---
## Step 2 — load the photograph

```python
pixels = imread('test1.jpg')
```

---
## Step 3 — load the pre-trained model

```python
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
```

---
## Step 4 — perform face detection

```python
bboxes = classifier.detectMultiScale(pixels)
```

---
## Step 5 — print bounding box for each detected face

```python
for box in bboxes:
```

---
## Step 6 — extract

```python
x, y, width, height = box
	x2, y2 = x + width, y + height
```

---
## Step 7 — draw a rectangle over the pixels

```python
rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
```

---
## Step 8 — show the image

```python
imshow('face detection', pixels)
```

---
## Step 9 — keep the window open until we press a key

```python
waitKey(0)
```

---
## Step 10 — close the window

```python
destroyAllWindows()
```

---
## Learning Notes / 学习笔记

- **概念**: plot photo with detected faces using opencv cascade classifier 是机器学习中的常用技术。  
  *plot photo with detected faces using opencv cascade classifier is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Opencv Face Detector Plot1 / 人脸识别
# Complete Code / 完整代码
# ===============================

# plot photo with detected faces using opencv cascade classifier
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
# load the photograph
pixels = imread('test1.jpg')
# load the pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
# perform face detection
bboxes = classifier.detectMultiScale(pixels)
# print bounding box for each detected face
for box in bboxes:
	# extract
	x, y, width, height = box
	x2, y2 = x + width, y + height
	# draw a rectangle over the pixels
	rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
# show the image
imshow('face detection', pixels)
# keep the window open until we press a key
waitKey(0)
# close the window
destroyAllWindows()
```

---

➡️ **Next / 下一步**: File 4 of 10

---

### Opencv Face Detector Plot2

# 04 — Opencv Face Detector Plot2 / 人脸识别

**Chapter 28 — File 4 of 10 / 第28章 — 第4个文件（共10个）**

---

## Summary / 总结

This script demonstrates **plot photo with detected faces using opencv cascade classifier**.

本脚本演示 **plot photo with detected faces using opencv cascade classifier**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — plot photo with detected faces using opencv cascade classifier

```python
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
```

---
## Step 2 — load the photograph

```python
pixels = imread('test2.jpg')
```

---
## Step 3 — load the pre-trained model

```python
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
```

---
## Step 4 — perform face detection

```python
bboxes = classifier.detectMultiScale(pixels)
```

---
## Step 5 — print bounding box for each detected face

```python
for box in bboxes:
```

---
## Step 6 — extract

```python
x, y, width, height = box
	x2, y2 = x + width, y + height
```

---
## Step 7 — draw a rectangle over the pixels

```python
rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
```

---
## Step 8 — show the image

```python
imshow('face detection', pixels)
```

---
## Step 9 — keep the window open until we press a key

```python
waitKey(0)
```

---
## Step 10 — close the window

```python
destroyAllWindows()
```

---
## Learning Notes / 学习笔记

- **概念**: plot photo with detected faces using opencv cascade classifier 是机器学习中的常用技术。  
  *plot photo with detected faces using opencv cascade classifier is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Opencv Face Detector Plot2 / 人脸识别
# Complete Code / 完整代码
# ===============================

# plot photo with detected faces using opencv cascade classifier
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
# load the photograph
pixels = imread('test2.jpg')
# load the pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
# perform face detection
bboxes = classifier.detectMultiScale(pixels)
# print bounding box for each detected face
for box in bboxes:
	# extract
	x, y, width, height = box
	x2, y2 = x + width, y + height
	# draw a rectangle over the pixels
	rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
# show the image
imshow('face detection', pixels)
# keep the window open until we press a key
waitKey(0)
# close the window
destroyAllWindows()
```

---

➡️ **Next / 下一步**: File 5 of 10

---

### Check Mtcnn Version

# 05 — Check Mtcnn Version / 库版本信息

**Chapter 28 — File 5 of 10 / 第28章 — 第5个文件（共10个）**

---

## Summary / 总结

This script demonstrates **confirm mtcnn was installed correctly**.

本脚本演示 **confirm mtcnn was installed correctly**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — confirm mtcnn was installed correctly

```python
import mtcnn
```

---
## Step 2 — print version

```python
print(mtcnn.__version__)
```

---
## Learning Notes / 学习笔记

- **概念**: confirm mtcnn was installed correctly 是机器学习中的常用技术。  
  *confirm mtcnn was installed correctly is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Check Mtcnn Version / 库版本信息
# Complete Code / 完整代码
# ===============================

# confirm mtcnn was installed correctly
import mtcnn
# print version
print(mtcnn.__version__)
```

---

➡️ **Next / 下一步**: File 6 of 10

---

### Mtcnn Face Detection

# 06 — Mtcnn Face Detection / 卷积神经网络

**Chapter 28 — File 6 of 10 / 第28章 — 第6个文件（共10个）**

---

## Summary / 总结

This script demonstrates **face detection with mtcnn on a photograph**.

本脚本演示 **face detection with mtcnn on a photograph**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — face detection with mtcnn on a photograph

```python
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
```

---
## Step 2 — load image from file

```python
filename = 'test1.jpg'
pixels = pyplot.imread(filename)
```

---
## Step 3 — create the detector, using default weights

```python
detector = MTCNN()
```

---
## Step 4 — detect faces in the image

```python
faces = detector.detect_faces(pixels)
for face in faces:
	print(face)
```

---
## Learning Notes / 学习笔记

- **概念**: face detection with mtcnn on a photograph 是机器学习中的常用技术。  
  *face detection with mtcnn on a photograph is a common technique in machine learning.*

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
# Mtcnn Face Detection / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# face detection with mtcnn on a photograph
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
# load image from file
filename = 'test1.jpg'
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
for face in faces:
	print(face)
```

---

➡️ **Next / 下一步**: File 7 of 10

---

### Mtcnn Face Detection Plot

# 07 — Mtcnn Face Detection Plot / 卷积神经网络

**Chapter 28 — File 7 of 10 / 第28章 — 第7个文件（共10个）**

---

## Summary / 总结

This script demonstrates **face detection with mtcnn on a photograph**.

本脚本演示 **face detection with mtcnn on a photograph**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — face detection with mtcnn on a photograph

```python
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
```

---
## Step 2 — draw an image with detected objects

```python
def draw_image_with_boxes(filename, result_list):
```

---
## Step 3 — load the image

```python
data = pyplot.imread(filename)
```

---
## Step 4 — plot the image

```python
pyplot.imshow(data)
```

---
## Step 5 — get the context for drawing boxes

```python
ax = pyplot.gca()
```

---
## Step 6 — plot each box

```python
for result in result_list:
```

---
## Step 7 — get coordinates

```python
x, y, width, height = result['box']
```

---
## Step 8 — create the shape

```python
rect = Rectangle((x, y), width, height, fill=False, color='red')
```

---
## Step 9 — draw the box

```python
ax.add_patch(rect)
```

---
## Step 10 — show the plot

```python
pyplot.show()

filename = 'test1.jpg'
```

---
## Step 11 — load image from file

```python
pixels = pyplot.imread(filename)
```

---
## Step 12 — create the detector, using default weights

```python
detector = MTCNN()
```

---
## Step 13 — detect faces in the image

```python
faces = detector.detect_faces(pixels)
```

---
## Step 14 — display faces on the original image

```python
draw_image_with_boxes(filename, faces)
```

---
## Learning Notes / 学习笔记

- **概念**: face detection with mtcnn on a photograph 是机器学习中的常用技术。  
  *face detection with mtcnn on a photograph is a common technique in machine learning.*

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
# Mtcnn Face Detection Plot / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN

# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
	# show the plot
	pyplot.show()

filename = 'test1.jpg'
# load image from file
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
# display faces on the original image
draw_image_with_boxes(filename, faces)
```

---

➡️ **Next / 下一步**: File 8 of 10

---

### Mtcnn Extract Faces

# 10 — Mtcnn Extract Faces / 卷积神经网络

**Chapter 28 — File 10 of 10 / 第28章 — 第10个文件（共10个）**

---

## Summary / 总结

This script demonstrates **extract and plot each detected face in a photograph**.

本脚本演示 **extract and plot each detected face in a photograph**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — extract and plot each detected face in a photograph

```python
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
```

---
## Step 2 — draw each face separately

```python
def draw_faces(filename, result_list):
```

---
## Step 3 — load the image

```python
data = pyplot.imread(filename)
```

---
## Step 4 — plot each face as a subplot

```python
for i in range(len(result_list)):
```

---
## Step 5 — get coordinates

```python
x1, y1, width, height = result_list[i]['box']
		x2, y2 = x1 + width, y1 + height
```

---
## Step 6 — define subplot

```python
pyplot.subplot(1, len(result_list), i+1)
		pyplot.axis('off')
```

---
## Step 7 — plot face

```python
pyplot.imshow(data[y1:y2, x1:x2])
```

---
## Step 8 — show the plot

```python
pyplot.show()

filename = 'test2.jpg'
```

---
## Step 9 — load image from file

```python
pixels = pyplot.imread(filename)
```

---
## Step 10 — create the detector, using default weights

```python
detector = MTCNN()
```

---
## Step 11 — detect faces in the image

```python
faces = detector.detect_faces(pixels)
```

---
## Step 12 — display faces on the original image

```python
draw_faces(filename, faces)
```

---
## Learning Notes / 学习笔记

- **概念**: extract and plot each detected face in a photograph 是机器学习中的常用技术。  
  *extract and plot each detected face in a photograph is a common technique in machine learning.*

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
# Mtcnn Extract Faces / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# extract and plot each detected face in a photograph
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

# draw each face separately
def draw_faces(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot each face as a subplot
	for i in range(len(result_list)):
		# get coordinates
		x1, y1, width, height = result_list[i]['box']
		x2, y2 = x1 + width, y1 + height
		# define subplot
		pyplot.subplot(1, len(result_list), i+1)
		pyplot.axis('off')
		# plot face
		pyplot.imshow(data[y1:y2, x1:x2])
	# show the plot
	pyplot.show()

filename = 'test2.jpg'
# load image from file
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
# display faces on the original image
draw_faces(filename, faces)
```

---

### Chapter Summary

# Chapter 28 Summary / 第28章总结

## Theme / 主题: Chapter 28 / Chapter 28

This chapter contains **10 code files** demonstrating chapter 28.

本章包含 **10 个代码文件**，演示Chapter 28。

---
## Evolution / 演化路线

  1. `01_confirm_opencv.ipynb` — Confirm Opencv
  2. `02_opencv_face_detector.ipynb` — Opencv Face Detector
  3. `03_opencv_face_detector_plot1.ipynb` — Opencv Face Detector Plot1
  4. `04_opencv_face_detector_plot2.ipynb` — Opencv Face Detector Plot2
  5. `05_check_mtcnn_version.ipynb` — Check Mtcnn Version
  6. `06_mtcnn_face_detection.ipynb` — Mtcnn Face Detection
  7. `07_mtcnn_face_detection_plot.ipynb` — Mtcnn Face Detection Plot
  8. `08_mtcnn_face_detection_plot_landmarks1.ipynb` — Mtcnn Face Detection Plot Landmarks1
  9. `09_mtcnn_face_detection_plot_landmarks2.ipynb` — Mtcnn Face Detection Plot Landmarks2
  10. `10_mtcnn_extract_faces.ipynb` — Mtcnn Extract Faces

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 28) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 28）是机器学习流水线中的基础构建块。

---
