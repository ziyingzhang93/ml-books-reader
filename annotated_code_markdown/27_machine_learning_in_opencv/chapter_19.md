# OpenCV ML
## Chapter 19

---

### Train Detector

# 08 — Train Detector / 08 Train Detector

**Chapter 19 — File 1 of 2 / 第19章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Define HOG parameters**.

本脚本演示 **Define HOG parameters**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model


---
## Step 1 — Step 1

```python
import pathlib
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np

def read_voc_xml(xmlfile):
    """read the Pascal VOC XML"""
    root = ET.parse(xmlfile).getroot()
    boxes = {"filename": root.find("filename").text,
             "objects": []}
    for box in root.iter('object'):
        bb = box.find('bndbox')
        obj = {
            "name": box.find('name').text,
            "xmin": int(bb.find("xmin").text),
            "ymin": int(bb.find("ymin").text),
            "xmax": int(bb.find("xmax").text),
            "ymax": int(bb.find("ymax").text),
        }
        boxes["objects"].append(obj)

    return boxes

def make_square(xmin, xmax, ymin, ymax):
    """Shrink the bounding box to square shape"""
    xcenter = (xmax + xmin) // 2
    ycenter = (ymax + ymin) // 2
    halfdim = min(xmax-xmin, ymax-ymin) // 2
    xmin, xmax = xcenter-halfdim, xcenter+halfdim
    ymin, ymax = ycenter-halfdim, ycenter+halfdim
    return xmin, xmax, ymin, ymax
```

---
## Step 2 — Define HOG parameters

```python
winSize = (64, 64)
blockSize = (32, 32)
blockStride = (16, 16)
cellSize = (16, 16)
nbins = 9

num_samples = 1000
```

---
## Step 3 — Load your dataset and corresponding bounding box annotations

```python
base_path = pathlib.Path("oxford-iiit-pet")
img_src = base_path / "images"
ann_src = base_path / "annotations" / "xmls"
```

---
## Step 4 — collect samples by cropping the images from dataset

```python
positive = []
negative = []
```

---
## Step 5 — collect positive and negative samples

```python
for xmlfile in ann_src.glob("*.xml"):
```

---
## Step 6 — load xml

```python
ann = read_voc_xml(str(xmlfile))
```

---
## Step 7 — cat for positive samples, other for negative samples

```python
if ann["objects"][0]["name"] == "cat":
        if len(positive) <= num_samples:
```

---
## Step 8 — adjust the bounding box to square

```python
box = ann["objects"][0]
            xmin, xmax, ymin, ymax = box["xmin"], box["xmax"], box["ymin"], box["ymax"]
            xmin, xmax, ymin, ymax = make_square(xmin, xmax, ymin, ymax)
```

---
## Step 9 — crop a positive sample

```python
img = cv2.imread(str(img_src / ann["filename"]))
            sample = img[ymin:ymax, xmin:xmax]
            sample = cv2.resize(sample, winSize)
            positive.append(sample)
    else:
        if len(negative) <= num_samples:
```

---
## Step 10 — random bounding box: at least the target size to avoid scaling up

```python
height, width = img.shape[:2]
            boxsize = random.randint(winSize[0], min(height, width))
            x = random.randint(0, width-boxsize)
            y = random.randint(0, height-boxsize)
            sample = img[y:y+boxsize, x:x+boxsize]
            assert tuple(sample.shape[:2]) == (boxsize, boxsize)
            sample = cv2.resize(sample, winSize)
            negative.append(sample)

images = positive + negative
labels = ([1] * len(positive)) + ([0] * len(negative))
```

---
## Step 11 — Create the HOG descriptor and the HOG from each image

```python
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
data = []
for img in images:
    features = hog.compute(img)
    data.append(features.flatten())
```

---
## Step 12 — Convert data and labels to numpy arrays

```python
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)
```

---
## Step 13 — Train the SVM

```python
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100000, 1e-8))

svm.train(data, cv2.ml.ROW_SAMPLE, labels)
```

---
## Step 14 — Save the SVM model

```python
svm.save('svm_model.yml')
print(svm.getSupportVectors())
```

---
## Learning Notes / 学习笔记

- **概念**: Define HOG parameters 是机器学习中的常用技术。  
  *Define HOG parameters is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `SVM` | 支持向量机 | Support Vector Machine |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Detector / 08 Train Detector
# Complete Code / 完整代码
# ===============================

import pathlib
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np

def read_voc_xml(xmlfile):
    """read the Pascal VOC XML"""
    root = ET.parse(xmlfile).getroot()
    boxes = {"filename": root.find("filename").text,
             "objects": []}
    for box in root.iter('object'):
        bb = box.find('bndbox')
        obj = {
            "name": box.find('name').text,
            "xmin": int(bb.find("xmin").text),
            "ymin": int(bb.find("ymin").text),
            "xmax": int(bb.find("xmax").text),
            "ymax": int(bb.find("ymax").text),
        }
        boxes["objects"].append(obj)

    return boxes

def make_square(xmin, xmax, ymin, ymax):
    """Shrink the bounding box to square shape"""
    xcenter = (xmax + xmin) // 2
    ycenter = (ymax + ymin) // 2
    halfdim = min(xmax-xmin, ymax-ymin) // 2
    xmin, xmax = xcenter-halfdim, xcenter+halfdim
    ymin, ymax = ycenter-halfdim, ycenter+halfdim
    return xmin, xmax, ymin, ymax

# Define HOG parameters
winSize = (64, 64)
blockSize = (32, 32)
blockStride = (16, 16)
cellSize = (16, 16)
nbins = 9

num_samples = 1000

# Load your dataset and corresponding bounding box annotations
base_path = pathlib.Path("oxford-iiit-pet")
img_src = base_path / "images"
ann_src = base_path / "annotations" / "xmls"

# collect samples by cropping the images from dataset
positive = []
negative = []

# collect positive and negative samples
for xmlfile in ann_src.glob("*.xml"):
    # load xml
    ann = read_voc_xml(str(xmlfile))
    # cat for positive samples, other for negative samples
    if ann["objects"][0]["name"] == "cat":
        if len(positive) <= num_samples:
            # adjust the bounding box to square
            box = ann["objects"][0]
            xmin, xmax, ymin, ymax = box["xmin"], box["xmax"], box["ymin"], box["ymax"]
            xmin, xmax, ymin, ymax = make_square(xmin, xmax, ymin, ymax)
            # crop a positive sample
            img = cv2.imread(str(img_src / ann["filename"]))
            sample = img[ymin:ymax, xmin:xmax]
            sample = cv2.resize(sample, winSize)
            positive.append(sample)
    else:
        if len(negative) <= num_samples:
            # random bounding box: at least the target size to avoid scaling up
            height, width = img.shape[:2]
            boxsize = random.randint(winSize[0], min(height, width))
            x = random.randint(0, width-boxsize)
            y = random.randint(0, height-boxsize)
            sample = img[y:y+boxsize, x:x+boxsize]
            assert tuple(sample.shape[:2]) == (boxsize, boxsize)
            sample = cv2.resize(sample, winSize)
            negative.append(sample)

images = positive + negative
labels = ([1] * len(positive)) + ([0] * len(negative))

# Create the HOG descriptor and the HOG from each image
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
data = []
for img in images:
    features = hog.compute(img)
    data.append(features.flatten())

# Convert data and labels to numpy arrays
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# Train the SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100000, 1e-8))

svm.train(data, cv2.ml.ROW_SAMPLE, labels)

# Save the SVM model
svm.save('svm_model.yml')
print(svm.getSupportVectors())
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Detect

# 10 — Detect / 10 Detect

**Chapter 19 — File 2 of 2 / 第19章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **load the SVM**.

本脚本演示 **load the SVM**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — Step 1

```python
import pathlib
import xml.etree.ElementTree as ET

import cv2
import numpy as np

def read_voc_xml(xmlfile: str) -> dict:
    """read the Pascal VOC XML and return (filename, object name, bounding box)
    where bounding box is a vector of (xmin, ymin, xmax, ymax). The pixel
    coordinates are 1-based.
    """
    root = ET.parse(xmlfile).getroot()
    boxes = {"filename": root.find("filename").text,
             "objects": []
            }
    for box in root.iter('object'):
        bb = box.find('bndbox')
        obj = {
            "name": box.find('name').text,
            "xmin": int(bb.find("xmin").text),
            "ymin": int(bb.find("ymin").text),
            "xmax": int(bb.find("xmax").text),
            "ymax": int(bb.find("ymax").text),
        }
        boxes["objects"].append(obj)

    return boxes
```

---
## Step 2 — load the SVM

```python
winSize = (64, 64)
blockSize = (32, 32)
blockStride = (16, 16)
cellSize = (16, 16)
nbins = 9

svm = cv2.ml.SVM_load('svm_model.yml')
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
hog.setSVMDetector(svm.getSupportVectors()[0])
```

---
## Step 3 — Run the SVM on each image

```python
base_path = pathlib.Path("oxford-iiit-pet")
img_src = base_path / "images"
ann_src = base_path / "annotations" / "xmls"

for xmlfile in ann_src.glob("*.xml"):
```

---
## Step 4 — load xml

```python
ann = read_voc_xml(str(xmlfile))
```

---
## Step 5 — read image and the groundtruth

```python
img = cv2.imread(str(img_src / ann["filename"]))
    bbox = ann["objects"][0]
    start_point = (bbox["xmin"], bbox["ymin"])
    end_point = (bbox["xmax"], bbox["ymax"])
```

---
## Step 6 — detect and draw

```python
locations, scores = hog.detectMultiScale(img)
    x, y, w, h = locations[np.argmax(scores.flatten())]
    cv2.rectangle(img, start_point, end_point, (0,0,255), 2)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 5)
    cv2.imshow(f"{ann['filename']}: {ann['objects'][0]['name']}", img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == 27:  # ESC key
        break
```

---
## Learning Notes / 学习笔记

- **概念**: load the SVM 是机器学习中的常用技术。  
  *load the SVM is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `SVM` | 支持向量机 | Support Vector Machine |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Detect / 10 Detect
# Complete Code / 完整代码
# ===============================

import pathlib
import xml.etree.ElementTree as ET

import cv2
import numpy as np

def read_voc_xml(xmlfile: str) -> dict:
    """read the Pascal VOC XML and return (filename, object name, bounding box)
    where bounding box is a vector of (xmin, ymin, xmax, ymax). The pixel
    coordinates are 1-based.
    """
    root = ET.parse(xmlfile).getroot()
    boxes = {"filename": root.find("filename").text,
             "objects": []
            }
    for box in root.iter('object'):
        bb = box.find('bndbox')
        obj = {
            "name": box.find('name').text,
            "xmin": int(bb.find("xmin").text),
            "ymin": int(bb.find("ymin").text),
            "xmax": int(bb.find("xmax").text),
            "ymax": int(bb.find("ymax").text),
        }
        boxes["objects"].append(obj)

    return boxes

# load the SVM
winSize = (64, 64)
blockSize = (32, 32)
blockStride = (16, 16)
cellSize = (16, 16)
nbins = 9

svm = cv2.ml.SVM_load('svm_model.yml')
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
hog.setSVMDetector(svm.getSupportVectors()[0])

# Run the SVM on each image
base_path = pathlib.Path("oxford-iiit-pet")
img_src = base_path / "images"
ann_src = base_path / "annotations" / "xmls"

for xmlfile in ann_src.glob("*.xml"):
    # load xml
    ann = read_voc_xml(str(xmlfile))
    # read image and the groundtruth
    img = cv2.imread(str(img_src / ann["filename"]))
    bbox = ann["objects"][0]
    start_point = (bbox["xmin"], bbox["ymin"])
    end_point = (bbox["xmax"], bbox["ymax"])
    # detect and draw
    locations, scores = hog.detectMultiScale(img)
    x, y, w, h = locations[np.argmax(scores.flatten())]
    cv2.rectangle(img, start_point, end_point, (0,0,255), 2)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 5)
    cv2.imshow(f"{ann['filename']}: {ann['objects'][0]['name']}", img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == 27:  # ESC key
        break
```

---
