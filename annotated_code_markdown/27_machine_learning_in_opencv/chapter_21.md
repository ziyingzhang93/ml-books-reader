# OpenCV ML
## Chapter 21

---

### Create Info File

# 10 — Create Info File / 10 Create Info File

**Chapter 21 — File 1 of 2 / 第21章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Read Pascal VOC and write data**.

本脚本演示 **Read Pascal VOC and write data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import pathlib
import xml.etree.ElementTree as ET

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
## Step 2 — Read Pascal VOC and write data

```python
base_path = pathlib.Path("oxford-iiit-pet")
img_src = base_path / "images"
ann_src = base_path / "annotations" / "xmls"

negative = []
positive = []
for xmlfile in ann_src.glob("*.xml"):
```

---
## Step 3 — load xml

```python
ann = read_voc_xml(str(xmlfile))
    if ann['objects'][0]['name'] == 'dog':
```

---
## Step 4 — negative sample (dog)

```python
negative.append(str(img_src / ann['filename']))
    else:
```

---
## Step 5 — positive sample (cats)

```python
bbox = []
        for obj in ann['objects']:
            x = obj['xmin']
            y = obj['ymin']
            w = obj['xmax'] - obj['xmin']
            h = obj['ymax'] - obj['ymin']
            bbox.append(f"{x} {y} {w} {h}")
        line = f"{str(img_src/ann['filename'])} {len(bbox)} {' '.join(bbox)}"
        positive.append(line)
```

---
## Step 6 — write the output to `negative.dat` and `postiive.dat`

```python
with open("negative.dat", "w") as fp:
    fp.write("\n".join(negative))

with open("positive.dat", "w") as fp:
    fp.write("\n".join(positive))
```

---
## Learning Notes / 学习笔记

- **概念**: Read Pascal VOC and write data 是机器学习中的常用技术。  
  *Read Pascal VOC and write data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Create Info File / 10 Create Info File
# Complete Code / 完整代码
# ===============================

import pathlib
import xml.etree.ElementTree as ET

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

# Read Pascal VOC and write data
base_path = pathlib.Path("oxford-iiit-pet")
img_src = base_path / "images"
ann_src = base_path / "annotations" / "xmls"

negative = []
positive = []
for xmlfile in ann_src.glob("*.xml"):
    # load xml
    ann = read_voc_xml(str(xmlfile))
    if ann['objects'][0]['name'] == 'dog':
        # negative sample (dog)
        negative.append(str(img_src / ann['filename']))
    else:
        # positive sample (cats)
        bbox = []
        for obj in ann['objects']:
            x = obj['xmin']
            y = obj['ymin']
            w = obj['xmax'] - obj['xmin']
            h = obj['ymax'] - obj['ymin']
            bbox.append(f"{x} {y} {w} {h}")
        line = f"{str(img_src/ann['filename'])} {len(bbox)} {' '.join(bbox)}"
        positive.append(line)

# write the output to `negative.dat` and `postiive.dat`
with open("negative.dat", "w") as fp:
    fp.write("\n".join(negative))

with open("positive.dat", "w") as fp:
    fp.write("\n".join(positive))
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Cascade

# 15 — Cascade / 15 Cascade

**Chapter 21 — File 2 of 2 / 第21章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Convert the image to grayscale**.

本脚本演示 **Convert the image to grayscale**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import cv2

image = 'oxford-iiit-pet/images/Abyssinian_88.jpg'
model = 'cat_detect/cascade.xml'

classifier = cv2.CascadeClassifier(model)
img = cv2.imread(image)
```

---
## Step 2 — Convert the image to grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---
## Step 3 — Perform object detection

```python
objects = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(30, 30))
```

---
## Step 4 — Draw rectangles around detected objects

```python
for (x, y, w, h) in objects:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
```

---
## Step 5 — Display the result

```python
cv2.imshow('Object Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---
## Learning Notes / 学习笔记

- **概念**: Convert the image to grayscale 是机器学习中的常用技术。  
  *Convert the image to grayscale is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cascade / 15 Cascade
# Complete Code / 完整代码
# ===============================

import cv2

image = 'oxford-iiit-pet/images/Abyssinian_88.jpg'
model = 'cat_detect/cascade.xml'

classifier = cv2.CascadeClassifier(model)
img = cv2.imread(image)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform object detection
objects = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(30, 30))

# Draw rectangles around detected objects
for (x, y, w, h) in objects:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the result
cv2.imshow('Object Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---
