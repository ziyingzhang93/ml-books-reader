# 计算机视觉深度学习 / Deep Learning for Computer Vision
## Chapter 25

---

### Localize Objects

# 01 — Localize Objects / 目标检测

**Chapter 25 — File 1 of 2 / 第25章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **example of inference with a pre-trained coco model**.

本脚本演示 **example of inference with a pre-trained coco model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — example of inference with a pre-trained coco model

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.patches import Rectangle
```

---
## Step 2 — draw an image with detected objects

```python
def draw_image_with_boxes(filename, boxes_list):
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
for box in boxes_list:
```

---
## Step 7 — get coordinates

```python
y1, x1, y2, x2 = box
```

---
## Step 8 — calculate width and height of the box

```python
width, height = x2 - x1, y2 - y1
```

---
## Step 9 — create the shape

```python
rect = Rectangle((x1, y1), width, height, fill=False, color='red')
```

---
## Step 10 — draw the box

```python
ax.add_patch(rect)
```

---
## Step 11 — show the plot

```python
pyplot.show()
```

---
## Step 12 — define the test configuration

```python
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80
```

---
## Step 13 — define the model

```python
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
```

---
## Step 14 — load coco model weights

```python
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
```

---
## Step 15 — load photograph

```python
img = load_img('elephant.jpg')
img = img_to_array(img)
```

---
## Step 16 — make prediction

```python
results = rcnn.detect([img], verbose=0)
```

---
## Step 17 — visualize the results

```python
draw_image_with_boxes('elephant.jpg', results[0]['rois'])
```

---
## Learning Notes / 学习笔记

- **概念**: example of inference with a pre-trained coco model 是机器学习中的常用技术。  
  *example of inference with a pre-trained coco model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Localize Objects / 目标检测
# Complete Code / 完整代码
# ===============================

# example of inference with a pre-trained coco model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.patches import Rectangle

# draw an image with detected objects
def draw_image_with_boxes(filename, boxes_list):
     # load the image
     data = pyplot.imread(filename)
     # plot the image
     pyplot.imshow(data)
     # get the context for drawing boxes
     ax = pyplot.gca()
     # plot each box
     for box in boxes_list:
          # get coordinates
          y1, x1, y2, x2 = box
          # calculate width and height of the box
          width, height = x2 - x1, y2 - y1
          # create the shape
          rect = Rectangle((x1, y1), width, height, fill=False, color='red')
          # draw the box
          ax.add_patch(rect)
     # show the plot
     pyplot.show()

# define the test configuration
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80

# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
# load photograph
img = load_img('elephant.jpg')
img = img_to_array(img)
# make prediction
results = rcnn.detect([img], verbose=0)
# visualize the results
draw_image_with_boxes('elephant.jpg', results[0]['rois'])
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Object Detection

# 02 — Object Detection / 目标检测

**Chapter 25 — File 2 of 2 / 第25章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **example of inference with a pre-trained coco model**.

本脚本演示 **example of inference with a pre-trained coco model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — example of inference with a pre-trained coco model

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
```

---
## Step 2 — define 81 classes that the coco model knowns about

```python
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
```

---
## Step 3 — define the test configuration

```python
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80
```

---
## Step 4 — define the model

```python
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
```

---
## Step 5 — load coco model weights

```python
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
```

---
## Step 6 — load photograph

```python
img = load_img('elephant.jpg')
img = img_to_array(img)
```

---
## Step 7 — make prediction

```python
results = rcnn.detect([img], verbose=0)
```

---
## Step 8 — get dictionary for first prediction

```python
r = results[0]
```

---
## Step 9 — show photo with bounding boxes, masks, class labels and scores

```python
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
```

---
## Learning Notes / 学习笔记

- **概念**: example of inference with a pre-trained coco model 是机器学习中的常用技术。  
  *example of inference with a pre-trained coco model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Object Detection / 目标检测
# Complete Code / 完整代码
# ===============================

# example of inference with a pre-trained coco model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# define 81 classes that the coco model knowns about
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# define the test configuration
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80

# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
# load photograph
img = load_img('elephant.jpg')
img = img_to_array(img)
# make prediction
results = rcnn.detect([img], verbose=0)
# get dictionary for first prediction
r = results[0]
# show photo with bounding boxes, masks, class labels and scores
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
```

---

### Chapter Summary / 章节总结

# Chapter 25 Summary / 第25章总结

## Theme / 主题: Chapter 25 / Chapter 25

This chapter contains **2 code files** demonstrating chapter 25.

本章包含 **2 个代码文件**，演示Chapter 25。

---
## Evolution / 演化路线

  1. `01_localize_objects.ipynb` — Localize Objects
  2. `02_object_detection.ipynb` — Object Detection

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 25) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 25）是机器学习流水线中的基础构建块。

---
