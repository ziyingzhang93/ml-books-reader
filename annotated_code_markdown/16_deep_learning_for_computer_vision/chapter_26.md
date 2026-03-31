# 计算机视觉深度学习 / Deep Learning for Computer Vision
## Chapter 26

---

### Extract Annotation

# 01 — Extract Annotation / 01 Extract Annotation

**Chapter 26 — File 1 of 9 / 第26章 — 第1个文件（共9个）**

---

## Summary / 总结

This script demonstrates **example of extracting bounding boxes from an annotation file**.

本脚本演示 **example of extracting bounding boxes from an annotation file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of extracting bounding boxes from an annotation file

```python
from xml.etree import ElementTree
```

---
## Step 2 — function to extract bounding boxes from an annotation file

```python
def extract_boxes(filename):
```

---
## Step 3 — load and parse the file

```python
tree = ElementTree.parse(filename)
```

---
## Step 4 — get the root of the document

```python
root = tree.getroot()
```

---
## Step 5 — extract each bounding box

```python
boxes = list()
	for box in root.findall('.//bndbox'):
		xmin = int(box.find('xmin').text)
		ymin = int(box.find('ymin').text)
		xmax = int(box.find('xmax').text)
		ymax = int(box.find('ymax').text)
		coors = [xmin, ymin, xmax, ymax]
  # 添加元素到列表末尾 / Append element to list end
		boxes.append(coors)
```

---
## Step 6 — extract image dimensions

```python
width = int(root.find('.//size/width').text)
	height = int(root.find('.//size/height').text)
	return boxes, width, height
```

---
## Step 7 — extract details form annotation file

```python
boxes, w, h = extract_boxes('kangaroo/annots/00001.xml')
```

---
## Step 8 — summarize extracted details

```python
# 打印输出 / Print output
print(boxes, w, h)
```

---
## Learning Notes / 学习笔记

- **概念**: example of extracting bounding boxes from an annotation file 是机器学习中的常用技术。  
  *example of extracting bounding boxes from an annotation file is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Extract Annotation / 01 Extract Annotation
# Complete Code / 完整代码
# ===============================

# example of extracting bounding boxes from an annotation file
from xml.etree import ElementTree

# function to extract bounding boxes from an annotation file
def extract_boxes(filename):
	# load and parse the file
	tree = ElementTree.parse(filename)
	# get the root of the document
	root = tree.getroot()
	# extract each bounding box
	boxes = list()
	for box in root.findall('.//bndbox'):
		xmin = int(box.find('xmin').text)
		ymin = int(box.find('ymin').text)
		xmax = int(box.find('xmax').text)
		ymax = int(box.find('ymax').text)
		coors = [xmin, ymin, xmax, ymax]
  # 添加元素到列表末尾 / Append element to list end
		boxes.append(coors)
	# extract image dimensions
	width = int(root.find('.//size/width').text)
	height = int(root.find('.//size/height').text)
	return boxes, width, height

# extract details form annotation file
boxes, w, h = extract_boxes('kangaroo/annots/00001.xml')
# summarize extracted details
# 打印输出 / Print output
print(boxes, w, h)
```

---

➡️ **Next / 下一步**: File 2 of 9

---

### Dataset Object

# 02 — Dataset Object / 目标检测

**Chapter 26 — File 2 of 9 / 第26章 — 第2个文件（共9个）**

---

## Summary / 总结

This script demonstrates **split into train and test set**.

本脚本演示 **split into train and test set**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — split into train and test set

```python
from os import listdir
from xml.etree import ElementTree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from mrcnn.utils import Dataset
```

---
## Step 2 — class that defines and loads the kangaroo dataset

```python
# 定义数据集 / Define dataset
class KangarooDataset(Dataset):
```

---
## Step 3 — load the dataset definitions

```python
def load_dataset(self, dataset_dir, is_train=True):
```

---
## Step 4 — define one class

```python
self.add_class("dataset", 1, "kangaroo")
```

---
## Step 5 — define data locations

```python
images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
```

---
## Step 6 — find all images

```python
for filename in listdir(images_dir):
```

---
## Step 7 — extract image id

```python
image_id = filename[:-4]
```

---
## Step 8 — skip bad images

```python
if image_id in ['00090']:
				continue
```

---
## Step 9 — skip all images after 150 if we are building the train set

```python
if is_train and int(image_id) >= 150:
				continue
```

---
## Step 10 — skip all images before 150 if we are building the test/val set

```python
if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
```

---
## Step 11 — add to dataset

```python
self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
```

---
## Step 12 — extract bounding boxes from an annotation file

```python
def extract_boxes(self, filename):
```

---
## Step 13 — load and parse the file

```python
tree = ElementTree.parse(filename)
```

---
## Step 14 — get the root of the document

```python
root = tree.getroot()
```

---
## Step 15 — extract each bounding box

```python
boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
   # 添加元素到列表末尾 / Append element to list end
			boxes.append(coors)
```

---
## Step 16 — extract image dimensions

```python
width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height
```

---
## Step 17 — load the masks for an image

```python
def load_mask(self, image_id):
```

---
## Step 18 — get details of image

```python
info = self.image_info[image_id]
```

---
## Step 19 — define box file location

```python
path = info['annotation']
```

---
## Step 20 — load XML

```python
boxes, w, h = self.extract_boxes(path)
```

---
## Step 21 — create one array for all masks, each on a different channel

```python
# 获取长度 / Get length
masks = zeros([h, w, len(boxes)], dtype='uint8')
```

---
## Step 22 — create masks

```python
class_ids = list()
  # 获取长度 / Get length
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
   # 添加元素到列表末尾 / Append element to list end
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')
```

---
## Step 23 — load an image reference

```python
def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']
```

---
## Step 24 — train set

```python
# 定义数据集 / Define dataset
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
# 打印输出 / Print output
print('Train: %d' % len(train_set.image_ids))
```

---
## Step 25 — test/val set

```python
# 定义数据集 / Define dataset
test_set = KangarooDataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
# 打印输出 / Print output
print('Test: %d' % len(test_set.image_ids))
```

---
## Learning Notes / 学习笔记

- **概念**: split into train and test set 是机器学习中的常用技术。  
  *split into train and test set is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dataset Object / 目标检测
# Complete Code / 完整代码
# ===============================

# split into train and test set
from os import listdir
from xml.etree import ElementTree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from mrcnn.utils import Dataset

# class that defines and loads the kangaroo dataset
# 定义数据集 / Define dataset
class KangarooDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "kangaroo")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			if image_id in ['00090']:
				continue
			# skip all images after 150 if we are building the train set
			if is_train and int(image_id) >= 150:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
   # 添加元素到列表末尾 / Append element to list end
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
  # 获取长度 / Get length
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
  # 获取长度 / Get length
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
   # 添加元素到列表末尾 / Append element to list end
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# train set
# 定义数据集 / Define dataset
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
# 打印输出 / Print output
print('Train: %d' % len(train_set.image_ids))

# test/val set
# 定义数据集 / Define dataset
test_set = KangarooDataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
# 打印输出 / Print output
print('Test: %d' % len(test_set.image_ids))
```

---

➡️ **Next / 下一步**: File 3 of 9

---

### Plot Photo With Mask

# 03 — Plot Photo With Mask / 03 Plot Photo With Mask

**Chapter 26 — File 3 of 9 / 第26章 — 第3个文件（共9个）**

---

## Summary / 总结

This script demonstrates **plot one photograph and mask**.

本脚本演示 **plot one photograph and mask**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — plot one photograph and mask

```python
from os import listdir
from xml.etree import ElementTree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from mrcnn.utils import Dataset
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — class that defines and loads the kangaroo dataset

```python
# 定义数据集 / Define dataset
class KangarooDataset(Dataset):
```

---
## Step 3 — load the dataset definitions

```python
def load_dataset(self, dataset_dir, is_train=True):
```

---
## Step 4 — define one class

```python
self.add_class("dataset", 1, "kangaroo")
```

---
## Step 5 — define data locations

```python
images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
```

---
## Step 6 — find all images

```python
for filename in listdir(images_dir):
```

---
## Step 7 — extract image id

```python
image_id = filename[:-4]
```

---
## Step 8 — skip bad images

```python
if image_id in ['00090']:
				continue
```

---
## Step 9 — skip all images after 150 if we are building the train set

```python
if is_train and int(image_id) >= 150:
				continue
```

---
## Step 10 — skip all images before 150 if we are building the test/val set

```python
if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
```

---
## Step 11 — add to dataset

```python
self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
```

---
## Step 12 — extract bounding boxes from an annotation file

```python
def extract_boxes(self, filename):
```

---
## Step 13 — load and parse the file

```python
tree = ElementTree.parse(filename)
```

---
## Step 14 — get the root of the document

```python
root = tree.getroot()
```

---
## Step 15 — extract each bounding box

```python
boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
   # 添加元素到列表末尾 / Append element to list end
			boxes.append(coors)
```

---
## Step 16 — extract image dimensions

```python
width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height
```

---
## Step 17 — load the masks for an image

```python
def load_mask(self, image_id):
```

---
## Step 18 — get details of image

```python
info = self.image_info[image_id]
```

---
## Step 19 — define box file location

```python
path = info['annotation']
```

---
## Step 20 — load XML

```python
boxes, w, h = self.extract_boxes(path)
```

---
## Step 21 — create one array for all masks, each on a different channel

```python
# 获取长度 / Get length
masks = zeros([h, w, len(boxes)], dtype='uint8')
```

---
## Step 22 — create masks

```python
class_ids = list()
  # 获取长度 / Get length
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
   # 添加元素到列表末尾 / Append element to list end
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')
```

---
## Step 23 — load an image reference

```python
def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']
```

---
## Step 24 — train set

```python
# 定义数据集 / Define dataset
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
```

---
## Step 25 — load an image

```python
image_id = 0
image = train_set.load_image(image_id)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(image.shape)
```

---
## Step 26 — load image mask

```python
mask, class_ids = train_set.load_mask(image_id)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(mask.shape)
```

---
## Step 27 — plot image

```python
pyplot.imshow(image)
```

---
## Step 28 — plot mask

```python
pyplot.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot one photograph and mask 是机器学习中的常用技术。  
  *plot one photograph and mask is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Photo With Mask / 03 Plot Photo With Mask
# Complete Code / 完整代码
# ===============================

# plot one photograph and mask
from os import listdir
from xml.etree import ElementTree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from mrcnn.utils import Dataset
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# class that defines and loads the kangaroo dataset
# 定义数据集 / Define dataset
class KangarooDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "kangaroo")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			if image_id in ['00090']:
				continue
			# skip all images after 150 if we are building the train set
			if is_train and int(image_id) >= 150:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
   # 添加元素到列表末尾 / Append element to list end
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
  # 获取长度 / Get length
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
  # 获取长度 / Get length
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
   # 添加元素到列表末尾 / Append element to list end
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# train set
# 定义数据集 / Define dataset
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
# load an image
image_id = 0
image = train_set.load_image(image_id)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(image.shape)
# load image mask
mask, class_ids = train_set.load_mask(image_id)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(mask.shape)
# plot image
pyplot.imshow(image)
# plot mask
pyplot.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 9

---

### Plot Multiple Photos With Mask

# 04 — Plot Multiple Photos With Mask / 04 Plot Multiple Photos With Mask

**Chapter 26 — File 4 of 9 / 第26章 — 第4个文件（共9个）**

---

## Summary / 总结

This script demonstrates **plot a number of photographs and their mask**.

本脚本演示 **plot a number of photographs and their mask**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — plot a number of photographs and their mask

```python
from os import listdir
from xml.etree import ElementTree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from mrcnn.utils import Dataset
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — class that defines and loads the kangaroo dataset

```python
# 定义数据集 / Define dataset
class KangarooDataset(Dataset):
```

---
## Step 3 — load the dataset definitions

```python
def load_dataset(self, dataset_dir, is_train=True):
```

---
## Step 4 — define one class

```python
self.add_class("dataset", 1, "kangaroo")
```

---
## Step 5 — define data locations

```python
images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
```

---
## Step 6 — find all images

```python
for filename in listdir(images_dir):
```

---
## Step 7 — extract image id

```python
image_id = filename[:-4]
```

---
## Step 8 — skip bad images

```python
if image_id in ['00090']:
				continue
```

---
## Step 9 — skip all images after 150 if we are building the train set

```python
if is_train and int(image_id) >= 150:
				continue
```

---
## Step 10 — skip all images before 150 if we are building the test/val set

```python
if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
```

---
## Step 11 — add to dataset

```python
self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
```

---
## Step 12 — extract bounding boxes from an annotation file

```python
def extract_boxes(self, filename):
```

---
## Step 13 — load and parse the file

```python
tree = ElementTree.parse(filename)
```

---
## Step 14 — get the root of the document

```python
root = tree.getroot()
```

---
## Step 15 — extract each bounding box

```python
boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
   # 添加元素到列表末尾 / Append element to list end
			boxes.append(coors)
```

---
## Step 16 — extract image dimensions

```python
width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height
```

---
## Step 17 — load the masks for an image

```python
def load_mask(self, image_id):
```

---
## Step 18 — get details of image

```python
info = self.image_info[image_id]
```

---
## Step 19 — define box file location

```python
path = info['annotation']
```

---
## Step 20 — load XML

```python
boxes, w, h = self.extract_boxes(path)
```

---
## Step 21 — create one array for all masks, each on a different channel

```python
# 获取长度 / Get length
masks = zeros([h, w, len(boxes)], dtype='uint8')
```

---
## Step 22 — create masks

```python
class_ids = list()
  # 获取长度 / Get length
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
   # 添加元素到列表末尾 / Append element to list end
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')
```

---
## Step 23 — load an image reference

```python
def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']
```

---
## Step 24 — train set

```python
# 定义数据集 / Define dataset
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
```

---
## Step 25 — load an image

```python
image_id = 0
image = train_set.load_image(image_id)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(image.shape)
```

---
## Step 26 — load image mask

```python
mask, class_ids = train_set.load_mask(image_id)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(mask.shape)
```

---
## Step 27 — plot first few images

```python
# 生成整数序列 / Generate integer sequence
for i in range(9):
```

---
## Step 28 — define subplot

```python
pyplot.subplot(330 + 1 + i)
```

---
## Step 29 — turn off axis labels

```python
pyplot.axis('off')
```

---
## Step 30 — plot raw pixel data

```python
image = train_set.load_image(i)
	pyplot.imshow(image)
```

---
## Step 31 — plot all masks

```python
mask, _ = train_set.load_mask(i)
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	for j in range(mask.shape[2]):
		pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
```

---
## Step 32 — show the figure

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot a number of photographs and their mask 是机器学习中的常用技术。  
  *plot a number of photographs and their mask is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Multiple Photos With Mask / 04 Plot Multiple Photos With Mask
# Complete Code / 完整代码
# ===============================

# plot a number of photographs and their mask
from os import listdir
from xml.etree import ElementTree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from mrcnn.utils import Dataset
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# class that defines and loads the kangaroo dataset
# 定义数据集 / Define dataset
class KangarooDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "kangaroo")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			if image_id in ['00090']:
				continue
			# skip all images after 150 if we are building the train set
			if is_train and int(image_id) >= 150:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
   # 添加元素到列表末尾 / Append element to list end
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
  # 获取长度 / Get length
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
  # 获取长度 / Get length
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
   # 添加元素到列表末尾 / Append element to list end
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# train set
# 定义数据集 / Define dataset
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
# load an image
image_id = 0
image = train_set.load_image(image_id)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(image.shape)
# load image mask
mask, class_ids = train_set.load_mask(image_id)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(mask.shape)
# plot first few images
# 生成整数序列 / Generate integer sequence
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# turn off axis labels
	pyplot.axis('off')
	# plot raw pixel data
	image = train_set.load_image(i)
	pyplot.imshow(image)
	# plot all masks
	mask, _ = train_set.load_mask(i)
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	for j in range(mask.shape[2]):
		pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
# show the figure
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 9

---

### Summarize Image Paths

# 05 — Summarize Image Paths / 图像处理

**Chapter 26 — File 5 of 9 / 第26章 — 第5个文件（共9个）**

---

## Summary / 总结

This script demonstrates **summarize the paths to images in the dataset**.

本脚本演示 **summarize the paths to images in the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — summarize the paths to images in the dataset

```python
from os import listdir
from xml.etree import ElementTree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from mrcnn.utils import Dataset
```

---
## Step 2 — class that defines and loads the kangaroo dataset

```python
# 定义数据集 / Define dataset
class KangarooDataset(Dataset):
```

---
## Step 3 — load the dataset definitions

```python
def load_dataset(self, dataset_dir, is_train=True):
```

---
## Step 4 — define one class

```python
self.add_class("dataset", 1, "kangaroo")
```

---
## Step 5 — define data locations

```python
images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
```

---
## Step 6 — find all images

```python
for filename in listdir(images_dir):
```

---
## Step 7 — extract image id

```python
image_id = filename[:-4]
```

---
## Step 8 — skip bad images

```python
if image_id in ['00090']:
				continue
```

---
## Step 9 — skip all images after 150 if we are building the train set

```python
if is_train and int(image_id) >= 150:
				continue
```

---
## Step 10 — skip all images before 150 if we are building the test/val set

```python
if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
```

---
## Step 11 — add to dataset

```python
self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
```

---
## Step 12 — extract bounding boxes from an annotation file

```python
def extract_boxes(self, filename):
```

---
## Step 13 — load and parse the file

```python
tree = ElementTree.parse(filename)
```

---
## Step 14 — get the root of the document

```python
root = tree.getroot()
```

---
## Step 15 — extract each bounding box

```python
boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
   # 添加元素到列表末尾 / Append element to list end
			boxes.append(coors)
```

---
## Step 16 — extract image dimensions

```python
width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height
```

---
## Step 17 — load the masks for an image

```python
def load_mask(self, image_id):
```

---
## Step 18 — get details of image

```python
info = self.image_info[image_id]
```

---
## Step 19 — define box file location

```python
path = info['annotation']
```

---
## Step 20 — load XML

```python
boxes, w, h = self.extract_boxes(path)
```

---
## Step 21 — create one array for all masks, each on a different channel

```python
# 获取长度 / Get length
masks = zeros([h, w, len(boxes)], dtype='uint8')
```

---
## Step 22 — create masks

```python
class_ids = list()
  # 获取长度 / Get length
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
   # 添加元素到列表末尾 / Append element to list end
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')
```

---
## Step 23 — load an image reference

```python
def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']
```

---
## Step 24 — train set

```python
# 定义数据集 / Define dataset
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
```

---
## Step 25 — load an image

```python
image_id = 0
image = train_set.load_image(image_id)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(image.shape)
```

---
## Step 26 — load image mask

```python
mask, class_ids = train_set.load_mask(image_id)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(mask.shape)
```

---
## Step 27 — enumerate all images in the dataset

```python
for image_id in train_set.image_ids:
```

---
## Step 28 — load image info

```python
info = train_set.image_info[image_id]
```

---
## Step 29 — display on the console

```python
# 打印输出 / Print output
print(info)
```

---
## Learning Notes / 学习笔记

- **概念**: summarize the paths to images in the dataset 是机器学习中的常用技术。  
  *summarize the paths to images in the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Summarize Image Paths / 图像处理
# Complete Code / 完整代码
# ===============================

# summarize the paths to images in the dataset
from os import listdir
from xml.etree import ElementTree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from mrcnn.utils import Dataset

# class that defines and loads the kangaroo dataset
# 定义数据集 / Define dataset
class KangarooDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "kangaroo")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			if image_id in ['00090']:
				continue
			# skip all images after 150 if we are building the train set
			if is_train and int(image_id) >= 150:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
   # 添加元素到列表末尾 / Append element to list end
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
  # 获取长度 / Get length
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
  # 获取长度 / Get length
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
   # 添加元素到列表末尾 / Append element to list end
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# train set
# 定义数据集 / Define dataset
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
# load an image
image_id = 0
image = train_set.load_image(image_id)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(image.shape)
# load image mask
mask, class_ids = train_set.load_mask(image_id)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(mask.shape)
# enumerate all images in the dataset
for image_id in train_set.image_ids:
	# load image info
	info = train_set.image_info[image_id]
	# display on the console
 # 打印输出 / Print output
	print(info)
```

---

➡️ **Next / 下一步**: File 6 of 9

---

### Plot Photo Mask Builtin

# 06 — Plot Photo Mask Builtin / 06 Plot Photo Mask Builtin

**Chapter 26 — File 6 of 9 / 第26章 — 第6个文件（共9个）**

---

## Summary / 总结

This script demonstrates **display image with masks and bounding boxes**.

本脚本演示 **display image with masks and bounding boxes**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — display image with masks and bounding boxes

```python
from os import listdir
from xml.etree import ElementTree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
```

---
## Step 2 — class that defines and loads the kangaroo dataset

```python
# 定义数据集 / Define dataset
class KangarooDataset(Dataset):
```

---
## Step 3 — load the dataset definitions

```python
def load_dataset(self, dataset_dir, is_train=True):
```

---
## Step 4 — define one class

```python
self.add_class("dataset", 1, "kangaroo")
```

---
## Step 5 — define data locations

```python
images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
```

---
## Step 6 — find all images

```python
for filename in listdir(images_dir):
```

---
## Step 7 — extract image id

```python
image_id = filename[:-4]
```

---
## Step 8 — skip bad images

```python
if image_id in ['00090']:
				continue
```

---
## Step 9 — skip all images after 150 if we are building the train set

```python
if is_train and int(image_id) >= 150:
				continue
```

---
## Step 10 — skip all images before 150 if we are building the test/val set

```python
if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
```

---
## Step 11 — add to dataset

```python
self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
```

---
## Step 12 — extract bounding boxes from an annotation file

```python
def extract_boxes(self, filename):
```

---
## Step 13 — load and parse the file

```python
tree = ElementTree.parse(filename)
```

---
## Step 14 — get the root of the document

```python
root = tree.getroot()
```

---
## Step 15 — extract each bounding box

```python
boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
   # 添加元素到列表末尾 / Append element to list end
			boxes.append(coors)
```

---
## Step 16 — extract image dimensions

```python
width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height
```

---
## Step 17 — load the masks for an image

```python
def load_mask(self, image_id):
```

---
## Step 18 — get details of image

```python
info = self.image_info[image_id]
```

---
## Step 19 — define box file location

```python
path = info['annotation']
```

---
## Step 20 — load XML

```python
boxes, w, h = self.extract_boxes(path)
```

---
## Step 21 — create one array for all masks, each on a different channel

```python
# 获取长度 / Get length
masks = zeros([h, w, len(boxes)], dtype='uint8')
```

---
## Step 22 — create masks

```python
class_ids = list()
  # 获取长度 / Get length
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
   # 添加元素到列表末尾 / Append element to list end
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')
```

---
## Step 23 — load an image reference

```python
def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']
```

---
## Step 24 — train set

```python
# 定义数据集 / Define dataset
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
```

---
## Step 25 — define image id

```python
image_id = 1
```

---
## Step 26 — load the image

```python
image = train_set.load_image(image_id)
```

---
## Step 27 — load the masks and the class ids

```python
mask, class_ids = train_set.load_mask(image_id)
```

---
## Step 28 — extract bounding boxes from the masks

```python
bbox = extract_bboxes(mask)
```

---
## Step 29 — display image with masks and bounding boxes

```python
display_instances(image, bbox, mask, class_ids, train_set.class_names)
```

---
## Learning Notes / 学习笔记

- **概念**: display image with masks and bounding boxes 是机器学习中的常用技术。  
  *display image with masks and bounding boxes is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Photo Mask Builtin / 06 Plot Photo Mask Builtin
# Complete Code / 完整代码
# ===============================

# display image with masks and bounding boxes
from os import listdir
from xml.etree import ElementTree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

# class that defines and loads the kangaroo dataset
# 定义数据集 / Define dataset
class KangarooDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "kangaroo")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			if image_id in ['00090']:
				continue
			# skip all images after 150 if we are building the train set
			if is_train and int(image_id) >= 150:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
   # 添加元素到列表末尾 / Append element to list end
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
  # 获取长度 / Get length
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
  # 获取长度 / Get length
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
   # 添加元素到列表末尾 / Append element to list end
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# train set
# 定义数据集 / Define dataset
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
# define image id
image_id = 1
# load the image
image = train_set.load_image(image_id)
# load the masks and the class ids
mask, class_ids = train_set.load_mask(image_id)
# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, train_set.class_names)
```

---

➡️ **Next / 下一步**: File 7 of 9

---

### Train Rcnn

# 07 — Train Rcnn / 卷积神经网络

**Chapter 26 — File 7 of 9 / 第26章 — 第7个文件（共9个）**

---

## Summary / 总结

This script demonstrates **fit a mask rcnn on the kangaroo dataset**.

本脚本演示 **fit a mask rcnn on the kangaroo dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — fit a mask rcnn on the kangaroo dataset

```python
from os import listdir
from xml.etree import ElementTree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
```

---
## Step 2 — class that defines and loads the kangaroo dataset

```python
# 定义数据集 / Define dataset
class KangarooDataset(Dataset):
```

---
## Step 3 — load the dataset definitions

```python
def load_dataset(self, dataset_dir, is_train=True):
```

---
## Step 4 — define one class

```python
self.add_class("dataset", 1, "kangaroo")
```

---
## Step 5 — define data locations

```python
images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
```

---
## Step 6 — find all images

```python
for filename in listdir(images_dir):
```

---
## Step 7 — extract image id

```python
image_id = filename[:-4]
```

---
## Step 8 — skip bad images

```python
if image_id in ['00090']:
				continue
```

---
## Step 9 — skip all images after 150 if we are building the train set

```python
if is_train and int(image_id) >= 150:
				continue
```

---
## Step 10 — skip all images before 150 if we are building the test/val set

```python
if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
```

---
## Step 11 — add to dataset

```python
self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
```

---
## Step 12 — extract bounding boxes from an annotation file

```python
def extract_boxes(self, filename):
```

---
## Step 13 — load and parse the file

```python
tree = ElementTree.parse(filename)
```

---
## Step 14 — get the root of the document

```python
root = tree.getroot()
```

---
## Step 15 — extract each bounding box

```python
boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
   # 添加元素到列表末尾 / Append element to list end
			boxes.append(coors)
```

---
## Step 16 — extract image dimensions

```python
width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height
```

---
## Step 17 — load the masks for an image

```python
def load_mask(self, image_id):
```

---
## Step 18 — get details of image

```python
info = self.image_info[image_id]
```

---
## Step 19 — define box file location

```python
path = info['annotation']
```

---
## Step 20 — load XML

```python
boxes, w, h = self.extract_boxes(path)
```

---
## Step 21 — create one array for all masks, each on a different channel

```python
# 获取长度 / Get length
masks = zeros([h, w, len(boxes)], dtype='uint8')
```

---
## Step 22 — create masks

```python
class_ids = list()
  # 获取长度 / Get length
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
   # 添加元素到列表末尾 / Append element to list end
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')
```

---
## Step 23 — load an image reference

```python
def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']
```

---
## Step 24 — define a configuration for the model

```python
class KangarooConfig(Config):
```

---
## Step 25 — define the name of the configuration

```python
NAME = "kangaroo_cfg"
```

---
## Step 26 — number of classes (background + kangaroo)

```python
NUM_CLASSES = 1 + 1
```

---
## Step 27 — number of training steps per epoch

```python
STEPS_PER_EPOCH = 131
```

---
## Step 28 — prepare train set

```python
# 定义数据集 / Define dataset
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
# 打印输出 / Print output
print('Train: %d' % len(train_set.image_ids))
```

---
## Step 29 — prepare test/val set

```python
# 定义数据集 / Define dataset
test_set = KangarooDataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
# 打印输出 / Print output
print('Test: %d' % len(test_set.image_ids))
```

---
## Step 30 — prepare config

```python
config = KangarooConfig()
config.display()
```

---
## Step 31 — define the model

```python
model = MaskRCNN(mode='training', model_dir='./', config=config)
```

---
## Step 32 — load weights (mscoco) and exclude the output layers

```python
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
```

---
## Step 33 — train weights (output layers or 'heads')

```python
# 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
```

---
## Learning Notes / 学习笔记

- **概念**: fit a mask rcnn on the kangaroo dataset 是机器学习中的常用技术。  
  *fit a mask rcnn on the kangaroo dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Rcnn / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# fit a mask rcnn on the kangaroo dataset
from os import listdir
from xml.etree import ElementTree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# class that defines and loads the kangaroo dataset
# 定义数据集 / Define dataset
class KangarooDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "kangaroo")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			if image_id in ['00090']:
				continue
			# skip all images after 150 if we are building the train set
			if is_train and int(image_id) >= 150:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
   # 添加元素到列表末尾 / Append element to list end
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
  # 获取长度 / Get length
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
  # 获取长度 / Get length
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
   # 添加元素到列表末尾 / Append element to list end
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# define a configuration for the model
class KangarooConfig(Config):
	# define the name of the configuration
	NAME = "kangaroo_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 1
	# number of training steps per epoch
	STEPS_PER_EPOCH = 131

# prepare train set
# 定义数据集 / Define dataset
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
# 打印输出 / Print output
print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
# 定义数据集 / Define dataset
test_set = KangarooDataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
# 打印输出 / Print output
print('Test: %d' % len(test_set.image_ids))
# prepare config
config = KangarooConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
# 切换到训练模式（启用Dropout等） / Switch to training mode (enable Dropout, etc.)
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
```

---

➡️ **Next / 下一步**: File 8 of 9

---

### Evaluate Rcnn

# 08 — Evaluate Rcnn / 卷积神经网络

**Chapter 26 — File 8 of 9 / 第26章 — 第8个文件（共9个）**

---

## Summary / 总结

This script demonstrates **evaluate the mask rcnn model on the kangaroo dataset**.

本脚本演示 **evaluate the mask rcnn model on the kangaroo dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — evaluate the mask rcnn model on the kangaroo dataset

```python
from os import listdir
from xml.etree import ElementTree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
```

---
## Step 2 — class that defines and loads the kangaroo dataset

```python
# 定义数据集 / Define dataset
class KangarooDataset(Dataset):
```

---
## Step 3 — load the dataset definitions

```python
def load_dataset(self, dataset_dir, is_train=True):
```

---
## Step 4 — define one class

```python
self.add_class("dataset", 1, "kangaroo")
```

---
## Step 5 — define data locations

```python
images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
```

---
## Step 6 — find all images

```python
for filename in listdir(images_dir):
```

---
## Step 7 — extract image id

```python
image_id = filename[:-4]
```

---
## Step 8 — skip bad images

```python
if image_id in ['00090']:
				continue
```

---
## Step 9 — skip all images after 150 if we are building the train set

```python
if is_train and int(image_id) >= 150:
				continue
```

---
## Step 10 — skip all images before 150 if we are building the test/val set

```python
if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
```

---
## Step 11 — add to dataset

```python
self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
```

---
## Step 12 — extract bounding boxes from an annotation file

```python
def extract_boxes(self, filename):
```

---
## Step 13 — load and parse the file

```python
tree = ElementTree.parse(filename)
```

---
## Step 14 — get the root of the document

```python
root = tree.getroot()
```

---
## Step 15 — extract each bounding box

```python
boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
   # 添加元素到列表末尾 / Append element to list end
			boxes.append(coors)
```

---
## Step 16 — extract image dimensions

```python
width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height
```

---
## Step 17 — load the masks for an image

```python
def load_mask(self, image_id):
```

---
## Step 18 — get details of image

```python
info = self.image_info[image_id]
```

---
## Step 19 — define box file location

```python
path = info['annotation']
```

---
## Step 20 — load XML

```python
boxes, w, h = self.extract_boxes(path)
```

---
## Step 21 — create one array for all masks, each on a different channel

```python
# 获取长度 / Get length
masks = zeros([h, w, len(boxes)], dtype='uint8')
```

---
## Step 22 — create masks

```python
class_ids = list()
  # 获取长度 / Get length
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
   # 添加元素到列表末尾 / Append element to list end
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')
```

---
## Step 23 — load an image reference

```python
def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']
```

---
## Step 24 — define the prediction configuration

```python
class PredictionConfig(Config):
```

---
## Step 25 — define the name of the configuration

```python
NAME = "kangaroo_cfg"
```

---
## Step 26 — number of classes (background + kangaroo)

```python
NUM_CLASSES = 1 + 1
```

---
## Step 27 — simplify GPU config

```python
GPU_COUNT = 1
	IMAGES_PER_GPU = 1
```

---
## Step 28 — calculate the mAP for a model on a given dataset

```python
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
```

---
## Step 29 — load image, bounding boxes and masks for the image id

```python
image, _, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
```

---
## Step 30 — convert pixel values (e.g. center)

```python
scaled_image = mold_image(image, cfg)
```

---
## Step 31 — convert image into one sample

```python
sample = expand_dims(scaled_image, 0)
```

---
## Step 32 — make prediction

```python
yhat = model.detect(sample, verbose=0)
```

---
## Step 33 — extract results for first sample

```python
r = yhat[0]
```

---
## Step 34 — calculate statistics, including AP

```python
AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
```

---
## Step 35 — store

```python
# 添加元素到列表末尾 / Append element to list end
APs.append(AP)
```

---
## Step 36 — calculate the mean AP across all images

```python
mAP = mean(APs)
	return mAP
```

---
## Step 37 — load the train dataset

```python
# 定义数据集 / Define dataset
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
# 打印输出 / Print output
print('Train: %d' % len(train_set.image_ids))
```

---
## Step 38 — load the test dataset

```python
# 定义数据集 / Define dataset
test_set = KangarooDataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
# 打印输出 / Print output
print('Test: %d' % len(test_set.image_ids))
```

---
## Step 39 — create config

```python
cfg = PredictionConfig()
```

---
## Step 40 — define the model

```python
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
```

---
## Step 41 — load model weights

```python
model.load_weights('mask_rcnn_kangaroo_cfg_0005.h5', by_name=True)
```

---
## Step 42 — evaluate model on training dataset

```python
train_mAP = evaluate_model(train_set, model, cfg)
# 打印输出 / Print output
print("Train mAP: %.3f" % train_mAP)
```

---
## Step 43 — evaluate model on test dataset

```python
test_mAP = evaluate_model(test_set, model, cfg)
# 打印输出 / Print output
print("Test mAP: %.3f" % test_mAP)
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate the mask rcnn model on the kangaroo dataset 是机器学习中的常用技术。  
  *evaluate the mask rcnn model on the kangaroo dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Rcnn / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# evaluate the mask rcnn model on the kangaroo dataset
from os import listdir
from xml.etree import ElementTree
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

# class that defines and loads the kangaroo dataset
# 定义数据集 / Define dataset
class KangarooDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "kangaroo")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			if image_id in ['00090']:
				continue
			# skip all images after 150 if we are building the train set
			if is_train and int(image_id) >= 150:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
   # 添加元素到列表末尾 / Append element to list end
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
  # 获取长度 / Get length
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
  # 获取长度 / Get length
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
   # 添加元素到列表末尾 / Append element to list end
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "kangaroo_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, _, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
  # 添加元素到列表末尾 / Append element to list end
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP

# load the train dataset
# 定义数据集 / Define dataset
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
# 打印输出 / Print output
print('Train: %d' % len(train_set.image_ids))
# load the test dataset
# 定义数据集 / Define dataset
test_set = KangarooDataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
# 打印输出 / Print output
print('Test: %d' % len(test_set.image_ids))
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model.load_weights('mask_rcnn_kangaroo_cfg_0005.h5', by_name=True)
# evaluate model on training dataset
train_mAP = evaluate_model(train_set, model, cfg)
# 打印输出 / Print output
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, cfg)
# 打印输出 / Print output
print("Test mAP: %.3f" % test_mAP)
```

---

➡️ **Next / 下一步**: File 9 of 9

---

### Predict Rcnn



---

### Chapter Summary / 章节总结

# Chapter 26 Summary / 第26章总结

## Theme / 主题: Chapter 26 / Chapter 26

This chapter contains **9 code files** demonstrating chapter 26.

本章包含 **9 个代码文件**，演示Chapter 26。

---
## Evolution / 演化路线

  1. `01_extract_annotation.ipynb` — Extract Annotation
  2. `02_dataset_object.ipynb` — Dataset Object
  3. `03_plot_photo_with_mask.ipynb` — Plot Photo With Mask
  4. `04_plot_multiple_photos_with_mask.ipynb` — Plot Multiple Photos With Mask
  5. `05_summarize_image_paths.ipynb` — Summarize Image Paths
  6. `06_plot_photo_mask_builtin.ipynb` — Plot Photo Mask Builtin
  7. `07_train_rcnn.ipynb` — Train Rcnn
  8. `08_evaluate_rcnn.ipynb` — Evaluate Rcnn
  9. `09_predict_rcnn.ipynb` — Predict Rcnn

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 26) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 26）是机器学习流水线中的基础构建块。

---
