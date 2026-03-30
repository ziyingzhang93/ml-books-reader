# CV深度学习
## Chapter 30

---

### Facenet Model

# 01 — Facenet Model / 人脸识别

**Chapter 30 — File 1 of 7 / 第30章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of loading the keras facenet model**.

本脚本演示 **example of loading the keras facenet model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of loading the keras facenet model

```python
from keras.models import load_model
```

---
## Step 2 — load the model

```python
model = load_model('facenet_keras.h5')
```

---
## Step 3 — summarize input and output shape

```python
print(model.inputs)
print(model.outputs)
```

---
## Learning Notes / 学习笔记

- **概念**: example of loading the keras facenet model 是机器学习中的常用技术。  
  *example of loading the keras facenet model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Facenet Model / 人脸识别
# Complete Code / 完整代码
# ===============================

# example of loading the keras facenet model
from keras.models import load_model
# load the model
model = load_model('facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Check Mtcnn Version

# 02 — Check Mtcnn Version / 库版本信息

**Chapter 30 — File 2 of 7 / 第30章 — 第2个文件（共7个）**

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

➡️ **Next / 下一步**: File 3 of 7

---

### Extract Faces Dataset

# 04 — Extract Faces Dataset / 人脸识别

**Chapter 30 — File 4 of 7 / 第30章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **face detection for the 5 Celebrity Faces Dataset**.

本脚本演示 **face detection for the 5 Celebrity Faces Dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture


---
## Step 1 — face detection for the 5 Celebrity Faces Dataset

```python
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
```

---
## Step 2 — extract a single face from a given photograph

```python
def extract_face(filename, required_size=(160, 160)):
```

---
## Step 3 — load image from file

```python
image = Image.open(filename)
```

---
## Step 4 — convert to RGB, if needed

```python
image = image.convert('RGB')
```

---
## Step 5 — convert to array

```python
pixels = asarray(image)
```

---
## Step 6 — create the detector, using default weights

```python
detector = MTCNN()
```

---
## Step 7 — detect faces in the image

```python
results = detector.detect_faces(pixels)
```

---
## Step 8 — extract the bounding box from the first face

```python
x1, y1, width, height = results[0]['box']
```

---
## Step 9 — bug fix

```python
x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
```

---
## Step 10 — extract the face

```python
face = pixels[y1:y2, x1:x2]
```

---
## Step 11 — resize pixels to the model size

```python
image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
```

---
## Step 12 — load images and extract faces for all images in a directory

```python
def load_faces(directory):
	faces = list()
```

---
## Step 13 — enumerate files

```python
for filename in listdir(directory):
```

---
## Step 14 — path

```python
path = directory + filename
```

---
## Step 15 — get face

```python
face = extract_face(path)
```

---
## Step 16 — store

```python
faces.append(face)
	return faces
```

---
## Step 17 — load a dataset that contains one subdir for each class that in turn contains images

```python
def load_dataset(directory):
	X, y = list(), list()
```

---
## Step 18 — enumerate folders, on per class

```python
for subdir in listdir(directory):
```

---
## Step 19 — path

```python
path = directory + subdir + '/'
```

---
## Step 20 — skip any files that might be in the dir

```python
if not isdir(path):
			continue
```

---
## Step 21 — load all faces in the subdirectory

```python
faces = load_faces(path)
```

---
## Step 22 — create labels

```python
labels = [subdir for _ in range(len(faces))]
```

---
## Step 23 — summarize progress

```python
print('>loaded %d examples for class: %s' % (len(faces), subdir))
```

---
## Step 24 — store

```python
X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)
```

---
## Step 25 — load train dataset

```python
trainX, trainy = load_dataset('5-celebrity-faces-dataset/train/')
print(trainX.shape, trainy.shape)
```

---
## Step 26 — load test dataset

```python
testX, testy = load_dataset('5-celebrity-faces-dataset/val/')
```

---
## Step 27 — save arrays to one file in compressed format

```python
savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)
```

---
## Learning Notes / 学习笔记

- **概念**: face detection for the 5 Celebrity Faces Dataset 是机器学习中的常用技术。  
  *face detection for the 5 Celebrity Faces Dataset is a common technique in machine learning.*

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
# Extract Faces Dataset / 人脸识别
# Complete Code / 完整代码
# ===============================

# face detection for the 5 Celebrity Faces Dataset
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

# load train dataset
trainX, trainy = load_dataset('5-celebrity-faces-dataset/train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('5-celebrity-faces-dataset/val/')
# save arrays to one file in compressed format
savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Face Classification

# 06 — Face Classification / 分类

**Chapter 30 — File 6 of 7 / 第30章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **develop a classifier for the 5 Celebrity Faces Dataset**.

本脚本演示 **develop a classifier for the 5 Celebrity Faces Dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — develop a classifier for the 5 Celebrity Faces Dataset

```python
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
```

---
## Step 2 — load dataset

```python
data = load('5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
```

---
## Step 3 — normalize input vectors

```python
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
```

---
## Step 4 — label encode targets

```python
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
```

---
## Step 5 — fit model

```python
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
```

---
## Step 6 — predict

```python
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
```

---
## Step 7 — score

```python
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
```

---
## Step 8 — summarize

```python
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
```

---
## Learning Notes / 学习笔记

- **概念**: develop a classifier for the 5 Celebrity Faces Dataset 是机器学习中的常用技术。  
  *develop a classifier for the 5 Celebrity Faces Dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SVM` | 支持向量机 | Support Vector Machine |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Face Classification / 分类
# Complete Code / 完整代码
# ===============================

# develop a classifier for the 5 Celebrity Faces Dataset
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
# load dataset
data = load('5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Chapter Summary

# Chapter 30 Summary / 第30章总结

## Theme / 主题: Chapter 30 / Chapter 30

This chapter contains **7 code files** demonstrating chapter 30.

本章包含 **7 个代码文件**，演示Chapter 30。

---
## Evolution / 演化路线

  1. `01_facenet_model.ipynb` — Facenet Model
  2. `02_check_mtcnn_version.ipynb` — Check Mtcnn Version
  3. `03_extract_faces.ipynb` — Extract Faces
  4. `04_extract_faces_dataset.ipynb` — Extract Faces Dataset
  5. `05_predict_face_embeddings.ipynb` — Predict Face Embeddings
  6. `06_face_classification.ipynb` — Face Classification
  7. `07_random_face_identity_classification.ipynb` — Random Face Identity Classification

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 30) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 30）是机器学习流水线中的基础构建块。

---
