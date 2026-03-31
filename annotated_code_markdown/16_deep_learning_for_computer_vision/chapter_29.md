# 计算机视觉深度学习 / Deep Learning for Computer Vision
## Chapter 29

---

### Check Vggface Version

# 01 — Check Vggface Version / 库版本信息

**Chapter 29 — File 1 of 7 / 第29章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **check version of keras_vggface**.

本脚本演示 **check version of keras_vggface**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — check version of keras_vggface

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
import keras_vggface
```

---
## Step 2 — print version

```python
# 打印输出 / Print output
print(keras_vggface.__version__)
```

---
## Learning Notes / 学习笔记

- **概念**: check version of keras_vggface 是机器学习中的常用技术。  
  *check version of keras_vggface is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Check Vggface Version / 库版本信息
# Complete Code / 完整代码
# ===============================

# check version of keras_vggface
# 导入Keras高级神经网络API / Import Keras high-level neural network API
import keras_vggface
# print version
# 打印输出 / Print output
print(keras_vggface.__version__)
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Check Mtcnn Version

# 02 — Check Mtcnn Version / 库版本信息

**Chapter 29 — File 2 of 7 / 第29章 — 第2个文件（共7个）**

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
# 打印输出 / Print output
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
# 打印输出 / Print output
print(mtcnn.__version__)
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Face Detection

# 03 — Face Detection / 人脸识别

**Chapter 29 — File 3 of 7 / 第29章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of face detection with mtcnn**.

本脚本演示 **example of face detection with mtcnn**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
## Step 1 — example of face detection with mtcnn

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from PIL import Image
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from mtcnn.mtcnn import MTCNN
```

---
## Step 2 — extract a single face from a given photograph

```python
def extract_face(filename, required_size=(224, 224)):
```

---
## Step 3 — load image from file

```python
pixels = pyplot.imread(filename)
```

---
## Step 4 — create the detector, using default weights

```python
detector = MTCNN()
```

---
## Step 5 — detect faces in the image

```python
results = detector.detect_faces(pixels)
```

---
## Step 6 — extract the bounding box from the first face

```python
x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
```

---
## Step 7 — extract the face

```python
face = pixels[y1:y2, x1:x2]
```

---
## Step 8 — resize pixels to the model size

```python
image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
```

---
## Step 9 — load the photo and extract the face

```python
pixels = extract_face('sharon_stone1.jpg')
```

---
## Step 10 — plot the extracted face

```python
pyplot.imshow(pixels)
```

---
## Step 11 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of face detection with mtcnn 是机器学习中的常用技术。  
  *example of face detection with mtcnn is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Face Detection / 人脸识别
# Complete Code / 完整代码
# ===============================

# example of face detection with mtcnn
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from PIL import Image
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from mtcnn.mtcnn import MTCNN

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# load the photo and extract the face
pixels = extract_face('sharon_stone1.jpg')
# plot the extracted face
pyplot.imshow(pixels)
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Vggface Model

# 04 — Vggface Model / 人脸识别

**Chapter 29 — File 4 of 7 / 第29章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of creating a face embedding**.

本脚本演示 **example of creating a face embedding**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of creating a face embedding

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras_vggface.vggface import VGGFace
```

---
## Step 2 — create a vggface2 model

```python
model = VGGFace(model='resnet50')
```

---
## Step 3 — summarize input and output shape

```python
# 打印输出 / Print output
print('Inputs: %s' % model.inputs)
# 打印输出 / Print output
print('Outputs: %s' % model.outputs)
```

---
## Learning Notes / 学习笔记

- **概念**: example of creating a face embedding 是机器学习中的常用技术。  
  *example of creating a face embedding is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Vggface Model / 人脸识别
# Complete Code / 完整代码
# ===============================

# example of creating a face embedding
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras_vggface.vggface import VGGFace
# create a vggface2 model
model = VGGFace(model='resnet50')
# summarize input and output shape
# 打印输出 / Print output
print('Inputs: %s' % model.inputs)
# 打印输出 / Print output
print('Outputs: %s' % model.outputs)
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Vggface Face Identification Stone



---

### Vggface Face Identification Tatum



---

### Face Verification

# 07 — Face Verification / 人脸识别

**Chapter 29 — File 7 of 7 / 第29章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **face verification with the VGGFace2 model**.

本脚本演示 **face verification with the VGGFace2 model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — face verification with the VGGFace2 model

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from PIL import Image
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras_vggface.vggface import VGGFace
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras_vggface.utils import preprocess_input
```

---
## Step 2 — extract a single face from a given photograph

```python
def extract_face(filename, required_size=(224, 224)):
```

---
## Step 3 — load image from file

```python
pixels = pyplot.imread(filename)
```

---
## Step 4 — create the detector, using default weights

```python
detector = MTCNN()
```

---
## Step 5 — detect faces in the image

```python
results = detector.detect_faces(pixels)
```

---
## Step 6 — extract the bounding box from the first face

```python
x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
```

---
## Step 7 — extract the face

```python
face = pixels[y1:y2, x1:x2]
```

---
## Step 8 — resize pixels to the model size

```python
image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
```

---
## Step 9 — extract faces and calculate face embeddings for a list of photo files

```python
def get_embeddings(filenames):
```

---
## Step 10 — extract faces

```python
faces = [extract_face(f) for f in filenames]
```

---
## Step 11 — convert into an array of samples

```python
samples = asarray(faces, 'float32')
```

---
## Step 12 — prepare the face for the model, e.g. center pixels

```python
samples = preprocess_input(samples, version=2)
```

---
## Step 13 — create a vggface model

```python
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
```

---
## Step 14 — perform prediction

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(samples)
	return yhat
```

---
## Step 15 — determine if a candidate face is a match for a known face

```python
def is_match(known_embedding, candidate_embedding, thresh=0.5):
```

---
## Step 16 — calculate distance between embeddings

```python
score = cosine(known_embedding, candidate_embedding)
	if score <= thresh:
  # 打印输出 / Print output
		print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
	else:
  # 打印输出 / Print output
		print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
```

---
## Step 17 — define filenames

```python
filenames = ['sharon_stone1.jpg', 'sharon_stone2.jpg', 'sharon_stone3.jpg', 'channing_tatum.jpg']
```

---
## Step 18 — get embeddings file filenames

```python
embeddings = get_embeddings(filenames)
```

---
## Step 19 — define sharon stone

```python
sharon_id = embeddings[0]
```

---
## Step 20 — verify known photos of sharon

```python
# 打印输出 / Print output
print('Positive Tests')
is_match(embeddings[0], embeddings[1])
is_match(embeddings[0], embeddings[2])
```

---
## Step 21 — verify known photos of other people

```python
# 打印输出 / Print output
print('Negative Tests')
is_match(embeddings[0], embeddings[3])
```

---
## Learning Notes / 学习笔记

- **概念**: face verification with the VGGFace2 model 是机器学习中的常用技术。  
  *face verification with the VGGFace2 model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `matplotlib` | 绑图库 | Plotting library |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Face Verification / 人脸识别
# Complete Code / 完整代码
# ===============================

# face verification with the VGGFace2 model
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from PIL import Image
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras_vggface.vggface import VGGFace
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras_vggface.utils import preprocess_input

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
	# extract faces
	faces = [extract_face(f) for f in filenames]
	# convert into an array of samples
	samples = asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	# create a vggface model
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
 # 用模型做预测 / Make predictions with model
	yhat = model.predict(samples)
	return yhat

# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
	# calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding)
	if score <= thresh:
  # 打印输出 / Print output
		print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
	else:
  # 打印输出 / Print output
		print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))

# define filenames
filenames = ['sharon_stone1.jpg', 'sharon_stone2.jpg', 'sharon_stone3.jpg', 'channing_tatum.jpg']
# get embeddings file filenames
embeddings = get_embeddings(filenames)
# define sharon stone
sharon_id = embeddings[0]
# verify known photos of sharon
# 打印输出 / Print output
print('Positive Tests')
is_match(embeddings[0], embeddings[1])
is_match(embeddings[0], embeddings[2])
# verify known photos of other people
# 打印输出 / Print output
print('Negative Tests')
is_match(embeddings[0], embeddings[3])
```

---

### Chapter Summary / 章节总结

# Chapter 29 Summary / 第29章总结

## Theme / 主题: Chapter 29 / Chapter 29

This chapter contains **7 code files** demonstrating chapter 29.

本章包含 **7 个代码文件**，演示Chapter 29。

---
## Evolution / 演化路线

  1. `01_check_vggface_version.ipynb` — Check Vggface Version
  2. `02_check_mtcnn_version.ipynb` — Check Mtcnn Version
  3. `03_face_detection.ipynb` — Face Detection
  4. `04_vggface_model.ipynb` — Vggface Model
  5. `05_vggface_face_identification_stone.ipynb` — Vggface Face Identification Stone
  6. `06_vggface_face_identification_tatum.ipynb` — Vggface Face Identification Tatum
  7. `07_face_verification.ipynb` — Face Verification

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 29) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 29）是机器学习流水线中的基础构建块。

---
