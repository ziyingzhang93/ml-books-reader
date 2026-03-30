# GAN
## Chapter 12

---

### Inception Score Confident

# 01 — Inception Score Confident / 01 Inception Score Confident

**Chapter 12 — File 1 of 4 / 第12章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **calculate inception score in numpy**.

本脚本演示 **calculate inception score in numpy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — calculate inception score in numpy

```python
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp
```

---
## Step 2 — calculate the inception score for p(y|x)

```python
def calculate_inception_score(p_yx, eps=1E-16):
```

---
## Step 3 — calculate p(y)

```python
p_y = expand_dims(p_yx.mean(axis=0), 0)
```

---
## Step 4 — kl divergence for each image

```python
kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
```

---
## Step 5 — sum over classes

```python
sum_kl_d = kl_d.sum(axis=1)
```

---
## Step 6 — average over images

```python
avg_kl_d = mean(sum_kl_d)
```

---
## Step 7 — undo the logs

```python
is_score = exp(avg_kl_d)
	return is_score
```

---
## Step 8 — conditional probabilities for high quality images

```python
p_yx = asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
score = calculate_inception_score(p_yx)
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: calculate inception score in numpy 是机器学习中的常用技术。  
  *calculate inception score in numpy is a common technique in machine learning.*

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
# Inception Score Confident / 01 Inception Score Confident
# Complete Code / 完整代码
# ===============================

# calculate inception score in numpy
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp

# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
	# calculate p(y)
	p_y = expand_dims(p_yx.mean(axis=0), 0)
	# kl divergence for each image
	kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
	# sum over classes
	sum_kl_d = kl_d.sum(axis=1)
	# average over images
	avg_kl_d = mean(sum_kl_d)
	# undo the logs
	is_score = exp(avg_kl_d)
	return is_score

# conditional probabilities for high quality images
p_yx = asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
score = calculate_inception_score(p_yx)
print(score)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Inception Score Uniform

# 02 — Inception Score Uniform / 02 Inception Score Uniform

**Chapter 12 — File 2 of 4 / 第12章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **calculate inception score in numpy**.

本脚本演示 **calculate inception score in numpy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — calculate inception score in numpy

```python
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp
```

---
## Step 2 — calculate the inception score for p(y|x)

```python
def calculate_inception_score(p_yx, eps=1E-16):
```

---
## Step 3 — calculate p(y)

```python
p_y = expand_dims(p_yx.mean(axis=0), 0)
```

---
## Step 4 — kl divergence for each image

```python
kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
```

---
## Step 5 — sum over classes

```python
sum_kl_d = kl_d.sum(axis=1)
```

---
## Step 6 — average over images

```python
avg_kl_d = mean(sum_kl_d)
```

---
## Step 7 — undo the logs

```python
is_score = exp(avg_kl_d)
	return is_score
```

---
## Step 8 — conditional probabilities for low quality images

```python
p_yx = asarray([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])
score = calculate_inception_score(p_yx)
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: calculate inception score in numpy 是机器学习中的常用技术。  
  *calculate inception score in numpy is a common technique in machine learning.*

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
# Inception Score Uniform / 02 Inception Score Uniform
# Complete Code / 完整代码
# ===============================

# calculate inception score in numpy
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp

# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
	# calculate p(y)
	p_y = expand_dims(p_yx.mean(axis=0), 0)
	# kl divergence for each image
	kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
	# sum over classes
	sum_kl_d = kl_d.sum(axis=1)
	# average over images
	avg_kl_d = mean(sum_kl_d)
	# undo the logs
	is_score = exp(avg_kl_d)
	return is_score

# conditional probabilities for low quality images
p_yx = asarray([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])
score = calculate_inception_score(p_yx)
print(score)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Inception Score Cifar10

# 04 — Inception Score Cifar10 / 04 Inception Score Cifar10

**Chapter 12 — File 4 of 4 / 第12章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **calculate inception score for cifar-10 in Keras**.

本脚本演示 **calculate inception score for cifar-10 in Keras**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — calculate inception score for cifar-10 in Keras

```python
from math import floor
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray
```

---
## Step 2 — scale an array of images to a new size

```python
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
```

---
## Step 3 — resize with nearest neighbor interpolation

```python
new_image = resize(image, new_shape, 0)
```

---
## Step 4 — store

```python
images_list.append(new_image)
	return asarray(images_list)
```

---
## Step 5 — assumes images have any shape and pixels in [0,255]

```python
def calculate_inception_score(images, n_split=10, eps=1E-16):
```

---
## Step 6 — load inception v3 model

```python
model = InceptionV3()
```

---
## Step 7 — enumerate splits of images/predictions

```python
scores = list()
	n_part = floor(images.shape[0] / n_split)
	for i in range(n_split):
```

---
## Step 8 — retrieve images

```python
ix_start, ix_end = i * n_part, (i+1) * n_part
		subset = images[ix_start:ix_end]
```

---
## Step 9 — convert from uint8 to float32

```python
subset = subset.astype('float32')
```

---
## Step 10 — scale images to the required size

```python
subset = scale_images(subset, (299,299,3))
```

---
## Step 11 — pre-process images, scale to [-1,1]

```python
subset = preprocess_input(subset)
```

---
## Step 12 — predict p(y|x)

```python
p_yx = model.predict(subset)
```

---
## Step 13 — calculate p(y)

```python
p_y = expand_dims(p_yx.mean(axis=0), 0)
```

---
## Step 14 — calculate KL divergence using log probabilities

```python
kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
```

---
## Step 15 — sum over classes

```python
sum_kl_d = kl_d.sum(axis=1)
```

---
## Step 16 — average over images

```python
avg_kl_d = mean(sum_kl_d)
```

---
## Step 17 — undo the log

```python
is_score = exp(avg_kl_d)
```

---
## Step 18 — store

```python
scores.append(is_score)
```

---
## Step 19 — average across images

```python
is_avg, is_std = mean(scores), std(scores)
	return is_avg, is_std
```

---
## Step 20 — load cifar10 images

```python
(images, _), (_, _) = cifar10.load_data()
```

---
## Step 21 — shuffle images

```python
shuffle(images)
print('loaded', images.shape)
```

---
## Step 22 — calculate inception score

```python
is_avg, is_std = calculate_inception_score(images)
print('score', is_avg, is_std)
```

---
## Learning Notes / 学习笔记

- **概念**: calculate inception score for cifar-10 in Keras 是机器学习中的常用技术。  
  *calculate inception score for cifar-10 in Keras is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Inception Score Cifar10 / 04 Inception Score Cifar10
# Complete Code / 完整代码
# ===============================

# calculate inception score for cifar-10 in Keras
from math import floor
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
	# load inception v3 model
	model = InceptionV3()
	# enumerate splits of images/predictions
	scores = list()
	n_part = floor(images.shape[0] / n_split)
	for i in range(n_split):
		# retrieve images
		ix_start, ix_end = i * n_part, (i+1) * n_part
		subset = images[ix_start:ix_end]
		# convert from uint8 to float32
		subset = subset.astype('float32')
		# scale images to the required size
		subset = scale_images(subset, (299,299,3))
		# pre-process images, scale to [-1,1]
		subset = preprocess_input(subset)
		# predict p(y|x)
		p_yx = model.predict(subset)
		# calculate p(y)
		p_y = expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = mean(sum_kl_d)
		# undo the log
		is_score = exp(avg_kl_d)
		# store
		scores.append(is_score)
	# average across images
	is_avg, is_std = mean(scores), std(scores)
	return is_avg, is_std

# load cifar10 images
(images, _), (_, _) = cifar10.load_data()
# shuffle images
shuffle(images)
print('loaded', images.shape)
# calculate inception score
is_avg, is_std = calculate_inception_score(images)
print('score', is_avg, is_std)
```

---

### Chapter Summary

# Chapter 12 Summary / 第12章总结

## Theme / 主题: Chapter 12 / Chapter 12

This chapter contains **4 code files** demonstrating chapter 12.

本章包含 **4 个代码文件**，演示Chapter 12。

---
## Evolution / 演化路线

  1. `01_inception_score_confident.ipynb` — Inception Score Confident
  2. `02_inception_score_uniform.ipynb` — Inception Score Uniform
  3. `03_inception_score_keras.ipynb` — Inception Score Keras
  4. `04_inception_score_cifar10.ipynb` — Inception Score Cifar10

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 12) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 12）是机器学习流水线中的基础构建块。

---
