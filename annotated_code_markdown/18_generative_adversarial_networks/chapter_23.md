# 生成对抗网络 / Generative Adversarial Networks
## Chapter 23

---

### Prepare Dataset

# 01 — Prepare Dataset / 数据准备

**Chapter 23 — File 1 of 6 / 第23章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **load, split and scale the maps dataset ready for training**.

本脚本演示 **load, split and scale the maps dataset ready for training**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — load, split and scale the maps dataset ready for training

```python
from os import listdir
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import savez_compressed
```

---
## Step 2 — load all images in a directory into memory

```python
def load_images(path, size=(256,512)):
	src_list, tar_list = list(), list()
```

---
## Step 3 — enumerate filenames in directory, assume all are images

```python
for filename in listdir(path):
```

---
## Step 4 — load and resize the image

```python
pixels = load_img(path + filename, target_size=size)
```

---
## Step 5 — convert to numpy array

```python
pixels = img_to_array(pixels)
```

---
## Step 6 — split into satellite and map

```python
sat_img, map_img = pixels[:, :256], pixels[:, 256:]
  # 添加元素到列表末尾 / Append element to list end
		src_list.append(sat_img)
  # 添加元素到列表末尾 / Append element to list end
		tar_list.append(map_img)
	return [asarray(src_list), asarray(tar_list)]
```

---
## Step 7 — dataset path

```python
path = 'maps/train/'
```

---
## Step 8 — load dataset

```python
[src_images, tar_images] = load_images(path)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Loaded: ', src_images.shape, tar_images.shape)
```

---
## Step 9 — save as compressed numpy array

```python
filename = 'maps_256.npz'
savez_compressed(filename, src_images, tar_images)
# 打印输出 / Print output
print('Saved dataset: ', filename)
```

---
## Learning Notes / 学习笔记

- **概念**: load, split and scale the maps dataset ready for training 是机器学习中的常用技术。  
  *load, split and scale the maps dataset ready for training is a common technique in machine learning.*

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
# Prepare Dataset / 数据准备
# Complete Code / 完整代码
# ===============================

# load, split and scale the maps dataset ready for training
from os import listdir
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import savez_compressed

# load all images in a directory into memory
def load_images(path, size=(256,512)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
  # 添加元素到列表末尾 / Append element to list end
		src_list.append(sat_img)
  # 添加元素到列表末尾 / Append element to list end
		tar_list.append(map_img)
	return [asarray(src_list), asarray(tar_list)]

# dataset path
path = 'maps/train/'
# load dataset
[src_images, tar_images] = load_images(path)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'maps_256.npz'
savez_compressed(filename, src_images, tar_images)
# 打印输出 / Print output
print('Saved dataset: ', filename)
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Load Plot Dataset

# 02 — Load Plot Dataset / 02 Load Plot Dataset

**Chapter 23 — File 2 of 6 / 第23章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **load the prepared dataset**.

本脚本演示 **load the prepared dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — load the prepared dataset

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import load
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load the dataset

```python
data = load('maps_256.npz')
src_images, tar_images = data['arr_0'], data['arr_1']
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Loaded: ', src_images.shape, tar_images.shape)
```

---
## Step 3 — plot source images

```python
n_samples = 3
# 生成整数序列 / Generate integer sequence
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
 # 转换数据类型 / Convert data type
	pyplot.imshow(src_images[i].astype('uint8'))
```

---
## Step 4 — plot target image

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + n_samples + i)
	pyplot.axis('off')
 # 转换数据类型 / Convert data type
	pyplot.imshow(tar_images[i].astype('uint8'))
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load the prepared dataset 是机器学习中的常用技术。  
  *load the prepared dataset is a common technique in machine learning.*

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
# Load Plot Dataset / 02 Load Plot Dataset
# Complete Code / 完整代码
# ===============================

# load the prepared dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import load
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load the dataset
data = load('maps_256.npz')
src_images, tar_images = data['arr_0'], data['arr_1']
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Loaded: ', src_images.shape, tar_images.shape)
# plot source images
n_samples = 3
# 生成整数序列 / Generate integer sequence
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
 # 转换数据类型 / Convert data type
	pyplot.imshow(src_images[i].astype('uint8'))
# plot target image
# 生成整数序列 / Generate integer sequence
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + n_samples + i)
	pyplot.axis('off')
 # 转换数据类型 / Convert data type
	pyplot.imshow(tar_images[i].astype('uint8'))
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Train Pix2Pix



---

### Inference Pix2Pix

# 04 — Inference Pix2Pix / 04 Inference Pix2Pix

**Chapter 23 — File 4 of 6 / 第23章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of loading a pix2pix model and using it for image to image translation**.

本脚本演示 **example of loading a pix2pix model and using it for image to image translation**。

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
## Step 1 — example of loading a pix2pix model and using it for image to image translation

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import load
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
```

---
## Step 2 — load and prepare training images

```python
def load_real_samples(filename):
```

---
## Step 3 — load the compressed arrays

```python
data = load(filename)
```

---
## Step 4 — unpack the arrays

```python
X1, X2 = data['arr_0'], data['arr_1']
```

---
## Step 5 — scale from [0,255] to [-1,1]

```python
X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
```

---
## Step 6 — plot source, generated and target images

```python
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
```

---
## Step 7 — scale from [-1,1] to [0,1]

```python
images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
```

---
## Step 8 — plot images row by row

```python
# 获取长度 / Get length
for i in range(len(images)):
```

---
## Step 9 — define subplot

```python
pyplot.subplot(1, 3, 1 + i)
```

---
## Step 10 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 11 — plot raw pixel data

```python
pyplot.imshow(images[i])
```

---
## Step 12 — show title

```python
pyplot.title(titles[i])
	pyplot.show()
```

---
## Step 13 — load dataset

```python
[X1, X2] = load_real_samples('maps_256.npz')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Loaded', X1.shape, X2.shape)
```

---
## Step 14 — load model

```python
# 从文件加载模型 / Load model from file
model = load_model('model_109600.h5')
```

---
## Step 15 — select random example

```python
# 获取长度 / Get length
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
```

---
## Step 16 — generate image from source

```python
# 用模型做预测 / Make predictions with model
gen_image = model.predict(src_image)
```

---
## Step 17 — plot all three images

```python
plot_images(src_image, gen_image, tar_image)
```

---
## Learning Notes / 学习笔记

- **概念**: example of loading a pix2pix model and using it for image to image translation 是机器学习中的常用技术。  
  *example of loading a pix2pix model and using it for image to image translation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Inference Pix2Pix / 04 Inference Pix2Pix
# Complete Code / 完整代码
# ===============================

# example of loading a pix2pix model and using it for image to image translation
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import load
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint

# load and prepare training images
def load_real_samples(filename):
	# load the compressed arrays
	data = load(filename)
	# unpack the arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	# plot images row by row
 # 获取长度 / Get length
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 3, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		pyplot.title(titles[i])
	pyplot.show()

# load dataset
[X1, X2] = load_real_samples('maps_256.npz')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Loaded', X1.shape, X2.shape)
# load model
# 从文件加载模型 / Load model from file
model = load_model('model_109600.h5')
# select random example
# 获取长度 / Get length
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
# generate image from source
# 用模型做预测 / Make predictions with model
gen_image = model.predict(src_image)
# plot all three images
plot_images(src_image, gen_image, tar_image)
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Translate Single Image

# 05 — Translate Single Image / 图像处理

**Chapter 23 — File 5 of 6 / 第23章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of loading a pix2pix model and using it for one-off image translation**.

本脚本演示 **example of loading a pix2pix model and using it for one-off image translation**。

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
## Step 1 — example of loading a pix2pix model and using it for one-off image translation

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load an image

```python
def load_image(filename, size=(256,256)):
```

---
## Step 3 — load image with the preferred size

```python
pixels = load_img(filename, target_size=size)
```

---
## Step 4 — convert to numpy array

```python
pixels = img_to_array(pixels)
```

---
## Step 5 — scale from [0,255] to [-1,1]

```python
pixels = (pixels - 127.5) / 127.5
```

---
## Step 6 — reshape to 1 sample

```python
pixels = expand_dims(pixels, 0)
	return pixels
```

---
## Step 7 — load source image

```python
src_image = load_image('satellite.jpg')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Loaded', src_image.shape)
```

---
## Step 8 — load model

```python
# 从文件加载模型 / Load model from file
model = load_model('model_109600.h5')
```

---
## Step 9 — generate image from source

```python
# 用模型做预测 / Make predictions with model
gen_image = model.predict(src_image)
```

---
## Step 10 — scale from [-1,1] to [0,1]

```python
gen_image = (gen_image + 1) / 2.0
```

---
## Step 11 — plot the image

```python
pyplot.imshow(gen_image[0])
pyplot.axis('off')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of loading a pix2pix model and using it for one-off image translation 是机器学习中的常用技术。  
  *example of loading a pix2pix model and using it for one-off image translation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Translate Single Image / 图像处理
# Complete Code / 完整代码
# ===============================

# example of loading a pix2pix model and using it for one-off image translation
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# load an image
def load_image(filename, size=(256,256)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels

# load source image
src_image = load_image('satellite.jpg')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Loaded', src_image.shape)
# load model
# 从文件加载模型 / Load model from file
model = load_model('model_109600.h5')
# generate image from source
# 用模型做预测 / Make predictions with model
gen_image = model.predict(src_image)
# scale from [-1,1] to [0,1]
gen_image = (gen_image + 1) / 2.0
# plot the image
pyplot.imshow(gen_image[0])
pyplot.axis('off')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Train Pix2Pix Reverse



---

### Chapter Summary / 章节总结

# Chapter 23 Summary / 第23章总结

## Theme / 主题: Chapter 23 / Chapter 23

This chapter contains **6 code files** demonstrating chapter 23.

本章包含 **6 个代码文件**，演示Chapter 23。

---
## Evolution / 演化路线

  1. `01_prepare_dataset.ipynb` — Prepare Dataset
  2. `02_load_plot_dataset.ipynb` — Load Plot Dataset
  3. `03_train_pix2pix.ipynb` — Train Pix2Pix
  4. `04_inference_pix2pix.ipynb` — Inference Pix2Pix
  5. `05_translate_single_image.ipynb` — Translate Single Image
  6. `06_train_pix2pix_reverse.ipynb` — Train Pix2Pix Reverse

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 23) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 23）是机器学习流水线中的基础构建块。

---
