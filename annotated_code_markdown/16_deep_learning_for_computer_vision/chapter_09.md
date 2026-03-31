# 计算机视觉深度学习 / Deep Learning for Computer Vision
## Chapter 09

---

### Horizontal Shift Augmentation

# 01 — Horizontal Shift Augmentation / 01 Horizontal Shift Augmentation

**Chapter 09 — File 1 of 6 / 第09章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of horizontal shift image augmentation**.

本脚本演示 **example of horizontal shift image augmentation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — example of horizontal shift image augmentation

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load the image

```python
img = load_img('bird.jpg')
```

---
## Step 3 — convert to numpy array

```python
data = img_to_array(img)
```

---
## Step 4 — expand dimension to one sample

```python
samples = expand_dims(data, 0)
```

---
## Step 5 — create image data augmentation generator

```python
datagen = ImageDataGenerator(width_shift_range=[-200,200])
```

---
## Step 6 — prepare iterator

```python
it = datagen.flow(samples, batch_size=1)
```

---
## Step 7 — generate samples and plot

```python
# 生成整数序列 / Generate integer sequence
for i in range(9):
```

---
## Step 8 — define subplot

```python
pyplot.subplot(330 + 1 + i)
```

---
## Step 9 — generate batch of images

```python
batch = it.next()
```

---
## Step 10 — convert to unsigned integers for viewing

```python
# 转换数据类型 / Convert data type
image = batch[0].astype('uint8')
```

---
## Step 11 — plot raw pixel data

```python
pyplot.imshow(image)
```

---
## Step 12 — show the figure

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of horizontal shift image augmentation 是机器学习中的常用技术。  
  *example of horizontal shift image augmentation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Horizontal Shift Augmentation / 01 Horizontal Shift Augmentation
# Complete Code / 完整代码
# ===============================

# example of horizontal shift image augmentation
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load the image
img = load_img('bird.jpg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(width_shift_range=[-200,200])
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
# 生成整数序列 / Generate integer sequence
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
 # 转换数据类型 / Convert data type
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Vertical Shift Augmentation

# 02 — Vertical Shift Augmentation / 02 Vertical Shift Augmentation

**Chapter 09 — File 2 of 6 / 第09章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of vertical shift image augmentation**.

本脚本演示 **example of vertical shift image augmentation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — example of vertical shift image augmentation

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load the image

```python
img = load_img('bird.jpg')
```

---
## Step 3 — convert to numpy array

```python
data = img_to_array(img)
```

---
## Step 4 — expand dimension to one sample

```python
samples = expand_dims(data, 0)
```

---
## Step 5 — create image data augmentation generator

```python
datagen = ImageDataGenerator(height_shift_range=0.5)
```

---
## Step 6 — prepare iterator

```python
it = datagen.flow(samples, batch_size=1)
```

---
## Step 7 — generate samples and plot

```python
# 生成整数序列 / Generate integer sequence
for i in range(9):
```

---
## Step 8 — define subplot

```python
pyplot.subplot(330 + 1 + i)
```

---
## Step 9 — generate batch of images

```python
batch = it.next()
```

---
## Step 10 — convert to unsigned integers for viewing

```python
# 转换数据类型 / Convert data type
image = batch[0].astype('uint8')
```

---
## Step 11 — plot raw pixel data

```python
pyplot.imshow(image)
```

---
## Step 12 — show the figure

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of vertical shift image augmentation 是机器学习中的常用技术。  
  *example of vertical shift image augmentation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Vertical Shift Augmentation / 02 Vertical Shift Augmentation
# Complete Code / 完整代码
# ===============================

# example of vertical shift image augmentation
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load the image
img = load_img('bird.jpg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(height_shift_range=0.5)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
# 生成整数序列 / Generate integer sequence
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
 # 转换数据类型 / Convert data type
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Horizontal Flip Augmentation

# 03 — Horizontal Flip Augmentation / 03 Horizontal Flip Augmentation

**Chapter 09 — File 3 of 6 / 第09章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of horizontal flip image augmentation**.

本脚本演示 **example of horizontal flip image augmentation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — example of horizontal flip image augmentation

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load the image

```python
img = load_img('bird.jpg')
```

---
## Step 3 — convert to numpy array

```python
data = img_to_array(img)
```

---
## Step 4 — expand dimension to one sample

```python
samples = expand_dims(data, 0)
```

---
## Step 5 — create image data augmentation generator

```python
datagen = ImageDataGenerator(horizontal_flip=True)
```

---
## Step 6 — prepare iterator

```python
it = datagen.flow(samples, batch_size=1)
```

---
## Step 7 — generate samples and plot

```python
# 生成整数序列 / Generate integer sequence
for i in range(9):
```

---
## Step 8 — define subplot

```python
pyplot.subplot(330 + 1 + i)
```

---
## Step 9 — generate batch of images

```python
batch = it.next()
```

---
## Step 10 — convert to unsigned integers for viewing

```python
# 转换数据类型 / Convert data type
image = batch[0].astype('uint8')
```

---
## Step 11 — plot raw pixel data

```python
pyplot.imshow(image)
```

---
## Step 12 — show the figure

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of horizontal flip image augmentation 是机器学习中的常用技术。  
  *example of horizontal flip image augmentation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Horizontal Flip Augmentation / 03 Horizontal Flip Augmentation
# Complete Code / 完整代码
# ===============================

# example of horizontal flip image augmentation
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load the image
img = load_img('bird.jpg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(horizontal_flip=True)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
# 生成整数序列 / Generate integer sequence
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
 # 转换数据类型 / Convert data type
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Rotation Augmentation

# 04 — Rotation Augmentation / 04 Rotation Augmentation

**Chapter 09 — File 4 of 6 / 第09章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of random rotation image augmentation**.

本脚本演示 **example of random rotation image augmentation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — example of random rotation image augmentation

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load the image

```python
img = load_img('bird.jpg')
```

---
## Step 3 — convert to numpy array

```python
data = img_to_array(img)
```

---
## Step 4 — expand dimension to one sample

```python
samples = expand_dims(data, 0)
```

---
## Step 5 — create image data augmentation generator

```python
datagen = ImageDataGenerator(rotation_range=90)
```

---
## Step 6 — prepare iterator

```python
it = datagen.flow(samples, batch_size=1)
```

---
## Step 7 — generate samples and plot

```python
# 生成整数序列 / Generate integer sequence
for i in range(9):
```

---
## Step 8 — define subplot

```python
pyplot.subplot(330 + 1 + i)
```

---
## Step 9 — generate batch of images

```python
batch = it.next()
```

---
## Step 10 — convert to unsigned integers for viewing

```python
# 转换数据类型 / Convert data type
image = batch[0].astype('uint8')
```

---
## Step 11 — plot raw pixel data

```python
pyplot.imshow(image)
```

---
## Step 12 — show the figure

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of random rotation image augmentation 是机器学习中的常用技术。  
  *example of random rotation image augmentation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Rotation Augmentation / 04 Rotation Augmentation
# Complete Code / 完整代码
# ===============================

# example of random rotation image augmentation
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load the image
img = load_img('bird.jpg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(rotation_range=90)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
# 生成整数序列 / Generate integer sequence
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
 # 转换数据类型 / Convert data type
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Brightness Augmentation

# 05 — Brightness Augmentation / 05 Brightness Augmentation

**Chapter 09 — File 5 of 6 / 第09章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of brighting image augmentation**.

本脚本演示 **example of brighting image augmentation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — example of brighting image augmentation

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load the image

```python
img = load_img('bird.jpg')
```

---
## Step 3 — convert to numpy array

```python
data = img_to_array(img)
```

---
## Step 4 — expand dimension to one sample

```python
samples = expand_dims(data, 0)
```

---
## Step 5 — create image data augmentation generator

```python
datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
```

---
## Step 6 — prepare iterator

```python
it = datagen.flow(samples, batch_size=1)
```

---
## Step 7 — generate samples and plot

```python
# 生成整数序列 / Generate integer sequence
for i in range(9):
```

---
## Step 8 — define subplot

```python
pyplot.subplot(330 + 1 + i)
```

---
## Step 9 — generate batch of images

```python
batch = it.next()
```

---
## Step 10 — convert to unsigned integers for viewing

```python
# 转换数据类型 / Convert data type
image = batch[0].astype('uint8')
```

---
## Step 11 — plot raw pixel data

```python
pyplot.imshow(image)
```

---
## Step 12 — show the figure

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of brighting image augmentation 是机器学习中的常用技术。  
  *example of brighting image augmentation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Brightness Augmentation / 05 Brightness Augmentation
# Complete Code / 完整代码
# ===============================

# example of brighting image augmentation
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load the image
img = load_img('bird.jpg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
# 生成整数序列 / Generate integer sequence
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
 # 转换数据类型 / Convert data type
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Zoom Augmentation

# 06 — Zoom Augmentation / 06 Zoom Augmentation

**Chapter 09 — File 6 of 6 / 第09章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of zoom image augmentation**.

本脚本演示 **example of zoom image augmentation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — example of zoom image augmentation

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load the image

```python
img = load_img('bird.jpg')
```

---
## Step 3 — convert to numpy array

```python
data = img_to_array(img)
```

---
## Step 4 — expand dimension to one sample

```python
samples = expand_dims(data, 0)
```

---
## Step 5 — create image data augmentation generator

```python
datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
```

---
## Step 6 — prepare iterator

```python
it = datagen.flow(samples, batch_size=1)
```

---
## Step 7 — generate samples and plot

```python
# 生成整数序列 / Generate integer sequence
for i in range(9):
```

---
## Step 8 — define subplot

```python
pyplot.subplot(330 + 1 + i)
```

---
## Step 9 — generate batch of images

```python
batch = it.next()
```

---
## Step 10 — convert to unsigned integers for viewing

```python
# 转换数据类型 / Convert data type
image = batch[0].astype('uint8')
```

---
## Step 11 — plot raw pixel data

```python
pyplot.imshow(image)
```

---
## Step 12 — show the figure

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of zoom image augmentation 是机器学习中的常用技术。  
  *example of zoom image augmentation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Zoom Augmentation / 06 Zoom Augmentation
# Complete Code / 完整代码
# ===============================

# example of zoom image augmentation
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load the image
img = load_img('bird.jpg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
# 生成整数序列 / Generate integer sequence
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
 # 转换数据类型 / Convert data type
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 09 Summary / 第09章总结

## Theme / 主题: Chapter 09 / Chapter 09

This chapter contains **6 code files** demonstrating chapter 09.

本章包含 **6 个代码文件**，演示Chapter 09。

---
## Evolution / 演化路线

  1. `01_horizontal_shift_augmentation.ipynb` — Horizontal Shift Augmentation
  2. `02_vertical_shift_augmentation.ipynb` — Vertical Shift Augmentation
  3. `03_horizontal_flip_augmentation.ipynb` — Horizontal Flip Augmentation
  4. `04_rotation_augmentation.ipynb` — Rotation Augmentation
  5. `05_brightness_augmentation.ipynb` — Brightness Augmentation
  6. `06_zoom_augmentation.ipynb` — Zoom Augmentation

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 09) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 09）是机器学习流水线中的基础构建块。

---
