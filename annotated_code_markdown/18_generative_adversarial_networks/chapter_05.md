# 生成对抗网络 / Generative Adversarial Networks
## Chapter 05

---

### Strided Downsample

# 01 — Strided Downsample / 01 Strided Downsample

**Chapter 05 — File 1 of 11 / 第05章 — 第1个文件（共11个）**

---

## Summary / 总结

This script demonstrates **example of downsampling with strided convolutions**.

本脚本演示 **example of downsampling with strided convolutions**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of downsampling with strided convolutions

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
```

---
## Step 2 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=(64,64,3)))
```

---
## Step 3 — summarize model

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example of downsampling with strided convolutions 是机器学习中的常用技术。  
  *example of downsampling with strided convolutions is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Strided Downsample / 01 Strided Downsample
# Complete Code / 完整代码
# ===============================

# example of downsampling with strided convolutions
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=(64,64,3)))
# summarize model
model.summary()
```

---

➡️ **Next / 下一步**: File 2 of 11

---

### Strided Upsample

# 02 — Strided Upsample / 02 Strided Upsample

**Chapter 05 — File 2 of 11 / 第05章 — 第2个文件（共11个）**

---

## Summary / 总结

This script demonstrates **example of upsampling with strided convolutions**.

本脚本演示 **example of upsampling with strided convolutions**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of upsampling with strided convolutions

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
```

---
## Step 2 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', input_shape=(64,64,3)))
```

---
## Step 3 — summarize model

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example of upsampling with strided convolutions 是机器学习中的常用技术。  
  *example of upsampling with strided convolutions is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Strided Upsample / 02 Strided Upsample
# Complete Code / 完整代码
# ===============================

# example of upsampling with strided convolutions
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', input_shape=(64,64,3)))
# summarize model
model.summary()
```

---

➡️ **Next / 下一步**: File 3 of 11

---

### Leaky Relu



---

### Batch Norm



---

### Gauss Weight Init

# 05 — Gauss Weight Init / 05 Gauss Weight Init

**Chapter 05 — File 5 of 11 / 第05章 — 第5个文件（共11个）**

---

## Summary / 总结

This script demonstrates **example of gaussian weight initialization in a generator model**.

本脚本演示 **example of gaussian weight initialization in a generator model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of gaussian weight initialization in a generator model

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.initializers import RandomNormal
```

---
## Step 2 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
init = RandomNormal(mean=0.0, stddev=0.02)
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=(64,64,3)))
```

---
## Learning Notes / 学习笔记

- **概念**: example of gaussian weight initialization in a generator model 是机器学习中的常用技术。  
  *example of gaussian weight initialization in a generator model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gauss Weight Init / 05 Gauss Weight Init
# Complete Code / 完整代码
# ===============================

# example of gaussian weight initialization in a generator model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.initializers import RandomNormal
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
init = RandomNormal(mean=0.0, stddev=0.02)
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=(64,64,3)))
```

---

➡️ **Next / 下一步**: File 6 of 11

---

### Adam Sgd



---

### Image Scaling

# 07 — Image Scaling / 图像处理

**Chapter 05 — File 7 of 11 / 第05章 — 第7个文件（共11个）**

---

## Summary / 总结

This script demonstrates **example of a function for scaling images**.

本脚本演示 **example of a function for scaling images**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of a function for scaling images

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
```

---
## Step 2 — scale image data from [0,255] to [-1,1]

```python
def scale_images(images):
```

---
## Step 3 — convert from unit8 to float32

```python
# 转换数据类型 / Convert data type
images = images.astype('float32')
```

---
## Step 4 — scale from [0,255] to [-1,1]

```python
images = (images - 127.5) / 127.5
	return images
```

---
## Step 5 — define one 28x28 color image

```python
images = randint(0, 256, 28 * 28 * 3)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
images = images.reshape((1, 28, 28, 3))
```

---
## Step 6 — summarize pixel values

```python
# 打印输出 / Print output
print(images.min(), images.max())
```

---
## Step 7 — scale

```python
scaled = scale_images(images)
```

---
## Step 8 — summarize pixel scaled values

```python
# 打印输出 / Print output
print(scaled.min(), scaled.max())
```

---
## Learning Notes / 学习笔记

- **概念**: example of a function for scaling images 是机器学习中的常用技术。  
  *example of a function for scaling images is a common technique in machine learning.*

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
# Image Scaling / 图像处理
# Complete Code / 完整代码
# ===============================

# example of a function for scaling images
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint

# scale image data from [0,255] to [-1,1]
def scale_images(images):
	# convert from unit8 to float32
 # 转换数据类型 / Convert data type
	images = images.astype('float32')
	# scale from [0,255] to [-1,1]
	images = (images - 127.5) / 127.5
	return images

# define one 28x28 color image
images = randint(0, 256, 28 * 28 * 3)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
images = images.reshape((1, 28, 28, 3))
# summarize pixel values
# 打印输出 / Print output
print(images.min(), images.max())
# scale
scaled = scale_images(images)
# summarize pixel scaled values
# 打印输出 / Print output
print(scaled.min(), scaled.max())
```

---

➡️ **Next / 下一步**: File 8 of 11

---

### Gauss Latent

# 08 — Gauss Latent / 08 Gauss Latent

**Chapter 05 — File 8 of 11 / 第05章 — 第8个文件（共11个）**

---

## Summary / 总结

This script demonstrates **example of sampling from a gaussian latent space**.

本脚本演示 **example of sampling from a gaussian latent space**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of sampling from a gaussian latent space

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
```

---
## Step 2 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n_samples):
```

---
## Step 3 — generate points in the latent space

```python
x_input = randn(latent_dim * n_samples)
```

---
## Step 4 — reshape into a batch of inputs for the network

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape((n_samples, latent_dim))
	return x_input
```

---
## Step 5 — size of latent space

```python
n_dim = 100
```

---
## Step 6 — number of samples to generate

```python
n_samples = 500
```

---
## Step 7 — generate samples

```python
samples = generate_latent_points(n_dim, n_samples)
```

---
## Step 8 — summarize

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(samples.shape, samples.mean(), samples.std())
```

---
## Learning Notes / 学习笔记

- **概念**: example of sampling from a gaussian latent space 是机器学习中的常用技术。  
  *example of sampling from a gaussian latent space is a common technique in machine learning.*

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
# Gauss Latent / 08 Gauss Latent
# Complete Code / 完整代码
# ===============================

# example of sampling from a gaussian latent space
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	x_input = x_input.reshape((n_samples, latent_dim))
	return x_input

# size of latent space
n_dim = 100
# number of samples to generate
n_samples = 500
# generate samples
samples = generate_latent_points(n_dim, n_samples)
# summarize
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(samples.shape, samples.mean(), samples.std())
```

---

➡️ **Next / 下一步**: File 9 of 11

---

### Positive Label Smoothing

# 09 — Positive Label Smoothing / 09 Positive Label Smoothing

**Chapter 05 — File 9 of 11 / 第05章 — 第9个文件（共11个）**

---

## Summary / 总结

This script demonstrates **example of positive label smoothing**.

本脚本演示 **example of positive label smoothing**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of positive label smoothing

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import random
```

---
## Step 2 — example of smoothing class=1 to [0.7, 1.2]

```python
def smooth_positive_labels(y):
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	return y - 0.3 + (random(y.shape) * 0.5)
```

---
## Step 3 — generate 'real' class labels (1)

```python
n_samples = 1000
y = ones((n_samples, 1))
```

---
## Step 4 — smooth labels

```python
y = smooth_positive_labels(y)
```

---
## Step 5 — summarize smooth labels

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(y.shape, y.min(), y.max())
```

---
## Learning Notes / 学习笔记

- **概念**: example of positive label smoothing 是机器学习中的常用技术。  
  *example of positive label smoothing is a common technique in machine learning.*

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
# Positive Label Smoothing / 09 Positive Label Smoothing
# Complete Code / 完整代码
# ===============================

# example of positive label smoothing
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import random

# example of smoothing class=1 to [0.7, 1.2]
def smooth_positive_labels(y):
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	return y - 0.3 + (random(y.shape) * 0.5)

# generate 'real' class labels (1)
n_samples = 1000
y = ones((n_samples, 1))
# smooth labels
y = smooth_positive_labels(y)
# summarize smooth labels
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(y.shape, y.min(), y.max())
```

---

➡️ **Next / 下一步**: File 10 of 11

---

### Negative Label Smoothing

# 10 — Negative Label Smoothing / 10 Negative Label Smoothing

**Chapter 05 — File 10 of 11 / 第05章 — 第10个文件（共11个）**

---

## Summary / 总结

This script demonstrates **example of negative label smoothing**.

本脚本演示 **example of negative label smoothing**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of negative label smoothing

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import random
```

---
## Step 2 — example of smoothing class=0 to [0.0, 0.3]

```python
def smooth_negative_labels(y):
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	return y + random(y.shape) * 0.3
```

---
## Step 3 — generate 'fake' class labels (0)

```python
n_samples = 1000
y = zeros((n_samples, 1))
```

---
## Step 4 — smooth labels

```python
y = smooth_negative_labels(y)
```

---
## Step 5 — summarize smooth labels

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(y.shape, y.min(), y.max())
```

---
## Learning Notes / 学习笔记

- **概念**: example of negative label smoothing 是机器学习中的常用技术。  
  *example of negative label smoothing is a common technique in machine learning.*

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
# Negative Label Smoothing / 10 Negative Label Smoothing
# Complete Code / 完整代码
# ===============================

# example of negative label smoothing
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import random

# example of smoothing class=0 to [0.0, 0.3]
def smooth_negative_labels(y):
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	return y + random(y.shape) * 0.3

# generate 'fake' class labels (0)
n_samples = 1000
y = zeros((n_samples, 1))
# smooth labels
y = smooth_negative_labels(y)
# summarize smooth labels
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(y.shape, y.min(), y.max())
```

---

➡️ **Next / 下一步**: File 11 of 11

---

### Noisy Labels



---

### Chapter Summary / 章节总结

# Chapter 05 Summary / 第05章总结

## Theme / 主题: Chapter 05 / Chapter 05

This chapter contains **11 code files** demonstrating chapter 05.

本章包含 **11 个代码文件**，演示Chapter 05。

---
## Evolution / 演化路线

  1. `01_strided_downsample.ipynb` — Strided Downsample
  2. `02_strided_upsample.ipynb` — Strided Upsample
  3. `03_leaky_relu.ipynb` — Leaky Relu
  4. `04_batch_norm.ipynb` — Batch Norm
  5. `05_gauss_weight_init.ipynb` — Gauss Weight Init
  6. `06_adam_sgd.ipynb` — Adam Sgd
  7. `07_image_scaling.ipynb` — Image Scaling
  8. `08_gauss_latent.ipynb` — Gauss Latent
  9. `09_positive_label_smoothing.ipynb` — Positive Label Smoothing
  10. `10_negative_label_smoothing.ipynb` — Negative Label Smoothing
  11. `11_noisy_labels.ipynb` — Noisy Labels

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 05) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 05）是机器学习流水线中的基础构建块。

---
