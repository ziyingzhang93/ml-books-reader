# 计算机视觉深度学习 / Deep Learning for Computer Vision
## Chapter 17

---

### One Vgg Block

# 01 — One Vgg Block / 01 One Vgg Block

**Chapter 17 — File 1 of 5 / 第17章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of creating a CNN model with a VGG block**.

本脚本演示 **Example of creating a CNN model with a VGG block**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Example of creating a CNN model with a VGG block

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import plot_model
```

---
## Step 2 — function for creating a vgg block

```python
def vgg_block(layer_in, n_filters, n_conv):
```

---
## Step 3 — add convolutional layers

```python
# 生成整数序列 / Generate integer sequence
for _ in range(n_conv):
  # 二维卷积层（Keras） / 2D convolution layer (Keras)
		layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)
```

---
## Step 4 — add max pooling layer

```python
# 最大池化层（Keras） / Max pooling layer (Keras)
layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
	return layer_in
```

---
## Step 5 — define model input

```python
visible = Input(shape=(256, 256, 3))
```

---
## Step 6 — add vgg module

```python
layer = vgg_block(visible, 64, 2)
```

---
## Step 7 — create model

```python
model = Model(inputs=visible, outputs=layer)
```

---
## Step 8 — summarize model

```python
model.summary()
```

---
## Step 9 — plot model architecture

```python
plot_model(model, show_shapes=True, to_file='vgg_block.png')
```

---
## Learning Notes / 学习笔记

- **概念**: Example of creating a CNN model with a VGG block 是机器学习中的常用技术。  
  *Example of creating a CNN model with a VGG block is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# One Vgg Block / 01 One Vgg Block
# Complete Code / 完整代码
# ===============================

# Example of creating a CNN model with a VGG block
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import plot_model

# function for creating a vgg block
def vgg_block(layer_in, n_filters, n_conv):
	# add convolutional layers
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_conv):
  # 二维卷积层（Keras） / 2D convolution layer (Keras)
		layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)
	# add max pooling layer
 # 最大池化层（Keras） / Max pooling layer (Keras)
	layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
	return layer_in

# define model input
visible = Input(shape=(256, 256, 3))
# add vgg module
layer = vgg_block(visible, 64, 2)
# create model
model = Model(inputs=visible, outputs=layer)
# summarize model
model.summary()
# plot model architecture
plot_model(model, show_shapes=True, to_file='vgg_block.png')
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Many Vgg Blocks

# 02 — Many Vgg Blocks / 02 Many Vgg Blocks

**Chapter 17 — File 2 of 5 / 第17章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of creating a CNN model with many VGG blocks**.

本脚本演示 **Example of creating a CNN model with many VGG blocks**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Example of creating a CNN model with many VGG blocks

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import plot_model
```

---
## Step 2 — function for creating a vgg block

```python
def vgg_block(layer_in, n_filters, n_conv):
```

---
## Step 3 — add convolutional layers

```python
# 生成整数序列 / Generate integer sequence
for _ in range(n_conv):
  # 二维卷积层（Keras） / 2D convolution layer (Keras)
		layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)
```

---
## Step 4 — add max pooling layer

```python
# 最大池化层（Keras） / Max pooling layer (Keras)
layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
	return layer_in
```

---
## Step 5 — define model input

```python
visible = Input(shape=(256, 256, 3))
```

---
## Step 6 — add vgg module

```python
layer = vgg_block(visible, 64, 2)
```

---
## Step 7 — add vgg module

```python
layer = vgg_block(layer, 128, 2)
```

---
## Step 8 — add vgg module

```python
layer = vgg_block(layer, 256, 4)
```

---
## Step 9 — create model

```python
model = Model(inputs=visible, outputs=layer)
```

---
## Step 10 — summarize model

```python
model.summary()
```

---
## Step 11 — plot model architecture

```python
plot_model(model, show_shapes=True, to_file='multiple_vgg_blocks.png')
```

---
## Learning Notes / 学习笔记

- **概念**: Example of creating a CNN model with many VGG blocks 是机器学习中的常用技术。  
  *Example of creating a CNN model with many VGG blocks is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Many Vgg Blocks / 02 Many Vgg Blocks
# Complete Code / 完整代码
# ===============================

# Example of creating a CNN model with many VGG blocks
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import plot_model

# function for creating a vgg block
def vgg_block(layer_in, n_filters, n_conv):
	# add convolutional layers
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_conv):
  # 二维卷积层（Keras） / 2D convolution layer (Keras)
		layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)
	# add max pooling layer
 # 最大池化层（Keras） / Max pooling layer (Keras)
	layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
	return layer_in

# define model input
visible = Input(shape=(256, 256, 3))
# add vgg module
layer = vgg_block(visible, 64, 2)
# add vgg module
layer = vgg_block(layer, 128, 2)
# add vgg module
layer = vgg_block(layer, 256, 4)
# create model
model = Model(inputs=visible, outputs=layer)
# summarize model
model.summary()
# plot model architecture
plot_model(model, show_shapes=True, to_file='multiple_vgg_blocks.png')
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Naive Inception

# 03 — Naive Inception / 03 Naive Inception

**Chapter 17 — File 3 of 5 / 第17章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of creating a CNN with an inception module**.

本脚本演示 **example of creating a CNN with an inception module**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of creating a CNN with an inception module

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.merge import concatenate
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import plot_model
```

---
## Step 2 — function for creating a naive inception block

```python
def naive_inception_module(layer_in, f1, f2, f3):
```

---
## Step 3 — 1x1 conv

```python
# 二维卷积层（Keras） / 2D convolution layer (Keras)
conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
```

---
## Step 4 — 3x3 conv

```python
# 二维卷积层（Keras） / 2D convolution layer (Keras)
conv3 = Conv2D(f2, (3,3), padding='same', activation='relu')(layer_in)
```

---
## Step 5 — 5x5 conv

```python
# 二维卷积层（Keras） / 2D convolution layer (Keras)
conv5 = Conv2D(f3, (5,5), padding='same', activation='relu')(layer_in)
```

---
## Step 6 — 3x3 max pooling

```python
# 最大池化层（Keras） / Max pooling layer (Keras)
pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
```

---
## Step 7 — concatenate filters, assumes filters/channels last

```python
layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out
```

---
## Step 8 — define model input

```python
visible = Input(shape=(256, 256, 3))
```

---
## Step 9 — add inception module

```python
layer = naive_inception_module(visible, 64, 128, 32)
```

---
## Step 10 — create model

```python
model = Model(inputs=visible, outputs=layer)
```

---
## Step 11 — summarize model

```python
model.summary()
```

---
## Step 12 — plot model architecture

```python
plot_model(model, show_shapes=True, to_file='naive_inception_module.png')
```

---
## Learning Notes / 学习笔记

- **概念**: example of creating a CNN with an inception module 是机器学习中的常用技术。  
  *example of creating a CNN with an inception module is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Naive Inception / 03 Naive Inception
# Complete Code / 完整代码
# ===============================

# example of creating a CNN with an inception module
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.merge import concatenate
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import plot_model

# function for creating a naive inception block
def naive_inception_module(layer_in, f1, f2, f3):
	# 1x1 conv
 # 二维卷积层（Keras） / 2D convolution layer (Keras)
	conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
 # 二维卷积层（Keras） / 2D convolution layer (Keras)
	conv3 = Conv2D(f2, (3,3), padding='same', activation='relu')(layer_in)
	# 5x5 conv
 # 二维卷积层（Keras） / 2D convolution layer (Keras)
	conv5 = Conv2D(f3, (5,5), padding='same', activation='relu')(layer_in)
	# 3x3 max pooling
 # 最大池化层（Keras） / Max pooling layer (Keras)
	pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

# define model input
visible = Input(shape=(256, 256, 3))
# add inception module
layer = naive_inception_module(visible, 64, 128, 32)
# create model
model = Model(inputs=visible, outputs=layer)
# summarize model
model.summary()
# plot model architecture
plot_model(model, show_shapes=True, to_file='naive_inception_module.png')
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Optimized Inception

# 04 — Optimized Inception / 优化

**Chapter 17 — File 4 of 5 / 第17章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of creating a CNN with an efficient inception module**.

本脚本演示 **example of creating a CNN with an efficient inception module**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of creating a CNN with an efficient inception module

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.merge import concatenate
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import plot_model
```

---
## Step 2 — function for creating a projected inception module

```python
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
```

---
## Step 3 — 1x1 conv

```python
# 二维卷积层（Keras） / 2D convolution layer (Keras)
conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
```

---
## Step 4 — 3x3 conv

```python
# 二维卷积层（Keras） / 2D convolution layer (Keras)
conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
 # 二维卷积层（Keras） / 2D convolution layer (Keras)
	conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
```

---
## Step 5 — 5x5 conv

```python
# 二维卷积层（Keras） / 2D convolution layer (Keras)
conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
 # 二维卷积层（Keras） / 2D convolution layer (Keras)
	conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
```

---
## Step 6 — 3x3 max pooling

```python
# 最大池化层（Keras） / Max pooling layer (Keras)
pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
 # 二维卷积层（Keras） / 2D convolution layer (Keras)
	pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
```

---
## Step 7 — concatenate filters, assumes filters/channels last

```python
layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out
```

---
## Step 8 — define model input

```python
visible = Input(shape=(256, 256, 3))
```

---
## Step 9 — add inception block 1

```python
layer = inception_module(visible, 64, 96, 128, 16, 32, 32)
```

---
## Step 10 — add inception block 1

```python
layer = inception_module(layer, 128, 128, 192, 32, 96, 64)
```

---
## Step 11 — create model

```python
model = Model(inputs=visible, outputs=layer)
```

---
## Step 12 — summarize model

```python
model.summary()
```

---
## Step 13 — plot model architecture

```python
plot_model(model, show_shapes=True, to_file='inception_module.png')
```

---
## Learning Notes / 学习笔记

- **概念**: example of creating a CNN with an efficient inception module 是机器学习中的常用技术。  
  *example of creating a CNN with an efficient inception module is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Optimized Inception / 优化
# Complete Code / 完整代码
# ===============================

# example of creating a CNN with an efficient inception module
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers.merge import concatenate
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import plot_model

# function for creating a projected inception module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
	# 1x1 conv
 # 二维卷积层（Keras） / 2D convolution layer (Keras)
	conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
 # 二维卷积层（Keras） / 2D convolution layer (Keras)
	conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
 # 二维卷积层（Keras） / 2D convolution layer (Keras)
	conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
	# 5x5 conv
 # 二维卷积层（Keras） / 2D convolution layer (Keras)
	conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
 # 二维卷积层（Keras） / 2D convolution layer (Keras)
	conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
	# 3x3 max pooling
 # 最大池化层（Keras） / Max pooling layer (Keras)
	pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
 # 二维卷积层（Keras） / 2D convolution layer (Keras)
	pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

# define model input
visible = Input(shape=(256, 256, 3))
# add inception block 1
layer = inception_module(visible, 64, 96, 128, 16, 32, 32)
# add inception block 1
layer = inception_module(layer, 128, 128, 192, 32, 96, 64)
# create model
model = Model(inputs=visible, outputs=layer)
# summarize model
model.summary()
# plot model architecture
plot_model(model, show_shapes=True, to_file='inception_module.png')
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Residual Module

# 05 — Residual Module / 05 Residual Module

**Chapter 17 — File 5 of 5 / 第17章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of a CNN model with an identity or projection residual module**.

本脚本演示 **example of a CNN model with an identity or projection residual module**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of a CNN model with an identity or projection residual module

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Activation
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import add
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import plot_model
```

---
## Step 2 — function for creating an identity or projection residual module

```python
def residual_module(layer_in, n_filters):
	merge_input = layer_in
```

---
## Step 3 — check if the number of filters needs to be increase, assumes channels last format

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
if layer_in.shape[-1] != n_filters:
  # 二维卷积层（Keras） / 2D convolution layer (Keras)
		merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
```

---
## Step 4 — conv1

```python
# 二维卷积层（Keras） / 2D convolution layer (Keras)
conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
```

---
## Step 5 — conv2

```python
# 二维卷积层（Keras） / 2D convolution layer (Keras)
conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
```

---
## Step 6 — add filters, assumes filters/channels last

```python
layer_out = add([conv2, merge_input])
```

---
## Step 7 — activation function

```python
layer_out = Activation('relu')(layer_out)
	return layer_out
```

---
## Step 8 — define model input

```python
visible = Input(shape=(256, 256, 3))
```

---
## Step 9 — add vgg module

```python
layer = residual_module(visible, 64)
```

---
## Step 10 — create model

```python
model = Model(inputs=visible, outputs=layer)
```

---
## Step 11 — summarize model

```python
model.summary()
```

---
## Step 12 — plot model architecture

```python
plot_model(model, show_shapes=True, to_file='residual_module.png')
```

---
## Learning Notes / 学习笔记

- **概念**: example of a CNN model with an identity or projection residual module 是机器学习中的常用技术。  
  *example of a CNN model with an identity or projection residual module is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Residual Module / 05 Residual Module
# Complete Code / 完整代码
# ===============================

# example of a CNN model with an identity or projection residual module
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Input
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Activation
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import add
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils import plot_model

# function for creating an identity or projection residual module
def residual_module(layer_in, n_filters):
	merge_input = layer_in
	# check if the number of filters needs to be increase, assumes channels last format
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	if layer_in.shape[-1] != n_filters:
  # 二维卷积层（Keras） / 2D convolution layer (Keras)
		merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv1
 # 二维卷积层（Keras） / 2D convolution layer (Keras)
	conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv2
 # 二维卷积层（Keras） / 2D convolution layer (Keras)
	conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
	# add filters, assumes filters/channels last
	layer_out = add([conv2, merge_input])
	# activation function
	layer_out = Activation('relu')(layer_out)
	return layer_out

# define model input
visible = Input(shape=(256, 256, 3))
# add vgg module
layer = residual_module(visible, 64)
# create model
model = Model(inputs=visible, outputs=layer)
# summarize model
model.summary()
# plot model architecture
plot_model(model, show_shapes=True, to_file='residual_module.png')
```

---

### Chapter Summary / 章节总结

# Chapter 17 Summary / 第17章总结

## Theme / 主题: Chapter 17 / Chapter 17

This chapter contains **5 code files** demonstrating chapter 17.

本章包含 **5 个代码文件**，演示Chapter 17。

---
## Evolution / 演化路线

  1. `01_one_vgg_block.ipynb` — One Vgg Block
  2. `02_many_vgg_blocks.ipynb` — Many Vgg Blocks
  3. `03_naive_inception.ipynb` — Naive Inception
  4. `04_optimized_inception.ipynb` — Optimized Inception
  5. `05_residual_module.ipynb` — Residual Module

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 17) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 17）是机器学习流水线中的基础构建块。

---
