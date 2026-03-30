# GAN
## Chapter 19

---

### Summarize Discriminator

# 01 — Summarize Discriminator / 01 Summarize Discriminator

**Chapter 19 — File 1 of 4 / 第19章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of defining the discriminator model**.

本脚本演示 **example of defining the discriminator model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of defining the discriminator model

```python
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
```

---
## Step 2 — define the standalone discriminator model

```python
def define_discriminator(in_shape=(28,28,1), n_classes=10):
```

---
## Step 3 — weight initialization

```python
init = RandomNormal(stddev=0.02)
```

---
## Step 4 — image input

```python
in_image = Input(shape=in_shape)
```

---
## Step 5 — downsample to 14x14

```python
fe = Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
```

---
## Step 6 — normal

```python
fe = Conv2D(64, (3,3), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
```

---
## Step 7 — downsample to 7x7

```python
fe = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
```

---
## Step 8 — normal

```python
fe = Conv2D(256, (3,3), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
```

---
## Step 9 — flatten feature maps

```python
fe = Flatten()(fe)
```

---
## Step 10 — real/fake output

```python
out1 = Dense(1, activation='sigmoid')(fe)
```

---
## Step 11 — class label output

```python
out2 = Dense(n_classes, activation='softmax')(fe)
```

---
## Step 12 — define model

```python
model = Model(in_image, [out1, out2])
```

---
## Step 13 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
	return model
```

---
## Step 14 — define the discriminator model

```python
model = define_discriminator()
```

---
## Step 15 — summarize the model

```python
model.summary()
```

---
## Step 16 — plot the model

```python
plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
```

---
## Learning Notes / 学习笔记

- **概念**: example of defining the discriminator model 是机器学习中的常用技术。  
  *example of defining the discriminator model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Summarize Discriminator / 01 Summarize Discriminator
# Complete Code / 完整代码
# ===============================

# example of defining the discriminator model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1), n_classes=10):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=in_shape)
	# downsample to 14x14
	fe = Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# normal
	fe = Conv2D(64, (3,3), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# downsample to 7x7
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# normal
	fe = Conv2D(256, (3,3), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# real/fake output
	out1 = Dense(1, activation='sigmoid')(fe)
	# class label output
	out2 = Dense(n_classes, activation='softmax')(fe)
	# define model
	model = Model(in_image, [out1, out2])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
	return model

# define the discriminator model
model = define_discriminator()
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Summarize Generator

# 02 — Summarize Generator / 02 Summarize Generator

**Chapter 19 — File 2 of 4 / 第19章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of defining the generator model**.

本脚本演示 **example of defining the generator model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of defining the generator model

```python
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.utils.vis_utils import plot_model
```

---
## Step 2 — define the standalone generator model

```python
def define_generator(latent_dim, n_classes=10):
```

---
## Step 3 — weight initialization

```python
init = RandomNormal(stddev=0.02)
```

---
## Step 4 — label input

```python
in_label = Input(shape=(1,))
```

---
## Step 5 — embedding for categorical input

```python
li = Embedding(n_classes, 50)(in_label)
```

---
## Step 6 — linear multiplication

```python
n_nodes = 7 * 7
	li = Dense(n_nodes, kernel_initializer=init)(li)
```

---
## Step 7 — reshape to additional channel

```python
li = Reshape((7, 7, 1))(li)
```

---
## Step 8 — image generator input

```python
in_lat = Input(shape=(latent_dim,))
```

---
## Step 9 — foundation for 7x7 image

```python
n_nodes = 384 * 7 * 7
	gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
	gen = Activation('relu')(gen)
	gen = Reshape((7, 7, 384))(gen)
```

---
## Step 10 — merge image gen and label input

```python
merge = Concatenate()([gen, li])
```

---
## Step 11 — upsample to 14x14

```python
gen = Conv2DTranspose(192, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(merge)
	gen = BatchNormalization()(gen)
	gen = Activation('relu')(gen)
```

---
## Step 12 — upsample to 28x28

```python
gen = Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	out_layer = Activation('tanh')(gen)
```

---
## Step 13 — define model

```python
model = Model([in_lat, in_label], out_layer)
	return model
```

---
## Step 14 — define the size of the latent space

```python
latent_dim = 100
```

---
## Step 15 — define the generator model

```python
model = define_generator(latent_dim)
```

---
## Step 16 — summarize the model

```python
model.summary()
```

---
## Step 17 — plot the model

```python
plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)
```

---
## Learning Notes / 学习笔记

- **概念**: example of defining the generator model 是机器学习中的常用技术。  
  *example of defining the generator model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Summarize Generator / 02 Summarize Generator
# Complete Code / 完整代码
# ===============================

# example of defining the generator model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.utils.vis_utils import plot_model

# define the standalone generator model
def define_generator(latent_dim, n_classes=10):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# linear multiplication
	n_nodes = 7 * 7
	li = Dense(n_nodes, kernel_initializer=init)(li)
	# reshape to additional channel
	li = Reshape((7, 7, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 384 * 7 * 7
	gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
	gen = Activation('relu')(gen)
	gen = Reshape((7, 7, 384))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x14
	gen = Conv2DTranspose(192, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(merge)
	gen = BatchNormalization()(gen)
	gen = Activation('relu')(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	out_layer = Activation('tanh')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model

# define the size of the latent space
latent_dim = 100
# define the generator model
model = define_generator(latent_dim)
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Inference Acgan

# 04 — Inference Acgan / 生成对抗网络

**Chapter 19 — File 4 of 4 / 第19章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of loading the generator model and generating images**.

本脚本演示 **example of loading the generator model and generating images**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
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
## Step 1 — example of loading the generator model and generating images

```python
from math import sqrt
from numpy import asarray
from numpy.random import randn
from keras.models import load_model
from matplotlib import pyplot
```

---
## Step 2 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n_samples, n_class):
```

---
## Step 3 — generate points in the latent space

```python
x_input = randn(latent_dim * n_samples)
```

---
## Step 4 — reshape into a batch of inputs for the network

```python
z_input = x_input.reshape(n_samples, latent_dim)
```

---
## Step 5 — generate labels

```python
labels = asarray([n_class for _ in range(n_samples)])
	return [z_input, labels]
```

---
## Step 6 — create and save a plot of generated images

```python
def save_plot(examples, n_examples):
```

---
## Step 7 — plot images

```python
for i in range(n_examples):
```

---
## Step 8 — define subplot

```python
pyplot.subplot(sqrt(n_examples), sqrt(n_examples), 1 + i)
```

---
## Step 9 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 10 — plot raw pixel data

```python
pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()
```

---
## Step 11 — load model

```python
model = load_model('model_93700.h5')
latent_dim = 100
n_examples = 100 # must be a square
n_class = 7 # sneaker
```

---
## Step 12 — generate images

```python
latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class)
```

---
## Step 13 — generate images

```python
X  = model.predict([latent_points, labels])
```

---
## Step 14 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
```

---
## Step 15 — plot the result

```python
save_plot(X, n_examples)
```

---
## Learning Notes / 学习笔记

- **概念**: example of loading the generator model and generating images 是机器学习中的常用技术。  
  *example of loading the generator model and generating images is a common technique in machine learning.*

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
# Inference Acgan / 生成对抗网络
# Complete Code / 完整代码
# ===============================

# example of loading the generator model and generating images
from math import sqrt
from numpy import asarray
from numpy.random import randn
from keras.models import load_model
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_class):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = asarray([n_class for _ in range(n_samples)])
	return [z_input, labels]

# create and save a plot of generated images
def save_plot(examples, n_examples):
	# plot images
	for i in range(n_examples):
		# define subplot
		pyplot.subplot(sqrt(n_examples), sqrt(n_examples), 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()

# load model
model = load_model('model_93700.h5')
latent_dim = 100
n_examples = 100 # must be a square
n_class = 7 # sneaker
# generate images
latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class)
# generate images
X  = model.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
save_plot(X, n_examples)
```

---

### Chapter Summary

# Chapter 19 Summary / 第19章总结

## Theme / 主题: Chapter 19 / Chapter 19

This chapter contains **4 code files** demonstrating chapter 19.

本章包含 **4 个代码文件**，演示Chapter 19。

---
## Evolution / 演化路线

  1. `01_summarize_discriminator.ipynb` — Summarize Discriminator
  2. `02_summarize_generator.ipynb` — Summarize Generator
  3. `03_train_acgan.ipynb` — Train Acgan
  4. `04_inference_acgan.ipynb` — Inference Acgan

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 19) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 19）是机器学习流水线中的基础构建块。

---
