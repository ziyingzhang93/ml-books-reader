# GAN
## Chapter 17

---

### Load Fasion Mnist

# 01 — Load Fasion Mnist / 01 Load Fasion Mnist

**Chapter 17 — File 1 of 6 / 第17章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of loading the fashion_mnist dataset**.

本脚本演示 **example of loading the fashion_mnist dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — example of loading the fashion_mnist dataset

```python
from keras.datasets.fashion_mnist import load_data
```

---
## Step 2 — load the images into memory

```python
(trainX, trainy), (testX, testy) = load_data()
```

---
## Step 3 — summarize the shape of the dataset

```python
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: example of loading the fashion_mnist dataset 是机器学习中的常用技术。  
  *example of loading the fashion_mnist dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Fasion Mnist / 01 Load Fasion Mnist
# Complete Code / 完整代码
# ===============================

# example of loading the fashion_mnist dataset
from keras.datasets.fashion_mnist import load_data
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Plot Fashion Mnist

# 02 — Plot Fashion Mnist / 02 Plot Fashion Mnist

**Chapter 17 — File 2 of 6 / 第17章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of loading the fashion_mnist dataset**.

本脚本演示 **example of loading the fashion_mnist dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — example of loading the fashion_mnist dataset

```python
from keras.datasets.fashion_mnist import load_data
from matplotlib import pyplot
```

---
## Step 2 — load the images into memory

```python
(trainX, trainy), (testX, testy) = load_data()
```

---
## Step 3 — plot images from the training dataset

```python
for i in range(100):
```

---
## Step 4 — define subplot

```python
pyplot.subplot(10, 10, 1 + i)
```

---
## Step 5 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 6 — plot raw pixel data

```python
pyplot.imshow(trainX[i], cmap='gray_r')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of loading the fashion_mnist dataset 是机器学习中的常用技术。  
  *example of loading the fashion_mnist dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Fashion Mnist / 02 Plot Fashion Mnist
# Complete Code / 完整代码
# ===============================

# example of loading the fashion_mnist dataset
from keras.datasets.fashion_mnist import load_data
from matplotlib import pyplot
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# plot images from the training dataset
for i in range(100):
	# define subplot
	pyplot.subplot(10, 10, 1 + i)
	# turn off axis
	pyplot.axis('off')
	# plot raw pixel data
	pyplot.imshow(trainX[i], cmap='gray_r')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Inference Unconditional Gan

# 04 — Inference Unconditional Gan / 生成对抗网络

**Chapter 17 — File 4 of 6 / 第17章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of loading the generator model and generating images**.

本脚本演示 **example of loading the generator model and generating images**。

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
## Step 1 — example of loading the generator model and generating images

```python
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
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
x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
```

---
## Step 5 — create and save a plot of generated images (reversed grayscale)

```python
def show_plot(examples, n):
```

---
## Step 6 — plot images

```python
for i in range(n * n):
```

---
## Step 7 — define subplot

```python
pyplot.subplot(n, n, 1 + i)
```

---
## Step 8 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 9 — plot raw pixel data

```python
pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()
```

---
## Step 10 — load model

```python
model = load_model('generator.h5')
```

---
## Step 11 — generate images

```python
latent_points = generate_latent_points(100, 100)
```

---
## Step 12 — generate images

```python
X = model.predict(latent_points)
```

---
## Step 13 — plot the result

```python
show_plot(X, 10)
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
# Inference Unconditional Gan / 生成对抗网络
# Complete Code / 完整代码
# ===============================

# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# create and save a plot of generated images (reversed grayscale)
def show_plot(examples, n):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()

# load model
model = load_model('generator.h5')
# generate images
latent_points = generate_latent_points(100, 100)
# generate images
X = model.predict(latent_points)
# plot the result
show_plot(X, 10)
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Train Conditional Gan

# 05 — Train Conditional Gan / 生成对抗网络

**Chapter 17 — File 5 of 6 / 第17章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of training an conditional gan on the fashion mnist dataset**.

本脚本演示 **example of training an conditional gan on the fashion mnist dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
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
## Step 1 — example of training an conditional gan on the fashion mnist dataset

```python
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
```

---
## Step 2 — define the standalone discriminator model

```python
def define_discriminator(in_shape=(28,28,1), n_classes=10):
```

---
## Step 3 — label input

```python
in_label = Input(shape=(1,))
```

---
## Step 4 — embedding for categorical input

```python
li = Embedding(n_classes, 50)(in_label)
```

---
## Step 5 — scale up to image dimensions with linear activation

```python
n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
```

---
## Step 6 — reshape to additional channel

```python
li = Reshape((in_shape[0], in_shape[1], 1))(li)
```

---
## Step 7 — image input

```python
in_image = Input(shape=in_shape)
```

---
## Step 8 — concat label as a channel

```python
merge = Concatenate()([in_image, li])
```

---
## Step 9 — downsample

```python
fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
```

---
## Step 10 — downsample

```python
fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
```

---
## Step 11 — flatten feature maps

```python
fe = Flatten()(fe)
```

---
## Step 12 — dropout

```python
fe = Dropout(0.4)(fe)
```

---
## Step 13 — output

```python
out_layer = Dense(1, activation='sigmoid')(fe)
```

---
## Step 14 — define model

```python
model = Model([in_image, in_label], out_layer)
```

---
## Step 15 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
```

---
## Step 16 — define the standalone generator model

```python
def define_generator(latent_dim, n_classes=10):
```

---
## Step 17 — label input

```python
in_label = Input(shape=(1,))
```

---
## Step 18 — embedding for categorical input

```python
li = Embedding(n_classes, 50)(in_label)
```

---
## Step 19 — linear multiplication

```python
n_nodes = 7 * 7
	li = Dense(n_nodes)(li)
```

---
## Step 20 — reshape to additional channel

```python
li = Reshape((7, 7, 1))(li)
```

---
## Step 21 — image generator input

```python
in_lat = Input(shape=(latent_dim,))
```

---
## Step 22 — foundation for 7x7 image

```python
n_nodes = 128 * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, 128))(gen)
```

---
## Step 23 — merge image gen and label input

```python
merge = Concatenate()([gen, li])
```

---
## Step 24 — upsample to 14x14

```python
gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
```

---
## Step 25 — upsample to 28x28

```python
gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
```

---
## Step 26 — output

```python
out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
```

---
## Step 27 — define model

```python
model = Model([in_lat, in_label], out_layer)
	return model
```

---
## Step 28 — define the combined generator and discriminator model, for updating the generator

```python
def define_gan(g_model, d_model):
```

---
## Step 29 — make weights in the discriminator not trainable

```python
d_model.trainable = False
```

---
## Step 30 — get noise and label inputs from generator model

```python
gen_noise, gen_label = g_model.input
```

---
## Step 31 — get image output from the generator model

```python
gen_output = g_model.output
```

---
## Step 32 — connect image output and label input from generator as inputs to discriminator

```python
gan_output = d_model([gen_output, gen_label])
```

---
## Step 33 — define gan model as taking noise and label and outputting a classification

```python
model = Model([gen_noise, gen_label], gan_output)
```

---
## Step 34 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
```

---
## Step 35 — load fashion mnist images

```python
def load_real_samples():
```

---
## Step 36 — load dataset

```python
(trainX, trainy), (_, _) = load_data()
```

---
## Step 37 — expand to 3d, e.g. add channels

```python
X = expand_dims(trainX, axis=-1)
```

---
## Step 38 — convert from ints to floats

```python
X = X.astype('float32')
```

---
## Step 39 — scale from [0,255] to [-1,1]

```python
X = (X - 127.5) / 127.5
	return [X, trainy]
```

---
## Step 40 — # select real samples

```python
def generate_real_samples(dataset, n_samples):
```

---
## Step 41 — split into images and labels

```python
images, labels = dataset
```

---
## Step 42 — choose random instances

```python
ix = randint(0, images.shape[0], n_samples)
```

---
## Step 43 — select images and labels

```python
X, labels = images[ix], labels[ix]
```

---
## Step 44 — generate class labels

```python
y = ones((n_samples, 1))
	return [X, labels], y
```

---
## Step 45 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n_samples, n_classes=10):
```

---
## Step 46 — generate points in the latent space

```python
x_input = randn(latent_dim * n_samples)
```

---
## Step 47 — reshape into a batch of inputs for the network

```python
z_input = x_input.reshape(n_samples, latent_dim)
```

---
## Step 48 — generate labels

```python
labels = randint(0, n_classes, n_samples)
	return [z_input, labels]
```

---
## Step 49 — use the generator to generate n fake examples, with class labels

```python
def generate_fake_samples(generator, latent_dim, n_samples):
```

---
## Step 50 — generate points in latent space

```python
z_input, labels_input = generate_latent_points(latent_dim, n_samples)
```

---
## Step 51 — predict outputs

```python
images = generator.predict([z_input, labels_input])
```

---
## Step 52 — create class labels

```python
y = zeros((n_samples, 1))
	return [images, labels_input], y
```

---
## Step 53 — train the generator and discriminator

```python
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
```

---
## Step 54 — manually enumerate epochs

```python
for i in range(n_epochs):
```

---
## Step 55 — enumerate batches over the training set

```python
for j in range(bat_per_epo):
```

---
## Step 56 — get randomly selected 'real' samples

```python
[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
```

---
## Step 57 — update discriminator model weights

```python
d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
```

---
## Step 58 — generate 'fake' examples

```python
[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
```

---
## Step 59 — update discriminator model weights

```python
d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
```

---
## Step 60 — prepare points in latent space as input for the generator

```python
[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
```

---
## Step 61 — create inverted labels for the fake samples

```python
y_gan = ones((n_batch, 1))
```

---
## Step 62 — update the generator via the discriminator's error

```python
g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
```

---
## Step 63 — summarize loss on this batch

```python
print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
```

---
## Step 64 — save the generator model

```python
g_model.save('cgan_generator.h5')
```

---
## Step 65 — size of the latent space

```python
latent_dim = 100
```

---
## Step 66 — create the discriminator

```python
d_model = define_discriminator()
```

---
## Step 67 — create the generator

```python
g_model = define_generator(latent_dim)
```

---
## Step 68 — create the gan

```python
gan_model = define_gan(g_model, d_model)
```

---
## Step 69 — load image data

```python
dataset = load_real_samples()
```

---
## Step 70 — train model

```python
train(g_model, d_model, gan_model, dataset, latent_dim)
```

---
## Learning Notes / 学习笔记

- **概念**: example of training an conditional gan on the fashion mnist dataset 是机器学习中的常用技术。  
  *example of training an conditional gan on the fashion mnist dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `embedding` | 嵌入：将离散数据映射为连续向量 | Embedding: map discrete data to continuous vectors |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Conditional Gan / 生成对抗网络
# Complete Code / 完整代码
# ===============================

# example of training an conditional gan on the fashion mnist dataset
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1), n_classes=10):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=10):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# linear multiplication
	n_nodes = 7 * 7
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((7, 7, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, 128))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load fashion mnist images
def load_real_samples():
	# load dataset
	(trainX, trainy), (_, _) = load_data()
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return [X, trainy]

# # select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			# generate 'fake' examples
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
			# prepare points in latent space as input for the generator
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
	# save the generator model
	g_model.save('cgan_generator.h5')

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Chapter Summary

# Chapter 17 Summary / 第17章总结

## Theme / 主题: Chapter 17 / Chapter 17

This chapter contains **6 code files** demonstrating chapter 17.

本章包含 **6 个代码文件**，演示Chapter 17。

---
## Evolution / 演化路线

  1. `01_load_fasion_mnist.ipynb` — Load Fasion Mnist
  2. `02_plot_fashion_mnist.ipynb` — Plot Fashion Mnist
  3. `03_train_unconditional_gan.ipynb` — Train Unconditional Gan
  4. `04_inference_unconditional_gan.ipynb` — Inference Unconditional Gan
  5. `05_train_conditional_gan.ipynb` — Train Conditional Gan
  6. `06_inference_conditional_gan.ipynb` — Inference Conditional Gan

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 17) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 17）是机器学习流水线中的基础构建块。

---
