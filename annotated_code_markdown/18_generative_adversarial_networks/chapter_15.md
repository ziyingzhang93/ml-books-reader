# GAN
## Chapter 15

---

### Train Lsgan

# 01 — Train Lsgan / 生成对抗网络

**Chapter 15 — File 1 of 2 / 第15章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **example of lsgan for mnist**.

本脚本演示 **example of lsgan for mnist**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — example of lsgan for mnist

```python
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from matplotlib import pyplot
```

---
## Step 2 — define the standalone discriminator model

```python
def define_discriminator(in_shape=(28,28,1)):
```

---
## Step 3 — weight initialization

```python
init = RandomNormal(stddev=0.02)
```

---
## Step 4 — define model

```python
model = Sequential()
```

---
## Step 5 — downsample to 14x14

```python
model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 6 — downsample to 7x7

```python
model.add(Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 7 — classifier

```python
model.add(Flatten())
	model.add(Dense(1, activation='linear', kernel_initializer=init))
```

---
## Step 8 — compile model with L2 loss

```python
model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5))
	return model
```

---
## Step 9 — define the standalone generator model

```python
def define_generator(latent_dim):
```

---
## Step 10 — weight initialization

```python
init = RandomNormal(stddev=0.02)
```

---
## Step 11 — define model

```python
model = Sequential()
```

---
## Step 12 — foundation for 7x7 image

```python
n_nodes = 256 * 7 * 7
	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Reshape((7, 7, 256)))
```

---
## Step 13 — upsample to 14x14

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
```

---
## Step 14 — upsample to 28x28

```python
model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
```

---
## Step 15 — output 28x28x1

```python
model.add(Conv2D(1, (7,7), padding='same', kernel_initializer=init))
	model.add(Activation('tanh'))
	return model
```

---
## Step 16 — define the combined generator and discriminator model, for updating the generator

```python
def define_gan(generator, discriminator):
```

---
## Step 17 — make weights in the discriminator not trainable

```python
for layer in discriminator.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
```

---
## Step 18 — connect them

```python
model = Sequential()
```

---
## Step 19 — add generator

```python
model.add(generator)
```

---
## Step 20 — add the discriminator

```python
model.add(discriminator)
```

---
## Step 21 — compile model with L2 loss

```python
model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5))
	return model
```

---
## Step 22 — load mnist images

```python
def load_real_samples():
```

---
## Step 23 — load dataset

```python
(trainX, _), (_, _) = load_data()
```

---
## Step 24 — expand to 3d, e.g. add channels

```python
X = expand_dims(trainX, axis=-1)
```

---
## Step 25 — convert from ints to floats

```python
X = X.astype('float32')
```

---
## Step 26 — scale from [0,255] to [-1,1]

```python
X = (X - 127.5) / 127.5
	return X
```

---
## Step 27 — # select real samples

```python
def generate_real_samples(dataset, n_samples):
```

---
## Step 28 — choose random instances

```python
ix = randint(0, dataset.shape[0], n_samples)
```

---
## Step 29 — select images

```python
X = dataset[ix]
```

---
## Step 30 — generate class labels

```python
y = ones((n_samples, 1))
	return X, y
```

---
## Step 31 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n_samples):
```

---
## Step 32 — generate points in the latent space

```python
x_input = randn(latent_dim * n_samples)
```

---
## Step 33 — reshape into a batch of inputs for the network

```python
x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
```

---
## Step 34 — use the generator to generate n fake examples, with class labels

```python
def generate_fake_samples(generator, latent_dim, n_samples):
```

---
## Step 35 — generate points in latent space

```python
x_input = generate_latent_points(latent_dim, n_samples)
```

---
## Step 36 — predict outputs

```python
X = generator.predict(x_input)
```

---
## Step 37 — create class labels

```python
y = zeros((n_samples, 1))
	return X, y
```

---
## Step 38 — generate samples and save as a plot and save the model

```python
def summarize_performance(step, g_model, latent_dim, n_samples=100):
```

---
## Step 39 — prepare fake examples

```python
X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
```

---
## Step 40 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
```

---
## Step 41 — plot images

```python
for i in range(10 * 10):
```

---
## Step 42 — define subplot

```python
pyplot.subplot(10, 10, 1 + i)
```

---
## Step 43 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 44 — plot raw pixel data

```python
pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
```

---
## Step 45 — save plot to file

```python
filename1 = 'generated_plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
```

---
## Step 46 — save the generator model

```python
filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('Saved %s and %s' % (filename1, filename2))
```

---
## Step 47 — create a line plot of loss for the gan and save to file

```python
def plot_history(d1_hist, d2_hist, g_hist):
	pyplot.plot(d1_hist, label='dloss1')
	pyplot.plot(d2_hist, label='dloss2')
	pyplot.plot(g_hist, label='gloss')
	pyplot.legend()
	filename = 'plot_line_plot_loss.png'
	pyplot.savefig(filename)
	pyplot.close()
	print('Saved %s' % (filename))
```

---
## Step 48 — train the generator and discriminator

```python
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=20, n_batch=64):
```

---
## Step 49 — calculate the number of batches per training epoch

```python
bat_per_epo = int(dataset.shape[0] / n_batch)
```

---
## Step 50 — calculate the number of training iterations

```python
n_steps = bat_per_epo * n_epochs
```

---
## Step 51 — calculate the size of half a batch of samples

```python
half_batch = int(n_batch / 2)
```

---
## Step 52 — lists for storing loss, for plotting later

```python
d1_hist, d2_hist, g_hist = list(), list(), list()
```

---
## Step 53 — manually enumerate epochs

```python
for i in range(n_steps):
```

---
## Step 54 — prepare real and fake samples

```python
X_real, y_real = generate_real_samples(dataset, half_batch)
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
```

---
## Step 55 — update discriminator model

```python
d_loss1 = d_model.train_on_batch(X_real, y_real)
		d_loss2 = d_model.train_on_batch(X_fake, y_fake)
```

---
## Step 56 — update the generator via the discriminator's error

```python
z_input = generate_latent_points(latent_dim, n_batch)
		y_real2 = ones((n_batch, 1))
		g_loss = gan_model.train_on_batch(z_input, y_real2)
```

---
## Step 57 — summarize loss on this batch

```python
print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, d_loss1, d_loss2, g_loss))
```

---
## Step 58 — record history

```python
d1_hist.append(d_loss1)
		d2_hist.append(d_loss2)
		g_hist.append(g_loss)
```

---
## Step 59 — evaluate the model performance every 'epoch'

```python
if (i+1) % (bat_per_epo * 1) == 0:
			summarize_performance(i, g_model, latent_dim)
```

---
## Step 60 — create line plot of training history

```python
plot_history(d1_hist, d2_hist, g_hist)
```

---
## Step 61 — size of the latent space

```python
latent_dim = 100
```

---
## Step 62 — create the discriminator

```python
discriminator = define_discriminator()
```

---
## Step 63 — create the generator

```python
generator = define_generator(latent_dim)
```

---
## Step 64 — create the gan

```python
gan_model = define_gan(generator, discriminator)
```

---
## Step 65 — load image data

```python
dataset = load_real_samples()
print(dataset.shape)
```

---
## Step 66 — train model

```python
train(generator, discriminator, gan_model, dataset, latent_dim)
```

---
## Learning Notes / 学习笔记

- **概念**: example of lsgan for mnist 是机器学习中的常用技术。  
  *example of lsgan for mnist is a common technique in machine learning.*

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
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Lsgan / 生成对抗网络
# Complete Code / 完整代码
# ===============================

# example of lsgan for mnist
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from matplotlib import pyplot

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
	model = Sequential()
	# downsample to 14x14
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 7x7
	model.add(Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# classifier
	model.add(Flatten())
	model.add(Dense(1, activation='linear', kernel_initializer=init))
	# compile model with L2 loss
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5))
	return model

# define the standalone generator model
def define_generator(latent_dim):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 256 * 7 * 7
	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Reshape((7, 7, 256)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	# upsample to 28x28
	model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	# output 28x28x1
	model.add(Conv2D(1, (7,7), padding='same', kernel_initializer=init))
	model.add(Activation('tanh'))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	for layer in discriminator.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model with L2 loss
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5))
	return model

# load mnist images
def load_real_samples():
	# load dataset
	(trainX, _), (_, _) = load_data()
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X

# # select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = zeros((n_samples, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
	# prepare fake examples
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	for i in range(10 * 10):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename1 = 'generated_plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('Saved %s and %s' % (filename1, filename2))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
	pyplot.plot(d1_hist, label='dloss1')
	pyplot.plot(d2_hist, label='dloss2')
	pyplot.plot(g_hist, label='gloss')
	pyplot.legend()
	filename = 'plot_line_plot_loss.png'
	pyplot.savefig(filename)
	pyplot.close()
	print('Saved %s' % (filename))

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=20, n_batch=64):
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# lists for storing loss, for plotting later
	d1_hist, d2_hist, g_hist = list(), list(), list()
	# manually enumerate epochs
	for i in range(n_steps):
		# prepare real and fake samples
		X_real, y_real = generate_real_samples(dataset, half_batch)
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator model
		d_loss1 = d_model.train_on_batch(X_real, y_real)
		d_loss2 = d_model.train_on_batch(X_fake, y_fake)
		# update the generator via the discriminator's error
		z_input = generate_latent_points(latent_dim, n_batch)
		y_real2 = ones((n_batch, 1))
		g_loss = gan_model.train_on_batch(z_input, y_real2)
		# summarize loss on this batch
		print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, d_loss1, d_loss2, g_loss))
		# record history
		d1_hist.append(d_loss1)
		d2_hist.append(d_loss2)
		g_hist.append(g_loss)
		# evaluate the model performance every 'epoch'
		if (i+1) % (bat_per_epo * 1) == 0:
			summarize_performance(i, g_model, latent_dim)
	# create line plot of training history
	plot_history(d1_hist, d2_hist, g_hist)

# size of the latent space
latent_dim = 100
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
print(dataset.shape)
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Inference Lsgan

# 02 — Inference Lsgan / 生成对抗网络

**Chapter 15 — File 2 of 2 / 第15章 — 第2个文件（共2个）**

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
## Step 5 — create a plot of generated images (reversed grayscale)

```python
def plot_generated(examples, n):
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
model = load_model('model_018740.h5')
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
plot_generated(X, 10)
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
# Inference Lsgan / 生成对抗网络
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

# create a plot of generated images (reversed grayscale)
def plot_generated(examples, n):
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
model = load_model('model_018740.h5')
# generate images
latent_points = generate_latent_points(100, 100)
# generate images
X = model.predict(latent_points)
# plot the result
plot_generated(X, 10)
```

---

### Chapter Summary

# Chapter 15 Summary / 第15章总结

## Theme / 主题: Chapter 15 / Chapter 15

This chapter contains **2 code files** demonstrating chapter 15.

本章包含 **2 个代码文件**，演示Chapter 15。

---
## Evolution / 演化路线

  1. `01_train_lsgan.ipynb` — Train Lsgan
  2. `02_inference_lsgan.ipynb` — Inference Lsgan

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 15) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 15）是机器学习流水线中的基础构建块。

---
