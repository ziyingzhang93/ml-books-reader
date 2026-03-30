# GAN
## Chapter 08

---

### Load Cifar10

# 01 — Load Cifar10 / 01 Load Cifar10

**Chapter 08 — File 1 of 10 / 第08章 — 第1个文件（共10个）**

---

## Summary / 总结

This script demonstrates **example of loading the cifar10 dataset**.

本脚本演示 **example of loading the cifar10 dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — example of loading the cifar10 dataset

```python
from keras.datasets.cifar10 import load_data
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

- **概念**: example of loading the cifar10 dataset 是机器学习中的常用技术。  
  *example of loading the cifar10 dataset is a common technique in machine learning.*

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
# Load Cifar10 / 01 Load Cifar10
# Complete Code / 完整代码
# ===============================

# example of loading the cifar10 dataset
from keras.datasets.cifar10 import load_data
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)
```

---

➡️ **Next / 下一步**: File 2 of 10

---

### Summarize Discriminator

# 03 — Summarize Discriminator / 03 Summarize Discriminator

**Chapter 08 — File 3 of 10 / 第08章 — 第3个文件（共10个）**

---

## Summary / 总结

This script demonstrates **example of defining the discriminator model**.

本脚本演示 **example of defining the discriminator model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of defining the discriminator model

```python
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model
```

---
## Step 2 — define the standalone discriminator model

```python
def define_discriminator(in_shape=(32,32,3)):
	model = Sequential()
```

---
## Step 3 — normal

```python
model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 4 — downsample

```python
model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 5 — downsample

```python
model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 6 — downsample

```python
model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 7 — classifier

```python
model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 8 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
```

---
## Step 9 — define model

```python
model = define_discriminator()
```

---
## Step 10 — summarize the model

```python
model.summary()
```

---
## Step 11 — plot the model

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
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Summarize Discriminator / 03 Summarize Discriminator
# Complete Code / 完整代码
# ===============================

# example of defining the discriminator model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model

# define the standalone discriminator model
def define_discriminator(in_shape=(32,32,3)):
	model = Sequential()
	# normal
	model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# classifier
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define model
model = define_discriminator()
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
```

---

➡️ **Next / 下一步**: File 4 of 10

---

### Train Discriminator

# 04 — Train Discriminator / 04 Train Discriminator

**Chapter 08 — File 4 of 10 / 第08章 — 第4个文件（共10个）**

---

## Summary / 总结

This script demonstrates **example of training the discriminator model on real and random cifar10 images**.

本脚本演示 **example of training the discriminator model on real and random cifar10 images**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of training the discriminator model on real and random cifar10 images

```python
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randint
from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
```

---
## Step 2 — define the standalone discriminator model

```python
def define_discriminator(in_shape=(32,32,3)):
	model = Sequential()
```

---
## Step 3 — normal

```python
model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 4 — downsample

```python
model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 5 — downsample

```python
model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 6 — downsample

```python
model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 7 — classifier

```python
model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 8 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
```

---
## Step 9 — load and prepare cifar10 training images

```python
def load_real_samples():
```

---
## Step 10 — load cifar10 dataset

```python
(trainX, _), (_, _) = load_data()
```

---
## Step 11 — convert from unsigned ints to floats

```python
X = trainX.astype('float32')
```

---
## Step 12 — scale from [0,255] to [-1,1]

```python
X = (X - 127.5) / 127.5
	return X
```

---
## Step 13 — select real samples

```python
def generate_real_samples(dataset, n_samples):
```

---
## Step 14 — choose random instances

```python
ix = randint(0, dataset.shape[0], n_samples)
```

---
## Step 15 — retrieve selected images

```python
X = dataset[ix]
```

---
## Step 16 — generate 'real' class labels (1)

```python
y = ones((n_samples, 1))
	return X, y
```

---
## Step 17 — generate n fake samples with class labels

```python
def generate_fake_samples(n_samples):
```

---
## Step 18 — generate uniform random numbers in [0,1]

```python
X = rand(32 * 32 * 3 * n_samples)
```

---
## Step 19 — update to have the range [-1, 1]

```python
X = -1 + X * 2
```

---
## Step 20 — reshape into a batch of color images

```python
X = X.reshape((n_samples, 32, 32, 3))
```

---
## Step 21 — generate 'fake' class labels (0)

```python
y = zeros((n_samples, 1))
	return X, y
```

---
## Step 22 — train the discriminator model

```python
def train_discriminator(model, dataset, n_iter=20, n_batch=128):
	half_batch = int(n_batch / 2)
```

---
## Step 23 — manually enumerate epochs

```python
for i in range(n_iter):
```

---
## Step 24 — get randomly selected 'real' samples

```python
X_real, y_real = generate_real_samples(dataset, half_batch)
```

---
## Step 25 — update discriminator on real samples

```python
_, real_acc = model.train_on_batch(X_real, y_real)
```

---
## Step 26 — generate 'fake' examples

```python
X_fake, y_fake = generate_fake_samples(half_batch)
```

---
## Step 27 — update discriminator on fake samples

```python
_, fake_acc = model.train_on_batch(X_fake, y_fake)
```

---
## Step 28 — summarize performance

```python
print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))
```

---
## Step 29 — define the discriminator model

```python
model = define_discriminator()
```

---
## Step 30 — load image data

```python
dataset = load_real_samples()
```

---
## Step 31 — fit the model

```python
train_discriminator(model, dataset)
```

---
## Learning Notes / 学习笔记

- **概念**: example of training the discriminator model on real and random cifar10 images 是机器学习中的常用技术。  
  *example of training the discriminator model on real and random cifar10 images is a common technique in machine learning.*

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
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Discriminator / 04 Train Discriminator
# Complete Code / 完整代码
# ===============================

# example of training the discriminator model on real and random cifar10 images
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randint
from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU

# define the standalone discriminator model
def define_discriminator(in_shape=(32,32,3)):
	model = Sequential()
	# normal
	model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# classifier
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# load and prepare cifar10 training images
def load_real_samples():
	# load cifar10 dataset
	(trainX, _), (_, _) = load_data()
	# convert from unsigned ints to floats
	X = trainX.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y

# generate n fake samples with class labels
def generate_fake_samples(n_samples):
	# generate uniform random numbers in [0,1]
	X = rand(32 * 32 * 3 * n_samples)
	# update to have the range [-1, 1]
	X = -1 + X * 2
	# reshape into a batch of color images
	X = X.reshape((n_samples, 32, 32, 3))
	# generate 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

# train the discriminator model
def train_discriminator(model, dataset, n_iter=20, n_batch=128):
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_iter):
		# get randomly selected 'real' samples
		X_real, y_real = generate_real_samples(dataset, half_batch)
		# update discriminator on real samples
		_, real_acc = model.train_on_batch(X_real, y_real)
		# generate 'fake' examples
		X_fake, y_fake = generate_fake_samples(half_batch)
		# update discriminator on fake samples
		_, fake_acc = model.train_on_batch(X_fake, y_fake)
		# summarize performance
		print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))

# define the discriminator model
model = define_discriminator()
# load image data
dataset = load_real_samples()
# fit the model
train_discriminator(model, dataset)
```

---

➡️ **Next / 下一步**: File 5 of 10

---

### Use Generator

# 06 — Use Generator / 06 Use Generator

**Chapter 08 — File 6 of 10 / 第08章 — 第6个文件（共10个）**

---

## Summary / 总结

This script demonstrates **example of defining and using the generator model**.

本脚本演示 **example of defining and using the generator model**。

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
## Step 1 — example of defining and using the generator model

```python
from numpy import zeros
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from matplotlib import pyplot
```

---
## Step 2 — define the standalone generator model

```python
def define_generator(latent_dim):
	model = Sequential()
```

---
## Step 3 — foundation for 4x4 image

```python
n_nodes = 256 * 4 * 4
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((4, 4, 256)))
```

---
## Step 4 — upsample to 8x8

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 5 — upsample to 16x16

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 6 — upsample to 32x32

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 7 — output layer

```python
model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
	return model
```

---
## Step 8 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n_samples):
```

---
## Step 9 — generate points in the latent space

```python
x_input = randn(latent_dim * n_samples)
```

---
## Step 10 — reshape into a batch of inputs for the network

```python
x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
```

---
## Step 11 — use the generator to generate n fake examples, with class labels

```python
def generate_fake_samples(g_model, latent_dim, n_samples):
```

---
## Step 12 — generate points in latent space

```python
x_input = generate_latent_points(latent_dim, n_samples)
```

---
## Step 13 — predict outputs

```python
X = g_model.predict(x_input)
```

---
## Step 14 — create 'fake' class labels (0)

```python
y = zeros((n_samples, 1))
	return X, y
```

---
## Step 15 — size of the latent space

```python
latent_dim = 100
```

---
## Step 16 — define the discriminator model

```python
model = define_generator(latent_dim)
```

---
## Step 17 — generate samples

```python
n_samples = 49
X, _ = generate_fake_samples(model, latent_dim, n_samples)
```

---
## Step 18 — scale pixel values from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
```

---
## Step 19 — plot the generated samples

```python
for i in range(n_samples):
```

---
## Step 20 — define subplot

```python
pyplot.subplot(7, 7, 1 + i)
```

---
## Step 21 — turn off axis labels

```python
pyplot.axis('off')
```

---
## Step 22 — plot single image

```python
pyplot.imshow(X[i])
```

---
## Step 23 — show the figure

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of defining and using the generator model 是机器学习中的常用技术。  
  *example of defining and using the generator model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `matplotlib` | 绑图库 | Plotting library |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Use Generator / 06 Use Generator
# Complete Code / 完整代码
# ===============================

# example of defining and using the generator model
from numpy import zeros
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from matplotlib import pyplot

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 4x4 image
	n_nodes = 256 * 4 * 4
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((4, 4, 256)))
	# upsample to 8x8
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 16x16
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 32x32
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# output layer
	model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
	return model

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

# size of the latent space
latent_dim = 100
# define the discriminator model
model = define_generator(latent_dim)
# generate samples
n_samples = 49
X, _ = generate_fake_samples(model, latent_dim, n_samples)
# scale pixel values from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the generated samples
for i in range(n_samples):
	# define subplot
	pyplot.subplot(7, 7, 1 + i)
	# turn off axis labels
	pyplot.axis('off')
	# plot single image
	pyplot.imshow(X[i])
# show the figure
pyplot.show()
```

---

➡️ **Next / 下一步**: File 7 of 10

---

### Summarize Composite

# 07 — Summarize Composite / 07 Summarize Composite

**Chapter 08 — File 7 of 10 / 第08章 — 第7个文件（共10个）**

---

## Summary / 总结

This script demonstrates **demonstrate creating the three models in the gan**.

本脚本演示 **demonstrate creating the three models in the gan**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — demonstrate creating the three models in the gan

```python
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model
```

---
## Step 2 — define the standalone discriminator model

```python
def define_discriminator(in_shape=(32,32,3)):
	model = Sequential()
```

---
## Step 3 — normal

```python
model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 4 — downsample

```python
model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 5 — downsample

```python
model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 6 — downsample

```python
model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 7 — classifier

```python
model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 8 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
```

---
## Step 9 — define the standalone generator model

```python
def define_generator(latent_dim):
	model = Sequential()
```

---
## Step 10 — foundation for 4x4 image

```python
n_nodes = 256 * 4 * 4
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((4, 4, 256)))
```

---
## Step 11 — upsample to 8x8

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 12 — upsample to 16x16

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 13 — upsample to 32x32

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 14 — output layer

```python
model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
	return model
```

---
## Step 15 — define the combined generator and discriminator model, for updating the generator

```python
def define_gan(g_model, d_model):
```

---
## Step 16 — make weights in the discriminator not trainable

```python
d_model.trainable = False
```

---
## Step 17 — connect them

```python
model = Sequential()
```

---
## Step 18 — add generator

```python
model.add(g_model)
```

---
## Step 19 — add the discriminator

```python
model.add(d_model)
```

---
## Step 20 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
```

---
## Step 21 — size of the latent space

```python
latent_dim = 100
```

---
## Step 22 — create the discriminator

```python
d_model = define_discriminator()
```

---
## Step 23 — create the generator

```python
g_model = define_generator(latent_dim)
```

---
## Step 24 — create the gan

```python
gan_model = define_gan(g_model, d_model)
```

---
## Step 25 — summarize gan model

```python
gan_model.summary()
```

---
## Step 26 — plot gan model

```python
plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)
```

---
## Learning Notes / 学习笔记

- **概念**: demonstrate creating the three models in the gan 是机器学习中的常用技术。  
  *demonstrate creating the three models in the gan is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Summarize Composite / 07 Summarize Composite
# Complete Code / 完整代码
# ===============================

# demonstrate creating the three models in the gan
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model

# define the standalone discriminator model
def define_discriminator(in_shape=(32,32,3)):
	model = Sequential()
	# normal
	model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# classifier
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 4x4 image
	n_nodes = 256 * 4 * 4
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((4, 4, 256)))
	# upsample to 8x8
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 16x16
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 32x32
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# output layer
	model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# summarize gan model
gan_model.summary()
# plot gan model
plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)
```

---

➡️ **Next / 下一步**: File 8 of 10

---

### Load Use Generator

# 09 — Load Use Generator / 09 Load Use Generator

**Chapter 08 — File 9 of 10 / 第08章 — 第9个文件（共10个）**

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
## Step 5 — create and save a plot of generated images

```python
def save_plot(examples, n):
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
pyplot.imshow(examples[i, :, :])
	pyplot.show()
```

---
## Step 10 — load model

```python
model = load_model('generator_model_200.h5')
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
## Step 13 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
```

---
## Step 14 — plot the result

```python
save_plot(X, 10)
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
# Load Use Generator / 09 Load Use Generator
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

# create and save a plot of generated images
def save_plot(examples, n):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :])
	pyplot.show()

# load model
model = load_model('generator_model_200.h5')
# generate images
latent_points = generate_latent_points(100, 100)
# generate images
X = model.predict(latent_points)
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
save_plot(X, 10)
```

---

➡️ **Next / 下一步**: File 10 of 10

---

### Chapter Summary

# Chapter 08 Summary / 第08章总结

## Theme / 主题: Chapter 08 / Chapter 08

This chapter contains **10 code files** demonstrating chapter 08.

本章包含 **10 个代码文件**，演示Chapter 08。

---
## Evolution / 演化路线

  1. `01_load_cifar10.ipynb` — Load Cifar10
  2. `02_plot_cifar10.ipynb` — Plot Cifar10
  3. `03_summarize_discriminator.ipynb` — Summarize Discriminator
  4. `04_train_discriminator.ipynb` — Train Discriminator
  5. `05_summarize_generator.ipynb` — Summarize Generator
  6. `06_use_generator.ipynb` — Use Generator
  7. `07_summarize_composite.ipynb` — Summarize Composite
  8. `08_complete_example.ipynb` — Complete Example
  9. `09_load_use_generator.ipynb` — Load Use Generator
  10. `10_generate_single_image.ipynb` — Generate Single Image

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 08) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 08）是机器学习流水线中的基础构建块。

---
