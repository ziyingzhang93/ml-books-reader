# GAN
## Chapter 07

---

### Load Mnist

# 01 — Load Mnist / 01 Load Mnist

**Chapter 07 — File 1 of 10 / 第07章 — 第1个文件（共10个）**

---

## Summary / 总结

This script demonstrates **example of loading the mnist dataset**.

本脚本演示 **example of loading the mnist dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — example of loading the mnist dataset

```python
from keras.datasets.mnist import load_data
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

- **概念**: example of loading the mnist dataset 是机器学习中的常用技术。  
  *example of loading the mnist dataset is a common technique in machine learning.*

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
# Load Mnist / 01 Load Mnist
# Complete Code / 完整代码
# ===============================

# example of loading the mnist dataset
from keras.datasets.mnist import load_data
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)
```

---

➡️ **Next / 下一步**: File 2 of 10

---

### Plot Mnist

# 02 — Plot Mnist / 02 Plot Mnist

**Chapter 07 — File 2 of 10 / 第07章 — 第2个文件（共10个）**

---

## Summary / 总结

This script demonstrates **example of loading the mnist dataset**.

本脚本演示 **example of loading the mnist dataset**。

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
## Step 1 — example of loading the mnist dataset

```python
from keras.datasets.mnist import load_data
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
for i in range(25):
```

---
## Step 4 — define subplot

```python
pyplot.subplot(5, 5, 1 + i)
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

- **概念**: example of loading the mnist dataset 是机器学习中的常用技术。  
  *example of loading the mnist dataset is a common technique in machine learning.*

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
# Plot Mnist / 02 Plot Mnist
# Complete Code / 完整代码
# ===============================

# example of loading the mnist dataset
from keras.datasets.mnist import load_data
from matplotlib import pyplot
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# plot images from the training dataset
for i in range(25):
	# define subplot
	pyplot.subplot(5, 5, 1 + i)
	# turn off axis
	pyplot.axis('off')
	# plot raw pixel data
	pyplot.imshow(trainX[i], cmap='gray_r')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 10

---

### Summarize Discriminator

# 03 — Summarize Discriminator / 03 Summarize Discriminator

**Chapter 07 — File 3 of 10 / 第07章 — 第3个文件（共10个）**

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
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 3 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
```

---
## Step 4 — define model

```python
model = define_discriminator()
```

---
## Step 5 — summarize the model

```python
model.summary()
```

---
## Step 6 — plot the model

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
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
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

**Chapter 07 — File 4 of 10 / 第07章 — 第4个文件（共10个）**

---

## Summary / 总结

This script demonstrates **example of training the discriminator model on real and random mnist images**.

本脚本演示 **example of training the discriminator model on real and random mnist images**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of training the discriminator model on real and random mnist images

```python
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randint
from keras.datasets.mnist import load_data
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
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 3 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
```

---
## Step 4 — load and prepare mnist training images

```python
def load_real_samples():
```

---
## Step 5 — load mnist dataset

```python
(trainX, _), (_, _) = load_data()
```

---
## Step 6 — expand to 3d, e.g. add channels dimension

```python
X = expand_dims(trainX, axis=-1)
```

---
## Step 7 — convert from unsigned ints to floats

```python
X = X.astype('float32')
```

---
## Step 8 — scale from [0,255] to [0,1]

```python
X = X / 255.0
	return X
```

---
## Step 9 — select real samples

```python
def generate_real_samples(dataset, n_samples):
```

---
## Step 10 — choose random instances

```python
ix = randint(0, dataset.shape[0], n_samples)
```

---
## Step 11 — retrieve selected images

```python
X = dataset[ix]
```

---
## Step 12 — generate 'real' class labels (1)

```python
y = ones((n_samples, 1))
	return X, y
```

---
## Step 13 — generate n fake samples with class labels

```python
def generate_fake_samples(n_samples):
```

---
## Step 14 — generate uniform random numbers in [0,1]

```python
X = rand(28 * 28 * n_samples)
```

---
## Step 15 — reshape into a batch of grayscale images

```python
X = X.reshape((n_samples, 28, 28, 1))
```

---
## Step 16 — generate 'fake' class labels (0)

```python
y = zeros((n_samples, 1))
	return X, y
```

---
## Step 17 — train the discriminator model

```python
def train_discriminator(model, dataset, n_iter=100, n_batch=256):
	half_batch = int(n_batch / 2)
```

---
## Step 18 — manually enumerate epochs

```python
for i in range(n_iter):
```

---
## Step 19 — get randomly selected 'real' samples

```python
X_real, y_real = generate_real_samples(dataset, half_batch)
```

---
## Step 20 — update discriminator on real samples

```python
_, real_acc = model.train_on_batch(X_real, y_real)
```

---
## Step 21 — generate 'fake' examples

```python
X_fake, y_fake = generate_fake_samples(half_batch)
```

---
## Step 22 — update discriminator on fake samples

```python
_, fake_acc = model.train_on_batch(X_fake, y_fake)
```

---
## Step 23 — summarize performance

```python
print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))
```

---
## Step 24 — define the discriminator model

```python
model = define_discriminator()
```

---
## Step 25 — load image data

```python
dataset = load_real_samples()
```

---
## Step 26 — fit the model

```python
train_discriminator(model, dataset)
```

---
## Learning Notes / 学习笔记

- **概念**: example of training the discriminator model on real and random mnist images 是机器学习中的常用技术。  
  *example of training the discriminator model on real and random mnist images is a common technique in machine learning.*

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

# example of training the discriminator model on real and random mnist images
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# load and prepare mnist training images
def load_real_samples():
	# load mnist dataset
	(trainX, _), (_, _) = load_data()
	# expand to 3d, e.g. add channels dimension
	X = expand_dims(trainX, axis=-1)
	# convert from unsigned ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [0,1]
	X = X / 255.0
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
	X = rand(28 * 28 * n_samples)
	# reshape into a batch of grayscale images
	X = X.reshape((n_samples, 28, 28, 1))
	# generate 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

# train the discriminator model
def train_discriminator(model, dataset, n_iter=100, n_batch=256):
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

### Summarize Generator

# 05 — Summarize Generator / 05 Summarize Generator

**Chapter 07 — File 5 of 10 / 第07章 — 第5个文件（共10个）**

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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model
```

---
## Step 2 — define the standalone generator model

```python
def define_generator(latent_dim):
	model = Sequential()
```

---
## Step 3 — foundation for 7x7 image

```python
n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
```

---
## Step 4 — upsample to 14x14

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 5 — upsample to 28x28

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model
```

---
## Step 6 — define the size of the latent space

```python
latent_dim = 100
```

---
## Step 7 — define the generator model

```python
model = define_generator(latent_dim)
```

---
## Step 8 — summarize the model

```python
model.summary()
```

---
## Step 9 — plot the model

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
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Summarize Generator / 05 Summarize Generator
# Complete Code / 完整代码
# ===============================

# example of defining the generator model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
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

➡️ **Next / 下一步**: File 6 of 10

---

### Use Generator

# 06 — Use Generator / 06 Use Generator

**Chapter 07 — File 6 of 10 / 第07章 — 第6个文件（共10个）**

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
## Step 3 — foundation for 7x7 image

```python
n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
```

---
## Step 4 — upsample to 14x14

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 5 — upsample to 28x28

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model
```

---
## Step 6 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n_samples):
```

---
## Step 7 — generate points in the latent space

```python
x_input = randn(latent_dim * n_samples)
```

---
## Step 8 — reshape into a batch of inputs for the network

```python
x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
```

---
## Step 9 — use the generator to generate n fake examples, with class labels

```python
def generate_fake_samples(g_model, latent_dim, n_samples):
```

---
## Step 10 — generate points in latent space

```python
x_input = generate_latent_points(latent_dim, n_samples)
```

---
## Step 11 — predict outputs

```python
X = g_model.predict(x_input)
```

---
## Step 12 — create 'fake' class labels (0)

```python
y = zeros((n_samples, 1))
	return X, y
```

---
## Step 13 — size of the latent space

```python
latent_dim = 100
```

---
## Step 14 — define the discriminator model

```python
model = define_generator(latent_dim)
```

---
## Step 15 — generate samples

```python
n_samples = 25
X, _ = generate_fake_samples(model, latent_dim, n_samples)
```

---
## Step 16 — plot the generated samples

```python
for i in range(n_samples):
```

---
## Step 17 — define subplot

```python
pyplot.subplot(5, 5, 1 + i)
```

---
## Step 18 — turn off axis labels

```python
pyplot.axis('off')
```

---
## Step 19 — plot single image

```python
pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
```

---
## Step 20 — show the figure

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
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
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
n_samples = 25
X, _ = generate_fake_samples(model, latent_dim, n_samples)
# plot the generated samples
for i in range(n_samples):
	# define subplot
	pyplot.subplot(5, 5, 1 + i)
	# turn off axis labels
	pyplot.axis('off')
	# plot single image
	pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
# show the figure
pyplot.show()
```

---

➡️ **Next / 下一步**: File 7 of 10

---

### Summarize Composite

# 07 — Summarize Composite / 07 Summarize Composite

**Chapter 07 — File 7 of 10 / 第07章 — 第7个文件（共10个）**

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
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 3 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
```

---
## Step 4 — define the standalone generator model

```python
def define_generator(latent_dim):
	model = Sequential()
```

---
## Step 5 — foundation for 7x7 image

```python
n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
```

---
## Step 6 — upsample to 14x14

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 7 — upsample to 28x28

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model
```

---
## Step 8 — define the combined generator and discriminator model, for updating the generator

```python
def define_gan(g_model, d_model):
```

---
## Step 9 — make weights in the discriminator not trainable

```python
d_model.trainable = False
```

---
## Step 10 — connect them

```python
model = Sequential()
```

---
## Step 11 — add generator

```python
model.add(g_model)
```

---
## Step 12 — add the discriminator

```python
model.add(d_model)
```

---
## Step 13 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
```

---
## Step 14 — size of the latent space

```python
latent_dim = 100
```

---
## Step 15 — create the discriminator

```python
d_model = define_discriminator()
```

---
## Step 16 — create the generator

```python
g_model = define_generator(latent_dim)
```

---
## Step 17 — create the gan

```python
gan_model = define_gan(g_model, d_model)
```

---
## Step 18 — summarize gan model

```python
gan_model.summary()
```

---
## Step 19 — plot gan model

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
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
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

### Complete Example

# 08 — Complete Example / 08 Complete Example

**Chapter 07 — File 8 of 10 / 第07章 — 第8个文件（共10个）**

---

## Summary / 总结

This script demonstrates **example of training a gan on mnist**.

本脚本演示 **example of training a gan on mnist**。

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
## Step 1 — example of training a gan on mnist

```python
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from matplotlib import pyplot
```

---
## Step 2 — define the standalone discriminator model

```python
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 3 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
```

---
## Step 4 — define the standalone generator model

```python
def define_generator(latent_dim):
	model = Sequential()
```

---
## Step 5 — foundation for 7x7 image

```python
n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
```

---
## Step 6 — upsample to 14x14

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 7 — upsample to 28x28

```python
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model
```

---
## Step 8 — define the combined generator and discriminator model, for updating the generator

```python
def define_gan(g_model, d_model):
```

---
## Step 9 — make weights in the discriminator not trainable

```python
d_model.trainable = False
```

---
## Step 10 — connect them

```python
model = Sequential()
```

---
## Step 11 — add generator

```python
model.add(g_model)
```

---
## Step 12 — add the discriminator

```python
model.add(d_model)
```

---
## Step 13 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
```

---
## Step 14 — load and prepare mnist training images

```python
def load_real_samples():
```

---
## Step 15 — load mnist dataset

```python
(trainX, _), (_, _) = load_data()
```

---
## Step 16 — expand to 3d, e.g. add channels dimension

```python
X = expand_dims(trainX, axis=-1)
```

---
## Step 17 — convert from unsigned ints to floats

```python
X = X.astype('float32')
```

---
## Step 18 — scale from [0,255] to [0,1]

```python
X = X / 255.0
	return X
```

---
## Step 19 — select real samples

```python
def generate_real_samples(dataset, n_samples):
```

---
## Step 20 — choose random instances

```python
ix = randint(0, dataset.shape[0], n_samples)
```

---
## Step 21 — retrieve selected images

```python
X = dataset[ix]
```

---
## Step 22 — generate 'real' class labels (1)

```python
y = ones((n_samples, 1))
	return X, y
```

---
## Step 23 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n_samples):
```

---
## Step 24 — generate points in the latent space

```python
x_input = randn(latent_dim * n_samples)
```

---
## Step 25 — reshape into a batch of inputs for the network

```python
x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
```

---
## Step 26 — use the generator to generate n fake examples, with class labels

```python
def generate_fake_samples(g_model, latent_dim, n_samples):
```

---
## Step 27 — generate points in latent space

```python
x_input = generate_latent_points(latent_dim, n_samples)
```

---
## Step 28 — predict outputs

```python
X = g_model.predict(x_input)
```

---
## Step 29 — create 'fake' class labels (0)

```python
y = zeros((n_samples, 1))
	return X, y
```

---
## Step 30 — create and save a plot of generated images (reversed grayscale)

```python
def save_plot(examples, epoch, n=10):
```

---
## Step 31 — plot images

```python
for i in range(n * n):
```

---
## Step 32 — define subplot

```python
pyplot.subplot(n, n, 1 + i)
```

---
## Step 33 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 34 — plot raw pixel data

```python
pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
```

---
## Step 35 — save plot to file

```python
filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()
```

---
## Step 36 — evaluate the discriminator, plot generated images, save generator model

```python
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
```

---
## Step 37 — prepare real samples

```python
X_real, y_real = generate_real_samples(dataset, n_samples)
```

---
## Step 38 — evaluate discriminator on real examples

```python
_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
```

---
## Step 39 — prepare fake examples

```python
x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
```

---
## Step 40 — evaluate discriminator on fake examples

```python
_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
```

---
## Step 41 — summarize discriminator performance

```python
print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
```

---
## Step 42 — save plot

```python
save_plot(x_fake, epoch)
```

---
## Step 43 — save the generator model tile file

```python
filename = 'generator_model_%03d.h5' % (epoch + 1)
	g_model.save(filename)
```

---
## Step 44 — train the generator and discriminator

```python
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
```

---
## Step 45 — manually enumerate epochs

```python
for i in range(n_epochs):
```

---
## Step 46 — enumerate batches over the training set

```python
for j in range(bat_per_epo):
```

---
## Step 47 — get randomly selected 'real' samples

```python
X_real, y_real = generate_real_samples(dataset, half_batch)
```

---
## Step 48 — generate 'fake' examples

```python
X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
```

---
## Step 49 — create training set for the discriminator

```python
X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
```

---
## Step 50 — update discriminator model weights

```python
d_loss, _ = d_model.train_on_batch(X, y)
```

---
## Step 51 — prepare points in latent space as input for the generator

```python
X_gan = generate_latent_points(latent_dim, n_batch)
```

---
## Step 52 — create inverted labels for the fake samples

```python
y_gan = ones((n_batch, 1))
```

---
## Step 53 — update the generator via the discriminator's error

```python
g_loss = gan_model.train_on_batch(X_gan, y_gan)
```

---
## Step 54 — summarize loss on this batch

```python
print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
```

---
## Step 55 — evaluate the model performance, sometimes

```python
if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)
```

---
## Step 56 — size of the latent space

```python
latent_dim = 100
```

---
## Step 57 — create the discriminator

```python
d_model = define_discriminator()
```

---
## Step 58 — create the generator

```python
g_model = define_generator(latent_dim)
```

---
## Step 59 — create the gan

```python
gan_model = define_gan(g_model, d_model)
```

---
## Step 60 — load image data

```python
dataset = load_real_samples()
```

---
## Step 61 — train model

```python
train(g_model, d_model, gan_model, dataset, latent_dim)
```

---
## Learning Notes / 学习笔记

- **概念**: example of training a gan on mnist 是机器学习中的常用技术。  
  *example of training a gan on mnist is a common technique in machine learning.*

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
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Complete Example / 08 Complete Example
# Complete Code / 完整代码
# ===============================

# example of training a gan on mnist
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from matplotlib import pyplot

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
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

# load and prepare mnist training images
def load_real_samples():
	# load mnist dataset
	(trainX, _), (_, _) = load_data()
	# expand to 3d, e.g. add channels dimension
	X = expand_dims(trainX, axis=-1)
	# convert from unsigned ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [0,1]
	X = X / 255.0
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

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	g_model.save(filename)

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# create training set for the discriminator
			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
		# evaluate the model performance, sometimes
		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)

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

➡️ **Next / 下一步**: File 9 of 10

---

### Load Use Generator

# 09 — Load Use Generator / 09 Load Use Generator

**Chapter 07 — File 9 of 10 / 第07章 — 第9个文件（共10个）**

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
pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()
```

---
## Step 10 — load model

```python
model = load_model('generator_model_100.h5')
```

---
## Step 11 — generate images

```python
latent_points = generate_latent_points(100, 25)
```

---
## Step 12 — generate images

```python
X = model.predict(latent_points)
```

---
## Step 13 — plot the result

```python
save_plot(X, 5)
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

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, n):
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
model = load_model('generator_model_100.h5')
# generate images
latent_points = generate_latent_points(100, 25)
# generate images
X = model.predict(latent_points)
# plot the result
save_plot(X, 5)
```

---

➡️ **Next / 下一步**: File 10 of 10

---

### Generate Single Image

# 10 — Generate Single Image / 图像处理

**Chapter 07 — File 10 of 10 / 第07章 — 第10个文件（共10个）**

---

## Summary / 总结

This script demonstrates **example of generating an image for a specific point in the latent space**.

本脚本演示 **example of generating an image for a specific point in the latent space**。

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
## Step 1 — example of generating an image for a specific point in the latent space

```python
from keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
```

---
## Step 2 — load model

```python
model = load_model('generator_model_100.h5')
```

---
## Step 3 — all 0s

```python
vector = asarray([[0.0 for _ in range(100)]])
```

---
## Step 4 — generate image

```python
X = model.predict(vector)
```

---
## Step 5 — plot the result

```python
pyplot.imshow(X[0, :, :, 0], cmap='gray_r')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of generating an image for a specific point in the latent space 是机器学习中的常用技术。  
  *example of generating an image for a specific point in the latent space is a common technique in machine learning.*

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
# Generate Single Image / 图像处理
# Complete Code / 完整代码
# ===============================

# example of generating an image for a specific point in the latent space
from keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
# load model
model = load_model('generator_model_100.h5')
# all 0s
vector = asarray([[0.0 for _ in range(100)]])
# generate image
X = model.predict(vector)
# plot the result
pyplot.imshow(X[0, :, :, 0], cmap='gray_r')
pyplot.show()
```

---

### Chapter Summary

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **10 code files** demonstrating chapter 07.

本章包含 **10 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `01_load_mnist.ipynb` — Load Mnist
  2. `02_plot_mnist.ipynb` — Plot Mnist
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

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---
