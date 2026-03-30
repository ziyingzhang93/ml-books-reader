# GAN
## Chapter 18

---

### Define Plot Models

# 01 — Define Plot Models / 01 Define Plot Models

**Chapter 18 — File 1 of 4 / 第18章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **create and plot the infogan model for mnist**.

本脚本演示 **create and plot the infogan model for mnist**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — create and plot the infogan model for mnist

```python
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.initializers import RandomNormal
from keras.utils.vis_utils import plot_model
```

---
## Step 2 — define the standalone discriminator model

```python
def define_discriminator(n_cat, in_shape=(28,28,1)):
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
d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.1)(d)
```

---
## Step 6 — downsample to 7x7

```python
d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = LeakyReLU(alpha=0.1)(d)
	d = BatchNormalization()(d)
```

---
## Step 7 — normal

```python
d = Conv2D(256, (4,4), padding='same', kernel_initializer=init)(d)
	d = LeakyReLU(alpha=0.1)(d)
	d = BatchNormalization()(d)
```

---
## Step 8 — flatten feature maps

```python
d = Flatten()(d)
```

---
## Step 9 — real/fake output

```python
out_classifier = Dense(1, activation='sigmoid')(d)
```

---
## Step 10 — define d model

```python
d_model = Model(in_image, out_classifier)
```

---
## Step 11 — compile d model

```python
d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
```

---
## Step 12 — create q model layers

```python
q = Dense(128)(d)
	q = BatchNormalization()(q)
	q = LeakyReLU(alpha=0.1)(q)
```

---
## Step 13 — q model output

```python
out_codes = Dense(n_cat, activation='softmax')(q)
```

---
## Step 14 — define q model

```python
q_model = Model(in_image, out_codes)
	return d_model, q_model
```

---
## Step 15 — define the standalone generator model

```python
def define_generator(gen_input_size):
```

---
## Step 16 — weight initialization

```python
init = RandomNormal(stddev=0.02)
```

---
## Step 17 — image generator input

```python
in_lat = Input(shape=(gen_input_size,))
```

---
## Step 18 — foundation for 7x7 image

```python
n_nodes = 512 * 7 * 7
	gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
	gen = Reshape((7, 7, 512))(gen)
```

---
## Step 19 — normal

```python
gen = Conv2D(128, (4,4), padding='same', kernel_initializer=init)(gen)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
```

---
## Step 20 — upsample to 14x14

```python
gen = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
```

---
## Step 21 — upsample to 28x28

```python
gen = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
```

---
## Step 22 — tanh output

```python
out_layer = Activation('tanh')(gen)
```

---
## Step 23 — define model

```python
model = Model(in_lat, out_layer)
	return model
```

---
## Step 24 — define the combined discriminator, generator and q network model

```python
def define_gan(g_model, d_model, q_model):
```

---
## Step 25 — make weights in the discriminator (some shared with the q model) as not trainable

```python
for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
```

---
## Step 26 — connect g outputs to d inputs

```python
d_output = d_model(g_model.output)
```

---
## Step 27 — connect g outputs to q inputs

```python
q_output = q_model(g_model.output)
```

---
## Step 28 — define composite model

```python
model = Model(g_model.input, [d_output, q_output])
```

---
## Step 29 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt)
	return model
```

---
## Step 30 — number of values for the categorical control code

```python
n_cat = 10
```

---
## Step 31 — size of the latent space

```python
latent_dim = 62
```

---
## Step 32 — create the discriminator

```python
d_model, q_model = define_discriminator(n_cat)
```

---
## Step 33 — create the generator

```python
gen_input_size = latent_dim + n_cat
g_model = define_generator(gen_input_size)
```

---
## Step 34 — create the gan

```python
gan_model = define_gan(g_model, d_model, q_model)
```

---
## Step 35 — plot the model

```python
plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)
```

---
## Learning Notes / 学习笔记

- **概念**: create and plot the infogan model for mnist 是机器学习中的常用技术。  
  *create and plot the infogan model for mnist is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Define Plot Models / 01 Define Plot Models
# Complete Code / 完整代码
# ===============================

# create and plot the infogan model for mnist
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.initializers import RandomNormal
from keras.utils.vis_utils import plot_model

# define the standalone discriminator model
def define_discriminator(n_cat, in_shape=(28,28,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=in_shape)
	# downsample to 14x14
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.1)(d)
	# downsample to 7x7
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = LeakyReLU(alpha=0.1)(d)
	d = BatchNormalization()(d)
	# normal
	d = Conv2D(256, (4,4), padding='same', kernel_initializer=init)(d)
	d = LeakyReLU(alpha=0.1)(d)
	d = BatchNormalization()(d)
	# flatten feature maps
	d = Flatten()(d)
	# real/fake output
	out_classifier = Dense(1, activation='sigmoid')(d)
	# define d model
	d_model = Model(in_image, out_classifier)
	# compile d model
	d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
	# create q model layers
	q = Dense(128)(d)
	q = BatchNormalization()(q)
	q = LeakyReLU(alpha=0.1)(q)
	# q model output
	out_codes = Dense(n_cat, activation='softmax')(q)
	# define q model
	q_model = Model(in_image, out_codes)
	return d_model, q_model

# define the standalone generator model
def define_generator(gen_input_size):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image generator input
	in_lat = Input(shape=(gen_input_size,))
	# foundation for 7x7 image
	n_nodes = 512 * 7 * 7
	gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
	gen = Reshape((7, 7, 512))(gen)
	# normal
	gen = Conv2D(128, (4,4), padding='same', kernel_initializer=init)(gen)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
	# upsample to 14x14
	gen = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	# tanh output
	out_layer = Activation('tanh')(gen)
	# define model
	model = Model(in_lat, out_layer)
	return model

# define the combined discriminator, generator and q network model
def define_gan(g_model, d_model, q_model):
	# make weights in the discriminator (some shared with the q model) as not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# connect g outputs to d inputs
	d_output = d_model(g_model.output)
	# connect g outputs to q inputs
	q_output = q_model(g_model.output)
	# define composite model
	model = Model(g_model.input, [d_output, q_output])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt)
	return model

# number of values for the categorical control code
n_cat = 10
# size of the latent space
latent_dim = 62
# create the discriminator
d_model, q_model = define_discriminator(n_cat)
# create the generator
gen_input_size = latent_dim + n_cat
g_model = define_generator(gen_input_size)
# create the gan
gan_model = define_gan(g_model, d_model, q_model)
# plot the model
plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Train Infogan

# 02 — Train Infogan / 生成对抗网络

**Chapter 18 — File 2 of 4 / 第18章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of training an infogan on mnist**.

本脚本演示 **example of training an infogan on mnist**。

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
## Step 1 — example of training an infogan on mnist

```python
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy import hstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Activation
from matplotlib import pyplot
```

---
## Step 2 — define the standalone discriminator model

```python
def define_discriminator(n_cat, in_shape=(28,28,1)):
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
d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.1)(d)
```

---
## Step 6 — downsample to 7x7

```python
d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = LeakyReLU(alpha=0.1)(d)
	d = BatchNormalization()(d)
```

---
## Step 7 — normal

```python
d = Conv2D(256, (4,4), padding='same', kernel_initializer=init)(d)
	d = LeakyReLU(alpha=0.1)(d)
	d = BatchNormalization()(d)
```

---
## Step 8 — flatten feature maps

```python
d = Flatten()(d)
```

---
## Step 9 — real/fake output

```python
out_classifier = Dense(1, activation='sigmoid')(d)
```

---
## Step 10 — define d model

```python
d_model = Model(in_image, out_classifier)
```

---
## Step 11 — compile d model

```python
d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
```

---
## Step 12 — create q model layers

```python
q = Dense(128)(d)
	q = BatchNormalization()(q)
	q = LeakyReLU(alpha=0.1)(q)
```

---
## Step 13 — q model output

```python
out_codes = Dense(n_cat, activation='softmax')(q)
```

---
## Step 14 — define q model

```python
q_model = Model(in_image, out_codes)
	return d_model, q_model
```

---
## Step 15 — define the standalone generator model

```python
def define_generator(gen_input_size):
```

---
## Step 16 — weight initialization

```python
init = RandomNormal(stddev=0.02)
```

---
## Step 17 — image generator input

```python
in_lat = Input(shape=(gen_input_size,))
```

---
## Step 18 — foundation for 7x7 image

```python
n_nodes = 512 * 7 * 7
	gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
	gen = Reshape((7, 7, 512))(gen)
```

---
## Step 19 — normal

```python
gen = Conv2D(128, (4,4), padding='same', kernel_initializer=init)(gen)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
```

---
## Step 20 — upsample to 14x14

```python
gen = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
```

---
## Step 21 — upsample to 28x28

```python
gen = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
```

---
## Step 22 — tanh output

```python
out_layer = Activation('tanh')(gen)
```

---
## Step 23 — define model

```python
model = Model(in_lat, out_layer)
	return model
```

---
## Step 24 — define the combined discriminator, generator and q network model

```python
def define_gan(g_model, d_model, q_model):
```

---
## Step 25 — make weights in the discriminator (some shared with the q model) as not trainable

```python
for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
```

---
## Step 26 — connect g outputs to d inputs

```python
d_output = d_model(g_model.output)
```

---
## Step 27 — connect g outputs to q inputs

```python
q_output = q_model(g_model.output)
```

---
## Step 28 — define composite model

```python
model = Model(g_model.input, [d_output, q_output])
```

---
## Step 29 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt)
	return model
```

---
## Step 30 — load images

```python
def load_real_samples():
```

---
## Step 31 — load dataset

```python
(trainX, _), (_, _) = load_data()
```

---
## Step 32 — expand to 3d, e.g. add channels

```python
X = expand_dims(trainX, axis=-1)
```

---
## Step 33 — convert from ints to floats

```python
X = X.astype('float32')
```

---
## Step 34 — scale from [0,255] to [-1,1]

```python
X = (X - 127.5) / 127.5
	print(X.shape)
	return X
```

---
## Step 35 — select real samples

```python
def generate_real_samples(dataset, n_samples):
```

---
## Step 36 — choose random instances

```python
ix = randint(0, dataset.shape[0], n_samples)
```

---
## Step 37 — select images and labels

```python
X = dataset[ix]
```

---
## Step 38 — generate class labels

```python
y = ones((n_samples, 1))
	return X, y
```

---
## Step 39 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n_cat, n_samples):
```

---
## Step 40 — generate points in the latent space

```python
z_latent = randn(latent_dim * n_samples)
```

---
## Step 41 — reshape into a batch of inputs for the network

```python
z_latent = z_latent.reshape(n_samples, latent_dim)
```

---
## Step 42 — generate categorical codes

```python
cat_codes = randint(0, n_cat, n_samples)
```

---
## Step 43 — one hot encode

```python
cat_codes = to_categorical(cat_codes, num_classes=n_cat)
```

---
## Step 44 — concatenate latent points and control codes

```python
z_input = hstack((z_latent, cat_codes))
	return [z_input, cat_codes]
```

---
## Step 45 — use the generator to generate n fake examples, with class labels

```python
def generate_fake_samples(generator, latent_dim, n_cat, n_samples):
```

---
## Step 46 — generate points in latent space and control codes

```python
z_input, _ = generate_latent_points(latent_dim, n_cat, n_samples)
```

---
## Step 47 — predict outputs

```python
images = generator.predict(z_input)
```

---
## Step 48 — create class labels

```python
y = zeros((n_samples, 1))
	return images, y
```

---
## Step 49 — generate samples and save as a plot and save the model

```python
def summarize_performance(step, g_model, gan_model, latent_dim, n_cat, n_samples=100):
```

---
## Step 50 — prepare fake examples

```python
X, _ = generate_fake_samples(g_model, latent_dim, n_cat, n_samples)
```

---
## Step 51 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
```

---
## Step 52 — plot images

```python
for i in range(100):
```

---
## Step 53 — define subplot

```python
pyplot.subplot(10, 10, 1 + i)
```

---
## Step 54 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 55 — plot raw pixel data

```python
pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
```

---
## Step 56 — save plot to file

```python
filename1 = 'generated_plot_%04d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
```

---
## Step 57 — save the generator model

```python
filename2 = 'model_%04d.h5' % (step+1)
	g_model.save(filename2)
```

---
## Step 58 — save the gan model

```python
filename3 = 'gan_model_%04d.h5' % (step+1)
	gan_model.save(filename3)
	print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))
```

---
## Step 59 — train the generator and discriminator

```python
def train(g_model, d_model, gan_model, dataset, latent_dim, n_cat, n_epochs=100, n_batch=64):
```

---
## Step 60 — calculate the number of batches per training epoch

```python
bat_per_epo = int(dataset.shape[0] / n_batch)
```

---
## Step 61 — calculate the number of training iterations

```python
n_steps = bat_per_epo * n_epochs
```

---
## Step 62 — calculate the size of half a batch of samples

```python
half_batch = int(n_batch / 2)
```

---
## Step 63 — manually enumerate epochs

```python
for i in range(n_steps):
```

---
## Step 64 — get randomly selected 'real' samples

```python
X_real, y_real = generate_real_samples(dataset, half_batch)
```

---
## Step 65 — update discriminator and q model weights

```python
d_loss1 = d_model.train_on_batch(X_real, y_real)
```

---
## Step 66 — generate 'fake' examples

```python
X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_cat, half_batch)
```

---
## Step 67 — update discriminator model weights

```python
d_loss2 = d_model.train_on_batch(X_fake, y_fake)
```

---
## Step 68 — prepare points in latent space as input for the generator

```python
z_input, cat_codes = generate_latent_points(latent_dim, n_cat, n_batch)
```

---
## Step 69 — create inverted labels for the fake samples

```python
y_gan = ones((n_batch, 1))
```

---
## Step 70 — update the g via the d and q error

```python
_,g_1,g_2 = gan_model.train_on_batch(z_input, [y_gan, cat_codes])
```

---
## Step 71 — summarize loss on this batch

```python
print('>%d, d[%.3f,%.3f], g[%.3f] q[%.3f]' % (i+1, d_loss1, d_loss2, g_1, g_2))
```

---
## Step 72 — evaluate the model performance every 'epoch'

```python
if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, gan_model, latent_dim, n_cat)
```

---
## Step 73 — number of values for the categorical control code

```python
n_cat = 10
```

---
## Step 74 — size of the latent space

```python
latent_dim = 62
```

---
## Step 75 — create the discriminator

```python
d_model, q_model = define_discriminator(n_cat)
```

---
## Step 76 — create the generator

```python
gen_input_size = latent_dim + n_cat
g_model = define_generator(gen_input_size)
```

---
## Step 77 — create the gan

```python
gan_model = define_gan(g_model, d_model, q_model)
```

---
## Step 78 — load image data

```python
dataset = load_real_samples()
```

---
## Step 79 — train model

```python
train(g_model, d_model, gan_model, dataset, latent_dim, n_cat)
```

---
## Learning Notes / 学习笔记

- **概念**: example of training an infogan on mnist 是机器学习中的常用技术。  
  *example of training an infogan on mnist is a common technique in machine learning.*

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
# Train Infogan / 生成对抗网络
# Complete Code / 完整代码
# ===============================

# example of training an infogan on mnist
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy import hstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Activation
from matplotlib import pyplot

# define the standalone discriminator model
def define_discriminator(n_cat, in_shape=(28,28,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=in_shape)
	# downsample to 14x14
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.1)(d)
	# downsample to 7x7
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = LeakyReLU(alpha=0.1)(d)
	d = BatchNormalization()(d)
	# normal
	d = Conv2D(256, (4,4), padding='same', kernel_initializer=init)(d)
	d = LeakyReLU(alpha=0.1)(d)
	d = BatchNormalization()(d)
	# flatten feature maps
	d = Flatten()(d)
	# real/fake output
	out_classifier = Dense(1, activation='sigmoid')(d)
	# define d model
	d_model = Model(in_image, out_classifier)
	# compile d model
	d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
	# create q model layers
	q = Dense(128)(d)
	q = BatchNormalization()(q)
	q = LeakyReLU(alpha=0.1)(q)
	# q model output
	out_codes = Dense(n_cat, activation='softmax')(q)
	# define q model
	q_model = Model(in_image, out_codes)
	return d_model, q_model

# define the standalone generator model
def define_generator(gen_input_size):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image generator input
	in_lat = Input(shape=(gen_input_size,))
	# foundation for 7x7 image
	n_nodes = 512 * 7 * 7
	gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
	gen = Reshape((7, 7, 512))(gen)
	# normal
	gen = Conv2D(128, (4,4), padding='same', kernel_initializer=init)(gen)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
	# upsample to 14x14
	gen = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	gen = Activation('relu')(gen)
	gen = BatchNormalization()(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	# tanh output
	out_layer = Activation('tanh')(gen)
	# define model
	model = Model(in_lat, out_layer)
	return model

# define the combined discriminator, generator and q network model
def define_gan(g_model, d_model, q_model):
	# make weights in the discriminator (some shared with the q model) as not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# connect g outputs to d inputs
	d_output = d_model(g_model.output)
	# connect g outputs to q inputs
	q_output = q_model(g_model.output)
	# define composite model
	model = Model(g_model.input, [d_output, q_output])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt)
	return model

# load images
def load_real_samples():
	# load dataset
	(trainX, _), (_, _) = load_data()
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	print(X.shape)
	return X

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images and labels
	X = dataset[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_cat, n_samples):
	# generate points in the latent space
	z_latent = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_latent = z_latent.reshape(n_samples, latent_dim)
	# generate categorical codes
	cat_codes = randint(0, n_cat, n_samples)
	# one hot encode
	cat_codes = to_categorical(cat_codes, num_classes=n_cat)
	# concatenate latent points and control codes
	z_input = hstack((z_latent, cat_codes))
	return [z_input, cat_codes]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_cat, n_samples):
	# generate points in latent space and control codes
	z_input, _ = generate_latent_points(latent_dim, n_cat, n_samples)
	# predict outputs
	images = generator.predict(z_input)
	# create class labels
	y = zeros((n_samples, 1))
	return images, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, gan_model, latent_dim, n_cat, n_samples=100):
	# prepare fake examples
	X, _ = generate_fake_samples(g_model, latent_dim, n_cat, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	for i in range(100):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename1 = 'generated_plot_%04d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%04d.h5' % (step+1)
	g_model.save(filename2)
	# save the gan model
	filename3 = 'gan_model_%04d.h5' % (step+1)
	gan_model.save(filename3)
	print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_cat, n_epochs=100, n_batch=64):
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_steps):
		# get randomly selected 'real' samples
		X_real, y_real = generate_real_samples(dataset, half_batch)
		# update discriminator and q model weights
		d_loss1 = d_model.train_on_batch(X_real, y_real)
		# generate 'fake' examples
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_cat, half_batch)
		# update discriminator model weights
		d_loss2 = d_model.train_on_batch(X_fake, y_fake)
		# prepare points in latent space as input for the generator
		z_input, cat_codes = generate_latent_points(latent_dim, n_cat, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the g via the d and q error
		_,g_1,g_2 = gan_model.train_on_batch(z_input, [y_gan, cat_codes])
		# summarize loss on this batch
		print('>%d, d[%.3f,%.3f], g[%.3f] q[%.3f]' % (i+1, d_loss1, d_loss2, g_1, g_2))
		# evaluate the model performance every 'epoch'
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, gan_model, latent_dim, n_cat)

# number of values for the categorical control code
n_cat = 10
# size of the latent space
latent_dim = 62
# create the discriminator
d_model, q_model = define_discriminator(n_cat)
# create the generator
gen_input_size = latent_dim + n_cat
g_model = define_generator(gen_input_size)
# create the gan
gan_model = define_gan(g_model, d_model, q_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim, n_cat)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Inference Control Code

# 04 — Inference Control Code / 04 Inference Control Code

**Chapter 18 — File 4 of 4 / 第18章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of testing different values of the categorical control variable**.

本脚本演示 **example of testing different values of the categorical control variable**。

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
## Step 1 — example of testing different values of the categorical control variable

```python
from math import sqrt
from numpy import asarray
from numpy import hstack
from numpy.random import randn
from keras.models import load_model
from keras.utils import to_categorical
from matplotlib import pyplot
```

---
## Step 2 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n_cat, n_samples, digit):
```

---
## Step 3 — generate points in the latent space

```python
z_latent = randn(latent_dim * n_samples)
```

---
## Step 4 — reshape into a batch of inputs for the network

```python
z_latent = z_latent.reshape(n_samples, latent_dim)
```

---
## Step 5 — define categorical codes

```python
cat_codes = asarray([digit for _ in range(n_samples)])
```

---
## Step 6 — one hot encode

```python
cat_codes = to_categorical(cat_codes, num_classes=n_cat)
```

---
## Step 7 — concatenate latent points and control codes

```python
z_input = hstack((z_latent, cat_codes))
	return [z_input, cat_codes]
```

---
## Step 8 — create and save a plot of generated images

```python
def save_plot(examples, n_examples):
```

---
## Step 9 — plot images

```python
for i in range(n_examples):
```

---
## Step 10 — define subplot

```python
pyplot.subplot(sqrt(n_examples), sqrt(n_examples), 1 + i)
```

---
## Step 11 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 12 — plot raw pixel data

```python
pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()
```

---
## Step 13 — load model

```python
model = load_model('model_93700.h5')
```

---
## Step 14 — number of categorical control codes

```python
n_cat = 10
```

---
## Step 15 — size of the latent space

```python
latent_dim = 62
```

---
## Step 16 — number of examples to generate

```python
n_samples = 25
```

---
## Step 17 — define digit

```python
digit = 1
```

---
## Step 18 — generate points in latent space and control codes

```python
z_input, _ = generate_latent_points(latent_dim, n_cat, n_samples, digit)
```

---
## Step 19 — predict outputs

```python
X = model.predict(z_input)
```

---
## Step 20 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
```

---
## Step 21 — plot the result

```python
save_plot(X, n_samples)
```

---
## Learning Notes / 学习笔记

- **概念**: example of testing different values of the categorical control variable 是机器学习中的常用技术。  
  *example of testing different values of the categorical control variable is a common technique in machine learning.*

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
# Inference Control Code / 04 Inference Control Code
# Complete Code / 完整代码
# ===============================

# example of testing different values of the categorical control variable
from math import sqrt
from numpy import asarray
from numpy import hstack
from numpy.random import randn
from keras.models import load_model
from keras.utils import to_categorical
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_cat, n_samples, digit):
	# generate points in the latent space
	z_latent = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_latent = z_latent.reshape(n_samples, latent_dim)
	# define categorical codes
	cat_codes = asarray([digit for _ in range(n_samples)])
	# one hot encode
	cat_codes = to_categorical(cat_codes, num_classes=n_cat)
	# concatenate latent points and control codes
	z_input = hstack((z_latent, cat_codes))
	return [z_input, cat_codes]

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
# number of categorical control codes
n_cat = 10
# size of the latent space
latent_dim = 62
# number of examples to generate
n_samples = 25
# define digit
digit = 1
# generate points in latent space and control codes
z_input, _ = generate_latent_points(latent_dim, n_cat, n_samples, digit)
# predict outputs
X = model.predict(z_input)
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
save_plot(X, n_samples)
```

---

### Chapter Summary

# Chapter 18 Summary / 第18章总结

## Theme / 主题: Chapter 18 / Chapter 18

This chapter contains **4 code files** demonstrating chapter 18.

本章包含 **4 个代码文件**，演示Chapter 18。

---
## Evolution / 演化路线

  1. `01_define_plot_models.ipynb` — Define Plot Models
  2. `02_train_infogan.ipynb` — Train Infogan
  3. `03_inference_infogan.ipynb` — Inference Infogan
  4. `04_inference_control_code.ipynb` — Inference Control Code

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 18) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 18）是机器学习流水线中的基础构建块。

---
