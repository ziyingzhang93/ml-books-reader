# 生成对抗网络 / Generative Adversarial Networks
## Chapter 10

---

### Normal Train Gain

# 01 — Normal Train Gain / 01 Normal Train Gain

**Chapter 10 — File 1 of 4 / 第10章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of training a stable gan for generating a handwritten digit**.

本脚本演示 **example of training a stable gan for generating a handwritten digit**。

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
## Step 1 — example of training a stable gan for generating a handwritten digit

```python
from os import makedirs
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets.mnist import load_data
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import Adam
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Reshape
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LeakyReLU
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.initializers import RandomNormal
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
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
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 5 — downsample to 14x14

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 6 — downsample to 7x7

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 7 — classifier

```python
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 8 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
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
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 12 — foundation for 7x7 image

```python
n_nodes = 128 * 7 * 7
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Reshape((7, 7, 128)))
```

---
## Step 13 — upsample to 14x14

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 14 — upsample to 28x28

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 15 — output 28x28x1

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
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
discriminator.trainable = False
```

---
## Step 18 — connect them

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 19 — add generator

```python
# 向模型添加一层 / Add a layer to the model
model.add(generator)
```

---
## Step 20 — add the discriminator

```python
# 向模型添加一层 / Add a layer to the model
model.add(discriminator)
```

---
## Step 21 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt)
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
# 加载数据集 / Load dataset
(trainX, trainy), (_, _) = load_data()
```

---
## Step 24 — expand to 3d, e.g. add channels

```python
X = expand_dims(trainX, axis=-1)
```

---
## Step 25 — select all of the examples for a given class

```python
selected_ix = trainy == 8
	X = X[selected_ix]
```

---
## Step 26 — convert from ints to floats

```python
# 转换数据类型 / Convert data type
X = X.astype('float32')
```

---
## Step 27 — scale from [0,255] to [-1,1]

```python
X = (X - 127.5) / 127.5
	return X
```

---
## Step 28 — select real samples

```python
def generate_real_samples(dataset, n_samples):
```

---
## Step 29 — choose random instances

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
ix = randint(0, dataset.shape[0], n_samples)
```

---
## Step 30 — select images

```python
X = dataset[ix]
```

---
## Step 31 — generate class labels

```python
y = ones((n_samples, 1))
	return X, y
```

---
## Step 32 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n_samples):
```

---
## Step 33 — generate points in the latent space

```python
x_input = randn(latent_dim * n_samples)
```

---
## Step 34 — reshape into a batch of inputs for the network

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
```

---
## Step 35 — use the generator to generate n fake examples, with class labels

```python
def generate_fake_samples(generator, latent_dim, n_samples):
```

---
## Step 36 — generate points in latent space

```python
x_input = generate_latent_points(latent_dim, n_samples)
```

---
## Step 37 — predict outputs

```python
X = generator.predict(x_input)
```

---
## Step 38 — create class labels

```python
y = zeros((n_samples, 1))
	return X, y
```

---
## Step 39 — generate samples and save as a plot and save the model

```python
def summarize_performance(step, g_model, latent_dim, n_samples=100):
```

---
## Step 40 — prepare fake examples

```python
X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
```

---
## Step 41 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
```

---
## Step 42 — plot images

```python
# 生成整数序列 / Generate integer sequence
for i in range(10 * 10):
```

---
## Step 43 — define subplot

```python
pyplot.subplot(10, 10, 1 + i)
```

---
## Step 44 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 45 — plot raw pixel data

```python
pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
```

---
## Step 46 — save plot to file

```python
pyplot.savefig('results_baseline/generated_plot_%03d.png' % (step+1))
	pyplot.close()
```

---
## Step 47 — save the generator model

```python
# 保存模型到文件 / Save model to file
g_model.save('results_baseline/model_%03d.h5' % (step+1))
```

---
## Step 48 — create a line plot of loss for the gan and save to file

```python
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
```

---
## Step 49 — plot loss

```python
pyplot.subplot(2, 1, 1)
	pyplot.plot(d1_hist, label='d-real')
	pyplot.plot(d2_hist, label='d-fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
```

---
## Step 50 — plot discriminator accuracy

```python
pyplot.subplot(2, 1, 2)
	pyplot.plot(a1_hist, label='acc-real')
	pyplot.plot(a2_hist, label='acc-fake')
	pyplot.legend()
```

---
## Step 51 — save plot to file

```python
pyplot.savefig('results_baseline/plot_line_plot_loss.png')
	pyplot.close()
```

---
## Step 52 — train the generator and discriminator

```python
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=128):
```

---
## Step 53 — calculate the number of batches per epoch

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
bat_per_epo = int(dataset.shape[0] / n_batch)
```

---
## Step 54 — calculate the total iterations based on batch and epoch

```python
n_steps = bat_per_epo * n_epochs
```

---
## Step 55 — calculate the number of samples in half a batch

```python
half_batch = int(n_batch / 2)
```

---
## Step 56 — prepare lists for storing stats each iteration

```python
d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
```

---
## Step 57 — manually enumerate epochs

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_steps):
```

---
## Step 58 — get randomly selected 'real' samples

```python
X_real, y_real = generate_real_samples(dataset, half_batch)
```

---
## Step 59 — update discriminator model weights

```python
d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
```

---
## Step 60 — generate 'fake' examples

```python
X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
```

---
## Step 61 — update discriminator model weights

```python
d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
```

---
## Step 62 — prepare points in latent space as input for the generator

```python
X_gan = generate_latent_points(latent_dim, n_batch)
```

---
## Step 63 — create inverted labels for the fake samples

```python
y_gan = ones((n_batch, 1))
```

---
## Step 64 — update the generator via the discriminator's error

```python
g_loss = gan_model.train_on_batch(X_gan, y_gan)
```

---
## Step 65 — summarize loss on this batch

```python
# 打印输出 / Print output
print('>%d, d1=%.3f, d2=%.3f g=%.3f, a1=%d, a2=%d' %
			(i+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
```

---
## Step 66 — record history

```python
# 添加元素到列表末尾 / Append element to list end
d1_hist.append(d_loss1)
  # 添加元素到列表末尾 / Append element to list end
		d2_hist.append(d_loss2)
  # 添加元素到列表末尾 / Append element to list end
		g_hist.append(g_loss)
  # 添加元素到列表末尾 / Append element to list end
		a1_hist.append(d_acc1)
  # 添加元素到列表末尾 / Append element to list end
		a2_hist.append(d_acc2)
```

---
## Step 67 — evaluate the model performance every 'epoch'

```python
if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim)
	plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)
```

---
## Step 68 — make folder for results

```python
makedirs('results_baseline', exist_ok=True)
```

---
## Step 69 — size of the latent space

```python
latent_dim = 50
```

---
## Step 70 — create the discriminator

```python
discriminator = define_discriminator()
```

---
## Step 71 — create the generator

```python
generator = define_generator(latent_dim)
```

---
## Step 72 — create the gan

```python
gan_model = define_gan(generator, discriminator)
```

---
## Step 73 — load image data

```python
dataset = load_real_samples()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
```

---
## Step 74 — train model

```python
train(generator, discriminator, gan_model, dataset, latent_dim)
```

---
## Learning Notes / 学习笔记

- **概念**: example of training a stable gan for generating a handwritten digit 是机器学习中的常用技术。  
  *example of training a stable gan for generating a handwritten digit is a common technique in machine learning.*

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
# Normal Train Gain / 01 Normal Train Gain
# Complete Code / 完整代码
# ===============================

# example of training a stable gan for generating a handwritten digit
from os import makedirs
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets.mnist import load_data
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import Adam
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Reshape
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LeakyReLU
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.initializers import RandomNormal
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# downsample to 14x14
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 7x7
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# classifier
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# output 28x28x1
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# add generator
 # 向模型添加一层 / Add a layer to the model
	model.add(generator)
	# add the discriminator
 # 向模型添加一层 / Add a layer to the model
	model.add(discriminator)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load mnist images
def load_real_samples():
	# load dataset
 # 加载数据集 / Load dataset
	(trainX, trainy), (_, _) = load_data()
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# select all of the examples for a given class
	selected_ix = trainy == 8
	X = X[selected_ix]
	# convert from ints to floats
 # 转换数据类型 / Convert data type
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
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
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
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
 # 生成整数序列 / Generate integer sequence
	for i in range(10 * 10):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	pyplot.savefig('results_baseline/generated_plot_%03d.png' % (step+1))
	pyplot.close()
	# save the generator model
 # 保存模型到文件 / Save model to file
	g_model.save('results_baseline/model_%03d.h5' % (step+1))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
	# plot loss
	pyplot.subplot(2, 1, 1)
	pyplot.plot(d1_hist, label='d-real')
	pyplot.plot(d2_hist, label='d-fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	# plot discriminator accuracy
	pyplot.subplot(2, 1, 2)
	pyplot.plot(a1_hist, label='acc-real')
	pyplot.plot(a2_hist, label='acc-fake')
	pyplot.legend()
	# save plot to file
	pyplot.savefig('results_baseline/plot_line_plot_loss.png')
	pyplot.close()

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=128):
	# calculate the number of batches per epoch
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the total iterations based on batch and epoch
	n_steps = bat_per_epo * n_epochs
	# calculate the number of samples in half a batch
	half_batch = int(n_batch / 2)
	# prepare lists for storing stats each iteration
	d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
	# manually enumerate epochs
 # 生成整数序列 / Generate integer sequence
	for i in range(n_steps):
		# get randomly selected 'real' samples
		X_real, y_real = generate_real_samples(dataset, half_batch)
		# update discriminator model weights
		d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
		# generate 'fake' examples
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator model weights
		d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
		# prepare points in latent space as input for the generator
		X_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		# summarize loss on this batch
  # 打印输出 / Print output
		print('>%d, d1=%.3f, d2=%.3f g=%.3f, a1=%d, a2=%d' %
			(i+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
		# record history
  # 添加元素到列表末尾 / Append element to list end
		d1_hist.append(d_loss1)
  # 添加元素到列表末尾 / Append element to list end
		d2_hist.append(d_loss2)
  # 添加元素到列表末尾 / Append element to list end
		g_hist.append(g_loss)
  # 添加元素到列表末尾 / Append element to list end
		a1_hist.append(d_acc1)
  # 添加元素到列表末尾 / Append element to list end
		a2_hist.append(d_acc2)
		# evaluate the model performance every 'epoch'
		if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim)
	plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)

# make folder for results
makedirs('results_baseline', exist_ok=True)
# size of the latent space
latent_dim = 50
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Mode Collapse

# 02 — Mode Collapse / 02 Mode Collapse

**Chapter 10 — File 2 of 4 / 第10章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of training an unstable gan for generating a handwritten digit**.

本脚本演示 **example of training an unstable gan for generating a handwritten digit**。

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
## Step 1 — example of training an unstable gan for generating a handwritten digit

```python
from os import makedirs
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets.mnist import load_data
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import Adam
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Reshape
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LeakyReLU
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.initializers import RandomNormal
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
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
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 5 — downsample to 14x14

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 6 — downsample to 7x7

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 7 — classifier

```python
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 8 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
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
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 12 — foundation for 7x7 image

```python
n_nodes = 128 * 7 * 7
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Reshape((7, 7, 128)))
```

---
## Step 13 — upsample to 14x14

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 14 — upsample to 28x28

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 15 — output 28x28x1

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
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
discriminator.trainable = False
```

---
## Step 18 — connect them

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 19 — add generator

```python
# 向模型添加一层 / Add a layer to the model
model.add(generator)
```

---
## Step 20 — add the discriminator

```python
# 向模型添加一层 / Add a layer to the model
model.add(discriminator)
```

---
## Step 21 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt)
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
# 加载数据集 / Load dataset
(trainX, trainy), (_, _) = load_data()
```

---
## Step 24 — expand to 3d, e.g. add channels

```python
X = expand_dims(trainX, axis=-1)
```

---
## Step 25 — select all of the examples for a given class

```python
selected_ix = trainy == 8
	X = X[selected_ix]
```

---
## Step 26 — convert from ints to floats

```python
# 转换数据类型 / Convert data type
X = X.astype('float32')
```

---
## Step 27 — scale from [0,255] to [-1,1]

```python
X = (X - 127.5) / 127.5
	return X
```

---
## Step 28 — # select real samples

```python
def generate_real_samples(dataset, n_samples):
```

---
## Step 29 — choose random instances

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
ix = randint(0, dataset.shape[0], n_samples)
```

---
## Step 30 — select images

```python
X = dataset[ix]
```

---
## Step 31 — generate class labels

```python
y = ones((n_samples, 1))
	return X, y
```

---
## Step 32 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n_samples):
```

---
## Step 33 — generate points in the latent space

```python
x_input = randn(latent_dim * n_samples)
```

---
## Step 34 — reshape into a batch of inputs for the network

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
```

---
## Step 35 — use the generator to generate n fake examples, with class labels

```python
def generate_fake_samples(generator, latent_dim, n_samples):
```

---
## Step 36 — generate points in latent space

```python
x_input = generate_latent_points(latent_dim, n_samples)
```

---
## Step 37 — predict outputs

```python
X = generator.predict(x_input)
```

---
## Step 38 — create class labels

```python
y = zeros((n_samples, 1))
	return X, y
```

---
## Step 39 — generate samples and save as a plot and save the model

```python
def summarize_performance(step, g_model, latent_dim, n_samples=100):
```

---
## Step 40 — prepare fake examples

```python
X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
```

---
## Step 41 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
```

---
## Step 42 — plot images

```python
# 生成整数序列 / Generate integer sequence
for i in range(10 * 10):
```

---
## Step 43 — define subplot

```python
pyplot.subplot(10, 10, 1 + i)
```

---
## Step 44 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 45 — plot raw pixel data

```python
pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
```

---
## Step 46 — save plot to file

```python
pyplot.savefig('results_collapse/generated_plot_%03d.png' % (step+1))
	pyplot.close()
```

---
## Step 47 — save the generator model

```python
# 保存模型到文件 / Save model to file
g_model.save('results_collapse/model_%03d.h5' % (step+1))
```

---
## Step 48 — create a line plot of loss for the gan and save to file

```python
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
```

---
## Step 49 — plot loss

```python
pyplot.subplot(2, 1, 1)
	pyplot.plot(d1_hist, label='d-real')
	pyplot.plot(d2_hist, label='d-fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
```

---
## Step 50 — plot discriminator accuracy

```python
pyplot.subplot(2, 1, 2)
	pyplot.plot(a1_hist, label='acc-real')
	pyplot.plot(a2_hist, label='acc-fake')
	pyplot.legend()
```

---
## Step 51 — save plot to file

```python
pyplot.savefig('results_collapse/plot_line_plot_loss.png')
	pyplot.close()
```

---
## Step 52 — train the generator and discriminator

```python
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=128):
```

---
## Step 53 — calculate the number of batches per epoch

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
bat_per_epo = int(dataset.shape[0] / n_batch)
```

---
## Step 54 — calculate the total iterations based on batch and epoch

```python
n_steps = bat_per_epo * n_epochs
```

---
## Step 55 — calculate the number of samples in half a batch

```python
half_batch = int(n_batch / 2)
```

---
## Step 56 — prepare lists for storing stats each iteration

```python
d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
```

---
## Step 57 — manually enumerate epochs

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_steps):
```

---
## Step 58 — get randomly selected 'real' samples

```python
X_real, y_real = generate_real_samples(dataset, half_batch)
```

---
## Step 59 — update discriminator model weights

```python
d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
```

---
## Step 60 — generate 'fake' examples

```python
X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
```

---
## Step 61 — update discriminator model weights

```python
d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
```

---
## Step 62 — prepare points in latent space as input for the generator

```python
X_gan = generate_latent_points(latent_dim, n_batch)
```

---
## Step 63 — create inverted labels for the fake samples

```python
y_gan = ones((n_batch, 1))
```

---
## Step 64 — update the generator via the discriminator's error

```python
g_loss = gan_model.train_on_batch(X_gan, y_gan)
```

---
## Step 65 — summarize loss on this batch

```python
# 打印输出 / Print output
print('>%d, d1=%.3f, d2=%.3f g=%.3f, a1=%d, a2=%d' %
			(i+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
```

---
## Step 66 — record history

```python
# 添加元素到列表末尾 / Append element to list end
d1_hist.append(d_loss1)
  # 添加元素到列表末尾 / Append element to list end
		d2_hist.append(d_loss2)
  # 添加元素到列表末尾 / Append element to list end
		g_hist.append(g_loss)
  # 添加元素到列表末尾 / Append element to list end
		a1_hist.append(d_acc1)
  # 添加元素到列表末尾 / Append element to list end
		a2_hist.append(d_acc2)
```

---
## Step 67 — evaluate the model performance every 'epoch'

```python
if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim)
	plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)
```

---
## Step 68 — make folder for results

```python
makedirs('results_collapse', exist_ok=True)
```

---
## Step 69 — size of the latent space

```python
latent_dim = 1
```

---
## Step 70 — create the discriminator

```python
discriminator = define_discriminator()
```

---
## Step 71 — create the generator

```python
generator = define_generator(latent_dim)
```

---
## Step 72 — create the gan

```python
gan_model = define_gan(generator, discriminator)
```

---
## Step 73 — load image data

```python
dataset = load_real_samples()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
```

---
## Step 74 — train model

```python
train(generator, discriminator, gan_model, dataset, latent_dim)
```

---
## Learning Notes / 学习笔记

- **概念**: example of training an unstable gan for generating a handwritten digit 是机器学习中的常用技术。  
  *example of training an unstable gan for generating a handwritten digit is a common technique in machine learning.*

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
# Mode Collapse / 02 Mode Collapse
# Complete Code / 完整代码
# ===============================

# example of training an unstable gan for generating a handwritten digit
from os import makedirs
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets.mnist import load_data
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import Adam
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Reshape
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LeakyReLU
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.initializers import RandomNormal
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# downsample to 14x14
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 7x7
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# classifier
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# output 28x28x1
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# add generator
 # 向模型添加一层 / Add a layer to the model
	model.add(generator)
	# add the discriminator
 # 向模型添加一层 / Add a layer to the model
	model.add(discriminator)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load mnist images
def load_real_samples():
	# load dataset
 # 加载数据集 / Load dataset
	(trainX, trainy), (_, _) = load_data()
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# select all of the examples for a given class
	selected_ix = trainy == 8
	X = X[selected_ix]
	# convert from ints to floats
 # 转换数据类型 / Convert data type
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X

# # select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
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
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
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
 # 生成整数序列 / Generate integer sequence
	for i in range(10 * 10):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	pyplot.savefig('results_collapse/generated_plot_%03d.png' % (step+1))
	pyplot.close()
	# save the generator model
 # 保存模型到文件 / Save model to file
	g_model.save('results_collapse/model_%03d.h5' % (step+1))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
	# plot loss
	pyplot.subplot(2, 1, 1)
	pyplot.plot(d1_hist, label='d-real')
	pyplot.plot(d2_hist, label='d-fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	# plot discriminator accuracy
	pyplot.subplot(2, 1, 2)
	pyplot.plot(a1_hist, label='acc-real')
	pyplot.plot(a2_hist, label='acc-fake')
	pyplot.legend()
	# save plot to file
	pyplot.savefig('results_collapse/plot_line_plot_loss.png')
	pyplot.close()

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=128):
	# calculate the number of batches per epoch
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the total iterations based on batch and epoch
	n_steps = bat_per_epo * n_epochs
	# calculate the number of samples in half a batch
	half_batch = int(n_batch / 2)
	# prepare lists for storing stats each iteration
	d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
	# manually enumerate epochs
 # 生成整数序列 / Generate integer sequence
	for i in range(n_steps):
		# get randomly selected 'real' samples
		X_real, y_real = generate_real_samples(dataset, half_batch)
		# update discriminator model weights
		d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
		# generate 'fake' examples
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator model weights
		d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
		# prepare points in latent space as input for the generator
		X_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		# summarize loss on this batch
  # 打印输出 / Print output
		print('>%d, d1=%.3f, d2=%.3f g=%.3f, a1=%d, a2=%d' %
			(i+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
		# record history
  # 添加元素到列表末尾 / Append element to list end
		d1_hist.append(d_loss1)
  # 添加元素到列表末尾 / Append element to list end
		d2_hist.append(d_loss2)
  # 添加元素到列表末尾 / Append element to list end
		g_hist.append(g_loss)
  # 添加元素到列表末尾 / Append element to list end
		a1_hist.append(d_acc1)
  # 添加元素到列表末尾 / Append element to list end
		a2_hist.append(d_acc2)
		# evaluate the model performance every 'epoch'
		if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim)
	plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)

# make folder for results
makedirs('results_collapse', exist_ok=True)
# size of the latent space
latent_dim = 1
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Convergence Failure

# 03 — Convergence Failure / 03 Convergence Failure

**Chapter 10 — File 3 of 4 / 第10章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of training an unstable gan for generating a handwritten digit**.

本脚本演示 **example of training an unstable gan for generating a handwritten digit**。

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
## Step 1 — example of training an unstable gan for generating a handwritten digit

```python
from os import makedirs
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets.mnist import load_data
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import Adam
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Reshape
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LeakyReLU
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.initializers import RandomNormal
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
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
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 5 — downsample to 14x14

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 6 — downsample to 7x7

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 7 — classifier

```python
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 8 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
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
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 12 — foundation for 7x7 image

```python
n_nodes = 128 * 7 * 7
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Reshape((7, 7, 128)))
```

---
## Step 13 — upsample to 14x14

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 14 — upsample to 28x28

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 15 — output 28x28x1

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
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
discriminator.trainable = False
```

---
## Step 18 — connect them

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 19 — add generator

```python
# 向模型添加一层 / Add a layer to the model
model.add(generator)
```

---
## Step 20 — add the discriminator

```python
# 向模型添加一层 / Add a layer to the model
model.add(discriminator)
```

---
## Step 21 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt)
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
# 加载数据集 / Load dataset
(trainX, trainy), (_, _) = load_data()
```

---
## Step 24 — expand to 3d, e.g. add channels

```python
X = expand_dims(trainX, axis=-1)
```

---
## Step 25 — select all of the examples for a given class

```python
selected_ix = trainy == 8
	X = X[selected_ix]
```

---
## Step 26 — convert from ints to floats

```python
# 转换数据类型 / Convert data type
X = X.astype('float32')
```

---
## Step 27 — scale from [0,255] to [-1,1]

```python
X = (X - 127.5) / 127.5
	return X
```

---
## Step 28 — # select real samples

```python
def generate_real_samples(dataset, n_samples):
```

---
## Step 29 — choose random instances

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
ix = randint(0, dataset.shape[0], n_samples)
```

---
## Step 30 — select images

```python
X = dataset[ix]
```

---
## Step 31 — generate class labels

```python
y = ones((n_samples, 1))
	return X, y
```

---
## Step 32 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n_samples):
```

---
## Step 33 — generate points in the latent space

```python
x_input = randn(latent_dim * n_samples)
```

---
## Step 34 — reshape into a batch of inputs for the network

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
```

---
## Step 35 — use the generator to generate n fake examples, with class labels

```python
def generate_fake_samples(generator, latent_dim, n_samples):
```

---
## Step 36 — generate points in latent space

```python
x_input = generate_latent_points(latent_dim, n_samples)
```

---
## Step 37 — predict outputs

```python
X = generator.predict(x_input)
```

---
## Step 38 — create class labels

```python
y = zeros((n_samples, 1))
	return X, y
```

---
## Step 39 — generate samples and save as a plot and save the model

```python
def summarize_performance(step, g_model, latent_dim, n_samples=100):
```

---
## Step 40 — prepare fake examples

```python
X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
```

---
## Step 41 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
```

---
## Step 42 — plot images

```python
# 生成整数序列 / Generate integer sequence
for i in range(10 * 10):
```

---
## Step 43 — define subplot

```python
pyplot.subplot(10, 10, 1 + i)
```

---
## Step 44 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 45 — plot raw pixel data

```python
pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
```

---
## Step 46 — save plot to file

```python
pyplot.savefig('results_convergence/generated_plot_%03d.png' % (step+1))
	pyplot.close()
```

---
## Step 47 — save the generator model

```python
# 保存模型到文件 / Save model to file
g_model.save('results_convergence/model_%03d.h5' % (step+1))
```

---
## Step 48 — create a line plot of loss for the gan and save to file

```python
def plot_history(d_hist, g_hist, a_hist):
```

---
## Step 49 — plot loss

```python
pyplot.subplot(2, 1, 1)
	pyplot.plot(d_hist, label='dis')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
```

---
## Step 50 — plot discriminator accuracy

```python
pyplot.subplot(2, 1, 2)
	pyplot.plot(a_hist, label='acc')
	pyplot.legend()
```

---
## Step 51 — save plot to file

```python
pyplot.savefig('results_convergence/plot_line_plot_loss.png')
	pyplot.close()
```

---
## Step 52 — train the generator and discriminator

```python
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=128):
```

---
## Step 53 — calculate the number of batches per epoch

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
bat_per_epo = int(dataset.shape[0] / n_batch)
```

---
## Step 54 — calculate the total iterations based on batch and epoch

```python
n_steps = bat_per_epo * n_epochs
```

---
## Step 55 — calculate the number of samples in half a batch

```python
half_batch = int(n_batch / 2)
```

---
## Step 56 — prepare lists for storing stats each iteration

```python
d_hist, g_hist, a_hist = list(), list(), list()
```

---
## Step 57 — manually enumerate epochs

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_steps):
```

---
## Step 58 — get randomly selected 'real' samples

```python
X_real, y_real = generate_real_samples(dataset, half_batch)
```

---
## Step 59 — generate 'fake' examples

```python
X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
```

---
## Step 60 — combine into one batch

```python
X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
```

---
## Step 61 — update discriminator model weights

```python
d_loss, d_acc = d_model.train_on_batch(X, y)
```

---
## Step 62 — prepare points in latent space as input for the generator

```python
X_gan = generate_latent_points(latent_dim, n_batch)
```

---
## Step 63 — create inverted labels for the fake samples

```python
y_gan = ones((n_batch, 1))
```

---
## Step 64 — update the generator via the discriminator's error

```python
g_loss = gan_model.train_on_batch(X_gan, y_gan)
```

---
## Step 65 — summarize loss on this batch

```python
# 打印输出 / Print output
print('>%d, d=%.3f, g=%.3f, a=%d' % (i+1, d_loss, g_loss, int(100*d_acc)))
```

---
## Step 66 — record history

```python
# 添加元素到列表末尾 / Append element to list end
d_hist.append(d_loss)
  # 添加元素到列表末尾 / Append element to list end
		g_hist.append(g_loss)
  # 添加元素到列表末尾 / Append element to list end
		a_hist.append(d_acc)
```

---
## Step 67 — evaluate the model performance every 'epoch'

```python
if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim)
	plot_history(d_hist, g_hist, a_hist)
```

---
## Step 68 — make folder for results

```python
makedirs('results_convergence', exist_ok=True)
```

---
## Step 69 — size of the latent space

```python
latent_dim = 50
```

---
## Step 70 — create the discriminator

```python
discriminator = define_discriminator()
```

---
## Step 71 — create the generator

```python
generator = define_generator(latent_dim)
```

---
## Step 72 — create the gan

```python
gan_model = define_gan(generator, discriminator)
```

---
## Step 73 — load image data

```python
dataset = load_real_samples()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
```

---
## Step 74 — train model

```python
train(generator, discriminator, gan_model, dataset, latent_dim)
```

---
## Learning Notes / 学习笔记

- **概念**: example of training an unstable gan for generating a handwritten digit 是机器学习中的常用技术。  
  *example of training an unstable gan for generating a handwritten digit is a common technique in machine learning.*

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
# Convergence Failure / 03 Convergence Failure
# Complete Code / 完整代码
# ===============================

# example of training an unstable gan for generating a handwritten digit
from os import makedirs
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets.mnist import load_data
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import Adam
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Reshape
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LeakyReLU
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.initializers import RandomNormal
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# downsample to 14x14
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 7x7
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# classifier
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# output 28x28x1
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# add generator
 # 向模型添加一层 / Add a layer to the model
	model.add(generator)
	# add the discriminator
 # 向模型添加一层 / Add a layer to the model
	model.add(discriminator)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load mnist images
def load_real_samples():
	# load dataset
 # 加载数据集 / Load dataset
	(trainX, trainy), (_, _) = load_data()
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# select all of the examples for a given class
	selected_ix = trainy == 8
	X = X[selected_ix]
	# convert from ints to floats
 # 转换数据类型 / Convert data type
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X

# # select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
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
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
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
 # 生成整数序列 / Generate integer sequence
	for i in range(10 * 10):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	pyplot.savefig('results_convergence/generated_plot_%03d.png' % (step+1))
	pyplot.close()
	# save the generator model
 # 保存模型到文件 / Save model to file
	g_model.save('results_convergence/model_%03d.h5' % (step+1))

# create a line plot of loss for the gan and save to file
def plot_history(d_hist, g_hist, a_hist):
	# plot loss
	pyplot.subplot(2, 1, 1)
	pyplot.plot(d_hist, label='dis')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	# plot discriminator accuracy
	pyplot.subplot(2, 1, 2)
	pyplot.plot(a_hist, label='acc')
	pyplot.legend()
	# save plot to file
	pyplot.savefig('results_convergence/plot_line_plot_loss.png')
	pyplot.close()

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=128):
	# calculate the number of batches per epoch
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the total iterations based on batch and epoch
	n_steps = bat_per_epo * n_epochs
	# calculate the number of samples in half a batch
	half_batch = int(n_batch / 2)
	# prepare lists for storing stats each iteration
	d_hist, g_hist, a_hist = list(), list(), list()
	# manually enumerate epochs
 # 生成整数序列 / Generate integer sequence
	for i in range(n_steps):
		# get randomly selected 'real' samples
		X_real, y_real = generate_real_samples(dataset, half_batch)
		# generate 'fake' examples
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# combine into one batch
		X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
		# update discriminator model weights
		d_loss, d_acc = d_model.train_on_batch(X, y)
		# prepare points in latent space as input for the generator
		X_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		# summarize loss on this batch
  # 打印输出 / Print output
		print('>%d, d=%.3f, g=%.3f, a=%d' % (i+1, d_loss, g_loss, int(100*d_acc)))
		# record history
  # 添加元素到列表末尾 / Append element to list end
		d_hist.append(d_loss)
  # 添加元素到列表末尾 / Append element to list end
		g_hist.append(g_loss)
  # 添加元素到列表末尾 / Append element to list end
		a_hist.append(d_acc)
		# evaluate the model performance every 'epoch'
		if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim)
	plot_history(d_hist, g_hist, a_hist)

# make folder for results
makedirs('results_convergence', exist_ok=True)
# size of the latent space
latent_dim = 50
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Different Convergence Failure

# 04 — Different Convergence Failure / 04 Different Convergence Failure

**Chapter 10 — File 4 of 4 / 第10章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of training an unstable gan for generating a handwritten digit**.

本脚本演示 **example of training an unstable gan for generating a handwritten digit**。

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
## Step 1 — example of training an unstable gan for generating a handwritten digit

```python
from os import makedirs
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets.mnist import load_data
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Reshape
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LeakyReLU
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.initializers import RandomNormal
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
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
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 5 — downsample to 14x14

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 6 — downsample to 7x7

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 7 — classifier

```python
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 8 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 12 — foundation for 7x7 image

```python
n_nodes = 128 * 7 * 7
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Reshape((7, 7, 128)))
```

---
## Step 13 — upsample to 14x14

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 14 — upsample to 28x28

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 15 — output 28x28x1

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
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
discriminator.trainable = False
```

---
## Step 18 — connect them

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 19 — add generator

```python
# 向模型添加一层 / Add a layer to the model
model.add(generator)
```

---
## Step 20 — add the discriminator

```python
# 向模型添加一层 / Add a layer to the model
model.add(discriminator)
```

---
## Step 21 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam')
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
# 加载数据集 / Load dataset
(trainX, trainy), (_, _) = load_data()
```

---
## Step 24 — expand to 3d, e.g. add channels

```python
X = expand_dims(trainX, axis=-1)
```

---
## Step 25 — select all of the examples for a given class

```python
selected_ix = trainy == 8
	X = X[selected_ix]
```

---
## Step 26 — convert from ints to floats

```python
# 转换数据类型 / Convert data type
X = X.astype('float32')
```

---
## Step 27 — scale from [0,255] to [-1,1]

```python
X = (X - 127.5) / 127.5
	return X
```

---
## Step 28 — select real samples

```python
def generate_real_samples(dataset, n_samples):
```

---
## Step 29 — choose random instances

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
ix = randint(0, dataset.shape[0], n_samples)
```

---
## Step 30 — select images

```python
X = dataset[ix]
```

---
## Step 31 — generate class labels

```python
y = ones((n_samples, 1))
	return X, y
```

---
## Step 32 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n_samples):
```

---
## Step 33 — generate points in the latent space

```python
x_input = randn(latent_dim * n_samples)
```

---
## Step 34 — reshape into a batch of inputs for the network

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
```

---
## Step 35 — use the generator to generate n fake examples, with class labels

```python
def generate_fake_samples(generator, latent_dim, n_samples):
```

---
## Step 36 — generate points in latent space

```python
x_input = generate_latent_points(latent_dim, n_samples)
```

---
## Step 37 — predict outputs

```python
X = generator.predict(x_input)
```

---
## Step 38 — create class labels

```python
y = zeros((n_samples, 1))
	return X, y
```

---
## Step 39 — generate samples and save as a plot and save the model

```python
def summarize_performance(step, g_model, latent_dim, n_samples=100):
```

---
## Step 40 — prepare fake examples

```python
X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
```

---
## Step 41 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
```

---
## Step 42 — plot images

```python
# 生成整数序列 / Generate integer sequence
for i in range(10 * 10):
```

---
## Step 43 — define subplot

```python
pyplot.subplot(10, 10, 1 + i)
```

---
## Step 44 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 45 — plot raw pixel data

```python
pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
```

---
## Step 46 — save plot to file

```python
pyplot.savefig('results_opt/generated_plot_%03d.png' % (step+1))
	pyplot.close()
```

---
## Step 47 — save the generator model

```python
# 保存模型到文件 / Save model to file
g_model.save('results_opt/model_%03d.h5' % (step+1))
```

---
## Step 48 — create a line plot of loss for the gan and save to file

```python
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
```

---
## Step 49 — plot loss

```python
pyplot.subplot(2, 1, 1)
	pyplot.plot(d1_hist, label='d-real')
	pyplot.plot(d2_hist, label='d-fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
```

---
## Step 50 — plot discriminator accuracy

```python
pyplot.subplot(2, 1, 2)
	pyplot.plot(a1_hist, label='acc-real')
	pyplot.plot(a2_hist, label='acc-fake')
	pyplot.legend()
```

---
## Step 51 — save plot to file

```python
pyplot.savefig('results_opt/plot_line_plot_loss.png')
	pyplot.close()
```

---
## Step 52 — train the generator and discriminator

```python
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=128):
```

---
## Step 53 — calculate the number of batches per epoch

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
bat_per_epo = int(dataset.shape[0] / n_batch)
```

---
## Step 54 — calculate the total iterations based on batch and epoch

```python
n_steps = bat_per_epo * n_epochs
```

---
## Step 55 — calculate the number of samples in half a batch

```python
half_batch = int(n_batch / 2)
```

---
## Step 56 — prepare lists for storing stats each iteration

```python
d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
```

---
## Step 57 — manually enumerate epochs

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_steps):
```

---
## Step 58 — get randomly selected 'real' samples

```python
X_real, y_real = generate_real_samples(dataset, half_batch)
```

---
## Step 59 — update discriminator model weights

```python
d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
```

---
## Step 60 — generate 'fake' examples

```python
X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
```

---
## Step 61 — update discriminator model weights

```python
d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
```

---
## Step 62 — prepare points in latent space as input for the generator

```python
X_gan = generate_latent_points(latent_dim, n_batch)
```

---
## Step 63 — create inverted labels for the fake samples

```python
y_gan = ones((n_batch, 1))
```

---
## Step 64 — update the generator via the discriminator's error

```python
g_loss = gan_model.train_on_batch(X_gan, y_gan)
```

---
## Step 65 — summarize loss on this batch

```python
# 打印输出 / Print output
print('>%d, d1=%.3f, d2=%.3f g=%.3f, a1=%d, a2=%d' %
			(i+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
```

---
## Step 66 — record history

```python
# 添加元素到列表末尾 / Append element to list end
d1_hist.append(d_loss1)
  # 添加元素到列表末尾 / Append element to list end
		d2_hist.append(d_loss2)
  # 添加元素到列表末尾 / Append element to list end
		g_hist.append(g_loss)
  # 添加元素到列表末尾 / Append element to list end
		a1_hist.append(d_acc1)
  # 添加元素到列表末尾 / Append element to list end
		a2_hist.append(d_acc2)
```

---
## Step 67 — evaluate the model performance every 'epoch'

```python
if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim)
	plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)
```

---
## Step 68 — make folder for results

```python
makedirs('results_opt', exist_ok=True)
```

---
## Step 69 — size of the latent space

```python
latent_dim = 50
```

---
## Step 70 — create the discriminator

```python
discriminator = define_discriminator()
```

---
## Step 71 — create the generator

```python
generator = define_generator(latent_dim)
```

---
## Step 72 — create the gan

```python
gan_model = define_gan(generator, discriminator)
```

---
## Step 73 — load image data

```python
dataset = load_real_samples()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
```

---
## Step 74 — train model

```python
train(generator, discriminator, gan_model, dataset, latent_dim)
```

---
## Learning Notes / 学习笔记

- **概念**: example of training an unstable gan for generating a handwritten digit 是机器学习中的常用技术。  
  *example of training an unstable gan for generating a handwritten digit is a common technique in machine learning.*

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
# Different Convergence Failure / 04 Different Convergence Failure
# Complete Code / 完整代码
# ===============================

# example of training an unstable gan for generating a handwritten digit
from os import makedirs
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets.mnist import load_data
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Reshape
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LeakyReLU
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.initializers import RandomNormal
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# downsample to 14x14
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 7x7
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# classifier
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# compile model
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# output 28x28x1
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# add generator
 # 向模型添加一层 / Add a layer to the model
	model.add(generator)
	# add the discriminator
 # 向模型添加一层 / Add a layer to the model
	model.add(discriminator)
	# compile model
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

# load mnist images
def load_real_samples():
	# load dataset
 # 加载数据集 / Load dataset
	(trainX, trainy), (_, _) = load_data()
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# select all of the examples for a given class
	selected_ix = trainy == 8
	X = X[selected_ix]
	# convert from ints to floats
 # 转换数据类型 / Convert data type
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
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
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
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
 # 生成整数序列 / Generate integer sequence
	for i in range(10 * 10):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	pyplot.savefig('results_opt/generated_plot_%03d.png' % (step+1))
	pyplot.close()
	# save the generator model
 # 保存模型到文件 / Save model to file
	g_model.save('results_opt/model_%03d.h5' % (step+1))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
	# plot loss
	pyplot.subplot(2, 1, 1)
	pyplot.plot(d1_hist, label='d-real')
	pyplot.plot(d2_hist, label='d-fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	# plot discriminator accuracy
	pyplot.subplot(2, 1, 2)
	pyplot.plot(a1_hist, label='acc-real')
	pyplot.plot(a2_hist, label='acc-fake')
	pyplot.legend()
	# save plot to file
	pyplot.savefig('results_opt/plot_line_plot_loss.png')
	pyplot.close()

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=128):
	# calculate the number of batches per epoch
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the total iterations based on batch and epoch
	n_steps = bat_per_epo * n_epochs
	# calculate the number of samples in half a batch
	half_batch = int(n_batch / 2)
	# prepare lists for storing stats each iteration
	d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
	# manually enumerate epochs
 # 生成整数序列 / Generate integer sequence
	for i in range(n_steps):
		# get randomly selected 'real' samples
		X_real, y_real = generate_real_samples(dataset, half_batch)
		# update discriminator model weights
		d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
		# generate 'fake' examples
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator model weights
		d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
		# prepare points in latent space as input for the generator
		X_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		# summarize loss on this batch
  # 打印输出 / Print output
		print('>%d, d1=%.3f, d2=%.3f g=%.3f, a1=%d, a2=%d' %
			(i+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
		# record history
  # 添加元素到列表末尾 / Append element to list end
		d1_hist.append(d_loss1)
  # 添加元素到列表末尾 / Append element to list end
		d2_hist.append(d_loss2)
  # 添加元素到列表末尾 / Append element to list end
		g_hist.append(g_loss)
  # 添加元素到列表末尾 / Append element to list end
		a1_hist.append(d_acc1)
  # 添加元素到列表末尾 / Append element to list end
		a2_hist.append(d_acc2)
		# evaluate the model performance every 'epoch'
		if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim)
	plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)

# make folder for results
makedirs('results_opt', exist_ok=True)
# size of the latent space
latent_dim = 50
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)
```

---

### Chapter Summary / 章节总结

# Chapter 10 Summary / 第10章总结

## Theme / 主题: Chapter 10 / Chapter 10

This chapter contains **4 code files** demonstrating chapter 10.

本章包含 **4 个代码文件**，演示Chapter 10。

---
## Evolution / 演化路线

  1. `01_normal_train_gain.ipynb` — Normal Train Gain
  2. `02_mode_collapse.ipynb` — Mode Collapse
  3. `03_convergence_failure.ipynb` — Convergence Failure
  4. `04_different_convergence_failure.ipynb` — Different Convergence Failure

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 10) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 10）是机器学习流水线中的基础构建块。

---
