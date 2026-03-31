# 生成对抗网络 / Generative Adversarial Networks
## Chapter 16

---

### Train Wgan

# 01 — Train Wgan / 生成对抗网络

**Chapter 16 — File 1 of 2 / 第16章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **example of a wgan for generating handwritten digits**.

本脚本演示 **example of a wgan for generating handwritten digits**。

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
## Step 1 — example of a wgan for generating handwritten digits

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets.mnist import load_data
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras import backend
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import RMSprop
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
from keras.layers import BatchNormalization
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.initializers import RandomNormal
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.constraints import Constraint
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — clip model weights to a given hypercube

```python
class ClipConstraint(Constraint):
```

---
## Step 3 — set clip value when initialized

```python
# 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
def __init__(self, clip_value):
		self.clip_value = clip_value
```

---
## Step 4 — clip model weights to hypercube

```python
def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
```

---
## Step 5 — get the config

```python
def get_config(self):
		return {'clip_value': self.clip_value}
```

---
## Step 6 — calculate wasserstein loss

```python
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)
```

---
## Step 7 — define the standalone critic model

```python
def define_critic(in_shape=(28,28,1)):
```

---
## Step 8 — weight initialization

```python
init = RandomNormal(stddev=0.02)
```

---
## Step 9 — weight constraint

```python
const = ClipConstraint(0.01)
```

---
## Step 10 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 11 — downsample to 14x14

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
 # 向模型添加一层 / Add a layer to the model
	model.add(BatchNormalization())
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 12 — downsample to 7x7

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
 # 向模型添加一层 / Add a layer to the model
	model.add(BatchNormalization())
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 13 — scoring, linear activation

```python
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1))
```

---
## Step 14 — compile model

```python
opt = RMSprop(lr=0.00005)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model
```

---
## Step 15 — define the standalone generator model

```python
def define_generator(latent_dim):
```

---
## Step 16 — weight initialization

```python
init = RandomNormal(stddev=0.02)
```

---
## Step 17 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 18 — foundation for 7x7 image

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
## Step 19 — upsample to 14x14

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(BatchNormalization())
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 20 — upsample to 28x28

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(BatchNormalization())
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 21 — output 28x28x1

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
	return model
```

---
## Step 22 — define the combined generator and critic model, for updating the generator

```python
def define_gan(generator, critic):
```

---
## Step 23 — make weights in the critic not trainable

```python
for layer in critic.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
```

---
## Step 24 — connect them

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 25 — add generator

```python
# 向模型添加一层 / Add a layer to the model
model.add(generator)
```

---
## Step 26 — add the critic

```python
# 向模型添加一层 / Add a layer to the model
model.add(critic)
```

---
## Step 27 — compile model

```python
opt = RMSprop(lr=0.00005)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model
```

---
## Step 28 — load images

```python
def load_real_samples():
```

---
## Step 29 — load dataset

```python
# 加载数据集 / Load dataset
(trainX, trainy), (_, _) = load_data()
```

---
## Step 30 — select all of the examples for a given class

```python
selected_ix = trainy == 7
	X = trainX[selected_ix]
```

---
## Step 31 — expand to 3d, e.g. add channels

```python
X = expand_dims(X, axis=-1)
```

---
## Step 32 — convert from ints to floats

```python
# 转换数据类型 / Convert data type
X = X.astype('float32')
```

---
## Step 33 — scale from [0,255] to [-1,1]

```python
X = (X - 127.5) / 127.5
	return X
```

---
## Step 34 — select real samples

```python
def generate_real_samples(dataset, n_samples):
```

---
## Step 35 — choose random instances

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
ix = randint(0, dataset.shape[0], n_samples)
```

---
## Step 36 — select images

```python
X = dataset[ix]
```

---
## Step 37 — generate class labels, -1 for 'real'

```python
y = -ones((n_samples, 1))
	return X, y
```

---
## Step 38 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n_samples):
```

---
## Step 39 — generate points in the latent space

```python
x_input = randn(latent_dim * n_samples)
```

---
## Step 40 — reshape into a batch of inputs for the network

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
```

---
## Step 41 — use the generator to generate n fake examples, with class labels

```python
def generate_fake_samples(generator, latent_dim, n_samples):
```

---
## Step 42 — generate points in latent space

```python
x_input = generate_latent_points(latent_dim, n_samples)
```

---
## Step 43 — predict outputs

```python
X = generator.predict(x_input)
```

---
## Step 44 — create class labels with 1.0 for 'fake'

```python
y = ones((n_samples, 1))
	return X, y
```

---
## Step 45 — generate samples and save as a plot and save the model

```python
def summarize_performance(step, g_model, latent_dim, n_samples=100):
```

---
## Step 46 — prepare fake examples

```python
X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
```

---
## Step 47 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
```

---
## Step 48 — plot images

```python
# 生成整数序列 / Generate integer sequence
for i in range(10 * 10):
```

---
## Step 49 — define subplot

```python
pyplot.subplot(10, 10, 1 + i)
```

---
## Step 50 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 51 — plot raw pixel data

```python
pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
```

---
## Step 52 — save plot to file

```python
filename1 = 'generated_plot_%04d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
```

---
## Step 53 — save the generator model

```python
filename2 = 'model_%04d.h5' % (step+1)
 # 保存模型到文件 / Save model to file
	g_model.save(filename2)
 # 打印输出 / Print output
	print('>Saved: %s and %s' % (filename1, filename2))
```

---
## Step 54 — create a line plot of loss for the gan and save to file

```python
def plot_history(d1_hist, d2_hist, g_hist):
```

---
## Step 55 — plot history

```python
pyplot.plot(d1_hist, label='crit_real')
	pyplot.plot(d2_hist, label='crit_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig('plot_line_plot_loss.png')
	pyplot.close()
```

---
## Step 56 — train the generator and critic

```python
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=64, n_critic=5):
```

---
## Step 57 — calculate the number of batches per training epoch

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
bat_per_epo = int(dataset.shape[0] / n_batch)
```

---
## Step 58 — calculate the number of training iterations

```python
n_steps = bat_per_epo * n_epochs
```

---
## Step 59 — calculate the size of half a batch of samples

```python
half_batch = int(n_batch / 2)
```

---
## Step 60 — lists for keeping track of loss

```python
c1_hist, c2_hist, g_hist = list(), list(), list()
```

---
## Step 61 — manually enumerate epochs

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_steps):
```

---
## Step 62 — update the critic more than the generator

```python
c1_tmp, c2_tmp = list(), list()
  # 生成整数序列 / Generate integer sequence
		for _ in range(n_critic):
```

---
## Step 63 — get randomly selected 'real' samples

```python
X_real, y_real = generate_real_samples(dataset, half_batch)
```

---
## Step 64 — update critic model weights

```python
c_loss1 = c_model.train_on_batch(X_real, y_real)
   # 添加元素到列表末尾 / Append element to list end
			c1_tmp.append(c_loss1)
```

---
## Step 65 — generate 'fake' examples

```python
X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
```

---
## Step 66 — update critic model weights

```python
c_loss2 = c_model.train_on_batch(X_fake, y_fake)
   # 添加元素到列表末尾 / Append element to list end
			c2_tmp.append(c_loss2)
```

---
## Step 67 — store critic loss

```python
# 添加元素到列表末尾 / Append element to list end
c1_hist.append(mean(c1_tmp))
  # 添加元素到列表末尾 / Append element to list end
		c2_hist.append(mean(c2_tmp))
```

---
## Step 68 — prepare points in latent space as input for the generator

```python
X_gan = generate_latent_points(latent_dim, n_batch)
```

---
## Step 69 — create inverted labels for the fake samples

```python
y_gan = -ones((n_batch, 1))
```

---
## Step 70 — update the generator via the critic's error

```python
g_loss = gan_model.train_on_batch(X_gan, y_gan)
  # 添加元素到列表末尾 / Append element to list end
		g_hist.append(g_loss)
```

---
## Step 71 — summarize loss on this batch

```python
# 打印输出 / Print output
print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))
```

---
## Step 72 — evaluate the model performance every 'epoch'

```python
if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim)
```

---
## Step 73 — line plots of loss

```python
plot_history(c1_hist, c2_hist, g_hist)
```

---
## Step 74 — size of the latent space

```python
latent_dim = 50
```

---
## Step 75 — create the critic

```python
critic = define_critic()
```

---
## Step 76 — create the generator

```python
generator = define_generator(latent_dim)
```

---
## Step 77 — create the gan

```python
gan_model = define_gan(generator, critic)
```

---
## Step 78 — load image data

```python
dataset = load_real_samples()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
```

---
## Step 79 — train model

```python
train(generator, critic, gan_model, dataset, latent_dim)
```

---
## Learning Notes / 学习笔记

- **概念**: example of a wgan for generating handwritten digits 是机器学习中的常用技术。  
  *example of a wgan for generating handwritten digits is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
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
# Train Wgan / 生成对抗网络
# Complete Code / 完整代码
# ===============================

# example of a wgan for generating handwritten digits
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.datasets.mnist import load_data
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras import backend
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import RMSprop
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
from keras.layers import BatchNormalization
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.initializers import RandomNormal
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.constraints import Constraint
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
 # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
	def __init__(self, clip_value):
		self.clip_value = clip_value

	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)

	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}

# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

# define the standalone critic model
def define_critic(in_shape=(28,28,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# weight constraint
	const = ClipConstraint(0.01)
	# define model
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# downsample to 14x14
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
 # 向模型添加一层 / Add a layer to the model
	model.add(BatchNormalization())
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 7x7
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
 # 向模型添加一层 / Add a layer to the model
	model.add(BatchNormalization())
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# scoring, linear activation
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1))
	# compile model
	opt = RMSprop(lr=0.00005)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss=wasserstein_loss, optimizer=opt)
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
	model.add(BatchNormalization())
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
 # 向模型添加一层 / Add a layer to the model
	model.add(BatchNormalization())
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# output 28x28x1
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
	return model

# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
	# make weights in the critic not trainable
	for layer in critic.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# connect them
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# add generator
 # 向模型添加一层 / Add a layer to the model
	model.add(generator)
	# add the critic
 # 向模型添加一层 / Add a layer to the model
	model.add(critic)
	# compile model
	opt = RMSprop(lr=0.00005)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model

# load images
def load_real_samples():
	# load dataset
 # 加载数据集 / Load dataset
	(trainX, trainy), (_, _) = load_data()
	# select all of the examples for a given class
	selected_ix = trainy == 7
	X = trainX[selected_ix]
	# expand to 3d, e.g. add channels
	X = expand_dims(X, axis=-1)
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
	# generate class labels, -1 for 'real'
	y = -ones((n_samples, 1))
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
	# create class labels with 1.0 for 'fake'
	y = ones((n_samples, 1))
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
	filename1 = 'generated_plot_%04d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%04d.h5' % (step+1)
 # 保存模型到文件 / Save model to file
	g_model.save(filename2)
 # 打印输出 / Print output
	print('>Saved: %s and %s' % (filename1, filename2))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
	# plot history
	pyplot.plot(d1_hist, label='crit_real')
	pyplot.plot(d2_hist, label='crit_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig('plot_line_plot_loss.png')
	pyplot.close()

# train the generator and critic
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=64, n_critic=5):
	# calculate the number of batches per training epoch
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# lists for keeping track of loss
	c1_hist, c2_hist, g_hist = list(), list(), list()
	# manually enumerate epochs
 # 生成整数序列 / Generate integer sequence
	for i in range(n_steps):
		# update the critic more than the generator
		c1_tmp, c2_tmp = list(), list()
  # 生成整数序列 / Generate integer sequence
		for _ in range(n_critic):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# update critic model weights
			c_loss1 = c_model.train_on_batch(X_real, y_real)
   # 添加元素到列表末尾 / Append element to list end
			c1_tmp.append(c_loss1)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update critic model weights
			c_loss2 = c_model.train_on_batch(X_fake, y_fake)
   # 添加元素到列表末尾 / Append element to list end
			c2_tmp.append(c_loss2)
		# store critic loss
  # 添加元素到列表末尾 / Append element to list end
		c1_hist.append(mean(c1_tmp))
  # 添加元素到列表末尾 / Append element to list end
		c2_hist.append(mean(c2_tmp))
		# prepare points in latent space as input for the generator
		X_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = -ones((n_batch, 1))
		# update the generator via the critic's error
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
  # 添加元素到列表末尾 / Append element to list end
		g_hist.append(g_loss)
		# summarize loss on this batch
  # 打印输出 / Print output
		print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))
		# evaluate the model performance every 'epoch'
		if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim)
	# line plots of loss
	plot_history(c1_hist, c2_hist, g_hist)

# size of the latent space
latent_dim = 50
# create the critic
critic = define_critic()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, critic)
# load image data
dataset = load_real_samples()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
# train model
train(generator, critic, gan_model, dataset, latent_dim)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Inference Wgan

# 02 — Inference Wgan / 生成对抗网络

**Chapter 16 — File 2 of 2 / 第16章 — 第2个文件（共2个）**

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
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
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
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
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
# 生成整数序列 / Generate integer sequence
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
# 从文件加载模型 / Load model from file
model = load_model('model_0970.h5')
```

---
## Step 11 — generate images

```python
latent_points = generate_latent_points(50, 25)
```

---
## Step 12 — generate images

```python
# 用模型做预测 / Make predictions with model
X = model.predict(latent_points)
```

---
## Step 13 — plot the result

```python
plot_generated(X, 5)
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
# Inference Wgan / 生成对抗网络
# Complete Code / 完整代码
# ===============================

# example of loading the generator model and generating images
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# create a plot of generated images (reversed grayscale)
def plot_generated(examples, n):
	# plot images
 # 生成整数序列 / Generate integer sequence
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()

# load model
# 从文件加载模型 / Load model from file
model = load_model('model_0970.h5')
# generate images
latent_points = generate_latent_points(50, 25)
# generate images
# 用模型做预测 / Make predictions with model
X = model.predict(latent_points)
# plot the result
plot_generated(X, 5)
```

---

### Chapter Summary / 章节总结

# Chapter 16 Summary / 第16章总结

## Theme / 主题: Chapter 16 / Chapter 16

This chapter contains **2 code files** demonstrating chapter 16.

本章包含 **2 个代码文件**，演示Chapter 16。

---
## Evolution / 演化路线

  1. `01_train_wgan.ipynb` — Train Wgan
  2. `02_inference_wgan.ipynb` — Inference Wgan

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 16) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 16）是机器学习流水线中的基础构建块。

---
