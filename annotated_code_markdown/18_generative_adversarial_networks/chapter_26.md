# GAN
## Chapter 26

---

### Load Prepare Dataset

# 01 — Load Prepare Dataset / 数据准备

**Chapter 26 — File 1 of 5 / 第26章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of preparing the horses and zebra dataset**.

本脚本演示 **example of preparing the horses and zebra dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of preparing the horses and zebra dataset

```python
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
```

---
## Step 2 — load all images in a directory into memory

```python
def load_images(path, size=(256,256)):
	data_list = list()
```

---
## Step 3 — enumerate filenames in directory, assume all are images

```python
for filename in listdir(path):
```

---
## Step 4 — load and resize the image

```python
pixels = load_img(path + filename, target_size=size)
```

---
## Step 5 — convert to numpy array

```python
pixels = img_to_array(pixels)
```

---
## Step 6 — store

```python
data_list.append(pixels)
	return asarray(data_list)
```

---
## Step 7 — dataset path

```python
path = 'horse2zebra/'
```

---
## Step 8 — load dataset A

```python
dataA1 = load_images(path + 'trainA/')
dataAB = load_images(path + 'testA/')
dataA = vstack((dataA1, dataAB))
print('Loaded dataA: ', dataA.shape)
```

---
## Step 9 — load dataset B

```python
dataB1 = load_images(path + 'trainB/')
dataB2 = load_images(path + 'testB/')
dataB = vstack((dataB1, dataB2))
print('Loaded dataB: ', dataB.shape)
```

---
## Step 10 — save as compressed numpy array

```python
filename = 'horse2zebra_256.npz'
savez_compressed(filename, dataA, dataB)
print('Saved dataset: ', filename)
```

---
## Learning Notes / 学习笔记

- **概念**: example of preparing the horses and zebra dataset 是机器学习中的常用技术。  
  *example of preparing the horses and zebra dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Prepare Dataset / 数据准备
# Complete Code / 完整代码
# ===============================

# example of preparing the horses and zebra dataset
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed

# load all images in a directory into memory
def load_images(path, size=(256,256)):
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return asarray(data_list)

# dataset path
path = 'horse2zebra/'
# load dataset A
dataA1 = load_images(path + 'trainA/')
dataAB = load_images(path + 'testA/')
dataA = vstack((dataA1, dataAB))
print('Loaded dataA: ', dataA.shape)
# load dataset B
dataB1 = load_images(path + 'trainB/')
dataB2 = load_images(path + 'testB/')
dataB = vstack((dataB1, dataB2))
print('Loaded dataB: ', dataB.shape)
# save as compressed numpy array
filename = 'horse2zebra_256.npz'
savez_compressed(filename, dataA, dataB)
print('Saved dataset: ', filename)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Load Plot Dataset

# 02 — Load Plot Dataset / 02 Load Plot Dataset

**Chapter 26 — File 2 of 5 / 第26章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **load and plot the prepared dataset**.

本脚本演示 **load and plot the prepared dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — load and plot the prepared dataset

```python
from numpy import load
from matplotlib import pyplot
```

---
## Step 2 — load the face dataset

```python
data = load('horse2zebra_256.npz')
dataA, dataB = data['arr_0'], data['arr_1']
print('Loaded: ', dataA.shape, dataB.shape)
```

---
## Step 3 — plot source images

```python
n_samples = 3
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(dataA[i].astype('uint8'))
```

---
## Step 4 — plot target image

```python
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + n_samples + i)
	pyplot.axis('off')
	pyplot.imshow(dataB[i].astype('uint8'))
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load and plot the prepared dataset 是机器学习中的常用技术。  
  *load and plot the prepared dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Plot Dataset / 02 Load Plot Dataset
# Complete Code / 完整代码
# ===============================

# load and plot the prepared dataset
from numpy import load
from matplotlib import pyplot
# load the face dataset
data = load('horse2zebra_256.npz')
dataA, dataB = data['arr_0'], data['arr_1']
print('Loaded: ', dataA.shape, dataB.shape)
# plot source images
n_samples = 3
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(dataA[i].astype('uint8'))
# plot target image
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + n_samples + i)
	pyplot.axis('off')
	pyplot.imshow(dataB[i].astype('uint8'))
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Train Cyclegan

# 03 — Train Cyclegan / 生成对抗网络

**Chapter 26 — File 3 of 5 / 第26章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of training a cyclegan on the horse2zebra dataset**.

本脚本演示 **example of training a cyclegan on the horse2zebra dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
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
## Step 1 — example of training a cyclegan on the horse2zebra dataset

```python
from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow_addons.layers import InstanceNormalization

from matplotlib import pyplot
```

---
## Step 2 — define the discriminator model

```python
def define_discriminator(image_shape):
```

---
## Step 3 — weight initialization

```python
init = RandomNormal(stddev=0.02)
```

---
## Step 4 — source image input

```python
in_image = Input(shape=image_shape)
```

---
## Step 5 — C64

```python
d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 6 — C128

```python
d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 7 — C256

```python
d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 8 — C512

```python
d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 9 — second last output layer

```python
d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 10 — patch output

```python
patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
```

---
## Step 11 — define model

```python
model = Model(in_image, patch_out)
```

---
## Step 12 — compile model

```python
model.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model
```

---
## Step 13 — generator a resnet block

```python
def resnet_block(n_filters, input_layer):
```

---
## Step 14 — weight initialization

```python
init = RandomNormal(stddev=0.02)
```

---
## Step 15 — first layer convolutional layer

```python
g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
```

---
## Step 16 — second convolutional layer

```python
g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
```

---
## Step 17 — concatenate merge channel-wise with input layer

```python
g = Concatenate()([g, input_layer])
	return g
```

---
## Step 18 — define the standalone generator model

```python
def define_generator(image_shape, n_resnet=9):
```

---
## Step 19 — weight initialization

```python
init = RandomNormal(stddev=0.02)
```

---
## Step 20 — image input

```python
in_image = Input(shape=image_shape)
```

---
## Step 21 — c7s1-64

```python
g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
```

---
## Step 22 — d128

```python
g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
```

---
## Step 23 — d256

```python
g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
```

---
## Step 24 — R256

```python
for _ in range(n_resnet):
		g = resnet_block(256, g)
```

---
## Step 25 — u128

```python
g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
```

---
## Step 26 — u64

```python
g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
```

---
## Step 27 — c7s1-3

```python
g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
```

---
## Step 28 — define model

```python
model = Model(in_image, out_image)
	return model
```

---
## Step 29 — define a composite model for updating generators by adversarial and cycle loss

```python
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    """Discriminators are compiled and will be trained separately, but generators are not compiled.
    """
```

---
## Step 30 — ensure the model we're updating is trainable

```python
g_model_1.trainable = True
```

---
## Step 31 — mark discriminator as not trainable

```python
d_model.trainable = False
```

---
## Step 32 — mark other generator model as not trainable

```python
g_model_2.trainable = False
```

---
## Step 33 — discriminator element

```python
input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
```

---
## Step 34 — identity element

```python
input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
```

---
## Step 35 — forward cycle

```python
output_f = g_model_2(gen1_out)
```

---
## Step 36 — backward cycle

```python
gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)
```

---
## Step 37 — define model graph

```python
model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
```

---
## Step 38 — define optimization algorithm configuration

```python
opt = Adam(lr=0.0002, beta_1=0.5)
```

---
## Step 39 — compile model with weighting of least squares loss and L1 loss

```python
model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
	return model
```

---
## Step 40 — load and prepare training images

```python
def load_real_samples(filename):
```

---
## Step 41 — load the dataset

```python
data = load(filename)
```

---
## Step 42 — unpack arrays

```python
X1, X2 = data['arr_0'], data['arr_1']
```

---
## Step 43 — scale from [0,255] to [-1,1]

```python
X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
```

---
## Step 44 — select a batch of random samples, returns images and target

```python
def generate_real_samples(dataset, n_samples, patch_shape):
```

---
## Step 45 — choose random instances

```python
ix = randint(0, dataset.shape[0], n_samples)
```

---
## Step 46 — retrieve selected images

```python
X = dataset[ix]
```

---
## Step 47 — generate 'real' class labels (1)

```python
y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y
```

---
## Step 48 — generate a batch of images, returns images and targets

```python
def generate_fake_samples(g_model, dataset, patch_shape):
```

---
## Step 49 — generate fake instance

```python
X = g_model.predict(dataset)
```

---
## Step 50 — create 'fake' class labels (0)

```python
y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y
```

---
## Step 51 — save the generator models to file

```python
def save_models(step, g_model_AtoB, g_model_BtoA):
```

---
## Step 52 — save the first generator model

```python
filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
	g_model_AtoB.save(filename1)
```

---
## Step 53 — save the second generator model

```python
filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
	g_model_BtoA.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
```

---
## Step 54 — generate samples and save as a plot and save the model

```python
def summarize_performance(step, g_model, trainX, name, n_samples=5):
```

---
## Step 55 — select a sample of input images

```python
X_in, _ = generate_real_samples(trainX, n_samples, 0)
```

---
## Step 56 — generate translated images

```python
X_out, _ = generate_fake_samples(g_model, X_in, 0)
```

---
## Step 57 — scale all pixels from [-1,1] to [0,1]

```python
X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0
```

---
## Step 58 — plot real images

```python
for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_in[i])
```

---
## Step 59 — plot translated image

```python
for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_out[i])
```

---
## Step 60 — save plot to file

```python
filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
	pyplot.savefig(filename1)
	pyplot.close()
```

---
## Step 61 — update image pool for fake images

```python
def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
```

---
## Step 62 — stock the pool

```python
pool.append(image)
			selected.append(image)
		elif random() < 0.5:
```

---
## Step 63 — use image, but don't add it to the pool

```python
selected.append(image)
		else:
```

---
## Step 64 — replace an existing image and use replaced image

```python
ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)
```

---
## Step 65 — train cyclegan models

```python
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
```

---
## Step 66 — define properties of the training run

```python
n_epochs, n_batch, = 100, 1
```

---
## Step 67 — determine the output square shape of the discriminator

```python
n_patch = d_model_A.output_shape[1]
```

---
## Step 68 — unpack dataset

```python
trainA, trainB = dataset
```

---
## Step 69 — prepare image pool for fakes

```python
poolA, poolB = list(), list()
```

---
## Step 70 — calculate the number of batches per training epoch

```python
bat_per_epo = int(len(trainA) / n_batch)
```

---
## Step 71 — calculate the number of training iterations

```python
n_steps = bat_per_epo * n_epochs
```

---
## Step 72 — manually enumerate epochs

```python
for i in range(n_steps):
```

---
## Step 73 — select a batch of real samples

```python
X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
		X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
```

---
## Step 74 — generate a batch of fake samples

```python
X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
		X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
```

---
## Step 75 — update fakes from pool

```python
X_fakeA = update_image_pool(poolA, X_fakeA)
		X_fakeB = update_image_pool(poolB, X_fakeB)
```

---
## Step 76 — update generator B->A via adversarial and cycle loss

```python
g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
```

---
## Step 77 — update discriminator for A -> [real/fake]

```python
dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
		dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
```

---
## Step 78 — update generator A->B via adversarial and cycle loss

```python
g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
```

---
## Step 79 — update discriminator for B -> [real/fake]

```python
dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
		dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
```

---
## Step 80 — summarize performance

```python
print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
```

---
## Step 81 — evaluate the model performance every so often

```python
if (i+1) % (bat_per_epo * 1) == 0:
```

---
## Step 82 — plot A->B translation

```python
summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
```

---
## Step 83 — plot B->A translation

```python
summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
		if (i+1) % (bat_per_epo * 5) == 0:
```

---
## Step 84 — save the models

```python
save_models(i, g_model_AtoB, g_model_BtoA)
```

---
## Step 85 — load image data

```python
dataset = load_real_samples('horse2zebra_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
```

---
## Step 86 — define input shape based on the loaded dataset

```python
image_shape = dataset[0].shape[1:]
```

---
## Step 87 — generator: A -> B

```python
g_model_AtoB = define_generator(image_shape)
```

---
## Step 88 — generator: B -> A

```python
g_model_BtoA = define_generator(image_shape)
```

---
## Step 89 — discriminator: A -> [real/fake]

```python
d_model_A = define_discriminator(image_shape)
```

---
## Step 90 — discriminator: B -> [real/fake]

```python
d_model_B = define_discriminator(image_shape)
```

---
## Step 91 — composite: A -> B -> [real/fake, A]

```python
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
```

---
## Step 92 — composite: B -> A -> [real/fake, B]

```python
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
```

---
## Step 93 — train models

```python
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)
```

---
## Learning Notes / 学习笔记

- **概念**: example of training a cyclegan on the horse2zebra dataset 是机器学习中的常用技术。  
  *example of training a cyclegan on the horse2zebra dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Cyclegan / 生成对抗网络
# Complete Code / 完整代码
# ===============================

# example of training a cyclegan on the horse2zebra dataset
from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow_addons.layers import InstanceNormalization

from matplotlib import pyplot

# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_image = Input(shape=image_shape)
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	# define model
	model = Model(in_image, patch_out)
	# compile model
	model.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model

# generator a resnet block
def resnet_block(n_filters, input_layer):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# first layer convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# second convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	# concatenate merge channel-wise with input layer
	g = Concatenate()([g, input_layer])
	return g

# define the standalone generator model
def define_generator(image_shape, n_resnet=9):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# c7s1-64
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for _ in range(n_resnet):
		g = resnet_block(256, g)
	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# c7s1-3
	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    """Discriminators are compiled and will be trained separately, but generators are not compiled.
    """
	# ensure the model we're updating is trainable
	g_model_1.trainable = True
	# mark discriminator as not trainable
	d_model.trainable = False
	# mark other generator model as not trainable
	g_model_2.trainable = False
	# discriminator element
	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	# identity element
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
	# forward cycle
	output_f = g_model_2(gen1_out)
	# backward cycle
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)
	# define model graph
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
	# define optimization algorithm configuration
	opt = Adam(lr=0.0002, beta_1=0.5)
	# compile model with weighting of least squares loss and L1 loss
	model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
	return model

# load and prepare training images
def load_real_samples(filename):
	# load the dataset
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
	# generate fake instance
	X = g_model.predict(dataset)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# save the generator models to file
def save_models(step, g_model_AtoB, g_model_BtoA):
	# save the first generator model
	filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
	g_model_AtoB.save(filename1)
	# save the second generator model
	filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
	g_model_BtoA.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, trainX, name, n_samples=5):
	# select a sample of input images
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	# generate translated images
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	# scale all pixels from [-1,1] to [0,1]
	X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0
	# plot real images
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_in[i])
	# plot translated image
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_out[i])
	# save plot to file
	filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
	pyplot.savefig(filename1)
	pyplot.close()

# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)

# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
	# define properties of the training run
	n_epochs, n_batch, = 100, 1
	# determine the output square shape of the discriminator
	n_patch = d_model_A.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# prepare image pool for fakes
	poolA, poolB = list(), list()
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
		X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
		X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
		# update fakes from pool
		X_fakeA = update_image_pool(poolA, X_fakeA)
		X_fakeB = update_image_pool(poolB, X_fakeB)
		# update generator B->A via adversarial and cycle loss
		g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
		# update discriminator for A -> [real/fake]
		dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
		dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
		# update generator A->B via adversarial and cycle loss
		g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
		# update discriminator for B -> [real/fake]
		dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
		dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
		# summarize performance
		print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
		# evaluate the model performance every so often
		if (i+1) % (bat_per_epo * 1) == 0:
			# plot A->B translation
			summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
			# plot B->A translation
			summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
		if (i+1) % (bat_per_epo * 5) == 0:
			# save the models
			save_models(i, g_model_AtoB, g_model_BtoA)

# load image data
dataset = load_real_samples('horse2zebra_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# generator: A -> B
g_model_AtoB = define_generator(image_shape)
# generator: B -> A
g_model_BtoA = define_generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Inference Cyclegan

# 04 — Inference Cyclegan / 生成对抗网络

**Chapter 26 — File 4 of 5 / 第26章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of using saved cyclegan models for image translation**.

本脚本演示 **example of using saved cyclegan models for image translation**。

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
## Step 1 — example of using saved cyclegan models for image translation

```python
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
```

---
## Step 2 — load and prepare training images

```python
def load_real_samples(filename):
```

---
## Step 3 — load the dataset

```python
data = load(filename)
```

---
## Step 4 — unpack arrays

```python
X1, X2 = data['arr_0'], data['arr_1']
```

---
## Step 5 — scale from [0,255] to [-1,1]

```python
X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
```

---
## Step 6 — select a random sample of images from the dataset

```python
def select_sample(dataset, n_samples):
```

---
## Step 7 — choose random instances

```python
ix = randint(0, dataset.shape[0], n_samples)
```

---
## Step 8 — retrieve selected images

```python
X = dataset[ix]
	return X
```

---
## Step 9 — plot the image, the translation, and the reconstruction

```python
def show_plot(imagesX, imagesY1, imagesY2):
	images = vstack((imagesX, imagesY1, imagesY2))
	titles = ['Real', 'Generated', 'Reconstructed']
```

---
## Step 10 — scale from [-1,1] to [0,1]

```python
images = (images + 1) / 2.0
```

---
## Step 11 — plot images row by row

```python
for i in range(len(images)):
```

---
## Step 12 — define subplot

```python
pyplot.subplot(1, len(images), 1 + i)
```

---
## Step 13 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 14 — plot raw pixel data

```python
pyplot.imshow(images[i])
```

---
## Step 15 — title

```python
pyplot.title(titles[i])
	pyplot.show()
```

---
## Step 16 — load dataset

```python
A_data, B_data = load_real_samples('horse2zebra_256.npz')
print('Loaded', A_data.shape, B_data.shape)
```

---
## Step 17 — load the models

```python
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('g_model_AtoB_089025.h5', cust)
model_BtoA = load_model('g_model_BtoA_089025.h5', cust)
```

---
## Step 18 — plot A->B->A

```python
A_real = select_sample(A_data, 1)
B_generated  = model_AtoB.predict(A_real)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real, B_generated, A_reconstructed)
```

---
## Step 19 — plot B->A->B

```python
B_real = select_sample(B_data, 1)
A_generated  = model_BtoA.predict(B_real)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real, A_generated, B_reconstructed)
```

---
## Learning Notes / 学习笔记

- **概念**: example of using saved cyclegan models for image translation 是机器学习中的常用技术。  
  *example of using saved cyclegan models for image translation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Inference Cyclegan / 生成对抗网络
# Complete Code / 完整代码
# ===============================

# example of using saved cyclegan models for image translation
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

# load and prepare training images
def load_real_samples(filename):
	# load the dataset
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	return X

# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
	images = vstack((imagesX, imagesY1, imagesY2))
	titles = ['Real', 'Generated', 'Reconstructed']
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, len(images), 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# title
		pyplot.title(titles[i])
	pyplot.show()

# load dataset
A_data, B_data = load_real_samples('horse2zebra_256.npz')
print('Loaded', A_data.shape, B_data.shape)
# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('g_model_AtoB_089025.h5', cust)
model_BtoA = load_model('g_model_BtoA_089025.h5', cust)
# plot A->B->A
A_real = select_sample(A_data, 1)
B_generated  = model_AtoB.predict(A_real)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real, B_generated, A_reconstructed)
# plot B->A->B
B_real = select_sample(B_data, 1)
A_generated  = model_BtoA.predict(B_real)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real, A_generated, B_reconstructed)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Translate Single Image

# 05 — Translate Single Image / 图像处理

**Chapter 26 — File 5 of 5 / 第26章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of using saved cyclegan models for image translation**.

本脚本演示 **example of using saved cyclegan models for image translation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
## Step 1 — example of using saved cyclegan models for image translation

```python
from numpy import expand_dims
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot
```

---
## Step 2 — load an image to the preferred size

```python
def load_image(filename, size=(256,256)):
```

---
## Step 3 — load and resize the image

```python
pixels = load_img(filename, target_size=size)
```

---
## Step 4 — convert to numpy array

```python
pixels = img_to_array(pixels)
```

---
## Step 5 — transform in a sample

```python
pixels = expand_dims(pixels, 0)
```

---
## Step 6 — scale from [0,255] to [-1,1]

```python
pixels = (pixels - 127.5) / 127.5
	return pixels
```

---
## Step 7 — load the image

```python
image_src = load_image('horse2zebra/trainA/n02381460_541.jpg')
```

---
## Step 8 — load the model

```python
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('g_model_AtoB_100895.h5', cust)
```

---
## Step 9 — translate image

```python
image_tar = model_AtoB.predict(image_src)
```

---
## Step 10 — scale from [-1,1] to [0,1]

```python
image_tar = (image_tar + 1) / 2.0
```

---
## Step 11 — plot the translated image

```python
pyplot.imshow(image_tar[0])
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of using saved cyclegan models for image translation 是机器学习中的常用技术。  
  *example of using saved cyclegan models for image translation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Translate Single Image / 图像处理
# Complete Code / 完整代码
# ===============================

# example of using saved cyclegan models for image translation
from numpy import expand_dims
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot

# load an image to the preferred size
def load_image(filename, size=(256,256)):
	# load and resize the image
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# transform in a sample
	pixels = expand_dims(pixels, 0)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	return pixels

# load the image
image_src = load_image('horse2zebra/trainA/n02381460_541.jpg')
# load the model
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('g_model_AtoB_100895.h5', cust)
# translate image
image_tar = model_AtoB.predict(image_src)
# scale from [-1,1] to [0,1]
image_tar = (image_tar + 1) / 2.0
# plot the translated image
pyplot.imshow(image_tar[0])
pyplot.show()
```

---

### Chapter Summary

# Chapter 26 Summary / 第26章总结

## Theme / 主题: Chapter 26 / Chapter 26

This chapter contains **5 code files** demonstrating chapter 26.

本章包含 **5 个代码文件**，演示Chapter 26。

---
## Evolution / 演化路线

  1. `01_load_prepare_dataset.ipynb` — Load Prepare Dataset
  2. `02_load_plot_dataset.ipynb` — Load Plot Dataset
  3. `03_train_cyclegan.ipynb` — Train Cyclegan
  4. `04_inference_cyclegan.ipynb` — Inference Cyclegan
  5. `05_translate_single_image.ipynb` — Translate Single Image

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 26) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 26）是机器学习流水线中的基础构建块。

---
