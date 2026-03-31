# 生成对抗网络 / Generative Adversarial Networks
## Chapter 09

---

### Plot Faces

# 01 — Plot Faces / 人脸识别

**Chapter 09 — File 1 of 11 / 第09章 — 第1个文件（共11个）**

---

## Summary / 总结

This script demonstrates **load and plot faces**.

本脚本演示 **load and plot faces**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — load and plot faces

```python
from os import listdir
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from PIL import Image
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load an image as an rgb numpy array

```python
def load_image(filename):
```

---
## Step 3 — load image from file

```python
image = Image.open(filename)
```

---
## Step 4 — convert to RGB, if needed

```python
image = image.convert('RGB')
```

---
## Step 5 — convert to array

```python
pixels = asarray(image)
	return pixels
```

---
## Step 6 — load images and extract faces for all images in a directory

```python
def load_faces(directory, n_faces):
	faces = list()
```

---
## Step 7 — enumerate files

```python
for filename in listdir(directory):
```

---
## Step 8 — load the image

```python
pixels = load_image(directory + filename)
```

---
## Step 9 — store

```python
# 添加元素到列表末尾 / Append element to list end
faces.append(pixels)
```

---
## Step 10 — stop once we have enough

```python
# 获取长度 / Get length
if len(faces) >= n_faces:
			break
	return asarray(faces)
```

---
## Step 11 — plot a list of loaded faces

```python
def plot_faces(faces, n):
 # 生成整数序列 / Generate integer sequence
	for i in range(n * n):
```

---
## Step 12 — define subplot

```python
pyplot.subplot(n, n, 1 + i)
```

---
## Step 13 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 14 — plot raw pixel data

```python
pyplot.imshow(faces[i])
	pyplot.show()
```

---
## Step 15 — directory that contains all images

```python
directory = 'img_align_celeba/'
```

---
## Step 16 — load and extract all faces

```python
faces = load_faces(directory, 25)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Loaded: ', faces.shape)
```

---
## Step 17 — plot faces

```python
plot_faces(faces, 5)
```

---
## Learning Notes / 学习笔记

- **概念**: load and plot faces 是机器学习中的常用技术。  
  *load and plot faces is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Faces / 人脸识别
# Complete Code / 完整代码
# ===============================

# load and plot faces
from os import listdir
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
from PIL import Image
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# load an image as an rgb numpy array
def load_image(filename):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	return pixels

# load images and extract faces for all images in a directory
def load_faces(directory, n_faces):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# load the image
		pixels = load_image(directory + filename)
		# store
  # 添加元素到列表末尾 / Append element to list end
		faces.append(pixels)
		# stop once we have enough
  # 获取长度 / Get length
		if len(faces) >= n_faces:
			break
	return asarray(faces)

# plot a list of loaded faces
def plot_faces(faces, n):
 # 生成整数序列 / Generate integer sequence
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(faces[i])
	pyplot.show()

# directory that contains all images
directory = 'img_align_celeba/'
# load and extract all faces
faces = load_faces(directory, 25)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Loaded: ', faces.shape)
# plot faces
plot_faces(faces, 5)
```

---

➡️ **Next / 下一步**: File 2 of 11

---

### Check Mtcnn

# 02 — Check Mtcnn / 卷积神经网络

**Chapter 09 — File 2 of 11 / 第09章 — 第2个文件（共11个）**

---

## Summary / 总结

This script demonstrates **confirm mtcnn was installed correctly**.

本脚本演示 **confirm mtcnn was installed correctly**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — confirm mtcnn was installed correctly

```python
import mtcnn
```

---
## Step 2 — show version

```python
# 打印输出 / Print output
print(mtcnn.__version__)
```

---
## Learning Notes / 学习笔记

- **概念**: confirm mtcnn was installed correctly 是机器学习中的常用技术。  
  *confirm mtcnn was installed correctly is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Check Mtcnn / 卷积神经网络
# Complete Code / 完整代码
# ===============================

# confirm mtcnn was installed correctly
import mtcnn
# show version
# 打印输出 / Print output
print(mtcnn.__version__)
```

---

➡️ **Next / 下一步**: File 3 of 11

---

### Prepare Dataset

# 03 — Prepare Dataset / 数据准备

**Chapter 09 — File 3 of 11 / 第09章 — 第3个文件（共11个）**

---

## Summary / 总结

This script demonstrates **example of extracting and resizing faces into a new dataset**.

本脚本演示 **example of extracting and resizing faces into a new dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — example of extracting and resizing faces into a new dataset

```python
from os import listdir
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import savez_compressed
from PIL import Image
from mtcnn.mtcnn import MTCNN
```

---
## Step 2 — load an image as an rgb numpy array

```python
def load_image(filename):
```

---
## Step 3 — load image from file

```python
image = Image.open(filename)
```

---
## Step 4 — convert to RGB, if needed

```python
image = image.convert('RGB')
```

---
## Step 5 — convert to array

```python
pixels = asarray(image)
	return pixels
```

---
## Step 6 — extract the face from a loaded image and resize

```python
def extract_face(model, pixels, required_size=(80, 80)):
```

---
## Step 7 — detect face in the image

```python
faces = model.detect_faces(pixels)
```

---
## Step 8 — skip cases where we could not detect a face

```python
# 获取长度 / Get length
if len(faces) == 0:
		return None
```

---
## Step 9 — extract details of the face

```python
x1, y1, width, height = faces[0]['box']
```

---
## Step 10 — force detected pixel values to be positive (bug fix)

```python
x1, y1 = abs(x1), abs(y1)
```

---
## Step 11 — convert into coordinates

```python
x2, y2 = x1 + width, y1 + height
```

---
## Step 12 — retrieve face pixels

```python
face_pixels = pixels[y1:y2, x1:x2]
```

---
## Step 13 — resize pixels to the model size

```python
image = Image.fromarray(face_pixels)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
```

---
## Step 14 — load images and extract faces for all images in a directory

```python
def load_faces(directory, n_faces):
```

---
## Step 15 — prepare model

```python
model = MTCNN()
	faces = list()
```

---
## Step 16 — enumerate files

```python
for filename in listdir(directory):
```

---
## Step 17 — load the image

```python
pixels = load_image(directory + filename)
```

---
## Step 18 — get face

```python
face = extract_face(model, pixels)
		if face is None:
			continue
```

---
## Step 19 — store

```python
# 添加元素到列表末尾 / Append element to list end
faces.append(face)
  # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
		print(len(faces), face.shape)
```

---
## Step 20 — stop once we have enough

```python
# 获取长度 / Get length
if len(faces) >= n_faces:
			break
	return asarray(faces)
```

---
## Step 21 — directory that contains all images

```python
directory = 'img_align_celeba/'
```

---
## Step 22 — load and extract all faces

```python
all_faces = load_faces(directory, 50000)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Loaded: ', all_faces.shape)
```

---
## Step 23 — save in compressed format

```python
savez_compressed('img_align_celeba.npz', all_faces)
```

---
## Learning Notes / 学习笔记

- **概念**: example of extracting and resizing faces into a new dataset 是机器学习中的常用技术。  
  *example of extracting and resizing faces into a new dataset is a common technique in machine learning.*

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
# Prepare Dataset / 数据准备
# Complete Code / 完整代码
# ===============================

# example of extracting and resizing faces into a new dataset
from os import listdir
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import savez_compressed
from PIL import Image
from mtcnn.mtcnn import MTCNN

# load an image as an rgb numpy array
def load_image(filename):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	return pixels

# extract the face from a loaded image and resize
def extract_face(model, pixels, required_size=(80, 80)):
	# detect face in the image
	faces = model.detect_faces(pixels)
	# skip cases where we could not detect a face
 # 获取长度 / Get length
	if len(faces) == 0:
		return None
	# extract details of the face
	x1, y1, width, height = faces[0]['box']
	# force detected pixel values to be positive (bug fix)
	x1, y1 = abs(x1), abs(y1)
	# convert into coordinates
	x2, y2 = x1 + width, y1 + height
	# retrieve face pixels
	face_pixels = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face_pixels)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# load images and extract faces for all images in a directory
def load_faces(directory, n_faces):
	# prepare model
	model = MTCNN()
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# load the image
		pixels = load_image(directory + filename)
		# get face
		face = extract_face(model, pixels)
		if face is None:
			continue
		# store
  # 添加元素到列表末尾 / Append element to list end
		faces.append(face)
  # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
		print(len(faces), face.shape)
		# stop once we have enough
  # 获取长度 / Get length
		if len(faces) >= n_faces:
			break
	return asarray(faces)

# directory that contains all images
directory = 'img_align_celeba/'
# load and extract all faces
all_faces = load_faces(directory, 50000)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Loaded: ', all_faces.shape)
# save in compressed format
savez_compressed('img_align_celeba.npz', all_faces)
```

---

➡️ **Next / 下一步**: File 4 of 11

---

### Load Saved Dataset

# 04 — Load Saved Dataset / 保存/加载模型

**Chapter 09 — File 4 of 11 / 第09章 — 第4个文件（共11个）**

---

## Summary / 总结

This script demonstrates **load the prepared dataset**.

本脚本演示 **load the prepared dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — load the prepared dataset

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import load
```

---
## Step 2 — load the face dataset

```python
data = load('img_align_celeba.npz')
faces = data['arr_0']
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Loaded: ', faces.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: load the prepared dataset 是机器学习中的常用技术。  
  *load the prepared dataset is a common technique in machine learning.*

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
# Load Saved Dataset / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# load the prepared dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import load
# load the face dataset
data = load('img_align_celeba.npz')
faces = data['arr_0']
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Loaded: ', faces.shape)
```

---

➡️ **Next / 下一步**: File 5 of 11

---

### Train Gan

# 05 — Train Gan / 生成对抗网络

**Chapter 09 — File 5 of 11 / 第09章 — 第5个文件（共11个）**

---

## Summary / 总结

This script demonstrates **example of a gan for generating faces**.

本脚本演示 **example of a gan for generating faces**。

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
## Step 1 — example of a gan for generating faces

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import load
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
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
from keras.layers import Dropout
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — define the standalone discriminator model

```python
def define_discriminator(in_shape=(80,80,3)):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
```

---
## Step 3 — normal

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(128, (5,5), padding='same', input_shape=in_shape))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 4 — downsample to 40x40

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 5 — downsample to 20x30

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 6 — downsample to 10x10

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 7 — downsample to 5x5

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 8 — classifier

```python
# 向模型添加一层 / Add a layer to the model
model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dropout(0.4))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 9 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
```

---
## Step 10 — define the standalone generator model

```python
def define_generator(latent_dim):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
```

---
## Step 11 — foundation for 5x5 feature maps

```python
n_nodes = 128 * 5 * 5
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Reshape((5, 5, 128)))
```

---
## Step 12 — upsample to 10x10

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 13 — upsample to 20x20

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 14 — upsample to 40x40

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 15 — upsample to 80x80

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
```

---
## Step 16 — output layer 80x80x3

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))
	return model
```

---
## Step 17 — define the combined generator and discriminator model, for updating the generator

```python
def define_gan(g_model, d_model):
```

---
## Step 18 — make weights in the discriminator not trainable

```python
d_model.trainable = False
```

---
## Step 19 — connect them

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 20 — add generator

```python
# 向模型添加一层 / Add a layer to the model
model.add(g_model)
```

---
## Step 21 — add the discriminator

```python
# 向模型添加一层 / Add a layer to the model
model.add(d_model)
```

---
## Step 22 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
```

---
## Step 23 — load and prepare training images

```python
def load_real_samples():
```

---
## Step 24 — load the face dataset

```python
data = load('img_align_celeba.npz')
	X = data['arr_0']
```

---
## Step 25 — convert from unsigned ints to floats

```python
# 转换数据类型 / Convert data type
X = X.astype('float32')
```

---
## Step 26 — scale from [0,255] to [-1,1]

```python
X = (X - 127.5) / 127.5
	return X
```

---
## Step 27 — select real samples

```python
def generate_real_samples(dataset, n_samples):
```

---
## Step 28 — choose random instances

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
ix = randint(0, dataset.shape[0], n_samples)
```

---
## Step 29 — retrieve selected images

```python
X = dataset[ix]
```

---
## Step 30 — generate 'real' class labels (1)

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
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
```

---
## Step 34 — use the generator to generate n fake examples, with class labels

```python
def generate_fake_samples(g_model, latent_dim, n_samples):
```

---
## Step 35 — generate points in latent space

```python
x_input = generate_latent_points(latent_dim, n_samples)
```

---
## Step 36 — predict outputs

```python
# 用模型做预测 / Make predictions with model
X = g_model.predict(x_input)
```

---
## Step 37 — create 'fake' class labels (0)

```python
y = zeros((n_samples, 1))
	return X, y
```

---
## Step 38 — create and save a plot of generated images

```python
def save_plot(examples, epoch, n=10):
```

---
## Step 39 — scale from [-1,1] to [0,1]

```python
examples = (examples + 1) / 2.0
```

---
## Step 40 — plot images

```python
# 生成整数序列 / Generate integer sequence
for i in range(n * n):
```

---
## Step 41 — define subplot

```python
pyplot.subplot(n, n, 1 + i)
```

---
## Step 42 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 43 — plot raw pixel data

```python
pyplot.imshow(examples[i])
```

---
## Step 44 — save plot to file

```python
filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()
```

---
## Step 45 — evaluate the discriminator, plot generated images, save generator model

```python
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
```

---
## Step 46 — prepare real samples

```python
X_real, y_real = generate_real_samples(dataset, n_samples)
```

---
## Step 47 — evaluate discriminator on real examples

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
```

---
## Step 48 — prepare fake examples

```python
x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
```

---
## Step 49 — evaluate discriminator on fake examples

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
```

---
## Step 50 — summarize discriminator performance

```python
# 打印输出 / Print output
print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
```

---
## Step 51 — save plot

```python
save_plot(x_fake, epoch)
```

---
## Step 52 — save the generator model tile file

```python
filename = 'generator_model_%03d.h5' % (epoch+1)
 # 保存模型到文件 / Save model to file
	g_model.save(filename)
```

---
## Step 53 — train the generator and discriminator

```python
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
```

---
## Step 54 — manually enumerate epochs

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_epochs):
```

---
## Step 55 — enumerate batches over the training set

```python
# 生成整数序列 / Generate integer sequence
for j in range(bat_per_epo):
```

---
## Step 56 — get randomly selected 'real' samples

```python
X_real, y_real = generate_real_samples(dataset, half_batch)
```

---
## Step 57 — update discriminator model weights

```python
d_loss1, _ = d_model.train_on_batch(X_real, y_real)
```

---
## Step 58 — generate 'fake' examples

```python
X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
```

---
## Step 59 — update discriminator model weights

```python
d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
```

---
## Step 60 — prepare points in latent space as input for the generator

```python
X_gan = generate_latent_points(latent_dim, n_batch)
```

---
## Step 61 — create inverted labels for the fake samples

```python
y_gan = ones((n_batch, 1))
```

---
## Step 62 — update the generator via the discriminator's error

```python
g_loss = gan_model.train_on_batch(X_gan, y_gan)
```

---
## Step 63 — summarize loss on this batch

```python
# 打印输出 / Print output
print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
```

---
## Step 64 — evaluate the model performance, sometimes

```python
if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)
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

- **概念**: example of a gan for generating faces 是机器学习中的常用技术。  
  *example of a gan for generating faces is a common technique in machine learning.*

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
# Train Gan / 生成对抗网络
# Complete Code / 完整代码
# ===============================

# example of a gan for generating faces
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import load
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
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
from keras.layers import Dropout
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# define the standalone discriminator model
def define_discriminator(in_shape=(80,80,3)):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# normal
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(128, (5,5), padding='same', input_shape=in_shape))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 40x40
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 20x30
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 10x10
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 5x5
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# classifier
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dropout(0.4))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# foundation for 5x5 feature maps
	n_nodes = 128 * 5 * 5
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_nodes, input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Reshape((5, 5, 128)))
	# upsample to 10x10
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 20x20
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 40x40
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 80x80
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(LeakyReLU(alpha=0.2))
	# output layer 80x80x3
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
	# add generator
 # 向模型添加一层 / Add a layer to the model
	model.add(g_model)
	# add the discriminator
 # 向模型添加一层 / Add a layer to the model
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load and prepare training images
def load_real_samples():
	# load the face dataset
	data = load('img_align_celeba.npz')
	X = data['arr_0']
	# convert from unsigned ints to floats
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
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
 # 用模型做预测 / Make predictions with model
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

# create and save a plot of generated images
def save_plot(examples, epoch, n=10):
	# scale from [-1,1] to [0,1]
	examples = (examples + 1) / 2.0
	# plot images
 # 生成整数序列 / Generate integer sequence
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i])
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
 # 评估模型在测试集上的表现 / Evaluate model on test set
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
 # 打印输出 / Print output
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch+1)
 # 保存模型到文件 / Save model to file
	g_model.save(filename)

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
 # 生成整数序列 / Generate integer sequence
	for i in range(n_epochs):
		# enumerate batches over the training set
  # 生成整数序列 / Generate integer sequence
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch(X_real, y_real)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
   # 打印输出 / Print output
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
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

➡️ **Next / 下一步**: File 6 of 11

---

### Load And Generate

# 06 — Load And Generate / 06 Load And Generate

**Chapter 09 — File 6 of 11 / 第09章 — 第6个文件（共11个）**

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
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
z_input = x_input.reshape(n_samples, latent_dim)
	return z_input
```

---
## Step 5 — create a plot of generated images

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
pyplot.imshow(examples[i, :, :])
	pyplot.show()
```

---
## Step 10 — load model

```python
# 从文件加载模型 / Load model from file
model = load_model('generator_model_030.h5')
```

---
## Step 11 — generate images

```python
latent_points = generate_latent_points(100, 25)
```

---
## Step 12 — generate images

```python
# 用模型做预测 / Make predictions with model
X  = model.predict(latent_points)
```

---
## Step 13 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
```

---
## Step 14 — plot the result

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
# Load And Generate / 06 Load And Generate
# Complete Code / 完整代码
# ===============================

# example of loading the generator model and generating images
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	z_input = x_input.reshape(n_samples, latent_dim)
	return z_input

# create a plot of generated images
def plot_generated(examples, n):
	# plot images
 # 生成整数序列 / Generate integer sequence
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :])
	pyplot.show()

# load model
# 从文件加载模型 / Load model from file
model = load_model('generator_model_030.h5')
# generate images
latent_points = generate_latent_points(100, 25)
# generate images
# 用模型做预测 / Make predictions with model
X  = model.predict(latent_points)
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
plot_generated(X, 5)
```

---

➡️ **Next / 下一步**: File 7 of 11

---

### Interpolate Latent

# 07 — Interpolate Latent / 07 Interpolate Latent

**Chapter 09 — File 7 of 11 / 第09章 — 第7个文件（共11个）**

---

## Summary / 总结

This script demonstrates **example of interpolating between generated faces**.

本脚本演示 **example of interpolating between generated faces**。

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
## Step 1 — example of interpolating between generated faces

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import linspace
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
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
z_input = x_input.reshape(n_samples, latent_dim)
	return z_input
```

---
## Step 5 — uniform interpolation between two points in latent space

```python
def interpolate_points(p1, p2, n_steps=10):
```

---
## Step 6 — interpolate ratios between the points

```python
ratios = linspace(0, 1, num=n_steps)
```

---
## Step 7 — linear interpolate vectors

```python
vectors = list()
	for ratio in ratios:
		v = (1.0 - ratio) * p1 + ratio * p2
  # 添加元素到列表末尾 / Append element to list end
		vectors.append(v)
	return asarray(vectors)
```

---
## Step 8 — create a plot of generated images

```python
def plot_generated(examples, n):
```

---
## Step 9 — plot images

```python
# 生成整数序列 / Generate integer sequence
for i in range(n):
```

---
## Step 10 — define subplot

```python
pyplot.subplot(1, n, 1 + i)
```

---
## Step 11 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 12 — plot raw pixel data

```python
pyplot.imshow(examples[i, :, :])
	pyplot.show()
```

---
## Step 13 — load model

```python
# 从文件加载模型 / Load model from file
model = load_model('generator_model_030.h5')
```

---
## Step 14 — generate points in latent space

```python
pts = generate_latent_points(100, 2)
```

---
## Step 15 — interpolate points in latent space

```python
interpolated = interpolate_points(pts[0], pts[1])
```

---
## Step 16 — generate images

```python
# 用模型做预测 / Make predictions with model
X = model.predict(interpolated)
```

---
## Step 17 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
```

---
## Step 18 — plot the result

```python
# 获取长度 / Get length
plot_generated(X, len(interpolated))
```

---
## Learning Notes / 学习笔记

- **概念**: example of interpolating between generated faces 是机器学习中的常用技术。  
  *example of interpolating between generated faces is a common technique in machine learning.*

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
# Interpolate Latent / 07 Interpolate Latent
# Complete Code / 完整代码
# ===============================

# example of interpolating between generated faces
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import linspace
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	z_input = x_input.reshape(n_samples, latent_dim)
	return z_input

# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=10):
	# interpolate ratios between the points
	ratios = linspace(0, 1, num=n_steps)
	# linear interpolate vectors
	vectors = list()
	for ratio in ratios:
		v = (1.0 - ratio) * p1 + ratio * p2
  # 添加元素到列表末尾 / Append element to list end
		vectors.append(v)
	return asarray(vectors)

# create a plot of generated images
def plot_generated(examples, n):
	# plot images
 # 生成整数序列 / Generate integer sequence
	for i in range(n):
		# define subplot
		pyplot.subplot(1, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :])
	pyplot.show()

# load model
# 从文件加载模型 / Load model from file
model = load_model('generator_model_030.h5')
# generate points in latent space
pts = generate_latent_points(100, 2)
# interpolate points in latent space
interpolated = interpolate_points(pts[0], pts[1])
# generate images
# 用模型做预测 / Make predictions with model
X = model.predict(interpolated)
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
# 获取长度 / Get length
plot_generated(X, len(interpolated))
```

---

➡️ **Next / 下一步**: File 8 of 11

---

### Multiple Interpolate Latent

# 08 — Multiple Interpolate Latent / 08 Multiple Interpolate Latent

**Chapter 09 — File 8 of 11 / 第09章 — 第8个文件（共11个）**

---

## Summary / 总结

This script demonstrates **example of interpolating between generated faces**.

本脚本演示 **example of interpolating between generated faces**。

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
## Step 1 — example of interpolating between generated faces

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import linspace
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
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
z_input = x_input.reshape(n_samples, latent_dim)
	return z_input
```

---
## Step 5 — uniform interpolation between two points in latent space

```python
def interpolate_points(p1, p2, n_steps=10):
```

---
## Step 6 — interpolate ratios between the points

```python
ratios = linspace(0, 1, num=n_steps)
```

---
## Step 7 — linear interpolate vectors

```python
vectors = list()
	for ratio in ratios:
		v = (1.0 - ratio) * p1 + ratio * p2
  # 添加元素到列表末尾 / Append element to list end
		vectors.append(v)
	return asarray(vectors)
```

---
## Step 8 — create a plot of generated images

```python
def plot_generated(examples, n):
```

---
## Step 9 — plot images

```python
# 生成整数序列 / Generate integer sequence
for i in range(n * n):
```

---
## Step 10 — define subplot

```python
pyplot.subplot(n, n, 1 + i)
```

---
## Step 11 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 12 — plot raw pixel data

```python
pyplot.imshow(examples[i, :, :])
	pyplot.show()
```

---
## Step 13 — load model

```python
# 从文件加载模型 / Load model from file
model = load_model('generator_model_030.h5')
```

---
## Step 14 — generate points in latent space

```python
n = 20
pts = generate_latent_points(100, n)
```

---
## Step 15 — interpolate pairs

```python
results = None
# 生成整数序列 / Generate integer sequence
for i in range(0, n, 2):
```

---
## Step 16 — interpolate points in latent space

```python
interpolated = interpolate_points(pts[i], pts[i+1])
```

---
## Step 17 — generate images

```python
# 用模型做预测 / Make predictions with model
X = model.predict(interpolated)
```

---
## Step 18 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
	if results is None:
		results = X
	else:
		results = vstack((results, X))
```

---
## Step 19 — plot the result

```python
plot_generated(results, 10)
```

---
## Learning Notes / 学习笔记

- **概念**: example of interpolating between generated faces 是机器学习中的常用技术。  
  *example of interpolating between generated faces is a common technique in machine learning.*

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
# Multiple Interpolate Latent / 08 Multiple Interpolate Latent
# Complete Code / 完整代码
# ===============================

# example of interpolating between generated faces
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import linspace
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	z_input = x_input.reshape(n_samples, latent_dim)
	return z_input

# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=10):
	# interpolate ratios between the points
	ratios = linspace(0, 1, num=n_steps)
	# linear interpolate vectors
	vectors = list()
	for ratio in ratios:
		v = (1.0 - ratio) * p1 + ratio * p2
  # 添加元素到列表末尾 / Append element to list end
		vectors.append(v)
	return asarray(vectors)

# create a plot of generated images
def plot_generated(examples, n):
	# plot images
 # 生成整数序列 / Generate integer sequence
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :])
	pyplot.show()

# load model
# 从文件加载模型 / Load model from file
model = load_model('generator_model_030.h5')
# generate points in latent space
n = 20
pts = generate_latent_points(100, n)
# interpolate pairs
results = None
# 生成整数序列 / Generate integer sequence
for i in range(0, n, 2):
	# interpolate points in latent space
	interpolated = interpolate_points(pts[i], pts[i+1])
	# generate images
 # 用模型做预测 / Make predictions with model
	X = model.predict(interpolated)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	if results is None:
		results = X
	else:
		results = vstack((results, X))
# plot the result
plot_generated(results, 10)
```

---

➡️ **Next / 下一步**: File 9 of 11

---

### Multiple Slerp Latent

# 09 — Multiple Slerp Latent / 09 Multiple Slerp Latent

**Chapter 09 — File 9 of 11 / 第09章 — 第9个文件（共11个）**

---

## Summary / 总结

This script demonstrates **example of interpolating between generated faces**.

本脚本演示 **example of interpolating between generated faces**。

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
## Step 1 — example of interpolating between generated faces

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arccos
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import clip
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import dot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import sin
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import linspace
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import norm
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
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
z_input = x_input.reshape(n_samples, latent_dim)
	return z_input
```

---
## Step 5 — spherical linear interpolation (slerp)

```python
def slerp(val, low, high):
	omega = arccos(clip(dot(low/norm(low), high/norm(high)), -1, 1))
	so = sin(omega)
	if so == 0:
```

---
## Step 6 — L'Hopital's rule/LERP

```python
return (1.0-val) * low + val * high
	return sin((1.0-val)*omega) / so * low + sin(val*omega) / so * high
```

---
## Step 7 — uniform interpolation between two points in latent space

```python
def interpolate_points(p1, p2, n_steps=10):
```

---
## Step 8 — interpolate ratios between the points

```python
ratios = linspace(0, 1, num=n_steps)
```

---
## Step 9 — linear interpolate vectors

```python
vectors = list()
	for ratio in ratios:
		v = slerp(ratio, p1, p2)
  # 添加元素到列表末尾 / Append element to list end
		vectors.append(v)
	return asarray(vectors)
```

---
## Step 10 — create a plot of generated images

```python
def plot_generated(examples, n):
```

---
## Step 11 — plot images

```python
# 生成整数序列 / Generate integer sequence
for i in range(n * n):
```

---
## Step 12 — define subplot

```python
pyplot.subplot(n, n, 1 + i)
```

---
## Step 13 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 14 — plot raw pixel data

```python
pyplot.imshow(examples[i, :, :])
	pyplot.show()
```

---
## Step 15 — load model

```python
# 从文件加载模型 / Load model from file
model = load_model('generator_model_030.h5')
```

---
## Step 16 — generate points in latent space

```python
n = 20
pts = generate_latent_points(100, n)
```

---
## Step 17 — interpolate pairs

```python
results = None
# 生成整数序列 / Generate integer sequence
for i in range(0, n, 2):
```

---
## Step 18 — interpolate points in latent space

```python
interpolated = interpolate_points(pts[i], pts[i+1])
```

---
## Step 19 — generate images

```python
# 用模型做预测 / Make predictions with model
X = model.predict(interpolated)
```

---
## Step 20 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
	if results is None:
		results = X
	else:
		results = vstack((results, X))
```

---
## Step 21 — plot the result

```python
plot_generated(results, 10)
```

---
## Learning Notes / 学习笔记

- **概念**: example of interpolating between generated faces 是机器学习中的常用技术。  
  *example of interpolating between generated faces is a common technique in machine learning.*

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
# Multiple Slerp Latent / 09 Multiple Slerp Latent
# Complete Code / 完整代码
# ===============================

# example of interpolating between generated faces
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arccos
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import clip
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import dot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import sin
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import linspace
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import norm
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	z_input = x_input.reshape(n_samples, latent_dim)
	return z_input

# spherical linear interpolation (slerp)
def slerp(val, low, high):
	omega = arccos(clip(dot(low/norm(low), high/norm(high)), -1, 1))
	so = sin(omega)
	if so == 0:
		# L'Hopital's rule/LERP
		return (1.0-val) * low + val * high
	return sin((1.0-val)*omega) / so * low + sin(val*omega) / so * high

# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=10):
	# interpolate ratios between the points
	ratios = linspace(0, 1, num=n_steps)
	# linear interpolate vectors
	vectors = list()
	for ratio in ratios:
		v = slerp(ratio, p1, p2)
  # 添加元素到列表末尾 / Append element to list end
		vectors.append(v)
	return asarray(vectors)

# create a plot of generated images
def plot_generated(examples, n):
	# plot images
 # 生成整数序列 / Generate integer sequence
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :])
	pyplot.show()

# load model
# 从文件加载模型 / Load model from file
model = load_model('generator_model_030.h5')
# generate points in latent space
n = 20
pts = generate_latent_points(100, n)
# interpolate pairs
results = None
# 生成整数序列 / Generate integer sequence
for i in range(0, n, 2):
	# interpolate points in latent space
	interpolated = interpolate_points(pts[i], pts[i+1])
	# generate images
 # 用模型做预测 / Make predictions with model
	X = model.predict(interpolated)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	if results is None:
		results = X
	else:
		results = vstack((results, X))
# plot the result
plot_generated(results, 10)
```

---

➡️ **Next / 下一步**: File 10 of 11

---

### Generate Random Faces

# 10 — Generate Random Faces / 人脸识别

**Chapter 09 — File 10 of 11 / 第09章 — 第10个文件（共11个）**

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import savez_compressed
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
z_input = x_input.reshape(n_samples, latent_dim)
	return z_input
```

---
## Step 5 — create a plot of generated images

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
pyplot.imshow(examples[i, :, :])
	pyplot.savefig('generated_faces.png')
	pyplot.close()
```

---
## Step 10 — load model

```python
# 从文件加载模型 / Load model from file
model = load_model('generator_model_030.h5')
```

---
## Step 11 — generate points in latent space

```python
latent_points = generate_latent_points(100, 100)
```

---
## Step 12 — save points

```python
savez_compressed('latent_points.npz', latent_points)
```

---
## Step 13 — generate images

```python
# 用模型做预测 / Make predictions with model
X  = model.predict(latent_points)
```

---
## Step 14 — scale from [-1,1] to [0,1]

```python
X = (X + 1) / 2.0
```

---
## Step 15 — save plot

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
# Generate Random Faces / 人脸识别
# Complete Code / 完整代码
# ===============================

# example of loading the generator model and generating images
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import savez_compressed

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	z_input = x_input.reshape(n_samples, latent_dim)
	return z_input

# create a plot of generated images
def plot_generated(examples, n):
	# plot images
 # 生成整数序列 / Generate integer sequence
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :])
	pyplot.savefig('generated_faces.png')
	pyplot.close()

# load model
# 从文件加载模型 / Load model from file
model = load_model('generator_model_030.h5')
# generate points in latent space
latent_points = generate_latent_points(100, 100)
# save points
savez_compressed('latent_points.npz', latent_points)
# generate images
# 用模型做预测 / Make predictions with model
X  = model.predict(latent_points)
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# save plot
plot_generated(X, 10)
```

---

➡️ **Next / 下一步**: File 11 of 11

---

### Vector Arithmetic

# 11 — Vector Arithmetic / 11 Vector Arithmetic

**Chapter 09 — File 11 of 11 / 第09章 — 第11个文件（共11个）**

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
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import load
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims
```

---
## Step 2 — average list of latent space vectors

```python
def average_points(points, ix):
```

---
## Step 3 — convert to zero offset points

```python
zero_ix = [i-1 for i in ix]
```

---
## Step 4 — retrieve required points

```python
vectors = points[zero_ix]
```

---
## Step 5 — average the vectors

```python
avg_vector = mean(vectors, axis=0)
```

---
## Step 6 — combine original and avg vectors

```python
all_vectors = vstack((vectors, avg_vector))
	return all_vectors
```

---
## Step 7 — create a plot of generated images

```python
def plot_generated(examples, rows, cols):
```

---
## Step 8 — plot images

```python
# 生成整数序列 / Generate integer sequence
for i in range(rows * cols):
```

---
## Step 9 — define subplot

```python
pyplot.subplot(rows, cols, 1 + i)
```

---
## Step 10 — turn off axis

```python
pyplot.axis('off')
```

---
## Step 11 — plot raw pixel data

```python
pyplot.imshow(examples[i, :, :])
	pyplot.show()
```

---
## Step 12 — load model

```python
# 从文件加载模型 / Load model from file
model = load_model('generator_model_030.h5')
```

---
## Step 13 — retrieve specific points

```python
smiling_woman_ix = [92, 98, 99]
neutral_woman_ix = [9, 21, 79]
neutral_man_ix = [10, 30, 45]
```

---
## Step 14 — load the saved latent points

```python
data = load('latent_points.npz')
points = data['arr_0']
```

---
## Step 15 — average vectors

```python
smiling_woman = average_points(points, smiling_woman_ix)
neutral_woman = average_points(points, neutral_woman_ix)
neutral_man = average_points(points, neutral_man_ix)
```

---
## Step 16 — combine all vectors

```python
all_vectors = vstack((smiling_woman, neutral_woman, neutral_man))
```

---
## Step 17 — generate images

```python
# 用模型做预测 / Make predictions with model
images = model.predict(all_vectors)
```

---
## Step 18 — scale pixel values

```python
images = (images + 1) / 2.0
plot_generated(images, 3, 4)
```

---
## Step 19 — smiling woman - neutral woman + neutral man = smiling man

```python
result_vector = smiling_woman[-1] - neutral_woman[-1] + neutral_man[-1]
```

---
## Step 20 — generate image

```python
result_vector = expand_dims(result_vector, 0)
# 用模型做预测 / Make predictions with model
result_image = model.predict(result_vector)
```

---
## Step 21 — scale pixel values

```python
result_image = (result_image + 1) / 2.0
pyplot.imshow(result_image[0])
pyplot.show()
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
# Vector Arithmetic / 11 Vector Arithmetic
# Complete Code / 完整代码
# ===============================

# example of loading the generator model and generating images
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import load
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import expand_dims

# average list of latent space vectors
def average_points(points, ix):
	# convert to zero offset points
	zero_ix = [i-1 for i in ix]
	# retrieve required points
	vectors = points[zero_ix]
	# average the vectors
	avg_vector = mean(vectors, axis=0)
	# combine original and avg vectors
	all_vectors = vstack((vectors, avg_vector))
	return all_vectors

# create a plot of generated images
def plot_generated(examples, rows, cols):
	# plot images
 # 生成整数序列 / Generate integer sequence
	for i in range(rows * cols):
		# define subplot
		pyplot.subplot(rows, cols, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :])
	pyplot.show()

# load model
# 从文件加载模型 / Load model from file
model = load_model('generator_model_030.h5')
# retrieve specific points
smiling_woman_ix = [92, 98, 99]
neutral_woman_ix = [9, 21, 79]
neutral_man_ix = [10, 30, 45]
# load the saved latent points
data = load('latent_points.npz')
points = data['arr_0']
# average vectors
smiling_woman = average_points(points, smiling_woman_ix)
neutral_woman = average_points(points, neutral_woman_ix)
neutral_man = average_points(points, neutral_man_ix)
# combine all vectors
all_vectors = vstack((smiling_woman, neutral_woman, neutral_man))
# generate images
# 用模型做预测 / Make predictions with model
images = model.predict(all_vectors)
# scale pixel values
images = (images + 1) / 2.0
plot_generated(images, 3, 4)
# smiling woman - neutral woman + neutral man = smiling man
result_vector = smiling_woman[-1] - neutral_woman[-1] + neutral_man[-1]
# generate image
result_vector = expand_dims(result_vector, 0)
# 用模型做预测 / Make predictions with model
result_image = model.predict(result_vector)
# scale pixel values
result_image = (result_image + 1) / 2.0
pyplot.imshow(result_image[0])
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 09 Summary / 第09章总结

## Theme / 主题: Chapter 09 / Chapter 09

This chapter contains **11 code files** demonstrating chapter 09.

本章包含 **11 个代码文件**，演示Chapter 09。

---
## Evolution / 演化路线

  1. `01_plot_faces.ipynb` — Plot Faces
  2. `02_check_mtcnn.ipynb` — Check Mtcnn
  3. `03_prepare_dataset.ipynb` — Prepare Dataset
  4. `04_load_saved_dataset.ipynb` — Load Saved Dataset
  5. `05_train_gan.ipynb` — Train Gan
  6. `06_load_and_generate.ipynb` — Load And Generate
  7. `07_interpolate_latent.ipynb` — Interpolate Latent
  8. `08_multiple_interpolate_latent.ipynb` — Multiple Interpolate Latent
  9. `09_multiple_slerp_latent.ipynb` — Multiple Slerp Latent
  10. `10_generate_random_faces.ipynb` — Generate Random Faces
  11. `11_vector_arithmetic.ipynb` — Vector Arithmetic

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 09) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 09）是机器学习流水线中的基础构建块。

---
