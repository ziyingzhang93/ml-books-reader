# GAN
## Chapter 22

---

### Receptive Field

# 01 — Receptive Field / 01 Receptive Field

**Chapter 22 — File 1 of 4 / 第22章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of calculating the receptive field for the PatchGAN**.

本脚本演示 **example of calculating the receptive field for the PatchGAN**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of calculating the receptive field for the PatchGAN
calculate the effective receptive field size

```python
def receptive_field(output_size, kernel_size, stride_size):
    return (output_size - 1) * stride_size + kernel_size
```

---
## Step 2 — output layer 1x1 pixel with 4x4 kernel and 1x1 stride

```python
rf = receptive_field(1, 4, 1)
print(rf)
```

---
## Step 3 — second last layer with 4x4 kernel and 1x1 stride

```python
rf = receptive_field(rf, 4, 1)
print(rf)
```

---
## Step 4 — 3 PatchGAN layers with 4x4 kernel and 2x2 stride

```python
rf = receptive_field(rf, 4, 2)
print(rf)
rf = receptive_field(rf, 4, 2)
print(rf)
rf = receptive_field(rf, 4, 2)
print(rf)
```

---
## Learning Notes / 学习笔记

- **概念**: example of calculating the receptive field for the PatchGAN 是机器学习中的常用技术。  
  *example of calculating the receptive field for the PatchGAN is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Receptive Field / 01 Receptive Field
# Complete Code / 完整代码
# ===============================

# example of calculating the receptive field for the PatchGAN

# calculate the effective receptive field size
def receptive_field(output_size, kernel_size, stride_size):
    return (output_size - 1) * stride_size + kernel_size

# output layer 1x1 pixel with 4x4 kernel and 1x1 stride
rf = receptive_field(1, 4, 1)
print(rf)
# second last layer with 4x4 kernel and 1x1 stride
rf = receptive_field(rf, 4, 1)
print(rf)
# 3 PatchGAN layers with 4x4 kernel and 2x2 stride
rf = receptive_field(rf, 4, 2)
print(rf)
rf = receptive_field(rf, 4, 2)
print(rf)
rf = receptive_field(rf, 4, 2)
print(rf)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Define Summarize Patchgan

# 02 — Define Summarize Patchgan / 生成对抗网络

**Chapter 22 — File 2 of 4 / 第22章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of defining a 70x70 patchgan discriminator model**.

本脚本演示 **example of defining a 70x70 patchgan discriminator model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of defining a 70x70 patchgan discriminator model

```python
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.utils.vis_utils import plot_model
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
in_src_image = Input(shape=image_shape)
```

---
## Step 5 — target image input

```python
in_target_image = Input(shape=image_shape)
```

---
## Step 6 — concatenate images channel-wise

```python
merged = Concatenate()([in_src_image, in_target_image])
```

---
## Step 7 — C64

```python
d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 8 — C128

```python
d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 9 — C256

```python
d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 10 — C512

```python
d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 11 — second last output layer

```python
d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 12 — patch output

```python
d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
```

---
## Step 13 — define model

```python
model = Model([in_src_image, in_target_image], patch_out)
```

---
## Step 14 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model
```

---
## Step 15 — define image shape

```python
image_shape = (256,256,3)
```

---
## Step 16 — create the model

```python
model = define_discriminator(image_shape)
```

---
## Step 17 — summarize the model

```python
model.summary()
```

---
## Step 18 — plot the model

```python
plot_model(model, to_file='discriminator_model_plot.png', show_shapes=True, show_layer_names=True)
```

---
## Learning Notes / 学习笔记

- **概念**: example of defining a 70x70 patchgan discriminator model 是机器学习中的常用技术。  
  *example of defining a 70x70 patchgan discriminator model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Define Summarize Patchgan / 生成对抗网络
# Complete Code / 完整代码
# ===============================

# example of defining a 70x70 patchgan discriminator model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.utils.vis_utils import plot_model

# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

# define image shape
image_shape = (256,256,3)
# create the model
model = define_discriminator(image_shape)
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='discriminator_model_plot.png', show_shapes=True, show_layer_names=True)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Define Summarize Composite

# 04 — Define Summarize Composite / 04 Define Summarize Composite

**Chapter 22 — File 4 of 4 / 第22章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of defining a composite model for training the generator model**.

本脚本演示 **example of defining a composite model for training the generator model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of defining a composite model for training the generator model

```python
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model
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
in_src_image = Input(shape=image_shape)
```

---
## Step 5 — target image input

```python
in_target_image = Input(shape=image_shape)
```

---
## Step 6 — concatenate images channel-wise

```python
merged = Concatenate()([in_src_image, in_target_image])
```

---
## Step 7 — C64

```python
d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 8 — C128

```python
d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 9 — C256

```python
d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 10 — C512

```python
d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 11 — second last output layer

```python
d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
```

---
## Step 12 — patch output

```python
d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
```

---
## Step 13 — define model

```python
model = Model([in_src_image, in_target_image], patch_out)
```

---
## Step 14 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model
```

---
## Step 15 — define an encoder block

```python
def define_encoder_block(layer_in, n_filters, batchnorm=True):
```

---
## Step 16 — weight initialization

```python
init = RandomNormal(stddev=0.02)
```

---
## Step 17 — add downsampling layer

```python
g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
```

---
## Step 18 — conditionally add batch normalization

```python
if batchnorm:
		g = BatchNormalization()(g, training=True)
```

---
## Step 19 — leaky relu activation

```python
g = LeakyReLU(alpha=0.2)(g)
	return g
```

---
## Step 20 — define a decoder block

```python
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
```

---
## Step 21 — weight initialization

```python
init = RandomNormal(stddev=0.02)
```

---
## Step 22 — add upsampling layer

```python
g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
```

---
## Step 23 — add batch normalization

```python
g = BatchNormalization()(g, training=True)
```

---
## Step 24 — conditionally add dropout

```python
if dropout:
		g = Dropout(0.5)(g, training=True)
```

---
## Step 25 — merge with skip connection

```python
g = Concatenate()([g, skip_in])
```

---
## Step 26 — relu activation

```python
g = Activation('relu')(g)
	return g
```

---
## Step 27 — define the standalone generator model

```python
def define_generator(image_shape=(256,256,3)):
```

---
## Step 28 — weight initialization

```python
init = RandomNormal(stddev=0.02)
```

---
## Step 29 — image input

```python
in_image = Input(shape=image_shape)
```

---
## Step 30 — encoder model: C64-C128-C256-C512-C512-C512-C512-C512

```python
e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
```

---
## Step 31 — bottleneck, no batch norm and relu

```python
b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
```

---
## Step 32 — decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128

```python
d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
```

---
## Step 33 — output

```python
g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
```

---
## Step 34 — define model

```python
model = Model(in_image, out_image)
	return model
```

---
## Step 35 — define the combined generator and discriminator model, for updating the generator

```python
def define_gan(g_model, d_model, image_shape):
```

---
## Step 36 — make weights in the discriminator not trainable

```python
for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
```

---
## Step 37 — define the source image

```python
in_src = Input(shape=image_shape)
```

---
## Step 38 — connect the source image to the generator input

```python
gen_out = g_model(in_src)
```

---
## Step 39 — connect the source input and generator output to the discriminator input

```python
dis_out = d_model([in_src, gen_out])
```

---
## Step 40 — src image as input, generated image and classification output

```python
model = Model(in_src, [dis_out, gen_out])
```

---
## Step 41 — compile model

```python
opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model
```

---
## Step 42 — define image shape

```python
image_shape = (256,256,3)
```

---
## Step 43 — define the models

```python
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
```

---
## Step 44 — define the composite model

```python
gan_model = define_gan(g_model, d_model, image_shape)
```

---
## Step 45 — summarize the model

```python
gan_model.summary()
```

---
## Step 46 — plot the model

```python
plot_model(gan_model, to_file='gan_model_plot.png', show_shapes=True, show_layer_names=True)
```

---
## Learning Notes / 学习笔记

- **概念**: example of defining a composite model for training the generator model 是机器学习中的常用技术。  
  *example of defining a composite model for training the generator model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Define Summarize Composite / 04 Define Summarize Composite
# Complete Code / 完整代码
# ===============================

# example of defining a composite model for training the generator model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model

# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

# define the standalone generator model
def define_generator(image_shape=(256,256,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model: C64-C128-C256-C512-C512-C512-C512-C512
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

# define image shape
image_shape = (256,256,3)
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# summarize the model
gan_model.summary()
# plot the model
plot_model(gan_model, to_file='gan_model_plot.png', show_shapes=True, show_layer_names=True)
```

---

### Chapter Summary

# Chapter 22 Summary / 第22章总结

## Theme / 主题: Chapter 22 / Chapter 22

This chapter contains **4 code files** demonstrating chapter 22.

本章包含 **4 个代码文件**，演示Chapter 22。

---
## Evolution / 演化路线

  1. `01_receptive_field.ipynb` — Receptive Field
  2. `02_define_summarize_patchgan.ipynb` — Define Summarize Patchgan
  3. `03_define_summarize_unet.ipynb` — Define Summarize Unet
  4. `04_define_summarize_composite.ipynb` — Define Summarize Composite

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 22) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 22）是机器学习流水线中的基础构建块。

---
