# GAN
## Chapter 20

---

### Summarize Separate Discriminators

# 02 — Summarize Separate Discriminators / 02 Summarize Separate Discriminators

**Chapter 20 — File 2 of 7 / 第20章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of defining semi-supervised discriminator model**.

本脚本演示 **example of defining semi-supervised discriminator model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of defining semi-supervised discriminator model

```python
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
```

---
## Step 2 — define the standalone supervised and unsupervised discriminator models

```python
def define_discriminator(in_shape=(28,28,1), n_classes=10):
```

---
## Step 3 — image input

```python
in_image = Input(shape=in_shape)
```

---
## Step 4 — downsample

```python
fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
```

---
## Step 5 — downsample

```python
fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
```

---
## Step 6 — downsample

```python
fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
```

---
## Step 7 — flatten feature maps

```python
fe = Flatten()(fe)
```

---
## Step 8 — dropout

```python
fe = Dropout(0.4)(fe)
```

---
## Step 9 — unsupervised output

```python
d_out_layer = Dense(1, activation='sigmoid')(fe)
```

---
## Step 10 — define and compile unsupervised discriminator model

```python
d_model = Model(in_image, d_out_layer)
	d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
```

---
## Step 11 — supervised output

```python
c_out_layer = Dense(n_classes, activation='softmax')(fe)
```

---
## Step 12 — define and compile supervised discriminator model

```python
c_model = Model(in_image, c_out_layer)
	c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
	return d_model, c_model
```

---
## Step 13 — create model

```python
d_model, c_model = define_discriminator()
```

---
## Step 14 — plot the model

```python
plot_model(d_model, to_file='discriminator1_plot.png', show_shapes=True, show_layer_names=True)
plot_model(c_model, to_file='discriminator2_plot.png', show_shapes=True, show_layer_names=True)
```

---
## Learning Notes / 学习笔记

- **概念**: example of defining semi-supervised discriminator model 是机器学习中的常用技术。  
  *example of defining semi-supervised discriminator model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Summarize Separate Discriminators / 02 Summarize Separate Discriminators
# Complete Code / 完整代码
# ===============================

# example of defining semi-supervised discriminator model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(28,28,1), n_classes=10):
	# image input
	in_image = Input(shape=in_shape)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# unsupervised output
	d_out_layer = Dense(1, activation='sigmoid')(fe)
	# define and compile unsupervised discriminator model
	d_model = Model(in_image, d_out_layer)
	d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
	# supervised output
	c_out_layer = Dense(n_classes, activation='softmax')(fe)
	# define and compile supervised discriminator model
	c_model = Model(in_image, c_out_layer)
	c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
	return d_model, c_model

# create model
d_model, c_model = define_discriminator()
# plot the model
plot_model(d_model, to_file='discriminator1_plot.png', show_shapes=True, show_layer_names=True)
plot_model(c_model, to_file='discriminator2_plot.png', show_shapes=True, show_layer_names=True)
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Summarize Multi Output Discriminator

# 03 — Summarize Multi Output Discriminator / 03 Summarize Multi Output Discriminator

**Chapter 20 — File 3 of 7 / 第20章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of defining semi-supervised discriminator model**.

本脚本演示 **example of defining semi-supervised discriminator model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of defining semi-supervised discriminator model

```python
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
```

---
## Step 2 — define the standalone supervised and unsupervised discriminator models

```python
def define_discriminator(in_shape=(28,28,1), n_classes=10):
```

---
## Step 3 — image input

```python
in_image = Input(shape=in_shape)
```

---
## Step 4 — downsample

```python
fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
```

---
## Step 5 — downsample

```python
fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
```

---
## Step 6 — downsample

```python
fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
```

---
## Step 7 — flatten feature maps

```python
fe = Flatten()(fe)
```

---
## Step 8 — dropout

```python
fe = Dropout(0.4)(fe)
```

---
## Step 9 — unsupervised output

```python
d_out_layer = Dense(1, activation='sigmoid')(fe)
```

---
## Step 10 — supervised output

```python
c_out_layer = Dense(n_classes + 1, activation='softmax')(fe)
```

---
## Step 11 — define and compile supervised discriminator model

```python
model = Model(in_image, [d_out_layer, c_out_layer])
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
	return model
```

---
## Step 12 — create model

```python
model = define_discriminator()
```

---
## Step 13 — plot the model

```python
plot_model(model, to_file='multioutput_discriminator_plot.png', show_shapes=True, show_layer_names=True)
```

---
## Learning Notes / 学习笔记

- **概念**: example of defining semi-supervised discriminator model 是机器学习中的常用技术。  
  *example of defining semi-supervised discriminator model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Summarize Multi Output Discriminator / 03 Summarize Multi Output Discriminator
# Complete Code / 完整代码
# ===============================

# example of defining semi-supervised discriminator model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(28,28,1), n_classes=10):
	# image input
	in_image = Input(shape=in_shape)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# unsupervised output
	d_out_layer = Dense(1, activation='sigmoid')(fe)
	# supervised output
	c_out_layer = Dense(n_classes + 1, activation='softmax')(fe)
	# define and compile supervised discriminator model
	model = Model(in_image, [d_out_layer, c_out_layer])
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
	return model

# create model
model = define_discriminator()
# plot the model
plot_model(model, to_file='multioutput_discriminator_plot.png', show_shapes=True, show_layer_names=True)
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Custom Activation

# 04 — Custom Activation / 04 Custom Activation

**Chapter 20 — File 4 of 7 / 第20章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of custom activation function**.

本脚本演示 **example of custom activation function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of custom activation function

```python
import numpy as np
```

---
## Step 2 — custom activation function

```python
def custom_activation(output):
	logexpsum = np.sum(np.exp(output))
	result = logexpsum / (logexpsum + 1.0)
	return result
```

---
## Step 3 — all -10s

```python
output = np.asarray([-10.0, -10.0, -10.0])
print(custom_activation(output))
```

---
## Step 4 — all -1s

```python
output = np.asarray([-1.0, -1.0, -1.0])
print(custom_activation(output))
```

---
## Step 5 — all 0s

```python
output = np.asarray([0.0, 0.0, 0.0])
print(custom_activation(output))
```

---
## Step 6 — all 1s

```python
output = np.asarray([1.0, 1.0, 1.0])
print(custom_activation(output))
```

---
## Step 7 — all 10s

```python
output = np.asarray([10.0, 10.0, 10.0])
print(custom_activation(output))
```

---
## Learning Notes / 学习笔记

- **概念**: example of custom activation function 是机器学习中的常用技术。  
  *example of custom activation function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Custom Activation / 04 Custom Activation
# Complete Code / 完整代码
# ===============================

# example of custom activation function
import numpy as np

# custom activation function
def custom_activation(output):
	logexpsum = np.sum(np.exp(output))
	result = logexpsum / (logexpsum + 1.0)
	return result

# all -10s
output = np.asarray([-10.0, -10.0, -10.0])
print(custom_activation(output))
# all -1s
output = np.asarray([-1.0, -1.0, -1.0])
print(custom_activation(output))
# all 0s
output = np.asarray([0.0, 0.0, 0.0])
print(custom_activation(output))
# all 1s
output = np.asarray([1.0, 1.0, 1.0])
print(custom_activation(output))
# all 10s
output = np.asarray([10.0, 10.0, 10.0])
print(custom_activation(output))
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Summarize Discriminator Cust Act

# 05 — Summarize Discriminator Cust Act / 05 Summarize Discriminator Cust Act

**Chapter 20 — File 5 of 7 / 第20章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of defining semi-supervised discriminator model**.

本脚本演示 **example of defining semi-supervised discriminator model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of defining semi-supervised discriminator model

```python
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras import backend
```

---
## Step 2 — custom activation function

```python
def custom_activation(output):
	logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
	result = logexpsum / (logexpsum + 1.0)
	return result
```

---
## Step 3 — define the standalone supervised and unsupervised discriminator models

```python
def define_discriminator(in_shape=(28,28,1), n_classes=10):
```

---
## Step 4 — image input

```python
in_image = Input(shape=in_shape)
```

---
## Step 5 — downsample

```python
fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
```

---
## Step 6 — downsample

```python
fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
```

---
## Step 7 — downsample

```python
fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
```

---
## Step 8 — flatten feature maps

```python
fe = Flatten()(fe)
```

---
## Step 9 — dropout

```python
fe = Dropout(0.4)(fe)
```

---
## Step 10 — output layer nodes

```python
fe = Dense(n_classes)(fe)
```

---
## Step 11 — supervised output

```python
c_out_layer = Activation('softmax')(fe)
```

---
## Step 12 — define and compile supervised discriminator model

```python
c_model = Model(in_image, c_out_layer)
	c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
```

---
## Step 13 — unsupervised output

```python
d_out_layer = Lambda(custom_activation)(fe)
```

---
## Step 14 — define and compile unsupervised discriminator model

```python
d_model = Model(in_image, d_out_layer)
	d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
	return d_model, c_model
```

---
## Step 15 — create model

```python
d_model, c_model = define_discriminator()
```

---
## Step 16 — plot the model

```python
plot_model(d_model, to_file='stacked_discriminator1_plot.png', show_shapes=True, show_layer_names=True)
plot_model(c_model, to_file='stacked_discriminator2_plot.png', show_shapes=True, show_layer_names=True)
```

---
## Learning Notes / 学习笔记

- **概念**: example of defining semi-supervised discriminator model 是机器学习中的常用技术。  
  *example of defining semi-supervised discriminator model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Summarize Discriminator Cust Act / 05 Summarize Discriminator Cust Act
# Complete Code / 完整代码
# ===============================

# example of defining semi-supervised discriminator model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras import backend

# custom activation function
def custom_activation(output):
	logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
	result = logexpsum / (logexpsum + 1.0)
	return result

# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(28,28,1), n_classes=10):
	# image input
	in_image = Input(shape=in_shape)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output layer nodes
	fe = Dense(n_classes)(fe)
	# supervised output
	c_out_layer = Activation('softmax')(fe)
	# define and compile supervised discriminator model
	c_model = Model(in_image, c_out_layer)
	c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
	# unsupervised output
	d_out_layer = Lambda(custom_activation)(fe)
	# define and compile unsupervised discriminator model
	d_model = Model(in_image, d_out_layer)
	d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
	return d_model, c_model

# create model
d_model, c_model = define_discriminator()
# plot the model
plot_model(d_model, to_file='stacked_discriminator1_plot.png', show_shapes=True, show_layer_names=True)
plot_model(c_model, to_file='stacked_discriminator2_plot.png', show_shapes=True, show_layer_names=True)
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Inference Sgan

# 07 — Inference Sgan / 生成对抗网络

**Chapter 20 — File 7 of 7 / 第20章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of loading the classifier model and generating images**.

本脚本演示 **example of loading the classifier model and generating images**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — example of loading the classifier model and generating images

```python
from numpy import expand_dims
from keras.models import load_model
from keras.datasets.mnist import load_data
```

---
## Step 2 — load the model

```python
model = load_model('c_model_7200.h5')
```

---
## Step 3 — load the dataset

```python
(trainX, trainy), (testX, testy) = load_data()
```

---
## Step 4 — expand to 3d, e.g. add channels

```python
trainX = expand_dims(trainX, axis=-1)
testX = expand_dims(testX, axis=-1)
```

---
## Step 5 — convert from ints to floats

```python
trainX = trainX.astype('float32')
testX = testX.astype('float32')
```

---
## Step 6 — scale from [0,255] to [-1,1]

```python
trainX = (trainX - 127.5) / 127.5
testX = (testX - 127.5) / 127.5
```

---
## Step 7 — evaluate the model

```python
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
print('Train Accuracy: %.3f%%' % (train_acc * 100))
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Test Accuracy: %.3f%%' % (test_acc * 100))
```

---
## Learning Notes / 学习笔记

- **概念**: example of loading the classifier model and generating images 是机器学习中的常用技术。  
  *example of loading the classifier model and generating images is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Inference Sgan / 生成对抗网络
# Complete Code / 完整代码
# ===============================

# example of loading the classifier model and generating images
from numpy import expand_dims
from keras.models import load_model
from keras.datasets.mnist import load_data
# load the model
model = load_model('c_model_7200.h5')
# load the dataset
(trainX, trainy), (testX, testy) = load_data()
# expand to 3d, e.g. add channels
trainX = expand_dims(trainX, axis=-1)
testX = expand_dims(testX, axis=-1)
# convert from ints to floats
trainX = trainX.astype('float32')
testX = testX.astype('float32')
# scale from [0,255] to [-1,1]
trainX = (trainX - 127.5) / 127.5
testX = (testX - 127.5) / 127.5
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
print('Train Accuracy: %.3f%%' % (train_acc * 100))
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Test Accuracy: %.3f%%' % (test_acc * 100))
```

---

### Chapter Summary

# Chapter 20 Summary / 第20章总结

## Theme / 主题: Chapter 20 / Chapter 20

This chapter contains **7 code files** demonstrating chapter 20.

本章包含 **7 个代码文件**，演示Chapter 20。

---
## Evolution / 演化路线

  1. `01_summarize_discriminator.ipynb` — Summarize Discriminator
  2. `02_summarize_separate_discriminators.ipynb` — Summarize Separate Discriminators
  3. `03_summarize_multi_output_discriminator.ipynb` — Summarize Multi Output Discriminator
  4. `04_custom_activation.ipynb` — Custom Activation
  5. `05_summarize_discriminator_cust_act.ipynb` — Summarize Discriminator Cust Act
  6. `06_train_sgan.ipynb` — Train Sgan
  7. `07_inference_sgan.ipynb` — Inference Sgan

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 20) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 20）是机器学习流水线中的基础构建块。

---
