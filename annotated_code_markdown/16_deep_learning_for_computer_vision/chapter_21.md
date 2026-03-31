# 计算机视觉深度学习 / Deep Learning for Computer Vision
## Chapter 21

---

### Plot Dog Photos

# 01 — Plot Dog Photos / 01 Plot Dog Photos

**Chapter 21 — File 1 of 14 / 第21章 — 第1个文件（共14个）**

---

## Summary / 总结

This script demonstrates **plot dog photos from the dogs vs cats dataset**.

本脚本演示 **plot dog photos from the dogs vs cats dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — plot dog photos from the dogs vs cats dataset

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.image import imread
```

---
## Step 2 — define location of dataset

```python
folder = 'train/'
```

---
## Step 3 — plot first few images

```python
# 生成整数序列 / Generate integer sequence
for i in range(9):
```

---
## Step 4 — define subplot

```python
pyplot.subplot(330 + 1 + i)
```

---
## Step 5 — define filename

```python
filename = folder + 'dog.' + str(i) + '.jpg'
```

---
## Step 6 — load image pixels

```python
image = imread(filename)
```

---
## Step 7 — plot raw pixel data

```python
pyplot.imshow(image)
```

---
## Step 8 — show the figure

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot dog photos from the dogs vs cats dataset 是机器学习中的常用技术。  
  *plot dog photos from the dogs vs cats dataset is a common technique in machine learning.*

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
# Plot Dog Photos / 01 Plot Dog Photos
# Complete Code / 完整代码
# ===============================

# plot dog photos from the dogs vs cats dataset
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.image import imread
# define location of dataset
folder = 'train/'
# plot first few images
# 生成整数序列 / Generate integer sequence
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + 'dog.' + str(i) + '.jpg'
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 14

---

### Plot Cat Photos

# 02 — Plot Cat Photos / 02 Plot Cat Photos

**Chapter 21 — File 2 of 14 / 第21章 — 第2个文件（共14个）**

---

## Summary / 总结

This script demonstrates **plot cat photos from the dogs vs cats dataset**.

本脚本演示 **plot cat photos from the dogs vs cats dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — plot cat photos from the dogs vs cats dataset

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.image import imread
```

---
## Step 2 — define location of dataset

```python
folder = 'train/'
```

---
## Step 3 — plot first few images

```python
# 生成整数序列 / Generate integer sequence
for i in range(9):
```

---
## Step 4 — define subplot

```python
pyplot.subplot(330 + 1 + i)
```

---
## Step 5 — define filename

```python
filename = folder + 'cat.' + str(i) + '.jpg'
```

---
## Step 6 — load image pixels

```python
image = imread(filename)
```

---
## Step 7 — plot raw pixel data

```python
pyplot.imshow(image)
```

---
## Step 8 — show the figure

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot cat photos from the dogs vs cats dataset 是机器学习中的常用技术。  
  *plot cat photos from the dogs vs cats dataset is a common technique in machine learning.*

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
# Plot Cat Photos / 02 Plot Cat Photos
# Complete Code / 完整代码
# ===============================

# plot cat photos from the dogs vs cats dataset
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.image import imread
# define location of dataset
folder = 'train/'
# plot first few images
# 生成整数序列 / Generate integer sequence
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + 'cat.' + str(i) + '.jpg'
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 14

---

### Preprocess Photo Sizes



---

### Load Preprocessed Photos

# 04 — Load Preprocessed Photos / 04 Load Preprocessed Photos

**Chapter 21 — File 4 of 14 / 第21章 — 第4个文件（共14个）**

---

## Summary / 总结

This script demonstrates **load and confirm the shape**.

本脚本演示 **load and confirm the shape**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — load and confirm the shape

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import load
photos = load('dogs_vs_cats_photos.npy')
labels = load('dogs_vs_cats_labels.npy')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(photos.shape, labels.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: load and confirm the shape 是机器学习中的常用技术。  
  *load and confirm the shape is a common technique in machine learning.*

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
# Load Preprocessed Photos / 04 Load Preprocessed Photos
# Complete Code / 完整代码
# ===============================

# load and confirm the shape
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import load
photos = load('dogs_vs_cats_photos.npy')
labels = load('dogs_vs_cats_labels.npy')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(photos.shape, labels.shape)
```

---

➡️ **Next / 下一步**: File 5 of 14

---

### Restructure Dataset



---

### Model Baseline1

# 06 — Model Baseline1 / 06 Model Baseline1

**Chapter 21 — File 6 of 14 / 第21章 — 第6个文件（共14个）**

---

## Summary / 总结

This script demonstrates **1-vgg block baseline model for the dogs vs cats dataset**.

本脚本演示 **1-vgg block baseline model for the dogs vs cats dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — 1-vgg block baseline model for the dogs vs cats dataset

```python
# 导入系统相关功能 / Import system utilities
import sys
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
```

---
## Step 2 — define cnn model

```python
def define_model():
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 3 — compile model

```python
opt = SGD(lr=0.001, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model
```

---
## Step 4 — plot diagnostic learning curves

```python
def summarize_diagnostics(history):
```

---
## Step 5 — plot loss

```python
pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
```

---
## Step 6 — plot accuracy

```python
pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
```

---
## Step 7 — save plot to file

```python
filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
```

---
## Step 8 — run the test harness for evaluating a model

```python
def run_test_harness():
```

---
## Step 9 — define model

```python
model = define_model()
```

---
## Step 10 — create data generator

```python
datagen = ImageDataGenerator(rescale=1.0/255.0)
```

---
## Step 11 — prepare iterators

```python
train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
```

---
## Step 12 — fit model

```python
# 获取长度 / Get length
history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
  # 获取长度 / Get length
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
```

---
## Step 13 — evaluate model

```python
# 获取长度 / Get length
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
 # 打印输出 / Print output
	print('> %.3f' % (acc * 100.0))
```

---
## Step 14 — learning curves

```python
summarize_diagnostics(history)
```

---
## Step 15 — entry point, run the test harness

```python
run_test_harness()
```

---
## Learning Notes / 学习笔记

- **概念**: 1-vgg block baseline model for the dogs vs cats dataset 是机器学习中的常用技术。  
  *1-vgg block baseline model for the dogs vs cats dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model Baseline1 / 06 Model Baseline1
# Complete Code / 完整代码
# ===============================

# 1-vgg block baseline model for the dogs vs cats dataset
# 导入系统相关功能 / Import system utilities
import sys
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator

# define cnn model
def define_model():
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	# fit model
 # 获取长度 / Get length
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
  # 获取长度 / Get length
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
	# evaluate model
 # 获取长度 / Get length
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
 # 打印输出 / Print output
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()
```

---

➡️ **Next / 下一步**: File 7 of 14

---

### Model Baseline2

# 07 — Model Baseline2 / 07 Model Baseline2

**Chapter 21 — File 7 of 14 / 第21章 — 第7个文件（共14个）**

---

## Summary / 总结

This script demonstrates **2-vgg block baseline model for the dogs vs cats dataset**.

本脚本演示 **2-vgg block baseline model for the dogs vs cats dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — 2-vgg block baseline model for the dogs vs cats dataset

```python
# 导入系统相关功能 / Import system utilities
import sys
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
```

---
## Step 2 — define cnn model

```python
def define_model():
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 3 — compile model

```python
opt = SGD(lr=0.001, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model
```

---
## Step 4 — plot diagnostic learning curves

```python
def summarize_diagnostics(history):
```

---
## Step 5 — plot loss

```python
pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
```

---
## Step 6 — plot accuracy

```python
pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
```

---
## Step 7 — save plot to file

```python
filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
```

---
## Step 8 — run the test harness for evaluating a model

```python
def run_test_harness():
```

---
## Step 9 — define model

```python
model = define_model()
```

---
## Step 10 — create data generator

```python
datagen = ImageDataGenerator(rescale=1.0/255.0)
```

---
## Step 11 — prepare iterators

```python
train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
```

---
## Step 12 — fit model

```python
# 获取长度 / Get length
history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
  # 获取长度 / Get length
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
```

---
## Step 13 — evaluate model

```python
# 获取长度 / Get length
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
 # 打印输出 / Print output
	print('> %.3f' % (acc * 100.0))
```

---
## Step 14 — learning curves

```python
summarize_diagnostics(history)
```

---
## Step 15 — entry point, run the test harness

```python
run_test_harness()
```

---
## Learning Notes / 学习笔记

- **概念**: 2-vgg block baseline model for the dogs vs cats dataset 是机器学习中的常用技术。  
  *2-vgg block baseline model for the dogs vs cats dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model Baseline2 / 07 Model Baseline2
# Complete Code / 完整代码
# ===============================

# 2-vgg block baseline model for the dogs vs cats dataset
# 导入系统相关功能 / Import system utilities
import sys
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator

# define cnn model
def define_model():
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	# fit model
 # 获取长度 / Get length
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
  # 获取长度 / Get length
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
	# evaluate model
 # 获取长度 / Get length
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
 # 打印输出 / Print output
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()
```

---

➡️ **Next / 下一步**: File 8 of 14

---

### Model Baseline3

# 08 — Model Baseline3 / 08 Model Baseline3

**Chapter 21 — File 8 of 14 / 第21章 — 第8个文件（共14个）**

---

## Summary / 总结

This script demonstrates **3-vgg block baseline model for the dogs vs cats dataset**.

本脚本演示 **3-vgg block baseline model for the dogs vs cats dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — 3-vgg block baseline model for the dogs vs cats dataset

```python
# 导入系统相关功能 / Import system utilities
import sys
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
```

---
## Step 2 — define cnn model

```python
def define_model():
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 3 — compile model

```python
opt = SGD(lr=0.001, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model
```

---
## Step 4 — plot diagnostic learning curves

```python
def summarize_diagnostics(history):
```

---
## Step 5 — plot loss

```python
pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
```

---
## Step 6 — plot accuracy

```python
pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
```

---
## Step 7 — save plot to file

```python
filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
```

---
## Step 8 — run the test harness for evaluating a model

```python
def run_test_harness():
```

---
## Step 9 — define model

```python
model = define_model()
```

---
## Step 10 — create data generator

```python
datagen = ImageDataGenerator(rescale=1.0/255.0)
```

---
## Step 11 — prepare iterators

```python
train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
```

---
## Step 12 — fit model

```python
# 获取长度 / Get length
history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
  # 获取长度 / Get length
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
```

---
## Step 13 — evaluate model

```python
# 获取长度 / Get length
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
 # 打印输出 / Print output
	print('> %.3f' % (acc * 100.0))
```

---
## Step 14 — learning curves

```python
summarize_diagnostics(history)
```

---
## Step 15 — entry point, run the test harness

```python
run_test_harness()
```

---
## Learning Notes / 学习笔记

- **概念**: 3-vgg block baseline model for the dogs vs cats dataset 是机器学习中的常用技术。  
  *3-vgg block baseline model for the dogs vs cats dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model Baseline3 / 08 Model Baseline3
# Complete Code / 完整代码
# ===============================

# 3-vgg block baseline model for the dogs vs cats dataset
# 导入系统相关功能 / Import system utilities
import sys
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator

# define cnn model
def define_model():
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	# fit model
 # 获取长度 / Get length
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
  # 获取长度 / Get length
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
	# evaluate model
 # 获取长度 / Get length
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
 # 打印输出 / Print output
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()
```

---

➡️ **Next / 下一步**: File 9 of 14

---

### Model Baseline3 Dropout

# 09 — Model Baseline3 Dropout / 随机失活

**Chapter 21 — File 9 of 14 / 第21章 — 第9个文件（共14个）**

---

## Summary / 总结

This script demonstrates **baseline model with dropout for the dogs vs cats dataset**.

本脚本演示 **baseline model with dropout for the dogs vs cats dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — baseline model with dropout for the dogs vs cats dataset

```python
# 导入系统相关功能 / Import system utilities
import sys
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dropout
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
```

---
## Step 2 — define cnn model

```python
def define_model():
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dropout(0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dropout(0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dropout(0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dropout(0.5))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 3 — compile model

```python
opt = SGD(lr=0.001, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model
```

---
## Step 4 — plot diagnostic learning curves

```python
def summarize_diagnostics(history):
```

---
## Step 5 — plot loss

```python
pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
```

---
## Step 6 — plot accuracy

```python
pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
```

---
## Step 7 — save plot to file

```python
filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
```

---
## Step 8 — run the test harness for evaluating a model

```python
def run_test_harness():
```

---
## Step 9 — define model

```python
model = define_model()
```

---
## Step 10 — create data generator

```python
datagen = ImageDataGenerator(rescale=1.0/255.0)
```

---
## Step 11 — prepare iterator

```python
train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
```

---
## Step 12 — fit model

```python
# 获取长度 / Get length
history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
  # 获取长度 / Get length
		validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=0)
```

---
## Step 13 — evaluate model

```python
# 获取长度 / Get length
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
 # 打印输出 / Print output
	print('> %.3f' % (acc * 100.0))
```

---
## Step 14 — learning curves

```python
summarize_diagnostics(history)
```

---
## Step 15 — entry point, run the test harness

```python
run_test_harness()
```

---
## Learning Notes / 学习笔记

- **概念**: baseline model with dropout for the dogs vs cats dataset 是机器学习中的常用技术。  
  *baseline model with dropout for the dogs vs cats dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `dropout` | 随机丢弃：训练时随机关闭部分神经元 | Randomly disable neurons during training |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model Baseline3 Dropout / 随机失活
# Complete Code / 完整代码
# ===============================

# baseline model with dropout for the dogs vs cats dataset
# 导入系统相关功能 / Import system utilities
import sys
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dropout
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator

# define cnn model
def define_model():
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dropout(0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dropout(0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dropout(0.2))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dropout(0.5))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterator
	train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	# fit model
 # 获取长度 / Get length
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
  # 获取长度 / Get length
		validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=0)
	# evaluate model
 # 获取长度 / Get length
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
 # 打印输出 / Print output
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()
```

---

➡️ **Next / 下一步**: File 10 of 14

---

### Model Baseline3 Data Aug

# 10 — Model Baseline3 Data Aug / 10 Model Baseline3 Data Aug

**Chapter 21 — File 10 of 14 / 第21章 — 第10个文件（共14个）**

---

## Summary / 总结

This script demonstrates **baseline model with data augmentation for the dogs vs cats dataset**.

本脚本演示 **baseline model with data augmentation for the dogs vs cats dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — baseline model with data augmentation for the dogs vs cats dataset

```python
# 导入系统相关功能 / Import system utilities
import sys
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
```

---
## Step 2 — define cnn model

```python
def define_model():
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 3 — compile model

```python
opt = SGD(lr=0.001, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model
```

---
## Step 4 — plot diagnostic learning curves

```python
def summarize_diagnostics(history):
```

---
## Step 5 — plot loss

```python
pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
```

---
## Step 6 — plot accuracy

```python
pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
```

---
## Step 7 — save plot to file

```python
filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
```

---
## Step 8 — run the test harness for evaluating a model

```python
def run_test_harness():
```

---
## Step 9 — define model

```python
model = define_model()
```

---
## Step 10 — create data generators

```python
train_datagen = ImageDataGenerator(rescale=1.0/255.0,
		width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	test_datagen = ImageDataGenerator(rescale=1.0/255.0)
```

---
## Step 11 — prepare iterators

```python
train_it = train_datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = test_datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
```

---
## Step 12 — fit model

```python
# 获取长度 / Get length
history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
  # 获取长度 / Get length
		validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=0)
```

---
## Step 13 — evaluate model

```python
# 获取长度 / Get length
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
 # 打印输出 / Print output
	print('> %.3f' % (acc * 100.0))
```

---
## Step 14 — learning curves

```python
summarize_diagnostics(history)
```

---
## Step 15 — entry point, run the test harness

```python
run_test_harness()
```

---
## Learning Notes / 学习笔记

- **概念**: baseline model with data augmentation for the dogs vs cats dataset 是机器学习中的常用技术。  
  *baseline model with data augmentation for the dogs vs cats dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model Baseline3 Data Aug / 10 Model Baseline3 Data Aug
# Complete Code / 完整代码
# ===============================

# baseline model with data augmentation for the dogs vs cats dataset
# 导入系统相关功能 / Import system utilities
import sys
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator

# define cnn model
def define_model():
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 # 向模型添加一层 / Add a layer to the model
	model.add(MaxPooling2D((2, 2)))
 # 向模型添加一层 / Add a layer to the model
	model.add(Flatten())
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
	# define model
	model = define_model()
	# create data generators
	train_datagen = ImageDataGenerator(rescale=1.0/255.0,
		width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	test_datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = train_datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = test_datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	# fit model
 # 获取长度 / Get length
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
  # 获取长度 / Get length
		validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=0)
	# evaluate model
 # 获取长度 / Get length
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
 # 打印输出 / Print output
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()
```

---

➡️ **Next / 下一步**: File 11 of 14

---

### Model Pretrained

# 11 — Model Pretrained / 预训练模型

**Chapter 21 — File 11 of 14 / 第21章 — 第11个文件（共14个）**

---

## Summary / 总结

This script demonstrates **vgg16 model used for transfer learning on the dogs and cats dataset**.

本脚本演示 **vgg16 model used for transfer learning on the dogs and cats dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — vgg16 model used for transfer learning on the dogs and cats dataset

```python
# 导入系统相关功能 / Import system utilities
import sys
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.applications.vgg16 import VGG16
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
```

---
## Step 2 — define cnn model

```python
def define_model():
```

---
## Step 3 — load model

```python
model = VGG16(include_top=False, input_shape=(224, 224, 3))
```

---
## Step 4 — mark loaded layers as not trainable

```python
for layer in model.layers:
		layer.trainable = False
```

---
## Step 5 — add new classifier layers

```python
# 展平层：多维→一维 / Flatten: multi-dim → 1D
flat1 = Flatten()(model.layers[-1].output)
 # 全连接层（Keras） / Fully connected layer (Keras)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
 # 全连接层（Keras） / Fully connected layer (Keras)
	output = Dense(1, activation='sigmoid')(class1)
```

---
## Step 6 — define new model

```python
model = Model(inputs=model.inputs, outputs=output)
```

---
## Step 7 — compile model

```python
opt = SGD(lr=0.001, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model
```

---
## Step 8 — plot diagnostic learning curves

```python
def summarize_diagnostics(history):
```

---
## Step 9 — plot loss

```python
pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
```

---
## Step 10 — plot accuracy

```python
pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
```

---
## Step 11 — save plot to file

```python
filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
```

---
## Step 12 — run the test harness for evaluating a model

```python
def run_test_harness():
```

---
## Step 13 — define model

```python
model = define_model()
```

---
## Step 14 — create data generator

```python
datagen = ImageDataGenerator(featurewise_center=True)
```

---
## Step 15 — specify imagenet mean values for centering

```python
datagen.mean = [123.68, 116.779, 103.939]
```

---
## Step 16 — prepare iterator

```python
train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
```

---
## Step 17 — fit model

```python
# 获取长度 / Get length
history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
  # 获取长度 / Get length
		validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=1)
```

---
## Step 18 — evaluate model

```python
# 获取长度 / Get length
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
 # 打印输出 / Print output
	print('> %.3f' % (acc * 100.0))
```

---
## Step 19 — learning curves

```python
summarize_diagnostics(history)
```

---
## Step 20 — entry point, run the test harness

```python
run_test_harness()
```

---
## Learning Notes / 学习笔记

- **概念**: vgg16 model used for transfer learning on the dogs and cats dataset 是机器学习中的常用技术。  
  *vgg16 model used for transfer learning on the dogs and cats dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model Pretrained / 预训练模型
# Complete Code / 完整代码
# ===============================

# vgg16 model used for transfer learning on the dogs and cats dataset
# 导入系统相关功能 / Import system utilities
import sys
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.applications.vgg16 import VGG16
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator

# define cnn model
def define_model():
	# load model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
 # 展平层：多维→一维 / Flatten: multi-dim → 1D
	flat1 = Flatten()(model.layers[-1].output)
 # 全连接层（Keras） / Fully connected layer (Keras)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
 # 全连接层（Keras） / Fully connected layer (Keras)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(featurewise_center=True)
	# specify imagenet mean values for centering
	datagen.mean = [123.68, 116.779, 103.939]
	# prepare iterator
	train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	# fit model
 # 获取长度 / Get length
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
  # 获取长度 / Get length
		validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=1)
	# evaluate model
 # 获取长度 / Get length
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
 # 打印输出 / Print output
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()
```

---

➡️ **Next / 下一步**: File 12 of 14

---

### Prepare Final Dataset

# 12 — Prepare Final Dataset / 数据准备

**Chapter 21 — File 12 of 14 / 第21章 — 第12个文件（共14个）**

---

## Summary / 总结

This script demonstrates **organize dataset into a useful structure**.

本脚本演示 **organize dataset into a useful structure**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — organize dataset into a useful structure

```python
from os import makedirs
from os import listdir
from shutil import copyfile
```

---
## Step 2 — create directories

```python
dataset_home = 'finalize_dogs_vs_cats/'
```

---
## Step 3 — create label subdirectories

```python
labeldirs = ['dogs/', 'cats/']
for labldir in labeldirs:
	newdir = dataset_home + labldir
	makedirs(newdir, exist_ok=True)
```

---
## Step 4 — copy training dataset images into subdirectories

```python
src_directory = 'dogs-vs-cats/train/'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	if file.startswith('cat'):
		dst = dataset_home + 'cats/'  + file
		copyfile(src, dst)
	elif file.startswith('dog'):
		dst = dataset_home + 'dogs/'  + file
		copyfile(src, dst)
```

---
## Learning Notes / 学习笔记

- **概念**: organize dataset into a useful structure 是机器学习中的常用技术。  
  *organize dataset into a useful structure is a common technique in machine learning.*

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
# Prepare Final Dataset / 数据准备
# Complete Code / 完整代码
# ===============================

# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
# create directories
dataset_home = 'finalize_dogs_vs_cats/'
# create label subdirectories
labeldirs = ['dogs/', 'cats/']
for labldir in labeldirs:
	newdir = dataset_home + labldir
	makedirs(newdir, exist_ok=True)
# copy training dataset images into subdirectories
src_directory = 'dogs-vs-cats/train/'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	if file.startswith('cat'):
		dst = dataset_home + 'cats/'  + file
		copyfile(src, dst)
	elif file.startswith('dog'):
		dst = dataset_home + 'dogs/'  + file
		copyfile(src, dst)
```

---

➡️ **Next / 下一步**: File 13 of 14

---

### Save Final Model

# 13 — Save Final Model / 保存/加载模型

**Chapter 21 — File 13 of 14 / 第21章 — 第13个文件（共14个）**

---

## Summary / 总结

This script demonstrates **save the final model to file**.

本脚本演示 **save the final model to file**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  💾 保存结果 / Save Results
```

---
## Step 1 — save the final model to file

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.applications.vgg16 import VGG16
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator
```

---
## Step 2 — define cnn model

```python
def define_model():
```

---
## Step 3 — load model

```python
model = VGG16(include_top=False, input_shape=(224, 224, 3))
```

---
## Step 4 — mark loaded layers as not trainable

```python
for layer in model.layers:
		layer.trainable = False
```

---
## Step 5 — add new classifier layers

```python
# 展平层：多维→一维 / Flatten: multi-dim → 1D
flat1 = Flatten()(model.layers[-1].output)
 # 全连接层（Keras） / Fully connected layer (Keras)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
 # 全连接层（Keras） / Fully connected layer (Keras)
	output = Dense(1, activation='sigmoid')(class1)
```

---
## Step 6 — define new model

```python
model = Model(inputs=model.inputs, outputs=output)
```

---
## Step 7 — compile model

```python
opt = SGD(lr=0.001, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model
```

---
## Step 8 — run the test harness for evaluating a model

```python
def run_test_harness():
```

---
## Step 9 — define model

```python
model = define_model()
```

---
## Step 10 — create data generator

```python
datagen = ImageDataGenerator(featurewise_center=True)
```

---
## Step 11 — specify imagenet mean values for centering

```python
datagen.mean = [123.68, 116.779, 103.939]
```

---
## Step 12 — prepare iterator

```python
train_it = datagen.flow_from_directory('finalize_dogs_vs_cats/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
```

---
## Step 13 — fit model

```python
# 获取长度 / Get length
model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=0)
```

---
## Step 14 — save model

```python
# 保存模型到文件 / Save model to file
model.save('final_model.h5')
```

---
## Step 15 — entry point, run the test harness

```python
run_test_harness()
```

---
## Learning Notes / 学习笔记

- **概念**: save the final model to file 是机器学习中的常用技术。  
  *save the final model to file is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Save Final Model / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# save the final model to file
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.applications.vgg16 import VGG16
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Flatten
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.optimizers import SGD
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import ImageDataGenerator

# define cnn model
def define_model():
	# load model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
 # 展平层：多维→一维 / Flatten: multi-dim → 1D
	flat1 = Flatten()(model.layers[-1].output)
 # 全连接层（Keras） / Fully connected layer (Keras)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
 # 全连接层（Keras） / Fully connected layer (Keras)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# run the test harness for evaluating a model
def run_test_harness():
# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(featurewise_center=True)
	# specify imagenet mean values for centering
	datagen.mean = [123.68, 116.779, 103.939]
	# prepare iterator
	train_it = datagen.flow_from_directory('finalize_dogs_vs_cats/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	# fit model
 # 获取长度 / Get length
	model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=0)
	# save model
 # 保存模型到文件 / Save model to file
	model.save('final_model.h5')

# entry point, run the test harness
run_test_harness()
```

---

➡️ **Next / 下一步**: File 14 of 14

---

### Predict Final Model

# 14 — Predict Final Model / 14 Predict Final Model

**Chapter 21 — File 14 of 14 / 第21章 — 第14个文件（共14个）**

---

## Summary / 总结

This script demonstrates **make a prediction for a new image.**.

本脚本演示 **make a prediction for a new image.**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — make a prediction for a new image.

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model
```

---
## Step 2 — load and prepare the image

```python
def load_image(filename):
```

---
## Step 3 — load the image

```python
img = load_img(filename, target_size=(224, 224))
```

---
## Step 4 — convert to array

```python
img = img_to_array(img)
```

---
## Step 5 — reshape into a single sample with 3 channels

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
img = img.reshape(1, 224, 224, 3)
```

---
## Step 6 — center pixel data

```python
# 转换数据类型 / Convert data type
img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img
```

---
## Step 7 — load an image and predict the class

```python
def run_example():
```

---
## Step 8 — load the image

```python
img = load_image('sample_image.jpg')
```

---
## Step 9 — load model

```python
# 从文件加载模型 / Load model from file
model = load_model('final_model.h5')
```

---
## Step 10 — predict the class

```python
# 用模型做预测 / Make predictions with model
result = model.predict(img)
 # 打印输出 / Print output
	print(result[0])
```

---
## Step 11 — entry point, run the example

```python
run_example()
```

---
## Learning Notes / 学习笔记

- **概念**: make a prediction for a new image. 是机器学习中的常用技术。  
  *make a prediction for a new image. is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Predict Final Model / 14 Predict Final Model
# Complete Code / 完整代码
# ===============================

# make a prediction for a new image.
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import load_img
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.preprocessing.image import img_to_array
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import load_model

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
 # 转换数据类型 / Convert data type
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img

# load an image and predict the class
def run_example():
	# load the image
	img = load_image('sample_image.jpg')
	# load model
 # 从文件加载模型 / Load model from file
	model = load_model('final_model.h5')
	# predict the class
 # 用模型做预测 / Make predictions with model
	result = model.predict(img)
 # 打印输出 / Print output
	print(result[0])

# entry point, run the example
run_example()
```

---

### Chapter Summary / 章节总结

# Chapter 21 Summary / 第21章总结

## Theme / 主题: Chapter 21 / Chapter 21

This chapter contains **14 code files** demonstrating chapter 21.

本章包含 **14 个代码文件**，演示Chapter 21。

---
## Evolution / 演化路线

  1. `01_plot_dog_photos.ipynb` — Plot Dog Photos
  2. `02_plot_cat_photos.ipynb` — Plot Cat Photos
  3. `03_preprocess_photo_sizes.ipynb` — Preprocess Photo Sizes
  4. `04_load_preprocessed_photos.ipynb` — Load Preprocessed Photos
  5. `05_restructure_dataset.ipynb` — Restructure Dataset
  6. `06_model_baseline1.ipynb` — Model Baseline1
  7. `07_model_baseline2.ipynb` — Model Baseline2
  8. `08_model_baseline3.ipynb` — Model Baseline3
  9. `09_model_baseline3_dropout.ipynb` — Model Baseline3 Dropout
  10. `10_model_baseline3_data_aug.ipynb` — Model Baseline3 Data Aug
  11. `11_model_pretrained.ipynb` — Model Pretrained
  12. `12_prepare_final_dataset.ipynb` — Prepare Final Dataset
  13. `13_save_final_model.ipynb` — Save Final Model
  14. `14_predict_final_model.ipynb` — Predict Final Model

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 21) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 21）是机器学习流水线中的基础构建块。

---
