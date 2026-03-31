# 生成对抗网络 / Generative Adversarial Networks
## Chapter 06

---

### Plot Target Function

# 01 — Plot Target Function / 01 Plot Target Function

**Chapter 06 — File 1 of 8 / 第06章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **demonstrate simple x^2 function**.

本脚本演示 **demonstrate simple x^2 function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — demonstrate simple x^2 function

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — simple function

```python
def calculate(x):
	return x * x
```

---
## Step 3 — define inputs

```python
inputs = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
```

---
## Step 4 — calculate outputs

```python
outputs = [calculate(x) for x in inputs]
```

---
## Step 5 — plot the result

```python
pyplot.plot(inputs, outputs)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: demonstrate simple x^2 function 是机器学习中的常用技术。  
  *demonstrate simple x^2 function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Target Function / 01 Plot Target Function
# Complete Code / 完整代码
# ===============================

# demonstrate simple x^2 function
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# simple function
def calculate(x):
	return x * x

# define inputs
inputs = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
# calculate outputs
outputs = [calculate(x) for x in inputs]
# plot the result
pyplot.plot(inputs, outputs)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Sample Target Function

# 02 — Sample Target Function / 02 Sample Target Function

**Chapter 06 — File 2 of 8 / 第06章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of generating random samples from X^2**.

本脚本演示 **example of generating random samples from X^2**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — example of generating random samples from X^2

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — generate randoms sample from x^2

```python
def generate_samples(n=100):
```

---
## Step 3 — generate random inputs in [-0.5, 0.5]

```python
X1 = rand(n) - 0.5
```

---
## Step 4 — generate outputs X^2 (quadratic)

```python
X2 = X1 * X1
```

---
## Step 5 — stack arrays

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X1 = X1.reshape(n, 1)
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X2 = X2.reshape(n, 1)
	return hstack((X1, X2))
```

---
## Step 6 — generate samples

```python
data = generate_samples()
```

---
## Step 7 — plot samples

```python
pyplot.scatter(data[:, 0], data[:, 1])
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of generating random samples from X^2 是机器学习中的常用技术。  
  *example of generating random samples from X^2 is a common technique in machine learning.*

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
# Sample Target Function / 02 Sample Target Function
# Complete Code / 完整代码
# ===============================

# example of generating random samples from X^2
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# generate randoms sample from x^2
def generate_samples(n=100):
	# generate random inputs in [-0.5, 0.5]
	X1 = rand(n) - 0.5
	# generate outputs X^2 (quadratic)
	X2 = X1 * X1
	# stack arrays
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X1 = X1.reshape(n, 1)
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X2 = X2.reshape(n, 1)
	return hstack((X1, X2))

# generate samples
data = generate_samples()
# plot samples
pyplot.scatter(data[:, 0], data[:, 1])
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Define Discriminator

# 03 — Define Discriminator / 03 Define Discriminator

**Chapter 06 — File 3 of 8 / 第06章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **define the discriminator model**.

本脚本演示 **define the discriminator model**。

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
```

---
## Step 1 — define the discriminator model

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils.vis_utils import plot_model
```

---
## Step 2 — define the standalone discriminator model

```python
def define_discriminator(n_inputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 3 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

---
## Step 4 — define the discriminator model

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

- **概念**: define the discriminator model 是机器学习中的常用技术。  
  *define the discriminator model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Define Discriminator / 03 Define Discriminator
# Complete Code / 完整代码
# ===============================

# define the discriminator model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils.vis_utils import plot_model

# define the standalone discriminator model
def define_discriminator(n_inputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# compile model
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# define the discriminator model
model = define_discriminator()
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Fit Discriminator

# 04 — Fit Discriminator / 04 Fit Discriminator

**Chapter 06 — File 4 of 8 / 第06章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **define and fit a discriminator model**.

本脚本演示 **define and fit a discriminator model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
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
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  🏋️ 训练模型 / Train Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — define and fit a discriminator model

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
```

---
## Step 2 — define the standalone discriminator model

```python
def define_discriminator(n_inputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 3 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

---
## Step 4 — generate n real samples with class labels

```python
def generate_real_samples(n):
```

---
## Step 5 — generate inputs in [-0.5, 0.5]

```python
X1 = rand(n) - 0.5
```

---
## Step 6 — generate outputs X^2

```python
X2 = X1 * X1
```

---
## Step 7 — stack arrays

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X1 = X1.reshape(n, 1)
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X2 = X2.reshape(n, 1)
	X = hstack((X1, X2))
```

---
## Step 8 — generate class labels

```python
y = ones((n, 1))
	return X, y
```

---
## Step 9 — generate n fake samples with class labels

```python
def generate_fake_samples(n):
```

---
## Step 10 — generate inputs in [-1, 1]

```python
X1 = -1 + rand(n) * 2
```

---
## Step 11 — generate outputs in [-1, 1]

```python
X2 = -1 + rand(n) * 2
```

---
## Step 12 — stack arrays

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X1 = X1.reshape(n, 1)
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X2 = X2.reshape(n, 1)
	X = hstack((X1, X2))
```

---
## Step 13 — generate class labels

```python
y = zeros((n, 1))
	return X, y
```

---
## Step 14 — train the discriminator model

```python
def train_discriminator(model, n_epochs=1000, n_batch=128):
	half_batch = int(n_batch / 2)
```

---
## Step 15 — run epochs manually

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_epochs):
```

---
## Step 16 — generate real examples

```python
X_real, y_real = generate_real_samples(half_batch)
```

---
## Step 17 — update model

```python
model.train_on_batch(X_real, y_real)
```

---
## Step 18 — generate fake examples

```python
X_fake, y_fake = generate_fake_samples(half_batch)
```

---
## Step 19 — update model

```python
model.train_on_batch(X_fake, y_fake)
```

---
## Step 20 — evaluate the model

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
_, acc_real = model.evaluate(X_real, y_real, verbose=0)
  # 评估模型在测试集上的表现 / Evaluate model on test set
		_, acc_fake = model.evaluate(X_fake, y_fake, verbose=0)
  # 打印输出 / Print output
		print(i, acc_real, acc_fake)
```

---
## Step 21 — define the discriminator model

```python
model = define_discriminator()
```

---
## Step 22 — fit the model

```python
train_discriminator(model)
```

---
## Learning Notes / 学习笔记

- **概念**: define and fit a discriminator model 是机器学习中的常用技术。  
  *define and fit a discriminator model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Fit Discriminator / 04 Fit Discriminator
# Complete Code / 完整代码
# ===============================

# define and fit a discriminator model
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense

# define the standalone discriminator model
def define_discriminator(n_inputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# compile model
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# generate n real samples with class labels
def generate_real_samples(n):
	# generate inputs in [-0.5, 0.5]
	X1 = rand(n) - 0.5
	# generate outputs X^2
	X2 = X1 * X1
	# stack arrays
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X1 = X1.reshape(n, 1)
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X2 = X2.reshape(n, 1)
	X = hstack((X1, X2))
	# generate class labels
	y = ones((n, 1))
	return X, y

# generate n fake samples with class labels
def generate_fake_samples(n):
	# generate inputs in [-1, 1]
	X1 = -1 + rand(n) * 2
	# generate outputs in [-1, 1]
	X2 = -1 + rand(n) * 2
	# stack arrays
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X1 = X1.reshape(n, 1)
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X2 = X2.reshape(n, 1)
	X = hstack((X1, X2))
	# generate class labels
	y = zeros((n, 1))
	return X, y

# train the discriminator model
def train_discriminator(model, n_epochs=1000, n_batch=128):
	half_batch = int(n_batch / 2)
	# run epochs manually
 # 生成整数序列 / Generate integer sequence
	for i in range(n_epochs):
		# generate real examples
		X_real, y_real = generate_real_samples(half_batch)
		# update model
		model.train_on_batch(X_real, y_real)
		# generate fake examples
		X_fake, y_fake = generate_fake_samples(half_batch)
		# update model
		model.train_on_batch(X_fake, y_fake)
		# evaluate the model
  # 评估模型在测试集上的表现 / Evaluate model on test set
		_, acc_real = model.evaluate(X_real, y_real, verbose=0)
  # 评估模型在测试集上的表现 / Evaluate model on test set
		_, acc_fake = model.evaluate(X_fake, y_fake, verbose=0)
  # 打印输出 / Print output
		print(i, acc_real, acc_fake)

# define the discriminator model
model = define_discriminator()
# fit the model
train_discriminator(model)
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Define Generator

# 05 — Define Generator / 05 Define Generator

**Chapter 06 — File 5 of 8 / 第06章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **define the generator model**.

本脚本演示 **define the generator model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — define the generator model

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils.vis_utils import plot_model
```

---
## Step 2 — define the standalone generator model

```python
def define_generator(latent_dim, n_outputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_outputs, activation='linear'))
	return model
```

---
## Step 3 — define the discriminator model

```python
model = define_generator(5)
```

---
## Step 4 — summarize the model

```python
model.summary()
```

---
## Step 5 — plot the model

```python
plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)
```

---
## Learning Notes / 学习笔记

- **概念**: define the generator model 是机器学习中的常用技术。  
  *define the generator model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Define Generator / 05 Define Generator
# Complete Code / 完整代码
# ===============================

# define the generator model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils.vis_utils import plot_model

# define the standalone generator model
def define_generator(latent_dim, n_outputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_outputs, activation='linear'))
	return model

# define the discriminator model
model = define_generator(5)
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Use Generator Model

# 06 — Use Generator Model / 06 Use Generator Model

**Chapter 06 — File 6 of 8 / 第06章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **define and use the generator model**.

本脚本演示 **define and use the generator model**。

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
## Step 1 — define and use the generator model

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — define the standalone generator model

```python
def define_generator(latent_dim, n_outputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_outputs, activation='linear'))
	return model
```

---
## Step 3 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n):
```

---
## Step 4 — generate points in the latent space

```python
x_input = randn(latent_dim * n)
```

---
## Step 5 — reshape into a batch of inputs for the network

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape(n, latent_dim)
	return x_input
```

---
## Step 6 — use the generator to generate n fake examples and plot the results

```python
def generate_fake_samples(generator, latent_dim, n):
```

---
## Step 7 — generate points in latent space

```python
x_input = generate_latent_points(latent_dim, n)
```

---
## Step 8 — predict outputs

```python
X = generator.predict(x_input)
```

---
## Step 9 — plot the results

```python
pyplot.scatter(X[:, 0], X[:, 1])
	pyplot.show()
```

---
## Step 10 — size of the latent space

```python
latent_dim = 5
```

---
## Step 11 — define the discriminator model

```python
model = define_generator(latent_dim)
```

---
## Step 12 — generate and plot generated samples

```python
generate_fake_samples(model, latent_dim, 100)
```

---
## Learning Notes / 学习笔记

- **概念**: define and use the generator model 是机器学习中的常用技术。  
  *define and use the generator model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Use Generator Model / 06 Use Generator Model
# Complete Code / 完整代码
# ===============================

# define and use the generator model
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# define the standalone generator model
def define_generator(latent_dim, n_outputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_outputs, activation='linear'))
	return model

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
	# generate points in the latent space
	x_input = randn(latent_dim * n)
	# reshape into a batch of inputs for the network
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	x_input = x_input.reshape(n, latent_dim)
	return x_input

# use the generator to generate n fake examples and plot the results
def generate_fake_samples(generator, latent_dim, n):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n)
	# predict outputs
	X = generator.predict(x_input)
	# plot the results
	pyplot.scatter(X[:, 0], X[:, 1])
	pyplot.show()

# size of the latent space
latent_dim = 5
# define the discriminator model
model = define_generator(latent_dim)
# generate and plot generated samples
generate_fake_samples(model, latent_dim, 100)
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Define Composite Model

# 07 — Define Composite Model / 07 Define Composite Model

**Chapter 06 — File 7 of 8 / 第06章 — 第7个文件（共8个）**

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
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — demonstrate creating the three models in the gan

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils.vis_utils import plot_model
```

---
## Step 2 — define the standalone discriminator model

```python
def define_discriminator(n_inputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 3 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

---
## Step 4 — define the standalone generator model

```python
def define_generator(latent_dim, n_outputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_outputs, activation='linear'))
	return model
```

---
## Step 5 — define the combined generator and discriminator model, for updating the generator

```python
def define_gan(generator, discriminator):
```

---
## Step 6 — make weights in the discriminator not trainable

```python
discriminator.trainable = False
```

---
## Step 7 — connect them

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 8 — add generator

```python
# 向模型添加一层 / Add a layer to the model
model.add(generator)
```

---
## Step 9 — add the discriminator

```python
# 向模型添加一层 / Add a layer to the model
model.add(discriminator)
```

---
## Step 10 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam')
	return model
```

---
## Step 11 — size of the latent space

```python
latent_dim = 5
```

---
## Step 12 — create the discriminator

```python
discriminator = define_discriminator()
```

---
## Step 13 — create the generator

```python
generator = define_generator(latent_dim)
```

---
## Step 14 — create the gan

```python
gan_model = define_gan(generator, discriminator)
```

---
## Step 15 — summarize gan model

```python
gan_model.summary()
```

---
## Step 16 — plot gan model

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
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Define Composite Model / 07 Define Composite Model
# Complete Code / 完整代码
# ===============================

# demonstrate creating the three models in the gan
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.utils.vis_utils import plot_model

# define the standalone discriminator model
def define_discriminator(n_inputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# compile model
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim, n_outputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_outputs, activation='linear'))
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

# size of the latent space
latent_dim = 5
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# summarize gan model
gan_model.summary()
# plot gan model
plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Complete Example

# 08 — Complete Example / 08 Complete Example

**Chapter 06 — File 8 of 8 / 第06章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **train a generative adversarial network on a one-dimensional function**.

本脚本演示 **train a generative adversarial network on a one-dimensional function**。

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
## Step 1 — train a generative adversarial network on a one-dimensional function

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — define the standalone discriminator model

```python
def define_discriminator(n_inputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
```

---
## Step 3 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

---
## Step 4 — define the standalone generator model

```python
def define_generator(latent_dim, n_outputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_outputs, activation='linear'))
	return model
```

---
## Step 5 — define the combined generator and discriminator model, for updating the generator

```python
def define_gan(generator, discriminator):
```

---
## Step 6 — make weights in the discriminator not trainable

```python
discriminator.trainable = False
```

---
## Step 7 — connect them

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 8 — add generator

```python
# 向模型添加一层 / Add a layer to the model
model.add(generator)
```

---
## Step 9 — add the discriminator

```python
# 向模型添加一层 / Add a layer to the model
model.add(discriminator)
```

---
## Step 10 — compile model

```python
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='binary_crossentropy', optimizer='adam')
	return model
```

---
## Step 11 — generate n real samples with class labels

```python
def generate_real_samples(n):
```

---
## Step 12 — generate inputs in [-0.5, 0.5]

```python
X1 = rand(n) - 0.5
```

---
## Step 13 — generate outputs X^2

```python
X2 = X1 * X1
```

---
## Step 14 — stack arrays

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X1 = X1.reshape(n, 1)
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X2 = X2.reshape(n, 1)
	X = hstack((X1, X2))
```

---
## Step 15 — generate class labels

```python
y = ones((n, 1))
	return X, y
```

---
## Step 16 — generate points in latent space as input for the generator

```python
def generate_latent_points(latent_dim, n):
```

---
## Step 17 — generate points in the latent space

```python
x_input = randn(latent_dim * n)
```

---
## Step 18 — reshape into a batch of inputs for the network

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x_input = x_input.reshape(n, latent_dim)
	return x_input
```

---
## Step 19 — use the generator to generate n fake examples, with class labels

```python
def generate_fake_samples(generator, latent_dim, n):
```

---
## Step 20 — generate points in latent space

```python
x_input = generate_latent_points(latent_dim, n)
```

---
## Step 21 — predict outputs

```python
X = generator.predict(x_input)
```

---
## Step 22 — create class labels

```python
y = zeros((n, 1))
	return X, y
```

---
## Step 23 — evaluate the discriminator and plot real and fake points

```python
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
```

---
## Step 24 — prepare real samples

```python
x_real, y_real = generate_real_samples(n)
```

---
## Step 25 — evaluate discriminator on real examples

```python
_, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
```

---
## Step 26 — prepare fake examples

```python
x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
```

---
## Step 27 — evaluate discriminator on fake examples

```python
_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
```

---
## Step 28 — summarize discriminator performance

```python
# 打印输出 / Print output
print(epoch, acc_real, acc_fake)
```

---
## Step 29 — scatter plot real and fake data points

```python
pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
	pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
```

---
## Step 30 — save plot to file

```python
filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()
```

---
## Step 31 — train the generator and discriminator

```python
def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000):
```

---
## Step 32 — determine half the size of one batch, for updating the discriminator

```python
half_batch = int(n_batch / 2)
```

---
## Step 33 — manually enumerate epochs

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_epochs):
```

---
## Step 34 — prepare real samples

```python
x_real, y_real = generate_real_samples(half_batch)
```

---
## Step 35 — prepare fake examples

```python
x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
```

---
## Step 36 — update discriminator

```python
d_model.train_on_batch(x_real, y_real)
		d_model.train_on_batch(x_fake, y_fake)
```

---
## Step 37 — prepare points in latent space as input for the generator

```python
x_gan = generate_latent_points(latent_dim, n_batch)
```

---
## Step 38 — create inverted labels for the fake samples

```python
y_gan = ones((n_batch, 1))
```

---
## Step 39 — update the generator via the discriminator's error

```python
gan_model.train_on_batch(x_gan, y_gan)
```

---
## Step 40 — evaluate the model every n_eval epochs

```python
if (i+1) % n_eval == 0:
			summarize_performance(i, g_model, d_model, latent_dim)
```

---
## Step 41 — size of the latent space

```python
latent_dim = 5
```

---
## Step 42 — create the discriminator

```python
discriminator = define_discriminator()
```

---
## Step 43 — create the generator

```python
generator = define_generator(latent_dim)
```

---
## Step 44 — create the gan

```python
gan_model = define_gan(generator, discriminator)
```

---
## Step 45 — train model

```python
train(generator, discriminator, gan_model, latent_dim)
```

---
## Learning Notes / 学习笔记

- **概念**: train a generative adversarial network on a one-dimensional function 是机器学习中的常用技术。  
  *train a generative adversarial network on a one-dimensional function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
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
# Complete Example / 08 Complete Example
# Complete Code / 完整代码
# ===============================

# train a generative adversarial network on a one-dimensional function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import zeros
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# define the standalone discriminator model
def define_discriminator(n_inputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(1, activation='sigmoid'))
	# compile model
 # 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim, n_outputs=2):
 # 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
	model = Sequential()
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
 # 向模型添加一层 / Add a layer to the model
	model.add(Dense(n_outputs, activation='linear'))
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

# generate n real samples with class labels
def generate_real_samples(n):
	# generate inputs in [-0.5, 0.5]
	X1 = rand(n) - 0.5
	# generate outputs X^2
	X2 = X1 * X1
	# stack arrays
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X1 = X1.reshape(n, 1)
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	X2 = X2.reshape(n, 1)
	X = hstack((X1, X2))
	# generate class labels
	y = ones((n, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
	# generate points in the latent space
	x_input = randn(latent_dim * n)
	# reshape into a batch of inputs for the network
 # 改变数组形状（不改变数据） / Reshape array (data unchanged)
	x_input = x_input.reshape(n, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = zeros((n, 1))
	return X, y

# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
	# prepare real samples
	x_real, y_real = generate_real_samples(n)
	# evaluate discriminator on real examples
	_, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
	# evaluate discriminator on fake examples
	_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
 # 打印输出 / Print output
	print(epoch, acc_real, acc_fake)
	# scatter plot real and fake data points
	pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
	pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000):
	# determine half the size of one batch, for updating the discriminator
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
 # 生成整数序列 / Generate integer sequence
	for i in range(n_epochs):
		# prepare real samples
		x_real, y_real = generate_real_samples(half_batch)
		# prepare fake examples
		x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator
		d_model.train_on_batch(x_real, y_real)
		d_model.train_on_batch(x_fake, y_fake)
		# prepare points in latent space as input for the generator
		x_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		gan_model.train_on_batch(x_gan, y_gan)
		# evaluate the model every n_eval epochs
		if (i+1) % n_eval == 0:
			summarize_performance(i, g_model, d_model, latent_dim)

# size of the latent space
latent_dim = 5
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# train model
train(generator, discriminator, gan_model, latent_dim)
```

---

### Chapter Summary / 章节总结



---
