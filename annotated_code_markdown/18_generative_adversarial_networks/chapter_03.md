# 生成对抗网络 / Generative Adversarial Networks
## Chapter 03

---

### Upsample Layer

# 01 — Upsample Layer / 01 Upsample Layer

**Chapter 03 — File 1 of 4 / 第03章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of using the upsampling layer**.

本脚本演示 **example of using the upsampling layer**。

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
## Step 1 — example of using the upsampling layer

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import UpSampling2D
```

---
## Step 2 — define input data

```python
X = asarray([[1, 2],
			 [3, 4]])
```

---
## Step 3 — show input data for context

```python
# 打印输出 / Print output
print(X)
```

---
## Step 4 — reshape input data into one sample a sample with a channel

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = X.reshape((1, 2, 2, 1))
```

---
## Step 5 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(UpSampling2D(input_shape=(2, 2, 1)))
```

---
## Step 6 — summarize the model

```python
model.summary()
```

---
## Step 7 — make a prediction with the model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X)
```

---
## Step 8 — reshape output to remove channel to make printing easier

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
yhat = yhat.reshape((4, 4))
```

---
## Step 9 — summarize output

```python
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: example of using the upsampling layer 是机器学习中的常用技术。  
  *example of using the upsampling layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Upsample Layer / 01 Upsample Layer
# Complete Code / 完整代码
# ===============================

# example of using the upsampling layer
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import UpSampling2D
# define input data
X = asarray([[1, 2],
			 [3, 4]])
# show input data for context
# 打印输出 / Print output
print(X)
# reshape input data into one sample a sample with a channel
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = X.reshape((1, 2, 2, 1))
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(UpSampling2D(input_shape=(2, 2, 1)))
# summarize the model
model.summary()
# make a prediction with the model
# 用模型做预测 / Make predictions with model
yhat = model.predict(X)
# reshape output to remove channel to make printing easier
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
yhat = yhat.reshape((4, 4))
# summarize output
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Upsample Generator

# 02 — Upsample Generator / 02 Upsample Generator

**Chapter 03 — File 2 of 4 / 第03章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of using upsampling in a simple generator model**.

本脚本演示 **example of using upsampling in a simple generator model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — example of using upsampling in a simple generator model

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Reshape
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import UpSampling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
```

---
## Step 2 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 3 — define input shape, output enough activations for for 128 5x5 image

```python
# 向模型添加一层 / Add a layer to the model
model.add(Dense(128 * 5 * 5, input_dim=100))
```

---
## Step 4 — reshape vector of activations into 128 feature maps with 5x5

```python
# 向模型添加一层 / Add a layer to the model
model.add(Reshape((5, 5, 128)))
```

---
## Step 5 — double input from 128 5x5 to 1 10x10 feature map

```python
# 向模型添加一层 / Add a layer to the model
model.add(UpSampling2D())
```

---
## Step 6 — fill in detail in the upsampled feature maps and output a single image

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (3,3), padding='same'))
```

---
## Step 7 — summarize model

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example of using upsampling in a simple generator model 是机器学习中的常用技术。  
  *example of using upsampling in a simple generator model is a common technique in machine learning.*

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
# Upsample Generator / 02 Upsample Generator
# Complete Code / 完整代码
# ===============================

# example of using upsampling in a simple generator model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Reshape
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import UpSampling2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# define input shape, output enough activations for for 128 5x5 image
# 向模型添加一层 / Add a layer to the model
model.add(Dense(128 * 5 * 5, input_dim=100))
# reshape vector of activations into 128 feature maps with 5x5
# 向模型添加一层 / Add a layer to the model
model.add(Reshape((5, 5, 128)))
# double input from 128 5x5 to 1 10x10 feature map
# 向模型添加一层 / Add a layer to the model
model.add(UpSampling2D())
# fill in detail in the upsampled feature maps and output a single image
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (3,3), padding='same'))
# summarize model
model.summary()
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Transpose Layer

# 03 — Transpose Layer / 03 Transpose Layer

**Chapter 03 — File 3 of 4 / 第03章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of using the transpose convolutional layer**.

本脚本演示 **example of using the transpose convolutional layer**。

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
## Step 1 — example of using the transpose convolutional layer

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
```

---
## Step 2 — define input data

```python
X = asarray([[1, 2],
			 [3, 4]])
```

---
## Step 3 — show input data for context

```python
# 打印输出 / Print output
print(X)
```

---
## Step 4 — reshape input data into one sample a sample with a channel

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = X.reshape((1, 2, 2, 1))
```

---
## Step 5 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(1, (1,1), strides=(2,2), input_shape=(2, 2, 1)))
```

---
## Step 6 — summarize the model

```python
model.summary()
```

---
## Step 7 — define weights that they do nothing

```python
weights = [asarray([[[[1]]]]), asarray([0])]
```

---
## Step 8 — store the weights in the model

```python
model.set_weights(weights)
```

---
## Step 9 — make a prediction with the model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X)
```

---
## Step 10 — reshape output to remove channel to make printing easier

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
yhat = yhat.reshape((4, 4))
```

---
## Step 11 — summarize output

```python
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: example of using the transpose convolutional layer 是机器学习中的常用技术。  
  *example of using the transpose convolutional layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Transpose Layer / 03 Transpose Layer
# Complete Code / 完整代码
# ===============================

# example of using the transpose convolutional layer
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
# define input data
X = asarray([[1, 2],
			 [3, 4]])
# show input data for context
# 打印输出 / Print output
print(X)
# reshape input data into one sample a sample with a channel
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = X.reshape((1, 2, 2, 1))
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(1, (1,1), strides=(2,2), input_shape=(2, 2, 1)))
# summarize the model
model.summary()
# define weights that they do nothing
weights = [asarray([[[[1]]]]), asarray([0])]
# store the weights in the model
model.set_weights(weights)
# make a prediction with the model
# 用模型做预测 / Make predictions with model
yhat = model.predict(X)
# reshape output to remove channel to make printing easier
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
yhat = yhat.reshape((4, 4))
# summarize output
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Transpose Generator

# 04 — Transpose Generator / 04 Transpose Generator

**Chapter 03 — File 4 of 4 / 第03章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of using transpose conv in a simple generator model**.

本脚本演示 **example of using transpose conv in a simple generator model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — example of using transpose conv in a simple generator model

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Reshape
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
```

---
## Step 2 — define model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
```

---
## Step 3 — define input shape, output enough activations for for 128 5x5 image

```python
# 向模型添加一层 / Add a layer to the model
model.add(Dense(128 * 5 * 5, input_dim=100))
```

---
## Step 4 — reshape vector of activations into 128 feature maps with 5x5

```python
# 向模型添加一层 / Add a layer to the model
model.add(Reshape((5, 5, 128)))
```

---
## Step 5 — double input from 128 5x5 to 1 10x10 feature map

```python
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding='same'))
```

---
## Step 6 — summarize model

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example of using transpose conv in a simple generator model 是机器学习中的常用技术。  
  *example of using transpose conv in a simple generator model is a common technique in machine learning.*

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
# Transpose Generator / 04 Transpose Generator
# Complete Code / 完整代码
# ===============================

# example of using transpose conv in a simple generator model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Reshape
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2DTranspose
# define model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# define input shape, output enough activations for for 128 5x5 image
# 向模型添加一层 / Add a layer to the model
model.add(Dense(128 * 5 * 5, input_dim=100))
# reshape vector of activations into 128 feature maps with 5x5
# 向模型添加一层 / Add a layer to the model
model.add(Reshape((5, 5, 128)))
# double input from 128 5x5 to 1 10x10 feature map
# 向模型添加一层 / Add a layer to the model
model.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding='same'))
# summarize model
model.summary()
```

---

### Chapter Summary / 章节总结

# Chapter 03 Summary / 第03章总结

## Theme / 主题: Chapter 03 / Chapter 03

This chapter contains **4 code files** demonstrating chapter 03.

本章包含 **4 个代码文件**，演示Chapter 03。

---
## Evolution / 演化路线

  1. `01_upsample_layer.ipynb` — Upsample Layer
  2. `02_upsample_generator.ipynb` — Upsample Generator
  3. `03_transpose_layer.ipynb` — Transpose Layer
  4. `04_transpose_generator.ipynb` — Transpose Generator

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 03) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 03）是机器学习流水线中的基础构建块。

---
