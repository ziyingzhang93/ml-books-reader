# 计算机视觉深度学习 / Deep Learning for Computer Vision
## Chapter 13

---

### 2D Vertical Line Filter Conv

# 01 — 2D Vertical Line Filter Conv / 01 2D Vertical Line Filter Conv

**Chapter 13 — File 1 of 4 / 第13章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of vertical line detection with a convolutional layer**.

本脚本演示 **example of vertical line detection with a convolutional layer**。

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
## Step 1 — example of vertical line detection with a convolutional layer

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
```

---
## Step 2 — define input data

```python
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
data = data.reshape(1, 8, 8, 1)
```

---
## Step 3 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
```

---
## Step 4 — summarize model

```python
model.summary()
```

---
## Step 5 — define a vertical line detector

```python
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
```

---
## Step 6 — store the weights in the model

```python
model.set_weights(weights)
```

---
## Step 7 — apply filter to input data

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(data)
```

---
## Step 8 — enumerate rows

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for r in range(yhat.shape[1]):
```

---
## Step 9 — print each column in the row

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

---
## Learning Notes / 学习笔记

- **概念**: example of vertical line detection with a convolutional layer 是机器学习中的常用技术。  
  *example of vertical line detection with a convolutional layer is a common technique in machine learning.*

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
# 2D Vertical Line Filter Conv / 01 2D Vertical Line Filter Conv
# Complete Code / 完整代码
# ===============================

# example of vertical line detection with a convolutional layer
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
data = data.reshape(1, 8, 8, 1)
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
# summarize model
model.summary()
# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# apply filter to input data
# 用模型做预测 / Make predictions with model
yhat = model.predict(data)
# enumerate rows
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for r in range(yhat.shape[1]):
	# print each column in the row
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Avg Pooling

# 02 — Avg Pooling / 02 Avg Pooling

**Chapter 13 — File 2 of 4 / 第13章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of average pooling**.

本脚本演示 **example of average pooling**。

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
## Step 1 — example of average pooling

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import AveragePooling2D
```

---
## Step 2 — define input data

```python
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
data = data.reshape(1, 8, 8, 1)
```

---
## Step 3 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
# 向模型添加一层 / Add a layer to the model
model.add(AveragePooling2D())
```

---
## Step 4 — summarize model

```python
model.summary()
```

---
## Step 5 — define a vertical line detector

```python
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
```

---
## Step 6 — store the weights in the model

```python
model.set_weights(weights)
```

---
## Step 7 — apply filter to input data

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(data)
```

---
## Step 8 — enumerate rows

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for r in range(yhat.shape[1]):
```

---
## Step 9 — print each column in the row

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

---
## Learning Notes / 学习笔记

- **概念**: example of average pooling 是机器学习中的常用技术。  
  *example of average pooling is a common technique in machine learning.*

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
# Avg Pooling / 02 Avg Pooling
# Complete Code / 完整代码
# ===============================

# example of average pooling
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import AveragePooling2D
# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
data = data.reshape(1, 8, 8, 1)
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
# 向模型添加一层 / Add a layer to the model
model.add(AveragePooling2D())
# summarize model
model.summary()
# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# apply filter to input data
# 用模型做预测 / Make predictions with model
yhat = model.predict(data)
# enumerate rows
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for r in range(yhat.shape[1]):
	# print each column in the row
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Max Pooling

# 03 — Max Pooling / 03 Max Pooling

**Chapter 13 — File 3 of 4 / 第13章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of max pooling**.

本脚本演示 **example of max pooling**。

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
## Step 1 — example of max pooling

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
```

---
## Step 2 — define input data

```python
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
data = data.reshape(1, 8, 8, 1)
```

---
## Step 3 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling2D())
```

---
## Step 4 — summarize model

```python
model.summary()
```

---
## Step 5 — define a vertical line detector

```python
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
```

---
## Step 6 — store the weights in the model

```python
model.set_weights(weights)
```

---
## Step 7 — apply filter to input data

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(data)
```

---
## Step 8 — enumerate rows

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for r in range(yhat.shape[1]):
```

---
## Step 9 — print each column in the row

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

---
## Learning Notes / 学习笔记

- **概念**: example of max pooling 是机器学习中的常用技术。  
  *example of max pooling is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Max Pooling / 03 Max Pooling
# Complete Code / 完整代码
# ===============================

# example of max pooling
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import MaxPooling2D
# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
data = data.reshape(1, 8, 8, 1)
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
# 向模型添加一层 / Add a layer to the model
model.add(MaxPooling2D())
# summarize model
model.summary()
# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# apply filter to input data
# 用模型做预测 / Make predictions with model
yhat = model.predict(data)
# enumerate rows
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for r in range(yhat.shape[1]):
	# print each column in the row
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Global Max Pooling

# 04 — Global Max Pooling / 04 Global Max Pooling

**Chapter 13 — File 4 of 4 / 第13章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of using global max pooling**.

本脚本演示 **example of using global max pooling**。

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
## Step 1 — example of using global max pooling

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import GlobalMaxPooling2D
```

---
## Step 2 — define input data

```python
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
data = data.reshape(1, 8, 8, 1)
```

---
## Step 3 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
# 向模型添加一层 / Add a layer to the model
model.add(GlobalMaxPooling2D())
```

---
## Step 4 — summarize model

```python
model.summary()
```

---
## Step 5 — define a vertical line detector

```python
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
```

---
## Step 6 — store the weights in the model

```python
model.set_weights(weights)
```

---
## Step 7 — apply filter to input data

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(data)
```

---
## Step 8 — show result

```python
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: example of using global max pooling 是机器学习中的常用技术。  
  *example of using global max pooling is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `MaxPooling2D` | 最大池化，缩小特征图 | Max pooling: downsample feature maps |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Global Max Pooling / 04 Global Max Pooling
# Complete Code / 完整代码
# ===============================

# example of using global max pooling
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import GlobalMaxPooling2D
# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
data = data.reshape(1, 8, 8, 1)
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
# 向模型添加一层 / Add a layer to the model
model.add(GlobalMaxPooling2D())
# summarize model
model.summary()
# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# apply filter to input data
# 用模型做预测 / Make predictions with model
yhat = model.predict(data)
# show result
# 打印输出 / Print output
print(yhat)
```

---

### Chapter Summary / 章节总结

# Chapter 13 Summary / 第13章总结

## Theme / 主题: Chapter 13 / Chapter 13

This chapter contains **4 code files** demonstrating chapter 13.

本章包含 **4 个代码文件**，演示Chapter 13。

---
## Evolution / 演化路线

  1. `01_2d_vertical_line_filter_conv.ipynb` — 2D Vertical Line Filter Conv
  2. `02_avg_pooling.ipynb` — Avg Pooling
  3. `03_max_pooling.ipynb` — Max Pooling
  4. `04_global_max_pooling.ipynb` — Global Max Pooling

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 13) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 13）是机器学习流水线中的基础构建块。

---
