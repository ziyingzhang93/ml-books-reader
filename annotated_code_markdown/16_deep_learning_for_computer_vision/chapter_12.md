# CV深度学习
## Chapter 12

---

### 2D Vertical Line Filter Conv

# 01 — 2D Vertical Line Filter Conv / 01 2D Vertical Line Filter Conv

**Chapter 12 — File 1 of 8 / 第12章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of using a single convolutional layer**.

本脚本演示 **example of using a single convolutional layer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of using a single convolutional layer

```python
from numpy import asarray
from keras.models import Sequential
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
data = data.reshape(1, 8, 8, 1)
```

---
## Step 3 — create model

```python
model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
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
yhat = model.predict(data)
```

---
## Step 8 — enumerate rows

```python
for r in range(yhat.shape[1]):
```

---
## Step 9 — print each column in the row

```python
print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

---
## Learning Notes / 学习笔记

- **概念**: example of using a single convolutional layer 是机器学习中的常用技术。  
  *example of using a single convolutional layer is a common technique in machine learning.*

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

# example of using a single convolutional layer
from numpy import asarray
from keras.models import Sequential
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
data = data.reshape(1, 8, 8, 1)
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
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
yhat = model.predict(data)
# enumerate rows
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Stacked Conv Layers

# 02 — Stacked Conv Layers / 堆叠方法

**Chapter 12 — File 2 of 8 / 第12章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of stacked convolutional layers**.

本脚本演示 **example of stacked convolutional layers**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of stacked convolutional layers

```python
from keras.models import Sequential
from keras.layers import Conv2D
```

---
## Step 2 — create model

```python
model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
model.add(Conv2D(1, (3,3)))
```

---
## Step 3 — summarize model

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example of stacked convolutional layers 是机器学习中的常用技术。  
  *example of stacked convolutional layers is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Stacked Conv Layers / 堆叠方法
# Complete Code / 完整代码
# ===============================

# example of stacked convolutional layers
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
model.add(Conv2D(1, (3,3)))
# summarize model
model.summary()
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Conv With 5X5 Kernel

# 03 — Conv With 5X5 Kernel / 03 Conv With 5X5 Kernel

**Chapter 12 — File 3 of 8 / 第12章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of a convolutional layer**.

本脚本演示 **example of a convolutional layer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of a convolutional layer

```python
from keras.models import Sequential
from keras.layers import Conv2D
```

---
## Step 2 — create model

```python
model = Sequential()
model.add(Conv2D(1, (5,5), input_shape=(8, 8, 1)))
```

---
## Step 3 — summarize model

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example of a convolutional layer 是机器学习中的常用技术。  
  *example of a convolutional layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Conv With 5X5 Kernel / 03 Conv With 5X5 Kernel
# Complete Code / 完整代码
# ===============================

# example of a convolutional layer
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (5,5), input_shape=(8, 8, 1)))
# summarize model
model.summary()
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Conv With 1X1 Kernel

# 04 — Conv With 1X1 Kernel / 04 Conv With 1X1 Kernel

**Chapter 12 — File 4 of 8 / 第12章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of a convolutional layer**.

本脚本演示 **example of a convolutional layer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of a convolutional layer

```python
from keras.models import Sequential
from keras.layers import Conv2D
```

---
## Step 2 — create model

```python
model = Sequential()
model.add(Conv2D(1, (1,1), input_shape=(8, 8, 1)))
```

---
## Step 3 — summarize model

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example of a convolutional layer 是机器学习中的常用技术。  
  *example of a convolutional layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Conv With 1X1 Kernel / 04 Conv With 1X1 Kernel
# Complete Code / 完整代码
# ===============================

# example of a convolutional layer
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (1,1), input_shape=(8, 8, 1)))
# summarize model
model.summary()
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Conv With 8X8 Kernel

# 05 — Conv With 8X8 Kernel / 05 Conv With 8X8 Kernel

**Chapter 12 — File 5 of 8 / 第12章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of a convolutional layer**.

本脚本演示 **example of a convolutional layer**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of a convolutional layer

```python
from keras.models import Sequential
from keras.layers import Conv2D
```

---
## Step 2 — create model

```python
model = Sequential()
model.add(Conv2D(1, (8,8), input_shape=(8, 8, 1)))
```

---
## Step 3 — summarize model

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example of a convolutional layer 是机器学习中的常用技术。  
  *example of a convolutional layer is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Conv With 8X8 Kernel / 05 Conv With 8X8 Kernel
# Complete Code / 完整代码
# ===============================

# example of a convolutional layer
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (8,8), input_shape=(8, 8, 1)))
# summarize model
model.summary()
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Conv With Padding

# 06 — Conv With Padding / 06 Conv With Padding

**Chapter 12 — File 6 of 8 / 第12章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example a convolutional layer with padding**.

本脚本演示 **example a convolutional layer with padding**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example a convolutional layer with padding

```python
from keras.models import Sequential
from keras.layers import Conv2D
```

---
## Step 2 — create model

```python
model = Sequential()
model.add(Conv2D(1, (3,3), padding='same', input_shape=(8, 8, 1)))
```

---
## Step 3 — summarize model

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example a convolutional layer with padding 是机器学习中的常用技术。  
  *example a convolutional layer with padding is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Conv With Padding / 06 Conv With Padding
# Complete Code / 完整代码
# ===============================

# example a convolutional layer with padding
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), padding='same', input_shape=(8, 8, 1)))
# summarize model
model.summary()
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Stacked Conv With Padding

# 07 — Stacked Conv With Padding / 堆叠方法

**Chapter 12 — File 7 of 8 / 第12章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example a deep cnn with padding**.

本脚本演示 **example a deep cnn with padding**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example a deep cnn with padding

```python
from keras.models import Sequential
from keras.layers import Conv2D
```

---
## Step 2 — create model

```python
model = Sequential()
model.add(Conv2D(1, (3,3), padding='same', input_shape=(8, 8, 1)))
model.add(Conv2D(1, (3,3), padding='same'))
model.add(Conv2D(1, (3,3), padding='same'))
```

---
## Step 3 — summarize model

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example a deep cnn with padding 是机器学习中的常用技术。  
  *example a deep cnn with padding is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Conv2D` | 二维卷积层（Keras） | 2D convolution layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Stacked Conv With Padding / 堆叠方法
# Complete Code / 完整代码
# ===============================

# example a deep cnn with padding
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), padding='same', input_shape=(8, 8, 1)))
model.add(Conv2D(1, (3,3), padding='same'))
model.add(Conv2D(1, (3,3), padding='same'))
# summarize model
model.summary()
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Conv With Larger Stride

# 08 — Conv With Larger Stride / 08 Conv With Larger Stride

**Chapter 12 — File 8 of 8 / 第12章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of vertical line filter with a stride of 2**.

本脚本演示 **example of vertical line filter with a stride of 2**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of vertical line filter with a stride of 2

```python
from numpy import asarray
from keras.models import Sequential
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
data = data.reshape(1, 8, 8, 1)
```

---
## Step 3 — create model

```python
model = Sequential()
model.add(Conv2D(1, (3,3), strides=(2, 2), input_shape=(8, 8, 1)))
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
yhat = model.predict(data)
```

---
## Step 8 — enumerate rows

```python
for r in range(yhat.shape[1]):
```

---
## Step 9 — print each column in the row

```python
print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

---
## Learning Notes / 学习笔记

- **概念**: example of vertical line filter with a stride of 2 是机器学习中的常用技术。  
  *example of vertical line filter with a stride of 2 is a common technique in machine learning.*

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
# Conv With Larger Stride / 08 Conv With Larger Stride
# Complete Code / 完整代码
# ===============================

# example of vertical line filter with a stride of 2
from numpy import asarray
from keras.models import Sequential
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
data = data.reshape(1, 8, 8, 1)
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), strides=(2, 2), input_shape=(8, 8, 1)))
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
yhat = model.predict(data)
# enumerate rows
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

---

### Chapter Summary

# Chapter 12 Summary / 第12章总结

## Theme / 主题: Chapter 12 / Chapter 12

This chapter contains **8 code files** demonstrating chapter 12.

本章包含 **8 个代码文件**，演示Chapter 12。

---
## Evolution / 演化路线

  1. `01_2d_vertical_line_filter_conv.ipynb` — 2D Vertical Line Filter Conv
  2. `02_stacked_conv_layers.ipynb` — Stacked Conv Layers
  3. `03_conv_with_5x5_kernel.ipynb` — Conv With 5X5 Kernel
  4. `04_conv_with_1x1_kernel.ipynb` — Conv With 1X1 Kernel
  5. `05_conv_with_8x8_kernel.ipynb` — Conv With 8X8 Kernel
  6. `06_conv_with_padding.ipynb` — Conv With Padding
  7. `07_stacked_conv_with_padding.ipynb` — Stacked Conv With Padding
  8. `08_conv_with_larger_stride.ipynb` — Conv With Larger Stride

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 12) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 12）是机器学习流水线中的基础构建块。

---
