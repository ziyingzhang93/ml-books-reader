# 计算机视觉深度学习 / Deep Learning for Computer Vision
## Chapter 16

---

### Normal Filter

# 01 — Normal Filter / 01 Normal Filter

**Chapter 16 — File 1 of 4 / 第16章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of simple cnn model**.

本脚本演示 **example of simple cnn model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of simple cnn model

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
```

---
## Step 2 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
```

---
## Step 3 — summarize model

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example of simple cnn model 是机器学习中的常用技术。  
  *example of simple cnn model is a common technique in machine learning.*

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
# Normal Filter / 01 Normal Filter
# Complete Code / 完整代码
# ===============================

# example of simple cnn model
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
# summarize model
model.summary()
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Project Feature Maps

# 02 — Project Feature Maps / 特征工程

**Chapter 16 — File 2 of 4 / 第16章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of a 1x1 filter for projection**.

本脚本演示 **example of a 1x1 filter for projection**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of a 1x1 filter for projection

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
```

---
## Step 2 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(512, (1,1), activation='relu'))
```

---
## Step 3 — summarize model

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example of a 1x1 filter for projection 是机器学习中的常用技术。  
  *example of a 1x1 filter for projection is a common technique in machine learning.*

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
# Project Feature Maps / 特征工程
# Complete Code / 完整代码
# ===============================

# example of a 1x1 filter for projection
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(512, (1,1), activation='relu'))
# summarize model
model.summary()
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Decrease Feature Maps

# 03 — Decrease Feature Maps / 特征工程

**Chapter 16 — File 3 of 4 / 第16章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of a 1x1 filter for dimensionality reduction**.

本脚本演示 **example of a 1x1 filter for dimensionality reduction**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of a 1x1 filter for dimensionality reduction

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
```

---
## Step 2 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(64, (1,1), activation='relu'))
```

---
## Step 3 — summarize model

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example of a 1x1 filter for dimensionality reduction 是机器学习中的常用技术。  
  *example of a 1x1 filter for dimensionality reduction is a common technique in machine learning.*

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
# Decrease Feature Maps / 特征工程
# Complete Code / 完整代码
# ===============================

# example of a 1x1 filter for dimensionality reduction
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(64, (1,1), activation='relu'))
# summarize model
model.summary()
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Increasing Feature Maps

# 04 — Increasing Feature Maps / 特征工程

**Chapter 16 — File 4 of 4 / 第16章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of a 1x1 filter to increase dimensionality**.

本脚本演示 **example of a 1x1 filter to increase dimensionality**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of a 1x1 filter to increase dimensionality

```python
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
```

---
## Step 2 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1024, (1,1), activation='relu'))
```

---
## Step 3 — summarize model

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example of a 1x1 filter to increase dimensionality 是机器学习中的常用技术。  
  *example of a 1x1 filter to increase dimensionality is a common technique in machine learning.*

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
# Increasing Feature Maps / 特征工程
# Complete Code / 完整代码
# ===============================

# example of a 1x1 filter to increase dimensionality
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv2D
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
# 向模型添加一层 / Add a layer to the model
model.add(Conv2D(1024, (1,1), activation='relu'))
# summarize model
model.summary()
```

---

### Chapter Summary / 章节总结

# Chapter 16 Summary / 第16章总结

## Theme / 主题: Chapter 16 / Chapter 16

This chapter contains **4 code files** demonstrating chapter 16.

本章包含 **4 个代码文件**，演示Chapter 16。

---
## Evolution / 演化路线

  1. `01_normal_filter.ipynb` — Normal Filter
  2. `02_project_feature_maps.ipynb` — Project Feature Maps
  3. `03_decrease_feature_maps.ipynb` — Decrease Feature Maps
  4. `04_increasing_feature_maps.ipynb` — Increasing Feature Maps

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 16) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 16）是机器学习流水线中的基础构建块。

---
