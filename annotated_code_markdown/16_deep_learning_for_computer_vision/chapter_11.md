# 计算机视觉深度学习 / Deep Learning for Computer Vision
## Chapter 11

---

### 1D Filter Model

# 01 — 1D Filter Model / 01 1D Filter Model

**Chapter 11 — File 1 of 4 / 第11章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of calculation 1d convolutions**.

本脚本演示 **example of calculation 1d convolutions**。

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
## Step 1 — example of calculation 1d convolutions

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv1D
```

---
## Step 2 — define input data

```python
data = asarray([0, 0, 0, 1, 1, 0, 0, 0])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
data = data.reshape(1, 8, 1)
```

---
## Step 3 — create model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(1, 3, input_shape=(8, 1)))
```

---
## Step 4 — define a vertical line detector

```python
weights = [asarray([[[0]],[[1]],[[0]]]), asarray([0.0])]
```

---
## Step 5 — store the weights in the model

```python
model.set_weights(weights)
```

---
## Step 6 — confirm they were stored

```python
# 打印输出 / Print output
print(model.get_weights())
```

---
## Step 7 — apply filter to input data

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(data)
# 打印输出 / Print output
print(yhat)
```

---
## Learning Notes / 学习笔记

- **概念**: example of calculation 1d convolutions 是机器学习中的常用技术。  
  *example of calculation 1d convolutions is a common technique in machine learning.*

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
# 1D Filter Model / 01 1D Filter Model
# Complete Code / 完整代码
# ===============================

# example of calculation 1d convolutions
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Conv1D
# define input data
data = asarray([0, 0, 0, 1, 1, 0, 0, 0])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
data = data.reshape(1, 8, 1)
# create model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Conv1D(1, 3, input_shape=(8, 1)))
# define a vertical line detector
weights = [asarray([[[0]],[[1]],[[0]]]), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# confirm they were stored
# 打印输出 / Print output
print(model.get_weights())
# apply filter to input data
# 用模型做预测 / Make predictions with model
yhat = model.predict(data)
# 打印输出 / Print output
print(yhat)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Maually Apply 1D Filter

# 02 — Maually Apply 1D Filter / 02 Maually Apply 1D Filter

**Chapter 11 — File 2 of 4 / 第11章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **manually apply a 1d filter**.

本脚本演示 **manually apply a 1d filter**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — manually apply a 1d filter

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 打印输出 / Print output
print(asarray([0, 1, 0]).dot(asarray([0, 0, 0])))
```

---
## Learning Notes / 学习笔记

- **概念**: manually apply a 1d filter 是机器学习中的常用技术。  
  *manually apply a 1d filter is a common technique in machine learning.*

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
# Maually Apply 1D Filter / 02 Maually Apply 1D Filter
# Complete Code / 完整代码
# ===============================

# manually apply a 1d filter
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 打印输出 / Print output
print(asarray([0, 1, 0]).dot(asarray([0, 0, 0])))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### 2D Filter Model

# 03 — 2D Filter Model / 03 2D Filter Model

**Chapter 11 — File 3 of 4 / 第11章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of calculation 2d convolutions**.

本脚本演示 **example of calculation 2d convolutions**。

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
## Step 1 — example of calculation 2d convolutions

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
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
```

---
## Step 4 — define a vertical line detector

```python
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
```

---
## Step 5 — store the weights in the model

```python
model.set_weights(weights)
```

---
## Step 6 — confirm they were stored

```python
# 打印输出 / Print output
print(model.get_weights())
```

---
## Step 7 — apply filter to input data

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(data)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for r in range(yhat.shape[1]):
```

---
## Step 8 — print each column in the row

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

---
## Learning Notes / 学习笔记

- **概念**: example of calculation 2d convolutions 是机器学习中的常用技术。  
  *example of calculation 2d convolutions is a common technique in machine learning.*

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
# 2D Filter Model / 03 2D Filter Model
# Complete Code / 完整代码
# ===============================

# example of calculation 2d convolutions
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
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# confirm they were stored
# 打印输出 / Print output
print(model.get_weights())
# apply filter to input data
# 用模型做预测 / Make predictions with model
yhat = model.predict(data)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for r in range(yhat.shape[1]):
	# print each column in the row
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Manually Apply 2D Filter

# 04 — Manually Apply 2D Filter / 04 Manually Apply 2D Filter

**Chapter 11 — File 4 of 4 / 第11章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of manually applying a 2d filter.**.

本脚本演示 **example of manually applying a 2d filter.**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of manually applying a 2d filter.

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import tensordot
m1 = asarray([[0, 1, 0],
			  [0, 1, 0],
			  [0, 1, 0]])
m2 = asarray([[0, 0, 0],
			  [0, 0, 0],
			  [0, 0, 0]])
# 打印输出 / Print output
print(tensordot(m1, m2))
```

---
## Learning Notes / 学习笔记

- **概念**: example of manually applying a 2d filter. 是机器学习中的常用技术。  
  *example of manually applying a 2d filter. is a common technique in machine learning.*

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
# Manually Apply 2D Filter / 04 Manually Apply 2D Filter
# Complete Code / 完整代码
# ===============================

# example of manually applying a 2d filter.
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import tensordot
m1 = asarray([[0, 1, 0],
			  [0, 1, 0],
			  [0, 1, 0]])
m2 = asarray([[0, 0, 0],
			  [0, 0, 0],
			  [0, 0, 0]])
# 打印输出 / Print output
print(tensordot(m1, m2))
```

---

### Chapter Summary / 章节总结

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **4 code files** demonstrating chapter 11.

本章包含 **4 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `01_1d_filter_model.ipynb` — 1D Filter Model
  2. `02_maually_apply_1d_filter.ipynb` — Maually Apply 1D Filter
  3. `03_2d_filter_model.ipynb` — 2D Filter Model
  4. `04_manually_apply_2d_filter.ipynb` — Manually Apply 2D Filter

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---
