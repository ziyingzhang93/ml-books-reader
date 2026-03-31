# Python 深度学习 / Deep Learning with Python
## Chapter 26

---

### Plot Mnist

# 05 — Plot Mnist / 05 Plot Mnist

**Chapter 26 — File 1 of 8 / 第26章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Plot images**.

本脚本演示 **Plot images**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Plot images

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
```

---
## Step 2 — load dbata

```python
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

---
## Step 3 — create a grid of 3x3 images

```python
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4,4))
# 生成整数序列 / Generate integer sequence
for i in range(3):
    # 生成整数序列 / Generate integer sequence
    for j in range(3):
        ax[i][j].imshow(X_train[i*3+j], cmap=plt.get_cmap("gray"))
```

---
## Step 4 — show the plot

```python
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Plot images 是机器学习中的常用技术。  
  *Plot images is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Mnist / 05 Plot Mnist
# Complete Code / 完整代码
# ===============================

# Plot images
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# load dbata
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# create a grid of 3x3 images
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4,4))
# 生成整数序列 / Generate integer sequence
for i in range(3):
    # 生成整数序列 / Generate integer sequence
    for j in range(3):
        ax[i][j].imshow(X_train[i*3+j], cmap=plt.get_cmap("gray"))
# show the plot
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Standardize



---

### Featurewise



---

### Zca



---

### Rotations



---

### Shifts

# 10 — Shifts / 10 Shifts

**Chapter 26 — File 6 of 8 / 第26章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Random Shifts**.

本脚本演示 **Random Shifts**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Random Shifts

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
```

---
## Step 2 — load data

```python
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

---
## Step 3 — reshape to be [samples][width][height][channels]

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
```

---
## Step 4 — convert from int to float

```python
# 转换数据类型 / Convert data type
X_train = X_train.astype('float32')
# 转换数据类型 / Convert data type
X_test = X_test.astype('float32')
```

---
## Step 5 — define data preparation

```python
shift = 0.2
datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
```

---
## Step 6 — configure batch size and retrieve one batch of images

```python
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, shuffle=False):
```

---
## Step 7 — create a grid of 3x3 images

```python
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4,4))
    # 生成整数序列 / Generate integer sequence
    for i in range(3):
        # 生成整数序列 / Generate integer sequence
        for j in range(3):
            # 改变数组形状（不改变数据） / Reshape array (data unchanged)
            ax[i][j].imshow(X_batch[i*3+j].reshape(28,28), cmap=plt.get_cmap("gray"))
```

---
## Step 8 — show the plot

```python
# 显示图表 / Display the plot
plt.show()
    break
```

---
## Learning Notes / 学习笔记

- **概念**: Random Shifts 是机器学习中的常用技术。  
  *Random Shifts is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `matplotlib` | 绑图库 | Plotting library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Shifts / 10 Shifts
# Complete Code / 完整代码
# ===============================

# Random Shifts
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# load data
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][width][height][channels]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
# convert from int to float
# 转换数据类型 / Convert data type
X_train = X_train.astype('float32')
# 转换数据类型 / Convert data type
X_test = X_test.astype('float32')
# define data preparation
shift = 0.2
datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, shuffle=False):
    # create a grid of 3x3 images
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4,4))
    # 生成整数序列 / Generate integer sequence
    for i in range(3):
        # 生成整数序列 / Generate integer sequence
        for j in range(3):
            # 改变数组形状（不改变数据） / Reshape array (data unchanged)
            ax[i][j].imshow(X_batch[i*3+j].reshape(28,28), cmap=plt.get_cmap("gray"))
    # show the plot
    # 显示图表 / Display the plot
    plt.show()
    break
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Flips

# 11 — Flips / 11 Flips

**Chapter 26 — File 7 of 8 / 第26章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Random Flips**.

本脚本演示 **Random Flips**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Random Flips

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
```

---
## Step 2 — load data

```python
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

---
## Step 3 — reshape to be [samples][width][height][channels]

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
```

---
## Step 4 — convert from int to float

```python
# 转换数据类型 / Convert data type
X_train = X_train.astype('float32')
# 转换数据类型 / Convert data type
X_test = X_test.astype('float32')
```

---
## Step 5 — define data preparation

```python
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
```

---
## Step 6 — configure batch size and retrieve one batch of images

```python
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, shuffle=False):
```

---
## Step 7 — create a grid of 3x3 images

```python
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4,4))
    # 生成整数序列 / Generate integer sequence
    for i in range(3):
        # 生成整数序列 / Generate integer sequence
        for j in range(3):
            # 改变数组形状（不改变数据） / Reshape array (data unchanged)
            ax[i][j].imshow(X_batch[i*3+j].reshape(28,28), cmap=plt.get_cmap("gray"))
```

---
## Step 8 — show the plot

```python
# 显示图表 / Display the plot
plt.show()
    break
```

---
## Learning Notes / 学习笔记

- **概念**: Random Flips 是机器学习中的常用技术。  
  *Random Flips is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `matplotlib` | 绑图库 | Plotting library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Flips / 11 Flips
# Complete Code / 完整代码
# ===============================

# Random Flips
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.datasets import mnist
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# load data
# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][width][height][channels]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
# convert from int to float
# 转换数据类型 / Convert data type
X_train = X_train.astype('float32')
# 转换数据类型 / Convert data type
X_test = X_test.astype('float32')
# define data preparation
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, shuffle=False):
    # create a grid of 3x3 images
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4,4))
    # 生成整数序列 / Generate integer sequence
    for i in range(3):
        # 生成整数序列 / Generate integer sequence
        for j in range(3):
            # 改变数组形状（不改变数据） / Reshape array (data unchanged)
            ax[i][j].imshow(X_batch[i*3+j].reshape(28,28), cmap=plt.get_cmap("gray"))
    # show the plot
    # 显示图表 / Display the plot
    plt.show()
    break
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Save



---

### Chapter Summary / 章节总结

# Chapter 26 Summary / 第26章总结

## Theme / 主题: Chapter 26 / Chapter 26

This chapter contains **8 code files** demonstrating chapter 26.

本章包含 **8 个代码文件**，演示Chapter 26。

---
## Evolution / 演化路线

  1. `05_plot_mnist.ipynb` — Plot Mnist
  2. `06_standardize.ipynb` — Standardize
  3. `07_featurewise.ipynb` — Featurewise
  4. `08_zca.ipynb` — Zca
  5. `09_rotations.ipynb` — Rotations
  6. `10_shifts.ipynb` — Shifts
  7. `11_flips.ipynb` — Flips
  8. `12_save.ipynb` — Save

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 26) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 26）是机器学习流水线中的基础构建块。

---
