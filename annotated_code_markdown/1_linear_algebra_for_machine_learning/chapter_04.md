# 线性代数与机器学习 / Linear Algebra for Machine Learning
## Chapter 04

---

### Create Array

## File 1 of 6 / 第1章 — 第1个文件（共6个）

### Summary / 总结
Learn how to create NumPy arrays from Python lists. Arrays are the fundamental data structure in NumPy and essential for linear algebra operations in machine learning.

学习如何从Python列表创建NumPy数组。数组是NumPy中的基本数据结构，是机器学习中线性代数运算的基础。

### Core Concept / 核心概念
**numpy.array(object)**: Converts a Python list into a NumPy array with attributes like `.shape` (dimensions) and `.dtype` (data type).

**numpy.array(object)**: 将Python列表转换为NumPy数组，具有`.shape`（维度）和`.dtype`（数据类型）等属性。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy and Create List / 导入NumPy并创建列表

First, we import the `array` function from NumPy and create a Python list of floating-point numbers.

首先，我们从NumPy导入`array`函数并创建一个浮点数的Python列表。

```python
# 导入NumPy数组函数 / Import NumPy array function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# 创建Python列表 / Create Python list
l = [1.0, 2.0, 3.0]
# 打印输出 / Print output
print("Original Python list:")
# 打印输出 / Print output
print(l)
```

## Step 2 — Convert List to NumPy Array / 将列表转换为NumPy数组

The `array()` function converts the Python list into a NumPy array, which provides efficient numerical operations.

`array()`函数将Python列表转换为NumPy数组，提供高效的数值运算。

```python
# 使用array()函数转换列表 / Convert list using array() function
a = array(l)
# 打印输出 / Print output
print("\nNumPy array:")
# 打印输出 / Print output
print(a)
```

## Step 3 — Inspect Array Shape / 检查数组形状

The `.shape` attribute returns a tuple indicating the dimensions of the array. This 1D array has shape (3,) meaning 3 elements.

`.shape`属性返回一个元组，表示数组的维度。这个一维数组的形状是(3,)，表示有3个元素。

```python
# 检查数组的形状 / Check array shape (dimensions)
# 打印输出 / Print output
print("\nArray shape:")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(a.shape)
```

## Step 4 — Inspect Data Type / 检查数据类型

The `.dtype` attribute shows the data type of array elements. Since we used floats (1.0, 2.0, 3.0), NumPy creates a float64 array.

`.dtype`属性显示数组元素的数据类型。由于我们使用浮点数(1.0, 2.0, 3.0)，NumPy创建一个float64数组。

```python
# 检查数组的数据类型 / Check array data type
# 打印输出 / Print output
print("\nArray data type:")
# 打印输出 / Print output
print(a.dtype)
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: A NumPy array is a homogeneous, n-dimensional container for numerical data. Unlike Python lists, all elements must be the same type, enabling vectorized operations.

NumPy数组是同类型的n维数值数据容器。与Python列表不同，所有元素必须是相同类型，这使得向量化运算成为可能。

• **ML Application / ML应用**: In machine learning, datasets are represented as 2D arrays (rows=samples, columns=features). Creating arrays efficiently is the first step in any ML pipeline.

在机器学习中，数据集表示为二维数组（行=样本，列=特征）。高效创建数组是任何ML流程的第一步。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 02_create_empty.ipynb** — Learn how to create arrays with uninitialized values using `empty()`.

## Complete Code / 完整代码一览

```python
# --- Array Creation Basics / 数组创建基础 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Create Python list / 创建Python列表
l = [1.0, 2.0, 3.0]
# 打印输出 / Print output
print("Original Python list:")
# 打印输出 / Print output
print(l)

# Convert to NumPy array / 转换为NumPy数组
a = array(l)
# 打印输出 / Print output
print("\nNumPy array:")
# 打印输出 / Print output
print(a)

# Check shape and dtype / 检查形状和数据类型
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("\nShape:", a.shape)
# 打印输出 / Print output
print("Data type:", a.dtype)
```

---

### Create Empty

## File 2 of 6 / 第1章 — 第2个文件（共6个）

### Summary / 总结
Learn how to create arrays with uninitialized values using `empty()`. This is useful when you want to allocate memory before filling it with values.

学习如何使用`empty()`创建未初始化的数组。当你想在填充值之前分配内存时很有用。

### Core Concept / 核心概念
**numpy.empty(shape)**: Creates an array of given shape with arbitrary (uninitialized) values. Faster than `zeros()` because it doesn't set values.

**numpy.empty(shape)**: 创建给定形状的数组，其值是任意的（未初始化的）。比`zeros()`更快，因为它不设置值。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import and Create Empty Array / 导入并创建空数组

We import `empty` from NumPy and create a 3x3 array. The values are garbage/arbitrary since the memory hasn't been initialized.

我们从NumPy导入`empty`并创建一个3x3数组。这些值是垃圾值/任意值，因为内存尚未初始化。

```python
# 导入empty函数 / Import empty function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import empty

# 创建3x3的未初始化数组 / Create 3x3 array with uninitialized values
a = empty([3, 3])
# 打印输出 / Print output
print("Empty array (arbitrary values):")
# 打印输出 / Print output
print(a)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("\nShape:", a.shape)
# 打印输出 / Print output
print("Data type:", a.dtype)
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Memory allocation in NumPy is separated from initialization. `empty()` allocates memory quickly without setting values, useful when you'll immediately overwrite the data.

NumPy中的内存分配与初始化分离。`empty()`快速分配内存而不设置值，当你会立即覆盖数据时很有用。

• **ML Application / ML应用**: In performance-critical operations like matrix computations, `empty()` can be faster than `zeros()`. However, always ensure values are set before using them.

在矩阵计算等性能关键的操作中，`empty()`比`zeros()`更快。但是，始终确保在使用值之前设置它们。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 03_create_zeros.ipynb** — Learn how to create arrays initialized to zero using `zeros()`.

## Complete Code / 完整代码一览

```python
# --- Empty Array Creation / 空数组创建 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import empty

# Create 3x3 empty array with uninitialized values / 创建3x3未初始化的数组
a = empty([3, 3])
# 打印输出 / Print output
print("Empty array (arbitrary values):")
# 打印输出 / Print output
print(a)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Shape:", a.shape)
# 打印输出 / Print output
print("Data type:", a.dtype)
```

---

### Create Zeros



---

### Create Ones

## File 4 of 6 / 第1章 — 第4个文件（共6个）

### Summary / 总结
Learn how to create arrays filled with ones using `ones()`. This complements `zeros()` for array initialization needs.

学习如何使用`ones()`创建填充为1的数组。这补充了`zeros()`以满足数组初始化的需求。

### Core Concept / 核心概念
**numpy.ones(shape)**: Creates an array of given shape filled with ones. Useful for creating identity-like structures and constant matrices.

**numpy.ones(shape)**: 创建给定形状的数组，填充为1。对创建单位矩阵结构和常数矩阵很有用。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import and Create Ones Array / 导入并创建单位数组

We import `ones` from NumPy and create a 1D array of length 5 filled with ones. This can be used for broadcasting or creating constant vectors.

我们从NumPy导入`ones`并创建一个长度为5、填充为1的一维数组。这可用于广播或创建常数向量。

```python
# 导入ones函数 / Import ones function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones

# 创建长度为5的全1数组 / Create 1D array of length 5 filled with ones
a = ones([5])
# 打印输出 / Print output
print("Array of ones:")
# 打印输出 / Print output
print(a)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("\nShape:", a.shape)
# 打印输出 / Print output
print("Data type:", a.dtype)
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: In linear algebra, ones vectors and ones matrices are used for summing, averaging, and creating uniform distributions. The all-ones vector is denoted as **1** in vector notation.

在线性代数中，单位向量和单位矩阵用于求和、平均化和创建均匀分布。全1向量在向量符号中表示为**1**。

• **ML Application / ML applicaton**: In ML, ones arrays are used for creating bias terms, implementing softmax normalization, feature augmentation (adding a constant feature column), and computing sums across dimensions.

在ML中，单位数组用于创建偏差项、实现softmax归一化、特征增强（添加常数特征列）和跨维度的求和。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 05_combine_vstack.ipynb** — Learn how to combine arrays vertically using `vstack()`.

## Complete Code / 完整代码一览

```python
# --- Ones Array Creation / 单位数组创建 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import ones

# Create 1D array of length 5 filled with ones / 创建长度为5的全1一维数组
a = ones([5])
# 打印输出 / Print output
print("Array of ones:")
# 打印输出 / Print output
print(a)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Shape:", a.shape)
# 打印输出 / Print output
print("Data type:", a.dtype)
```

---

### Combine Vstack



---

### Combine Hstack



---

### Chapter Summary / 章节总结

# Chapter 04 Summary / 第04章总结：NumPy Array Creation

## Theme / 主题

NumPy arrays are the foundation of numerical computing. This chapter teaches you how to create arrays from different sources—from Python lists, from scratch with predefined values, and how to combine multiple arrays into larger structures.

NumPy数组是数值计算的基础。本章教您如何从不同的源创建数组—从Python列表、从零开始使用预定义的值，以及如何将多个数组合并为更大的结构。

## Evolution / 演化路线

```
01_create_array.ipynb
    └─ Convert Python list → NumPy array (最基础的创建方式)
    
02_create_empty.ipynb
    └─ Create uninitialized array (快速分配内存)
    
03_create_zeros.ipynb
    └─ Fill with zeros (初始化为零)
    
04_create_ones.ipynb
    └─ Fill with ones (初始化为一)
    
05_combine_vstack.ipynb
    └─ Stack arrays vertically / 竖直堆积
    
06_combine_hstack.ipynb
    └─ Stack arrays horizontally / 水平堆积
```

## Progression Logic / 进度逻辑

The chapter progresses from **source-based creation** to **value-based creation** to **combination**:

1. **From existing data**: `np.array()` converts your Python lists
2. **From scratch**: `empty()`, `zeros()`, `ones()` allocate memory with different initial values
3. **Combining**: `vstack()` and `hstack()` merge multiple arrays into larger ones

This progression teaches you how to go from "I have some data" → "I need raw memory" → "I need to combine datasets".

进度从**基于源的创建**到**基于值的创建**再到**合并**：

1. **从现有数据**：`np.array()`转换您的Python列表
2. **从零开始**：`empty()`, `zeros()`, `ones()`用不同的初始值分配内存
3. **合并**：`vstack()`和`hstack()`将多个数组合并为较大的数组

这个进度教您如何从"我有一些数据"→"我需要原始内存"→"我需要合并数据集"。

## ML Relevance / 机器学习相关性

In machine learning:
- **`np.array()`**: Load your dataset from CSV, JSON, or in-memory lists
- **`zeros()`, `ones()`**: Initialize weights, biases, and model parameters
- **`vstack()`, `hstack()`**: Concatenate training batches, merge feature sets, combine datasets from multiple sources

All of these are essential for data preparation and model initialization.

在机器学习中：
- **`np.array()`**：从CSV、JSON或内存列表加载您的数据集
- **`zeros()`, `ones()`**：初始化权重、偏置和模型参数
- **`vstack()`, `hstack()`**：连接训练批次、合并特征集、合并来自多个源的数据集

所有这些对于数据准备和模型初始化都是必不可少的。

---
