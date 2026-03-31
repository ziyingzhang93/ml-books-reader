# 线性代数与机器学习 / Linear Algebra for Machine Learning
## Chapter 05

---

### Create 1D



---

### Create 2D

## File 2 of 16 / 第2章 — 第2个文件（共16个）

### Summary / 总结
Learn how to create 2D arrays (matrices) from nested Python lists. This is essential for representing datasets and performing matrix operations.

学习如何从嵌套Python列表创建二维数组（矩阵）。这对于表示数据集和执行矩阵操作至关重要。

### Core Concept / 核心概念
**2D Array (Matrix)**: A two-dimensional array with rows and columns. In linear algebra: **A** = [[a₁₁, a₁₂], [a₂₁, a₂₂], [a₃₁, a₃₂]]

**二维数组（矩阵）**: 具有行和列的二维数组。在线性代数中：**A** = [[a₁₁, a₁₂], [a₂₁, a₂₂], [a₃₁, a₃₂]]

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create Nested Python List / 创建嵌套Python列表

We create a Python list of lists, where each inner list represents a row. This structure naturally maps to a 2D array.

我们创建一个列表的列表，其中每个内部列表代表一行。这个结构自然地映射到二维数组。

```python
# 导入array函数 / Import array function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# 创建嵌套列表 / Create nested list (list of lists)
data = [[11, 22], [33, 44], [55, 66]]
# 打印输出 / Print output
print("Python list of lists:")
# 打印输出 / Print output
print(data)
```

## Step 2 — Convert to NumPy 2D Array / 转换为NumPy二维数组

We convert the nested list to a 2D NumPy array. This creates a matrix with 3 rows and 2 columns.

我们将嵌套列表转换为二维NumPy数组。这创建一个具有3行2列的矩阵。

```python
# 转换为2D NumPy数组 / Convert to 2D NumPy array
data = array(data)
# 打印输出 / Print output
print("\nNumPy 2D array (matrix):")
# 打印输出 / Print output
print(data)
```

## Step 3 — Inspect Type and Properties / 检查类型和属性

We verify that data is a 2D NumPy array and inspect its shape. The shape (3, 2) indicates 3 rows and 2 columns.

我们验证数据是二维NumPy数组并检查其形状。形状(3, 2)表示3行2列。

```python
# 检查类型 / Check type
# 打印输出 / Print output
print("\nType:")
# 打印输出 / Print output
print(type(data))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("\nShape:", data.shape)
# 打印输出 / Print output
print("Data type:", data.dtype)
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: A matrix is a rectangular array of numbers arranged in rows and columns. In machine learning, matrices represent datasets where rows are samples and columns are features. The shape (m, n) means m rows and n columns.

矩阵是按行和列排列的数字的矩形阵列。在机器学习中，矩阵表示数据集，其中行是样本，列是特征。形状(m, n)表示m行n列。

• **ML Application / ML应用**: In ML datasets, each row represents one training sample and each column represents one feature. A dataset with 3 samples and 2 features creates a 3×2 matrix, perfect for training algorithms.

在ML数据集中，每行表示一个训练样本，每列表示一个特征。具有3个样本和2个特征的数据集创建3×2矩阵，非常适合训练算法。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 03_index_1d.ipynb** — Learn how to access individual elements in 1D arrays using indexing.

## Complete Code / 完整代码一览

```python
# --- Create 2D Array / 创建二维数组 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Create nested list / 创建嵌套列表
data = [[11, 22], [33, 44], [55, 66]]
# 打印输出 / Print output
print("Python list of lists:")
# 打印输出 / Print output
print(data)

# Convert to 2D NumPy array / 转换为二维NumPy数组
data = array(data)
# 打印输出 / Print output
print("\nNumPy 2D array (matrix):")
# 打印输出 / Print output
print(data)

# Check type and properties / 检查类型和属性
# 打印输出 / Print output
print("\nType:", type(data))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Shape:", data.shape)
# 打印输出 / Print output
print("Data type:", data.dtype)
```

---

### Index 1D

## File 3 of 16 / 第2章 — 第3个文件（共16个）

### Summary / 总结
Learn how to access individual elements in 1D arrays using integer indexing. Indexing is the foundation for array manipulation and element access.

学习如何使用整数索引访问一维数组中的单个元素。索引是数组操作和元素访问的基础。

### Core Concept / 核心概念
**Array Indexing**: Access elements using zero-based indices. For array **v** = [11, 22, 33, 44, 55], index 0 gives 11, index 1 gives 22, etc.

**数组索引**: 使用从零开始的索引访问元素。对于数组**v** = [11, 22, 33, 44, 55]，索引0给出11，索引1给出22，等等。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 1D Array / 创建一维数组

We create a 1D array with 5 elements. Array indices in NumPy are zero-based: first element is at index 0, last is at index 4.

我们创建一个有5个元素的一维数组。NumPy中的数组索引从零开始：第一个元素在索引0处，最后一个在索引4处。

```python
# 导入array函数 / Import array function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# 创建一维数组 / Create 1D array
data = array([11, 22, 33, 44, 55])
# 打印输出 / Print output
print("Array:")
# 打印输出 / Print output
print(data)
```

## Step 2 — Access First Element / 访问第一个元素

We use index 0 to access the first element. In NumPy (and most programming languages), indexing starts at 0.

我们使用索引0来访问第一个元素。在NumPy（和大多数编程语言）中，索引从0开始。

```python
# 访问第一个元素（索引0） / Access first element (index 0)
# 打印输出 / Print output
print("\nFirst element (index 0):")
# 打印输出 / Print output
print(data[0])
```

## Step 3 — Access Last Element / 访问最后一个元素

We use index 4 to access the last element. Since the array has 5 elements, the last valid index is 4 (5 - 1).

我们使用索引4来访问最后一个元素。由于数组有5个元素，最后一个有效索引是4（5 - 1）。

```python
# 访问最后一个元素（索引4） / Access last element (index 4)
# 打印输出 / Print output
print("\nLast element (index 4):")
# 打印输出 / Print output
print(data[4])
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: In vector notation, the i-th element of vector **v** is denoted v_i. In programming, we use v[i-1] because indexing is zero-based. Understanding indexing is critical for selecting features, accessing data points, and matrix operations.

在向量符号中，向量**v**的第i个元素表示为v_i。在编程中，我们使用v[i-1]，因为索引从零开始。理解索引对于选择特征、访问数据点和矩阵操作至关重要。

• **ML Application / ML应用**: In ML, indexing is used to access specific features from a sample vector, retrieve individual training examples from a dataset, or extract predictions from model outputs. Efficient indexing is essential for data manipulation.

在ML中，索引用于从样本向量访问特定特征、从数据集检索个别训练示例或从模型输出提取预测。高效索引对数据操作至关重要。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 04_index_error_1d.ipynb** — Learn what happens when you try to access an out-of-bounds index.

## Complete Code / 完整代码一览

```python
# --- Index 1D Array / 索引一维数组 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Create 1D array / 创建一维数组
data = array([11, 22, 33, 44, 55])
# 打印输出 / Print output
print("Array:")
# 打印输出 / Print output
print(data)

# Access first element / 访问第一个元素
# 打印输出 / Print output
print("\nFirst element (index 0):")
# 打印输出 / Print output
print(data[0])

# Access last element / 访问最后一个元素
# 打印输出 / Print output
print("\nLast element (index 4):")
# 打印输出 / Print output
print(data[4])
```

---

### Index Error 1D

## File 4 of 16 / 第2章 — 第4个文件（共16个）

### Summary / 总结
Learn what happens when you try to access an array element with an index that is out of bounds. Understanding error handling is important for debugging code.

学习当你尝试使用超出边界的索引访问数组元素时会发生什么。理解错误处理对于调试代码很重要。

### Core Concept / 核心概念
**IndexError**: Raised when trying to access an index that doesn't exist. For an array of length n, valid indices are 0 to n-1.

**IndexError**: 当尝试访问不存在的索引时引发。对于长度为n的数组，有效索引为0到n-1。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 1D Array / 创建一维数组

We create a 1D array with 5 elements (indices 0-4). Any index 5 or higher is out of bounds.

我们创建一个有5个元素（索引0-4）的一维数组。任何索引5或更高的都超出边界。

```python
# 导入array函数 / Import array function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# 创建一维数组 / Create 1D array
data = array([11, 22, 33, 44, 55])
# 打印输出 / Print output
print("Array:")
# 打印输出 / Print output
print(data)
# 打印输出 / Print output
print("Length:", len(data))
# 打印输出 / Print output
print("Valid indices: 0, 1, 2, 3, 4")
```

## Step 2 — Try to Access Out-of-Bounds Index / 尝试访问超出边界的索引

We attempt to access index 5, which doesn't exist. This raises an IndexError because the array only has indices 0-4.

我们尝试访问索引5，它不存在。这引发IndexError，因为数组只有索引0-4。

```python
# 尝试访问超出边界的索引 / Try to access out-of-bounds index
try:
    # 打印输出 / Print output
    print("\nAttempting to access index 5:")
    # 打印输出 / Print output
    print(data[5])
except IndexError as e:
    # 打印输出 / Print output
    print(f"Error: {e}")
    # 打印输出 / Print output
    print("Index 5 is out of bounds!")
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Array bounds are fundamental to array access. For an array of size n, valid indices range from 0 to n-1. This "zero-based indexing" is used in most programming languages and prevents accessing undefined memory locations.

数组边界对数组访问至关重要。对于大小为n的数组，有效索引范围为0到n-1。这种"从零开始的索引"在大多数编程语言中使用，并防止访问未定义的内存位置。

• **ML Application / ML应用**: When accessing datasets, ensure your loop indices stay within bounds. Off-by-one errors are common bugs in data processing pipelines. Always verify array dimensions before indexing.

访问数据集时，确保循环索引保持在边界内。差一错误是数据处理管道中的常见错误。在索引前始终验证数组维度。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 05_index_negative_1d.ipynb** — Learn how to use negative indices to access elements from the end of an array.

## Complete Code / 完整代码一览

```python
# --- Index Error Demonstration / 索引错误演示 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Create 1D array / 创建一维数组
data = array([11, 22, 33, 44, 55])
# 打印输出 / Print output
print("Array:")
# 打印输出 / Print output
print(data)
# 打印输出 / Print output
print("Length:", len(data))

# Try to access out-of-bounds index / 尝试访问超出边界的索引
try:
    # 打印输出 / Print output
    print("\nAttempting to access index 5:")
    # 打印输出 / Print output
    print(data[5])
except IndexError as e:
    # 打印输出 / Print output
    print(f"Error: {e}")
    # 打印输出 / Print output
    print("Index 5 is out of bounds!")
```

---

### Index Negative 1D

## File 5 of 16 / 第2章 — 第5个文件（共16个）

### Summary / 总结
Learn how to use negative indices to access array elements counting from the end. Negative indexing provides a convenient way to work with array elements without knowing the array size.

学习如何使用负索引从末尾开始计数访问数组元素。负索引提供了一种方便的方式来处理数组元素，而不需要知道数组大小。

### Core Concept / 核心概念
**Negative Indexing**: Access elements from the end using negative indices. Index -1 gives the last element, -2 the second-to-last, etc. Mapping: index -i = index n-i

**负索引**: 使用负索引从末尾访问元素。索引-1给出最后一个元素，-2给出倒数第二个，等等。映射：索引-i = 索引n-i

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 1D Array / 创建一维数组

We create a 1D array with 5 elements. In NumPy, negative indices count from the end: -1 is the last element, -5 is the first.

我们创建一个有5个元素的一维数组。在NumPy中，负索引从末尾开始计数：-1是最后一个元素，-5是第一个。

```python
# 导入array函数 / Import array function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# 创建一维数组 / Create 1D array
data = array([11, 22, 33, 44, 55])
# 打印输出 / Print output
print("Array:")
# 打印输出 / Print output
print(data)
# 打印输出 / Print output
print("\nIndexing scheme:")
# 打印输出 / Print output
print("Positive: 0    1    2    3    4")
# 打印输出 / Print output
print("Negative: -5   -4   -3   -2   -1")
```

## Step 2 — Access Last Element Using -1 / 使用-1访问最后一个元素

We use negative index -1 to access the last element (55). This is equivalent to index 4 for this 5-element array.

我们使用负索引-1来访问最后一个元素(55)。对于这个5元素数组，这等同于索引4。

```python
# 使用负索引-1访问最后一个元素 / Access last element using -1
# 打印输出 / Print output
print("\nLast element (index -1):")
# 打印输出 / Print output
print(data[-1])
```

## Step 3 — Access First Element Using -5 / 使用-5访问第一个元素

We use negative index -5 to access the first element (11). This is equivalent to index 0 for this 5-element array.

我们使用负索引-5来访问第一个元素(11)。对于这个5元素数组，这等同于索引0。

```python
# 使用负索引-5访问第一个元素 / Access first element using -5
# 打印输出 / Print output
print("\nFirst element (index -5):")
# 打印输出 / Print output
print(data[-5])
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Negative indexing is a convenience feature that maps index -i to position n-i in the array. It's useful for accessing elements relative to the end without computing n-i explicitly. Mathematically: data[-i] = data[n-i].

负索引是一个方便的特性，将索引-i映射到数组中的位置n-i。对于相对于末尾访问元素而不显式计算n-i很有用。数学上：data[-i] = data[n-i]。

• **ML Application / ML应用**: Negative indexing is useful in data processing: access the last sample (data[-1]), remove the last feature (data[:-1]), or get the last few predictions from a model output without knowing the exact array size.

负索引在数据处理中很有用：访问最后一个样本(data[-1])，删除最后一个特征(data[:-1])，或从模型输出获取最后几个预测，而不需要知道确切的数组大小。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 06_index_2d.ipynb** — Learn how to access elements in 2D arrays (matrices) using row and column indices.

## Complete Code / 完整代码一览

```python
# --- Negative Indexing / 负索引 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Create 1D array / 创建一维数组
data = array([11, 22, 33, 44, 55])
# 打印输出 / Print output
print("Array:")
# 打印输出 / Print output
print(data)
# 打印输出 / Print output
print("\nIndexing scheme:")
# 打印输出 / Print output
print("Positive: 0    1    2    3    4")
# 打印输出 / Print output
print("Negative: -5   -4   -3   -2   -1")

# Access last element / 访问最后一个元素
# 打印输出 / Print output
print("\nLast element (index -1):")
# 打印输出 / Print output
print(data[-1])

# Access first element / 访问第一个元素
# 打印输出 / Print output
print("\nFirst element (index -5):")
# 打印输出 / Print output
print(data[-5])
```

---

### Index 2D

## File 6 of 16 / 第2章 — 第6个文件（共16个）

### Summary / 总结
Learn how to access individual elements in 2D arrays (matrices) using row and column indices. 2D indexing is essential for matrix operations.

学习如何使用行和列索引访问二维数组（矩阵）中的单个元素。二维索引对矩阵操作至关重要。

### Core Concept / 核心概念
**2D Indexing**: Access matrix element using [row, column] notation. For matrix A[i, j], i is row index and j is column index, both zero-based.

**二维索引**: 使用[行，列]符号访问矩阵元素。对于矩阵A[i, j]，i是行索引，j是列索引，都从零开始。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 2D Array / 创建二维数组

We create a 3x2 matrix (3 rows, 2 columns). Each element can be accessed using [row_index, column_index].

我们创建一个3x2矩阵（3行，2列）。每个元素都可以使用[row_index, column_index]访问。

```python
# 导入array函数 / Import array function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# 创建二维数组 / Create 2D array
data = array([[11, 22], [33, 44], [55, 66]])
# 打印输出 / Print output
print("Matrix:")
# 打印输出 / Print output
print(data)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("\nShape:", data.shape)
```

## Step 2 — Access Specific Element / 访问特定元素

We access the element at row 0, column 0 (the first row, first column), which is 11. The syntax is [row, column].

我们访问第0行第0列（第一行，第一列）的元素，即11。语法是[行，列]。

```python
# 访问第一行第一列的元素 / Access element at row 0, column 0
# 打印输出 / Print output
print("\nElement at row 0, column 0:")
# 打印输出 / Print output
print(data[0, 0])
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: In matrix notation, element A_{ij} denotes the element in the i-th row and j-th column. In NumPy, we use A[i, j] where both indices are zero-based. For a 3×2 matrix, valid indices are: rows 0-2, columns 0-1.

在矩阵符号中，元素A_{ij}表示第i行第j列的元素。在NumPy中，我们使用A[i, j]，其中两个索引都从零开始。对于3×2矩阵，有效索引为：行0-2，列0-1。

• **ML Application / ML应用**: In ML, accessing specific matrix elements is fundamental. For a dataset matrix where rows are samples and columns are features, data[i, j] gives the j-th feature of the i-th sample. This is used for slicing subsets of data.

在ML中，访问特定矩阵元素是基本的。对于数据集矩阵，其中行是样本，列是特征，data[i, j]给出第i个样本的第j个特征。这用于切割数据子集。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 07_index_first_row_2d.ipynb** — Learn how to access entire rows in 2D arrays.

## Complete Code / 完整代码一览

```python
# --- Index 2D Array / 索引二维数组 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Create 2D array / 创建二维数组
data = array([[11, 22], [33, 44], [55, 66]])
# 打印输出 / Print output
print("Matrix:")
# 打印输出 / Print output
print(data)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Shape:", data.shape)

# Access element at row 0, column 0 / 访问第0行第0列的元素
# 打印输出 / Print output
print("\nElement at row 0, column 0:")
# 打印输出 / Print output
print(data[0, 0])
```

---

### Index First Row 2D



---

### Slice 1D

## File 8 of 16 / 第2章 — 第8个文件（共16个）

### Summary / 总结
Learn how to slice 1D arrays to extract all elements or specific subsets. Array slicing with ":" is a fundamental operation for data manipulation.

学习如何切片一维数组以提取所有元素或特定子集。使用":"进行数组切片是数据操作的基本操作。

### Core Concept / 核心概念
**Slicing**: Use [start:stop] notation to extract a subset of elements. [:] means all elements. The stop index is exclusive.

**切片**: 使用[start:stop]符号提取元素的子集。[:]表示所有元素。stop索引是排他的。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 1D Array / 创建一维数组

We create a 1D array with 5 elements. Now we'll demonstrate slicing with [:], which returns all elements.

我们创建一个有5个元素的一维数组。现在我们将演示使用[:]的切片，它返回所有元素。

```python
# 导入array函数 / Import array function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# 创建一维数组 / Create 1D array
data = array([11, 22, 33, 44, 55])
# 打印输出 / Print output
print("Original array:")
# 打印输出 / Print output
print(data)
```

## Step 2 — Slice All Elements / 切片所有元素

We use [:] to select all elements. This is equivalent to copying the entire array, returning all 5 elements.

我们使用[:]来选择所有元素。这等同于复制整个数组，返回所有5个元素。

```python
# 使用[:]选择所有元素 / Select all elements using [:]
# 打印输出 / Print output
print("\nAll elements (using [:]):")
# 打印输出 / Print output
print(data[:])
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Slicing is equivalent to selecting a contiguous subvector from a vector. The notation [start:stop] selects elements from index start up to (but not including) index stop. Using [:] selects the entire array.

切片等同于从向量中选择连续的子向量。符号[start:stop]选择从索引start到（但不包括）索引stop的元素。使用[:]选择整个数组。

• **ML Application / ML applicaton**: Array slicing is essential for data manipulation: split training/test data, extract feature subsets, remove outlier samples, or access temporal sequences in time-series data.

数组切片对于数据操作至关重要：分割训练/测试数据、提取特征子集、删除离群值样本或访问时间序列数据中的时间序列。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 08_slice_subset_1d.ipynb** — Learn how to slice specific subsets of array elements.

## Complete Code / 完整代码一览

```python
# --- Slice All Elements / 切片所有元素 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Create 1D array / 创建一维数组
data = array([11, 22, 33, 44, 55])
# 打印输出 / Print output
print("Original array:")
# 打印输出 / Print output
print(data)

# Select all elements using [:] / 使用[:]选择所有元素
# 打印输出 / Print output
print("\nAll elements (using [:]):")
# 打印输出 / Print output
print(data[:])
```

---

### Slice Subset 1D



---

### Slice Negative 1D

## File 10 of 16 / 第2章 — 第10个文件（共16个）

### Summary / 总结
Learn how to slice arrays using negative indices. Negative indices in slicing are useful for accessing elements from the end of an array.

学习如何使用负索引切片数组。切片中的负索引对于访问数组末尾的元素很有用。

### Core Concept / 核心概念
**Negative Range Slicing**: [-n:] extracts the last n elements. For example, [-2:] returns the last 2 elements without needing to know array length.

**负范围切片**: [-n:]提取最后n个元素。例如，[-2:]返回最后2个元素，而不需要知道数组长度。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 1D Array / 创建一维数组

We create a 1D array with 5 elements. Now we'll extract the last 2 elements using [-2:], which is more convenient than computing the exact indices.

我们创建一个有5个元素的一维数组。现在我们将使用[-2:]提取最后2个元素，这比计算确切索引更方便。

```python
# 导入array函数 / Import array function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# 创建一维数组 / Create 1D array
data = array([11, 22, 33, 44, 55])
# 打印输出 / Print output
print("Original array:")
# 打印输出 / Print output
print(data)
```

## Step 2 — Slice Last 2 Elements / 切片最后2个元素

We use [-2:] to extract the last 2 elements (44 and 55). This is equivalent to [3:5] for a 5-element array, but more flexible since it doesn't depend on array size.

我们使用[-2:]提取最后2个元素(44和55)。对于5元素数组，这等同于[3:5]，但更灵活，因为它不依赖于数组大小。

```python
# 使用[-2:]切片最后两个元素 / Slice last 2 elements using [-2:]
# 打印输出 / Print output
print("\nLast 2 elements (using [-2:]):")
# 打印输出 / Print output
print(data[-2:])
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Negative slicing is syntactic sugar for selecting elements relative to the array's end. [-n:] selects the last n elements without explicit array size calculation. Mathematically: data[-n:] = data[len(data)-n:].

负切片是相对于数组末尾选择元素的语法糖。[-n:]选择最后n个元素，无需显式数组大小计算。数学上：data[-n:] = data[len(data)-n:]。

• **ML Application / ML application**: Negative slicing is useful for: removing last feature (data[:-1]), selecting last k predictions, accessing recent data in sliding windows, or excluding the last batch in train/val splits.

负切片对以下情况很有用：删除最后特征(data[:-1])、选择最后k个预测、在滑动窗口中访问最近数据或在训练/验证分割中排除最后批次。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 10_seprate_input_output.ipynb** — Learn how to separate input features from output labels in datasets.

## Complete Code / 完整代码一览

```python
# --- Slice with Negative Indices / 使用负索引切片 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Create 1D array / 创建一维数组
data = array([11, 22, 33, 44, 55])
# 打印输出 / Print output
print("Original array:")
# 打印输出 / Print output
print(data)

# Slice last 2 elements / 切片最后2个元素
# 打印输出 / Print output
print("\nLast 2 elements (using [-2:]):")
# 打印输出 / Print output
print(data[-2:])
```

---

### Seprate Input Output



---

### Separate Train Test Sets

## File 12 of 16 / 第2章 — 第12个文件（共16个）

### Summary / 总结
Learn how to split a dataset into training and test subsets. This is essential for validating machine learning models on unseen data.

学习如何将数据集分割为训练和测试子集。这对于在看不见的数据上验证机器学习模型至关重要。

### Core Concept / 核心概念
**Train-Test Split**: Use row slicing to partition data. train = data[:split, :] uses first rows for training. test = data[split:, :] uses remaining rows for testing.

**训练-测试分割**: 使用行切片对数据进行分区。train = data[:split, :]使用前几行进行训练。test = data[split:, :]使用剩余行进行测试。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create Dataset / 创建数据集

We create a dataset with 3 samples and 3 columns (2 features + 1 label). We'll split after the first 2 rows, creating a 2-sample train set and 1-sample test set.

我们创建一个具有3个样本和3列（2个特征+1个标签）的数据集。我们将在前2行后分割，创建2样本训练集和1样本测试集。

```python
# 导入array函数 / Import array function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# 创建数据集 / Create dataset
data = array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])
# 打印输出 / Print output
print("Full dataset (3 samples, 3 columns):")
# 打印输出 / Print output
print(data)
```

## Step 2 — Define Split Point / 定义分割点

We set split = 2, meaning the first 2 rows (indices 0-1) go to training, and remaining rows (index 2) go to testing.

我们设置split = 2，意味着前2行（索引0-1）进入训练，剩余行（索引2）进入测试。

```python
# 定义分割点 / Define split point
split = 2
# 打印输出 / Print output
print("\nSplit point: first {} samples for training".format(split))
```

## Step 3 — Split Training Data / 分割训练数据

We extract rows 0 to split (0-1 inclusive), which gives us the first 2 samples for training.

我们提取第0行到分割点（0-1包含），得到前2个样本用于训练。

```python
# 分割训练数据 / Split training data
train = data[:split, :]
# 打印输出 / Print output
print("\nTraining data (first {} samples):".format(split))
# 打印输出 / Print output
print(train)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Shape:", train.shape)
```

## Step 4 — Split Test Data / 分割测试数据

We extract rows from split onward, which gives us the remaining 1 sample for testing.

我们提取从分割点开始的行，得到剩余的1个样本用于测试。

```python
# 分割测试数据 / Split test data
test = data[split:, :]
# 打印输出 / Print output
print("\nTest data (remaining samples):")
# 打印输出 / Print output
print(test)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Shape:", test.shape)
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Train-test split partitions the dataset into disjoint subsets. If total samples = m and split = k, then training set has k samples and test set has m-k samples. This ensures no information leakage from test to training.

训练-测试分割将数据集分区为不相交的子集。如果总样本= m，分割= k，则训练集有k个样本，测试集有m-k个样本。这确保了没有信息从测试泄漏到训练。

• **ML Application / ML应用**: Train-test split is critical for unbiased model evaluation. Models train on the training set and are evaluated on the test set, which they've never seen. This prevents overfitting and provides a realistic measure of generalization performance.

训练-测试分割对于无偏模型评估至关重要。模型在训练集上训练，在测试集上评估，它们从未见过。这可以防止过拟合并提供泛化性能的真实度量。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 12_shape_1d.ipynb** — Learn how to inspect the shape of 1D arrays.

## Complete Code / 完整代码一览

```python
# --- Train-Test Split / 训练-测试分割 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Create dataset / 创建数据集
data = array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])
# 打印输出 / Print output
print("Full dataset:")
# 打印输出 / Print output
print(data)

# Define split point / 定义分割点
split = 2
# 打印输出 / Print output
print("\nSplit point: first {} samples for training".format(split))

# Split into training and test / 分割为训练和测试
train = data[:split, :]  # First 2 rows / 前2行
test = data[split:, :]   # Remaining rows / 剩余行

# 打印输出 / Print output
print("\nTraining data:")
# 打印输出 / Print output
print(train)

# 打印输出 / Print output
print("\nTest data:")
# 打印输出 / Print output
print(test)
```

---

### Shape 1D

## File 13 of 16 / 第2章 — 第13个文件（共16个）

### Summary / 总结
Learn how to inspect the shape of 1D arrays. The shape attribute tells you the dimensions of an array, essential for understanding data structure.

学习如何检查一维数组的形状。shape属性告诉你数组的维度，对于理解数据结构至关重要。

### Core Concept / 核心概念
**.shape attribute**: For 1D arrays, shape returns a tuple (n,) where n is the number of elements. For example, a vector with 5 elements has shape (5,).

**.shape属性**: 对于一维数组，shape返回一个元组(n,)，其中n是元素数。例如，具有5个元素的向量的形状是(5,)。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 1D Array / 创建一维数组

We create a 1D array with 5 elements. The shape will reflect this dimensionality.

我们创建一个有5个元素的一维数组。形状将反映这个维度。

```python
# 导入array函数 / Import array function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# 创建一维数组 / Create 1D array
data = array([11, 22, 33, 44, 55])
# 打印输出 / Print output
print("1D array:")
# 打印输出 / Print output
print(data)
```

## Step 2 — Inspect Shape / 检查形状

We use the .shape attribute to check the array's dimensions. For a 1D array with 5 elements, shape is (5,).

我们使用.shape属性来检查数组的维度。对于有5个元素的一维数组，形状是(5,)。

```python
# 检查形状 / Check shape
# 打印输出 / Print output
print("\nShape:")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(data.shape)
# 打印输出 / Print output
print("\nType of shape:")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(type(data.shape))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("\nNumber of elements: {}".format(data.shape[0]))
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: The shape of an array describes its dimensions. For a 1D array (vector), shape = (n,) means a vector with n components. Understanding shape is fundamental for understanding array structure and performing compatible operations.

数组的形状描述了它的维度。对于一维数组（向量），shape = (n,)表示具有n个分量的向量。理解形状对于理解数组结构和执行兼容操作至关重要。

• **ML Application / ML application**: Checking array shape is crucial for debugging. When a function expects shape (n, m) but you provide (n,), it will fail. Always verify shapes: if X.shape = (m, n), you have m samples with n features.

检查数组形状对于调试至关重要。当函数期望shape (n, m)但你提供(n,)时，它将失败。始终验证形状：如果X.shape = (m, n)，你有m个样本，每个n个特征。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 13_shape_2d.ipynb** — Learn how to inspect the shape of 2D arrays.

## Complete Code / 完整代码一览

```python
# --- Inspect 1D Array Shape / 检查一维数组形状 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Create 1D array / 创建一维数组
data = array([11, 22, 33, 44, 55])
# 打印输出 / Print output
print("1D array:")
# 打印输出 / Print output
print(data)

# Check shape / 检查形状
# 打印输出 / Print output
print("\nShape:")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(data.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("\nNumber of elements: {}".format(data.shape[0]))
```

---

### Shape 2D



---

### Shape Rows Cols 2D

## File 15 of 16 / 第2章 — 第15个文件（共16个）

### Summary / 总结
Learn how to extract and interpret individual dimensions (rows and columns) from a 2D array's shape. This helps you understand dataset sizes.

学习如何从二维数组的形状中提取和解释单个维度（行和列）。这有助于你理解数据集大小。

### Core Concept / 核心概念
**Shape indexing**: shape[0] gives number of rows, shape[1] gives number of columns. From shape (m, n), access via tuple indexing.

**形状索引**: shape[0]给出行数，shape[1]给出列数。从形状(m, n)，通过元组索引访问。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 2D Array / 创建二维数组

We create a 3×2 matrix (3 rows, 2 columns) and will extract these dimensions separately.

我们创建一个3×2矩阵（3行，2列）并分别提取这些维度。

```python
# 导入array函数 / Import array function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# 创建二维数组 / Create 2D array
data = [[11, 22], [33, 44], [55, 66]]
data = array(data)
# 打印输出 / Print output
print("2D array:")
# 打印输出 / Print output
print(data)
# 打印输出 / Print output
print("\nShape:")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(data.shape)
```

## Step 2 — Extract Number of Rows / 提取行数

We access shape[0] to get the number of rows. For this matrix, shape[0] = 3.

我们访问shape[0]来获取行数。对于此矩阵，shape[0] = 3。

```python
# 获取行数 / Get number of rows
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
rows = data.shape[0]
# 打印输出 / Print output
print("\nNumber of rows (shape[0]): {}".format(rows))
```

## Step 3 — Extract Number of Columns / 提取列数

We access shape[1] to get the number of columns. For this matrix, shape[1] = 2.

我们访问shape[1]来获取列数。对于此矩阵，shape[1] = 2。

```python
# 获取列数 / Get number of columns
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
cols = data.shape[1]
# 打印输出 / Print output
print("Number of columns (shape[1]): {}".format(cols))
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Shape is a tuple, and we access components via indexing. For shape (m, n), index [0] gives m (rows) and index [1] gives n (columns). This is crucial for dynamic code that works with arrays of any size.

形状是一个元组，我们通过索引访问其分量。对于形状(m, n)，索引[0]给出m（行），索引[1]给出n（列）。这对于适用于任何大小数组的动态代码至关重要。

• **ML Application / ML application**: In ML, accessing shape dimensions is fundamental: if X.shape[0] is the number of samples and X.shape[1] is the number of features, you can write code that works for any dataset size. This is how ML frameworks validate input dimensions.

在ML中，访问形状维度是基本的：如果X.shape[0]是样本数，X.shape[1]是特征数，你可以编写适用于任何数据集大小的代码。这是ML框架验证输入维度的方式。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 15_reshape_1d_to_2d.ipynb** — Learn how to reshape arrays from one dimensionality to another.

## Complete Code / 完整代码一览

```python
# --- Extract Rows and Columns from Shape / 从形状中提取行和列 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Create 2D array / 创建二维数组
data = [[11, 22], [33, 44], [55, 66]]
data = array(data)
# 打印输出 / Print output
print("2D array:")
# 打印输出 / Print output
print(data)

# Extract rows and columns / 提取行和列
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
rows = data.shape[0]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
cols = data.shape[1]

# 打印输出 / Print output
print("\nRows: {}".format(rows))
# 打印输出 / Print output
print("Cols: {}".format(cols))
```

---

### Reshape 1D To 2D

## File 16 of 16 / 第2章 — 第16个文件（共16个）

### Summary / 总结
Learn how to reshape 1D arrays into 2D arrays. Reshaping is essential for converting between different data representations without changing element values.

学习如何将一维数组重新整形为二维数组。重新整形对于在不同数据表示之间转换而不改变元素值至关重要。

### Core Concept / 核心概念
**.reshape(shape)**: Transforms array to new shape without changing data. Example: (5,) → (5, 1) converts vector to column matrix.

**.reshape(shape)**: 将数组转换为新形状而不改变数据。示例：(5,) → (5, 1)将向量转换为列矩阵。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 1D Array / 创建一维数组

We create a 1D array with 5 elements. We'll reshape it into a 5×1 column matrix (2D array).

我们创建一个有5个元素的一维数组。我们将其重新整形为5×1列矩阵（二维数组）。

```python
# 导入array函数 / Import array function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# 创建一维数组 / Create 1D array
data = array([11, 22, 33, 44, 55])
# 打印输出 / Print output
print("Original 1D array:")
# 打印输出 / Print output
print(data)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Shape:", data.shape)
```

## Step 2 — Reshape to 2D / 重新整形为二维

We use .reshape() to convert the 1D array into a 2D array with shape (5, 1). This creates a column vector (5 rows, 1 column).

我们使用.reshape()将一维数组转换为形状为(5, 1)的二维数组。这创建一个列向量（5行，1列）。

```python
# 重新整形为二维 / Reshape to 2D
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
data = data.reshape((data.shape[0], 1))
# 打印输出 / Print output
print("\nReshaped 2D array (column vector):")
# 打印输出 / Print output
print(data)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Shape:", data.shape)
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Reshaping changes the dimensionality of an array without changing the data. A vector can be reshaped to a column matrix: (n,) → (n, 1). The total number of elements must remain the same: reshape doesn't add or remove data.

重新整形改变数组的维度而不改变数据。向量可以重新整形为列矩阵：(n,) → (n, 1)。元素总数必须保持不变：reshape不添加或删除数据。

• **ML Application / ML application**: In ML, reshaping is critical: scikit-learn requires 2D input, so a 1D prediction array must be reshaped from (n,) to (n, 1) before passing to certain functions. Broadcasting also sometimes requires explicit reshapes.

在ML中，重新整形至关重要：scikit-learn需要二维输入，所以一维预测数组必须从(n,)重新整形为(n, 1)才能传递给某些函数。广播有时也需要显式重新整形。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 16_reshape_2d_to_3d.ipynb** — Learn how to reshape 2D arrays into 3D arrays.

## Complete Code / 完整代码一览

```python
# --- Reshape 1D to 2D / 从一维重新整形为二维 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Create 1D array / 创建一维数组
data = array([11, 22, 33, 44, 55])
# 打印输出 / Print output
print("Original 1D array:")
# 打印输出 / Print output
print(data)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Shape:", data.shape)

# Reshape to 2D column vector / 重新整形为二维列向量
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
data = data.reshape((data.shape[0], 1))
# 打印输出 / Print output
print("\nReshaped 2D array (column vector):")
# 打印输出 / Print output
print(data)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Shape:", data.shape)
```

---

### Reshape 2D To 3D



---

### Chapter Summary / 章节总结

# Chapter 05 Summary / 第05章总结：Indexing, Slicing, and Reshaping

## Theme / 主题

Once you have arrays, you need to extract, rearrange, and prepare data. This chapter covers how to access individual elements, extract subsets (slicing), and reshape arrays to fit ML pipeline requirements.

一旦您拥有数组，您就需要提取、重新排列和准备数据。本章涵盖如何访问单个元素、提取子集（切片）以及重新整形数组以适应ML管道要求。

## Evolution / 演化路线

```
01_create_1d.ipynb
    └─ 1D array indexing (一维数组索引)
    
02_create_2d.ipynb
    └─ 2D array indexing (二维数组索引)
    
03_negative_indexing.ipynb
    └─ Access from end with negative indices (负索引从末尾访问)
    
04_2d_indexing.ipynb
    └─ Row and column access in 2D (二维行列访问)
    
05_slicing_1d.ipynb
    └─ Extract ranges from 1D (从一维提取范围)
    
06_slicing_2d.ipynb
    └─ Extract submatrices (提取子矩阵)
    
07_input_output_split.ipynb
    └─ Separate features (X) and labels (y) (分离特征和标签)
    
08_train_test_split.ipynb
    └─ Create train/test subsets (创建训练/测试子集)
    
09_2d_shape.ipynb
    └─ Inspect dimensions (检查维度)
    
10_reshape_1d_to_2d.ipynb
    └─ Convert row vectors to column vectors (转换行向量为列向量)
    
11_reshape_2d_to_1d.ipynb
    └─ Flatten matrices (展平矩阵)
    
12_reshape_other.ipynb
    └─ Arbitrary dimension changes (任意维度改变)
    
13_reshape_infer.ipynb
    └─ Use -1 to auto-calculate dimensions (使用-1自动计算维度)
    
14_shape_stacking.ipynb
    └─ Reshape for batch operations (为批操作重新整形)
    
15_data_preparation.ipynb
    └─ Combine all concepts for ML (组合所有概念用于ML)
    
16_advanced_indexing.ipynb
    └─ Boolean and fancy indexing (布尔和花式索引)
```

## Progression Logic / 进度逻辑

The chapter progresses through **accessing** → **extracting** → **preparing for ML**:

1. **Indexing basics**: 1D → 2D → negative indices (all ways to get one element or a few)
2. **Slicing & splitting**: Extract ranges, separate X from y, split train/test
3. **Reshaping**: Change dimensions to match ML pipeline requirements
4. **Advanced indexing**: Filter data with boolean masks, use computed indices

This trains you to think about data flow: get raw data → extract what you need → reshape to ML format.

进度通过**访问** → **提取** → **为ML准备**：

1. **索引基础**：1D → 2D → 负索引（所有获取一个或几个元素的方式）
2. **切片和拆分**：提取范围、分离X和y、分割训练/测试
3. **重新整形**：更改维度以匹配ML管道要求
4. **高级索引**：使用布尔掩码过滤数据、使用计算的索引

这训练您思考数据流：获取原始数据→提取您需要的→重新整形为ML格式。

## ML Relevance / 机器学习相关性

In machine learning:
- **Indexing & slicing**: Extract features, labels, and subsets from raw datasets
- **Train/test splits**: `X_train[:80%]`, `y_test[80%:]` — essential for model evaluation
- **Reshaping**: Convert flat CSV columns into (n_samples, n_features) matrices required by scikit-learn
- **Boolean indexing**: Filter outliers, select specific classes, apply masks

Data preparation is 80% of ML work, and these tools are your foundation.

在机器学习中：
- **索引和切片**：从原始数据集中提取特征、标签和子集
- **训练/测试拆分**：`X_train[:80%]`, `y_test[80%:]` — 对模型评估至关重要
- **重新整形**：将平面CSV列转换为scikit-learn所需的(n_samples, n_features)矩阵
- **布尔索引**：过滤异常值、选择特定类、应用掩码

数据准备占ML工作的80%，这些工具是您的基础。

---
