# 线性代数与机器学习
## Chapter 07

---

### Rows

## File 2 of 6 / 第4章 — 第2个文件（共6个）

### Summary / 总结
Learn how to iterate through rows of a 2D array. Understanding row-wise iteration is essential before applying row-wise aggregate operations.

学习如何遍历二维数组的行。在应用按行聚合操作前，理解按行迭代至关重要。

### Core Concept / 核心概念
**Row Iteration**: Use shape[0] to loop through rows. Iterate from 0 to shape[0]-1, accessing each row with data[i, :].

**行迭代**: 使用shape[0]来循环遍历行。从0迭代到shape[0]-1，用data[i, :]访问每一行。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 2D Array / 创建二维数组

We create a 2×3 array with 2 rows. We'll iterate through each row and display it.

我们创建一个有2行的2×3数组。我们将遍历每一行并显示它。

```python
# 导入asarray函数 / Import asarray function
from numpy import asarray

# 创建二维数组 / Create 2D array
data = [[1, 2, 3], [4, 5, 6]]
data = asarray(data)
print("Array:")
print(data)
print("\nShape:", data.shape)
```

## Step 2 — Iterate Through Rows / 遍历行

We loop from 0 to shape[0] (0 to 2), accessing each row with data[row, :]. This prints each row as a 1D array.

我们从0循环到shape[0]（0到2），用data[row, :]访问每一行。这将每一行作为一维数组打印。

```python
# 遍历行 / Iterate through rows
print("\nIterating through rows:")
for row in range(data.shape[0]):
    print("Row {}: {}".format(row, data[row, :]))
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Iteration through rows is the basis for row-wise operations. For a matrix with m rows, iterating i from 0 to m-1 and accessing row i as **a** = [a_i1, a_i2, ..., a_in] enables element-wise operations on each row.

通过行的迭代是按行操作的基础。对于具有m行的矩阵，从0到m-1迭代i并将第i行作为**a** = [a_i1, a_i2, ..., a_in]访问，使得对每行的逐元素操作成为可能。

• **ML Application / ML application**: Row-wise iteration is fundamental in ML: processing training samples one at a time, computing per-sample statistics, or implementing algorithms that operate on individual samples. This is often wrapped in vectorized operations for efficiency.

按行迭代在ML中是基本的：一次处理一个训练样本、计算每个样本的统计量或实现在单个样本上操作的算法。这通常被包装在向量化操作中以提高效率。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 03_columns.ipynb** — Learn how to iterate through columns of an array.

## Complete Code / 完整代码一览

```python
# --- Iterate Through Rows / 遍历行 ---
from numpy import asarray

# Create 2D array / 创建二维数组
data = [[1, 2, 3], [4, 5, 6]]
data = asarray(data)
print("Array:")
print(data)
print("Shape:", data.shape)

# Iterate through rows / 遍历行
print("\nIterating through rows:")
for row in range(data.shape[0]):
    print("Row {}: {}".format(row, data[row, :]))
```

---

### Columns

## File 3 of 6 / 第4章 — 第3个文件（共6个）

### Summary / 总结
Learn how to iterate through columns of a 2D array. Column-wise iteration is useful for operations like feature-by-feature analysis.

学习如何遍历二维数组的列。列迭代对于逐特征分析等操作很有用。

### Core Concept / 核心概念
**Column Iteration**: Use shape[1] to loop through columns. Iterate from 0 to shape[1]-1, accessing each column with data[:, col].

**列迭代**: 使用shape[1]来循环遍历列。从0迭代到shape[1]-1，用data[:, col]访问每一列。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 2D Array / 创建二维数组

We create a 2×3 array with 3 columns. We'll iterate through each column and display it.

我们创建一个有3列的2×3数组。我们将遍历每一列并显示它。

```python
# 导入asarray函数 / Import asarray function
from numpy import asarray

# 创建二维数组 / Create 2D array
data = [[1, 2, 3], [4, 5, 6]]
data = asarray(data)
print("Array:")
print(data)
print("\nShape:", data.shape)
```

## Step 2 — Iterate Through Columns / 遍历列

We loop from 0 to shape[1] (0 to 3), accessing each column with data[:, col]. This prints each column as a 1D array.

我们从0循环到shape[1]（0到3），用data[:, col]访问每一列。这将每一列作为一维数组打印。

```python
# 遍历列 / Iterate through columns
print("\nIterating through columns:")
for col in range(data.shape[1]):
    print("Column {}: {}".format(col, data[:, col]))
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Column-wise iteration extracts individual features or dimensions. For a matrix with n columns, iterating j from 0 to n-1 and accessing column j gives the j-th feature vector. This is essential for feature-wise analysis in statistics.

列迭代提取单个特征或维度。对于具有n列的矩阵，从0到n-1迭代j并访问第j列给出第j个特征向量。这对于统计学中的特征分析至关重要。

• **ML Application / ML application**: In ML, column-wise operations are common: computing per-feature statistics (mean, std), feature scaling, feature selection, or detecting missing values in specific columns. Vectorized operations replace this loop for efficiency.

在ML中，按列操作很常见：计算每个特征的统计量（均值、标准差）、特征缩放、特征选择或检测特定列中的缺失值。向量化操作替代此循环以提高效率。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 04_axis_none.ipynb** — Learn how to compute aggregate functions across all elements (axis=None).

## Complete Code / 完整代码一览

```python
# --- Iterate Through Columns / 遍历列 ---
from numpy import asarray

# Create 2D array / 创建二维数组
data = [[1, 2, 3], [4, 5, 6]]
data = asarray(data)
print("Array:")
print(data)
print("Shape:", data.shape)

# Iterate through columns / 遍历列
print("\nIterating through columns:")
for col in range(data.shape[1]):
    print("Column {}: {}".format(col, data[:, col]))
```

---

### Axis None

## File 4 of 6 / 第4章 — 第4个文件（共6个）

### Summary / 总结
Learn how to compute aggregate functions across all array elements using axis=None. This computes a single aggregate value for the entire array.

学习如何使用axis=None计算所有数组元素的聚合函数。这计算整个数组的单个聚合值。

### Core Concept / 核心概念
**axis=None**: Aggregates across all elements. sum(axis=None) returns a single scalar sum of all elements. This is the default behavior for most aggregate functions.

**axis=None**: 在所有元素上聚合。sum(axis=None)返回所有元素的单个标量和。这是大多数聚合函数的默认行为。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 2D Array / 创建二维数组

We create a 2×3 array. We'll compute the sum of all elements regardless of row or column structure.

我们创建一个2×3数组。我们将计算所有元素的和，而不管行或列结构。

```python
# 导入asarray函数 / Import asarray function
from numpy import asarray

# 创建二维数组 / Create 2D array
data = [[1, 2, 3], [4, 5, 6]]
data = asarray(data)
print("Array:")
print(data)
```

## Step 2 — Compute Sum Across All Elements / 计算所有元素的和

We use .sum(axis=None) to add all elements: 1+2+3+4+5+6 = 21. This treats the entire array as a single collection.

我们使用.sum(axis=None)来相加所有元素：1+2+3+4+5+6 = 21。这将整个数组视为单个集合。

```python
# 显示数组 / Display array
print("\nArray:")
print(data)

# 计算所有元素的和 / Sum all elements
result = data.sum(axis=None)
print("\nSum (axis=None) - aggregate all elements:")
print(result)
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Aggregating with axis=None computes a scalar result from all array elements. For matrix **A** ∈ ℝ^(m×n), sum(axis=None) = Σᵢ Σⱼ A_{ij}. This is the most fundamental aggregate operation.

使用axis=None聚合从所有数组元素计算标量结果。对于矩阵**A** ∈ ℝ^(m×n)，sum(axis=None) = Σᵢ Σⱼ A_{ij}。这是最基本的聚合操作。

• **ML Application / ML application**: In ML, computing single aggregate values is common: total loss across all samples, total number of parameters, or overall accuracy. axis=None is also default behavior—many functions use it implicitly when axis is not specified.

在ML中，计算单个聚合值很常见：所有样本的总损失、总参数数或整体准确度。axis=None也是默认行为——许多函数在未指定axis时隐式使用它。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 05_axis_0.ipynb** — Learn how to aggregate along axis 0 (down rows).

## Complete Code / 完整代码一览

```python
# --- Aggregate All Elements (axis=None) / 聚合所有元素 ---
from numpy import asarray

# Create 2D array / 创建二维数组
data = [[1, 2, 3], [4, 5, 6]]
data = asarray(data)
print("Array:")
print(data)

# Sum all elements / 求和所有元素
result = data.sum(axis=None)
print("\nSum (axis=None) - aggregate all elements:")
print(result)
```

---

### Axis 0

## File 5 of 6 / 第4章 — 第5个文件（共6个）

### Summary / 总结
Learn how to aggregate along axis 0 (down rows). This computes per-column aggregates, useful for column-wise statistics in datasets.

学习如何沿轴0（向下行）聚合。这计算每列聚合，对于数据集中的列统计很有用。

### Core Concept / 核心概念
**axis=0**: Aggregates down rows (along the vertical axis). For shape (m, n) array, result has shape (n,). Example: sum(axis=0) = [col1_sum, col2_sum, col3_sum].

**axis=0**: 沿行向下聚合（沿垂直轴）。对于形状(m, n)的数组，结果形状为(n,)。示例：sum(axis=0) = [col1_sum, col2_sum, col3_sum]。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 2D Array / 创建二维数组

We create a 2×3 array. We'll sum down each column (axis 0), giving us one result per column.

我们创建一个2×3数组。我们将沿每列向下求和(轴0)，给我们每列一个结果。

```python
# 导入asarray函数 / Import asarray function
from numpy import asarray

# 创建二维数组 / Create 2D array
data = [[1, 2, 3], [4, 5, 6]]
data = asarray(data)
print("Array:")
print(data)
```

## Step 2 — Sum Along axis=0 / 沿axis=0求和

We use .sum(axis=0) to add down each column: col1=[1+4], col2=[2+5], col3=[3+6] = [5, 7, 9].

我们使用.sum(axis=0)来沿每列向下求和：col1=[1+4]，col2=[2+5]，col3=[3+6] = [5, 7, 9]。

```python
# 显示数组 / Display array
print("\nArray:")
print(data)

# 沿axis=0求和 / Sum along axis 0
result = data.sum(axis=0)
print("\nSum (axis=0) - sum down each column:")
print(result)
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Axis 0 represents rows (dimension 0). Aggregating along axis 0 means collapsing the row dimension. For matrix **A** ∈ ℝ^(m×n), sum(axis=0) returns a vector of length n where element j = Σᵢ A_{ij}.

轴0代表行（维度0）。沿轴0聚合意味着折叠行维度。对于矩阵**A** ∈ ℝ^(m×n)，sum(axis=0)返回长度为n的向量，其中元素j = Σᵢ A_{ij}。

• **ML Application / ML application**: In ML, axis=0 aggregation computes per-feature statistics: mean per feature (for normalization), sum per feature (for importance), or count per column (for missing value detection). This is essential for feature analysis in tabular data.

在ML中，axis=0聚合计算每个特征的统计量：每个特征的平均值（用于规范化）、每个特征的总和（用于重要性）或每列的计数（用于缺失值检测）。这对于表格数据中的特征分析至关重要。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 05_axis_1.ipynb** — Learn how to aggregate along axis 1 (across columns).

## Complete Code / 完整代码一览

```python
# --- Aggregate Along axis=0 / 沿axis=0聚合 ---
from numpy import asarray

# Create 2D array / 创建二维数组
data = [[1, 2, 3], [4, 5, 6]]
data = asarray(data)
print("Array:")
print(data)

# Sum along axis 0 / 沿轴0求和
result = data.sum(axis=0)
print("\nSum (axis=0) - sum down each column:")
print(result)
```

---

### Axis 1

## File 6 of 6 / 第4章 — 第6个文件（共6个）

### Summary / 总结
Learn how to aggregate along axis 1 (across columns). This computes per-row aggregates, useful for row-wise statistics in datasets like sample-wise sums.

学习如何沿轴1（跨列）聚合。这计算每行聚合，对于数据集中的行统计（如样本和）很有用。

### Core Concept / 核心概念
**axis=1**: Aggregates across columns (along the horizontal axis). For shape (m, n) array, result has shape (m,). Example: sum(axis=1) = [row1_sum, row2_sum].

**axis=1**: 跨列聚合（沿水平轴）。对于形状(m, n)的数组，结果形状为(m,)。示例：sum(axis=1) = [row1_sum, row2_sum]。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 2D Array / 创建二维数组

We create a 2×3 array. We'll sum across each row (axis 1), giving us one result per row.

我们创建一个2×3数组。我们将沿每行横向求和(轴1)，给我们每行一个结果。

```python
# 导入asarray函数 / Import asarray function
from numpy import asarray

# 创建二维数组 / Create 2D array
data = [[1, 2, 3], [4, 5, 6]]
data = asarray(data)
print("Array:")
print(data)
```

## Step 2 — Sum Along axis=1 / 沿axis=1求和

We use .sum(axis=1) to add across each row: row1=[1+2+3], row2=[4+5+6] = [6, 15].

我们使用.sum(axis=1)来沿每行横向求和：row1=[1+2+3]，row2=[4+5+6] = [6, 15]。

```python
# 显示数组 / Display array
print("\nArray:")
print(data)

# 沿axis=1求和 / Sum along axis 1
result = data.sum(axis=1)
print("\nSum (axis=1) - sum across each row:")
print(result)
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Axis 1 represents columns (dimension 1). Aggregating along axis 1 means collapsing the column dimension. For matrix **A** ∈ ℝ^(m×n), sum(axis=1) returns a vector of length m where element i = Σⱼ A_{ij}.

轴1代表列（维度1）。沿轴1聚合意味着折叠列维度。对于矩阵**A** ∈ ℝ^(m×n)，sum(axis=1)返回长度为m的向量，其中元素i = Σⱼ A_{ij}。

• **ML Application / ML application**: In ML, axis=1 aggregation computes per-sample statistics: total feature value per sample (useful for magnitude/norm), average prediction per sample, or sum of feature importance per sample. This is essential for sample-wise analysis.

在ML中，axis=1聚合计算每个样本的统计量：每个样本的总特征值（用于幅度/范数）、每个样本的平均预测或每个样本的特征重要性总和。这对于样本分析至关重要。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

🏁 **End of Chapter 7** — You've mastered array aggregation with different axes!

## Complete Code / 完整代码一览

```python
# --- Aggregate Along axis=1 / 沿axis=1聚合 ---
from numpy import asarray

# Create 2D array / 创建二维数组
data = [[1, 2, 3], [4, 5, 6]]
data = asarray(data)
print("Array:")
print(data)

# Sum along axis 1 / 沿轴1求和
result = data.sum(axis=1)
print("\nSum (axis=1) - sum across each row:")
print(result)
```

---

### Chapter Summary

# Chapter 07 Summary / 第07章总结：Aggregate Functions

## Theme / 主题

Aggregate functions reduce arrays to single values or vectors. The critical concept here is the **axis** parameter, which controls whether you aggregate across rows, columns, or the entire array. Understanding axis semantics is essential for ML data processing.

聚合函数将数组减少到单个值或向量。这里的关键概念是**轴**参数，它控制您是否在行、列或整个数组中聚合。理解轴语义对于ML数据处理至关重要。

## Evolution / 演化路线

```
01_shape.ipynb
    └─ Understand (n_samples, n_features) shape (理解矩阵形状)
    
02_iterate_rows.ipynb
    └─ Loop over rows (按行遍历)
    
03_iterate_cols.ipynb
    └─ Loop over columns (按列遍历)
    
04_sum_all.ipynb
    └─ sum(axis=None) → scalar (总和所有元素)
    
05_sum_axis0.ipynb
    └─ sum(axis=0) → (n_features,) vector (沿行求和 = 每列一个值)
    
06_sum_axis1.ipynb
    └─ sum(axis=1) → (n_samples,) vector (沿列求和 = 每行一个值)
```

## Progression Logic / 进度逻辑

Axis semantics are learned by **starting concrete, ending abstract**:

1. **Shape inspection**: See (n_samples, n_features) structure
2. **Manual loops**: Iterate to understand what axis=0 and axis=1 mean
3. **No axis (None)**: Sum everything → one scalar
4. **axis=0**: Sum down columns → (n_features,) — useful for column statistics
5. **axis=1**: Sum across rows → (n_samples,) — useful for per-sample statistics

This trains your mental model: axis=0 means "collapse rows", axis=1 means "collapse columns".

轴语义通过**从具体开始，以抽象结束**来学习：

1. **形状检查**：查看(n_samples, n_features)结构
2. **手动循环**：迭代以理解axis=0和axis=1的含义
3. **无轴(None)**：求和所有内容→一个标量
4. **axis=0**：沿列求和→(n_features,) — 对列统计有用
5. **axis=1**：按行求和→(n_samples,) — 对每样本统计有用

这训练您的心智模型：axis=0表示"折叠行"，axis=1表示"折叠列"。

## ML Relevance / 机器学习相关性

In machine learning:
- **Column statistics** (`axis=0`): Compute feature means, stdevs for normalization → shape matches features
- **Row statistics** (`axis=1`): Compute per-sample scores, losses → shape matches samples
- **Batch processing**: When data comes in shape (batch_size, n_features), axis=0 operations aggregate across samples
- **Feature engineering**: Sum/mean features for new features

Confusing axes causes:
- Normalizing by wrong dimension
- Broadcasting shape mismatches
- Computing statistics on wrong subset

Getting axis right is foundational to avoiding bugs.

在机器学习中：
- **列统计**（`axis=0`）：计算特征均值、标准差以进行归一化 → 形状与特征匹配
- **行统计**（`axis=1`）：计算每样本的分数、损失 → 形状与样本匹配
- **批处理**：当数据以(batch_size, n_features)形状出现时，axis=0操作在样本中聚合
- **特征工程**：求和/平均特征以获得新特征

混淆轴会导致：
- 按错误的维度进行归一化
- 广播形状不匹配
- 在错误的子集上计算统计

获得正确的轴是避免bug的基础。

---
