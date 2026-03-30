# 线性代数与机器学习
## Chapter 18

---

### Vector Mean

# 18.1 — Vector Mean / 向量均值

**Chapter 18 — File 1 of 8 / 第18章 — 第1个文件（共8个）**

## Summary / 总结

Learn to compute the mean (average) of a vector using NumPy's `mean()` function. The mean is a fundamental statistic that represents the central tendency of data.

学习使用NumPy的 `mean()` 函数计算向量的均值（平均值）。均值是代表数据中心趋势的基本统计量。

## Core Formula / 核心公式

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

where $n$ is the number of elements and $x_i$ are the vector elements.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import array creation and mean calculation functions
# 导入数组创建和均值计算函数
from numpy import array
from numpy import mean
```

## Step 2 — Create a Vector / 创建向量

```python
# Define a simple vector with 6 elements
# 定义一个包含6个元素的简单向量
v = array([1, 2, 3, 4, 5, 6])
print(f"Vector: {v}")
```

## Step 3 — Calculate the Mean / 计算均值

```python
# Compute the mean of the vector
# Mean = (1+2+3+4+5+6)/6 = 21/6 = 3.5
# 计算向量的均值
# 均值 = (1+2+3+4+5+6)/6 = 21/6 = 3.5
result = mean(v)
print(f"Mean of vector: {result}")
```

## Learning Notes / 学习笔记

- **Math Essence**: The mean is the sum of all elements divided by the count. It represents the "center of mass" of the data distribution.
  
  **数学本质**：均值是所有元素之和除以个数。它代表数据分布的"质心"。

- **ML Application**: The mean is used for feature scaling (zero-centering), baseline models, and understanding dataset properties before training.
  
  **ML应用**：均值用于特征缩放（零中心化）、基线模型和训练前理解数据集属性。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `02_matrix_mean.ipynb` — Computing column and row means for matrices

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
from numpy import array
from numpy import mean

# --- Vector Mean Computation / 向量均值计算 ---
v = array([1, 2, 3, 4, 5, 6])
result = mean(v)
print(result)
```

---

### Matrix Mean

# 18.2 — Matrix Mean / 矩阵均值

**Chapter 18 — File 2 of 8 / 第18章 — 第2个文件（共8个）**

## Summary / 总结

Compute column-wise and row-wise means of matrices using the `axis` parameter. This is essential for normalizing features in machine learning.

使用 `axis` 参数计算矩阵的列均值和行均值。这对于机器学习中的特征归一化至关重要。

## Core Formula / 核心公式

**Column mean**: $\bar{x}_j = \frac{1}{m}\sum_{i=1}^{m} x_{ij}$

**Row mean**: $\bar{x}_i = \frac{1}{n}\sum_{j=1}^{n} x_{ij}$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import array and mean function
# 导入数组和均值函数
from numpy import array
from numpy import mean
```

## Step 2 — Create a Matrix / 创建矩阵

```python
# Define a 2x6 matrix
# 定义一个2x6的矩阵
M = array([[1, 2, 3, 4, 5, 6],
           [1, 2, 3, 4, 5, 6]])
print("Matrix M:")
print(M)
```

## Step 3 — Calculate Column Means / 计算列均值

```python
# Compute mean along axis=0 (columns)
# axis=0 means compute mean down each column
# 沿axis=0计算均值（沿列方向）
# axis=0 表示沿着每一列计算均值
col_mean = mean(M, axis=0)
print(f"Column means: {col_mean}")
```

## Step 4 — Calculate Row Means / 计算行均值

```python
# Compute mean along axis=1 (rows)
# axis=1 means compute mean across each row
# 沿axis=1计算均值（沿行方向）
# axis=1 表示沿着每一行计算均值
row_mean = mean(M, axis=1)
print(f"Row means: {row_mean}")
```

## Learning Notes / 学习笔记

- **Math Essence**: The `axis` parameter determines the direction of averaging. axis=0 aggregates rows (produces column statistics), axis=1 aggregates columns (produces row statistics).
  
  **数学本质**：`axis` 参数确定平均方向。axis=0 聚合行（产生列统计），axis=1 聚合列（产生行统计）。

- **ML Application**: Feature normalization requires computing column means to center each feature independently. This is the first step in standardization.
  
  **ML应用**：特征归一化需要计算列均值来独立地中心化每个特征。这是标准化的第一步。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `03_vector_variance.ipynb` — Computing variance of vectors

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
from numpy import array
from numpy import mean

# --- Matrix Mean Computation / 矩阵均值计算 ---
M = array([[1, 2, 3, 4, 5, 6],
           [1, 2, 3, 4, 5, 6]])
col_mean = mean(M, axis=0)
print(col_mean)

# --- Row Mean Computation / 行均值计算 ---
row_mean = mean(M, axis=1)
print(row_mean)
```

---

### Vector Variance

# 18.3 — Vector Variance / 向量方差

**Chapter 18 — File 3 of 8 / 第18章 — 第3个文件（共8个）**

## Summary / 总结

Calculate the variance of a vector, which measures how spread out the values are from the mean. The `ddof=1` parameter computes sample variance (unbiased estimator).

计算向量的方差，衡量值与均值的分散程度。`ddof=1` 参数计算样本方差（无偏估计量）。

## Core Formula / 核心公式

**Sample Variance**: $$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

**Population Variance**: $$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import array and variance function
# 导入数组和方差函数
from numpy import array
from numpy import var
```

## Step 2 — Create a Vector / 创建向量

```python
# Define a vector
# 定义一个向量
v = array([1, 2, 3, 4, 5, 6])
print(f"Vector: {v}")
```

## Step 3 — Calculate Sample Variance / 计算样本方差

```python
# Compute sample variance with ddof=1 (degrees of freedom correction)
# ddof=1 divides by (n-1) instead of n, providing unbiased estimate
# 计算样本方差，ddof=1（自由度修正）
# ddof=1 除以(n-1)而不是n，提供无偏估计
result = var(v, ddof=1)
print(f"Sample Variance (ddof=1): {result}")

# For comparison: population variance (ddof=0, default)
# 对比：总体方差（ddof=0，默认值）
pop_var = var(v, ddof=0)
print(f"Population Variance (ddof=0): {pop_var}")
```

## Learning Notes / 学习笔记

- **Math Essence**: Variance measures the average squared distance from the mean. Sample variance (ddof=1) is preferred for statistical estimation because it provides an unbiased estimator.
  
  **数学本质**：方差衡量与均值的平均平方距离。样本方差（ddof=1）因其提供无偏估计量而优于总体方差。

- **ML Application**: Variance is used for feature scaling and understanding data spread. High variance features dominate distance-based algorithms (KNN, SVM), so normalization is critical.
  
  **ML应用**：方差用于特征缩放和理解数据分布。高方差特征在基于距离的算法（KNN、SVM）中占主导，所以归一化至关重要。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `04_matrix_variance.ipynb` — Computing variance for matrices

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
from numpy import array
from numpy import var

# --- Vector Variance Computation / 向量方差计算 ---
v = array([1, 2, 3, 4, 5, 6])
result = var(v, ddof=1)
print(result)
```

---

### Matrix Stdev

# 18.5 — Matrix Standard Deviation / 矩阵标准差

**Chapter 18 — File 5 of 8 / 第18章 — 第5个文件（共8个）**

## Summary / 总结

Calculate the standard deviation of a matrix. Standard deviation is the square root of variance, expressed in the same units as the original data.

计算矩阵的标准差。标准差是方差的平方根，用与原始数据相同的单位表示。

## Core Formula / 核心公式

**Standard Deviation**: $$s = \sqrt{s^2} = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import array and standard deviation function
# 导入数组和标准差函数
from numpy import array
from numpy import std
```

## Step 2 — Create a Matrix / 创建矩阵

```python
# Define a 2x6 matrix
# 定义一个2x6的矩阵
M = array([[1, 2, 3, 4, 5, 6],
           [1, 2, 3, 4, 5, 6]])
print("Matrix M:")
print(M)
```

## Step 3 — Calculate Column Standard Deviations / 计算列标准差

```python
# Compute standard deviation along axis=0 (columns) with ddof=1
# 沿axis=0计算标准差（列），ddof=1用于样本标准差
col_std = std(M, ddof=1, axis=0)
print(f"Column standard deviations: {col_std}")
```

## Step 4 — Calculate Row Standard Deviations / 计算行标准差

```python
# Compute standard deviation along axis=1 (rows) with ddof=1
# 沿axis=1计算标准差（行），ddof=1用于样本标准差
row_std = std(M, ddof=1, axis=1)
print(f"Row standard deviations: {row_std}")
```

## Step 5 — Relationship between Variance and StDev / 方差与标准差的关系

```python
# Standard deviation is the square root of variance
# 标准差是方差的平方根
from numpy import var, sqrt
col_var = var(M, ddof=1, axis=0)
col_std_from_var = sqrt(col_var)
print(f"Std from variance: {col_std_from_var}")
print(f"Std directly: {col_std}")
print(f"They are equal: {all(col_std == col_std_from_var)}")
```

## Learning Notes / 学习笔记

- **Math Essence**: Standard deviation is the square root of variance. Unlike variance, it has the same unit as the original data, making it more interpretable for humans.
  
  **数学本质**：标准差是方差的平方根。与方差不同，它与原始数据具有相同的单位，对人类更容易解释。

- **ML Application**: Standard deviation is used in normalization (z-score normalization: $(x - \mu) / \sigma$). Most preprocessing pipelines use both mean and standard deviation for feature scaling.
  
  **ML应用**：标准差用于归一化（z分数归一化：$(x - \mu) / \sigma$）。大多数预处理管道使用均值和标准差进行特征缩放。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `06_vector_covariance.ipynb` — Computing covariance between vectors

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
from numpy import array
from numpy import std

# --- Matrix Standard Deviation Computation / 矩阵标准差计算 ---
M = array([[1, 2, 3, 4, 5, 6],
           [1, 2, 3, 4, 5, 6]])
col_std = std(M, ddof=1, axis=0)
print(col_std)

# --- Row Standard Deviation Computation / 行标准差计算 ---
row_std = std(M, ddof=1, axis=1)
print(row_std)
```

---

### Vector Covariance

# 18.6 — Vector Covariance / 向量协方差

**Chapter 18 — File 6 of 8 / 第18章 — 第6个文件（共8个）**

## Summary / 总结

Calculate the covariance between two vectors. Covariance measures how two variables change together, indicating the strength and direction of their linear relationship.

计算两个向量之间的协方差。协方差衡量两个变量如何一起变化，表示它们线性关系的强度和方向。

## Core Formula / 核心公式

$$\text{Cov}(X, Y) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$

Positive covariance: variables increase together. Negative covariance: one increases as the other decreases.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import array and covariance function
# 导入数组和协方差函数
from numpy import array
from numpy import cov
```

## Step 2 — Create Two Vectors / 创建两个向量

```python
# Define two vectors with opposite trends
# x increases from 1 to 9
# y decreases from 9 to 1
# 定义两个趋势相反的向量
# x从1增加到9
# y从9减少到1
x = array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = array([9, 8, 7, 6, 5, 4, 3, 2, 1])
print(f"x: {x}")
print(f"y: {y}")
```

## Step 3 — Calculate Covariance / 计算协方差

```python
# Compute covariance matrix using cov()
# cov(x, y) returns a 2x2 matrix:
# [[var(x), cov(x,y)]
#  [cov(y,x), var(y)]]
# The off-diagonal element [0,1] is the covariance
# 使用cov()计算协方差矩阵
# cov(x, y)返回2x2矩阵
# 非对角元素[0,1]是协方差
cov_matrix = cov(x, y)
print("Covariance matrix:")
print(cov_matrix)

# Extract the covariance between x and y
# 提取x和y之间的协方差
Sigma = cov_matrix[0, 1]
print(f"\nCovariance between x and y: {Sigma}")
```

## Step 4 — Interpret the Result / 解释结果

```python
# The negative covariance indicates inverse relationship
# As x increases, y decreases
# 负协方差表示反向关系
# 当x增加时，y减少
if Sigma < 0:
    print("Negative covariance: variables have inverse relationship")
elif Sigma > 0:
    print("Positive covariance: variables move together")
else:
    print("Zero covariance: no linear relationship")
```

## Learning Notes / 学习笔记

- **Math Essence**: Covariance extends variance to two variables. It measures the joint variability. The covariance matrix (for multiple variables) contains variances on the diagonal and covariances on off-diagonals.
  
  **数学本质**：协方差将方差扩展到两个变量。它衡量联合变异性。协方差矩阵的对角线包含方差，非对角线包含协方差。

- **ML Application**: The covariance matrix is fundamental in PCA (Principal Component Analysis), linear discriminant analysis (LDA), and other unsupervised learning methods. It captures the structure of feature relationships.
  
  **ML应用**：协方差矩阵在主成分分析（PCA）、线性判别分析（LDA）和其他无监督学习方法中至关重要。它捕捉特征关系的结构。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `07_vector_correlation.ipynb` — Computing correlation between vectors

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
from numpy import array
from numpy import cov

# --- Vector Covariance Computation / 向量协方差计算 ---
x = array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = array([9, 8, 7, 6, 5, 4, 3, 2, 1])
Sigma = cov(x, y)[0, 1]
print(Sigma)
```

---

### Vector Correlation

# 18.7 — Vector Correlation / 向量相关性

**Chapter 18 — File 7 of 8 / 第18章 — 第7个文件（共8个）**

## Summary / 总结

Calculate the correlation coefficient between two vectors. Correlation is a normalized measure of covariance, ranging from -1 to 1, making it scale-invariant and easier to interpret.

计算两个向量之间的相关系数。相关性是协方差的归一化度量，范围从-1到1，使其尺度不变且易于解释。

## Core Formula / 核心公式

**Pearson Correlation**: $$r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

where $\sigma_X$ and $\sigma_Y$ are the standard deviations of X and Y.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import array and correlation function
# 导入数组和相关性函数
from numpy import array
from numpy import corrcoef
```

## Step 2 — Create Two Vectors / 创建两个向量

```python
# Define two vectors
# 定义两个向量
x = array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = array([9, 8, 7, 6, 5, 4, 3, 2, 1])
print(f"x: {x}")
print(f"y: {y}")
```

## Step 3 — Calculate Correlation / 计算相关性

```python
# Compute correlation matrix using corrcoef()
# corrcoef(x, y) returns a 2x2 matrix:
# [[1, r(x,y)]
#  [r(y,x), 1]]
# Diagonal is always 1 (variable correlates perfectly with itself)
# 使用corrcoef()计算相关矩阵
# 对角线总是1（变量与自身完全相关）
corr_matrix = corrcoef(x, y)
print("Correlation matrix:")
print(corr_matrix)

# Extract the correlation between x and y
# 提取x和y之间的相关性
corr = corr_matrix[0, 1]
print(f"\nCorrelation coefficient: {corr}")
```

## Step 4 — Interpret Correlation / 解释相关性

```python
# Correlation ranges from -1 to 1
# -1: perfect negative correlation (perfect inverse)
# 0: no linear correlation
# 1: perfect positive correlation (perfect agreement)
# 相关性范围从-1到1
# -1：完全负相关
# 0：没有线性相关
# 1：完全正相关
if corr < -0.7:
    print("Strong negative correlation: variables move in opposite directions")
elif corr < -0.3:
    print("Moderate negative correlation")
elif corr < 0.3:
    print("Weak or no correlation")
elif corr < 0.7:
    print("Moderate positive correlation")
else:
    print("Strong positive correlation: variables move together")

print(f"\nInterpretation: r = {corr:.4f}")
```

## Learning Notes / 学习笔记

- **Math Essence**: Correlation is covariance divided by the product of standard deviations. This normalization makes correlation scale-invariant and bounded in [-1, 1], making it easier to compare relationships across different datasets.
  
  **数学本质**：相关性是协方差除以标准差的乘积。这个归一化使相关性尺度不变且有界在[-1, 1]，便于比较不同数据集间的关系。

- **ML Application**: Correlation is used in feature selection and exploratory data analysis to identify multicollinearity. High correlation between features (>0.8 or <-0.8) suggests redundancy and feature removal may be beneficial.
  
  **ML应用**：相关性用于特征选择和探索性数据分析，识别多重共线性。高相关性（>0.8或<-0.8）表明冗余，移除特征可能有益。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `08_covariance_matrix.ipynb` — Computing covariance matrix for multiple variables

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
from numpy import array
from numpy import corrcoef

# --- Vector Correlation Computation / 向量相关性计算 ---
x = array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = array([9, 8, 7, 6, 5, 4, 3, 2, 1])
corr = corrcoef(x, y)[0, 1]
print(corr)
```

---

### Covariance Matrix

# 18.8 — Covariance Matrix / 协方差矩阵

**Chapter 18 — File 8 of 8 / 第18章 — 第8个文件（共8个）**

## Summary / 总结

Compute the covariance matrix for multiple variables. The covariance matrix is a square symmetric matrix that captures all pairwise covariances and variances of a dataset.

计算多个变量的协方差矩阵。协方差矩阵是一个方形对称矩阵，捕捉数据集的所有成对协方差和方差。

## Core Formula / 核心公式

**Covariance Matrix**: $$\Sigma = \begin{bmatrix} \text{Var}(X_1) & \text{Cov}(X_1, X_2) & \cdots & \text{Cov}(X_1, X_n) \\ \text{Cov}(X_2, X_1) & \text{Var}(X_2) & \cdots & \text{Cov}(X_2, X_n) \\ \vdots & \vdots & \ddots & \vdots \\ \text{Cov}(X_n, X_1) & \text{Cov}(X_n, X_2) & \cdots & \text{Var}(X_n) \end{bmatrix}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import array and covariance function
# 导入数组和协方差函数
from numpy import array
from numpy import cov
```

## Step 2 — Create a Multivariate Dataset / 创建多变量数据集

```python
# Define a 5x3 dataset (5 observations, 3 features)
# 定义5x3数据集（5个观测，3个特征）
X = array([[1, 5, 8],
           [3, 5, 11],
           [2, 4, 9],
           [3, 6, 10],
           [1, 5, 10]])
print("Dataset X (5 observations, 3 features):")
print(X)
print(f"Shape: {X.shape}")
```

## Step 3 — Calculate Covariance Matrix / 计算协方差矩阵

```python
# Compute covariance matrix
# X.T transposes the matrix so each row becomes a variable
# cov() expects each row to be a variable
# 计算协方差矩阵
# X.T转置矩阵，使每行成为一个变量
# cov()期望每行是一个变量
Sigma = cov(X.T)
print("Covariance Matrix:")
print(Sigma)
print(f"Shape: {Sigma.shape}")
```

## Step 4 — Interpret the Covariance Matrix / 解释协方差矩阵

```python
# The diagonal contains variances of each feature
# 对角线包含每个特征的方差
print("Diagonal (variances of each feature):")
for i in range(Sigma.shape[0]):
    print(f"  Var(X_{i+1}) = {Sigma[i, i]:.4f}")

# Off-diagonal elements contain covariances between features
# 非对角线元素包含特征间的协方差
print("\nOff-diagonal (covariances between features):")
for i in range(Sigma.shape[0]):
    for j in range(i+1, Sigma.shape[1]):
        print(f"  Cov(X_{i+1}, X_{j+1}) = {Sigma[i, j]:.4f}")
```

## Step 5 — Properties of the Covariance Matrix / 协方差矩阵的性质

```python
# The covariance matrix is symmetric: Cov(X_i, X_j) = Cov(X_j, X_i)
# 协方差矩阵是对称的
print("Is the matrix symmetric?")
import numpy as np
is_symmetric = np.allclose(Sigma, Sigma.T)
print(f"  {is_symmetric}")

# The matrix is positive semi-definite (all eigenvalues >= 0)
# Used in PCA and other algorithms
# 矩阵是半正定的（所有特征值≥0）
eigenvalues = np.linalg.eigvalsh(Sigma)
print(f"\nEigenvalues (all should be >= 0): {eigenvalues}")
```

## Learning Notes / 学习笔记

- **Math Essence**: The covariance matrix is a symmetric matrix where diagonal entries are variances and off-diagonal entries are covariances. It's positive semi-definite, meaning all eigenvalues are non-negative. This property is crucial for PCA and other linear algebra methods.
  
  **数学本质**：协方差矩阵是对称矩阵，对角元素是方差，非对角元素是协方差。它是半正定的，意味着所有特征值非负。这对PCA和其他线性代数方法至关重要。

- **ML Application**: The covariance matrix is the foundation of Principal Component Analysis (PCA). Eigendecomposition of the covariance matrix reveals the principal directions of data variance, used for dimensionality reduction and feature extraction.
  
  **ML应用**：协方差矩阵是主成分分析（PCA）的基础。协方差矩阵的特征分解揭示数据方差的主方向，用于降维和特征提取。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `np.linalg` | 线性代数运算 | Linear algebra operations |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `../chapter_19/01_pca.ipynb` — Principal Component Analysis (PCA)

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
from numpy import array
from numpy import cov

# --- Covariance Matrix Computation / 协方差矩阵计算 ---
X = array([[1, 5, 8],
           [3, 5, 11],
           [2, 4, 9],
           [3, 6, 10],
           [1, 5, 10]])
Sigma = cov(X.T)
print(Sigma)
```

---

### Chapter Summary

# Chapter 18 Summary / 第18章总结：Statistics

## Theme / 主题

Statistics quantify data properties: location (mean), spread (variance, stdev), and relationships (covariance, correlation). This chapter covers computing these statistics on vectors and matrices, with emphasis on understanding axis semantics—computing stats per-column (axis=0) vs. per-row (axis=1). The foundation for data preprocessing, feature normalization, and understanding covariance structure.

统计量量化数据属性：位置（均值）、传播（方差、标准差）和关系（协方差、相关性）。本章涵盖在向量和矩阵上计算这些统计量，强调理解轴语义——按列计算统计量（axis=0）vs.按行（axis=1）。数据预处理、特征归一化和理解协方差结构的基础。

## Evolution / 演化路线

```
01_vector_mean.ipynb
    └─ Mean of 1D array (一维数组的均值)
    
02_matrix_mean_all.ipynb
    └─ Mean of all elements (所有元素的均值)
    
03_matrix_mean_axis.ipynb
    └─ Mean per column (axis=0) and per row (axis=1) (按列和按行的均值)
    
04_vector_variance.ipynb
    └─ Variance of 1D array (一维数组的方差)
    
05_matrix_variance.ipynb
    └─ Variance per feature / per sample (按特征/按样本的方差)
    
06_standard_deviation.ipynb
    └─ Stdev = sqrt(variance) (标准差)
    
07_covariance_pair.ipynb
    └─ Covariance between two variables (两个变量之间的协方差)
    
08_covariance_matrix.ipynb
    └─ Full covariance structure (完整协方差结构)
```

## Progression Logic / 进度逻辑

Statistics are learned from **single number → axis-wise → pairwise relationships**:

1. **Mean**: Average of all values (scalar), then per-column and per-row
2. **Variance**: Spread around mean → stdev is square root
3. **Covariance**: How two variables move together → tells about linear relationship
4. **Covariance matrix**: Full structure of all pairwise relationships

Key insight: **axis=0 gives per-feature statistics** (what we typically use for normalization). **axis=1 gives per-sample statistics** (less common, but useful for understanding individual samples).

统计量从**单个数字→轴向→成对关系**学习：

1. **均值**：所有值的平均值（标量），然后按列和按行
2. **方差**：围绕均值的传播→标准差是平方根
3. **协方差**：两个变量如何一起移动→告诉关于线性关系
4. **协方差矩阵**：所有成对关系的完整结构

关键见解：**axis=0给出每特征统计量**（我们通常用于归一化的）。**axis=1给出每样本统计量**（不太常见，但对理解单个样本有用）。

## ML Relevance / 机器学习相关性

In machine learning:
- **Mean (per feature, axis=0)**:
  - Feature centering: `X_centered = X - X.mean(axis=0)` (critical for PCA)
  - Bias in linear models: offset learned separately
  - Batch normalization: subtract batch mean

- **Variance/Stdev (per feature, axis=0)**:
  - Feature scaling: `X_scaled = X / X.std(axis=0)` (standardization)
  - Identifying low-variance features: remove non-informative features
  - Regularization: helps optimization (better-conditioned loss)

- **Covariance (pairwise)**:
  - Identifies which features are correlated
  - High covariance = redundancy = multicollinearity problem
  - Solution: PCA or feature selection to decorrelate

- **Covariance matrix** (full structure):
  - Cov(X) = X^T · X / n is (n_features, n_features)
  - Symmetric positive semi-definite
  - Eigendecomposition of Cov = PCA
  - Used in Gaussian distributions, covariance estimation

- **Normalization for training**:
  - Standardize: `(X - mean) / std` → zero mean, unit variance
  - Whitening: `X · Q · Λ^(-1/2)` (decorrelate via eigendecomposition)
  - Batch norm: normalize within minibatches

- **Covariance in probabilistic models**:
  - Gaussian: parameterized by mean and covariance
  - Gaussian process: covariance function between data points
  - Multivariate normal: sampling requires covariance matrix

**Why it matters**: Covariance structure determines both optimization difficulty and feature redundancy. Understanding covariance is key to preprocessing and model diagnostics.

在机器学习中：
- **均值（每特征，axis=0）**：
  - 特征中心化：`X_centered = X - X.mean(axis=0)`（对PCA至关重要）
  - 线性模型中的偏差：单独学习的偏移
  - 批归一化：减去批均值

- **方差/标准差（每特征，axis=0）**：
  - 特征缩放：`X_scaled = X / X.std(axis=0)`（标准化）
  - 识别低方差特征：删除非信息特征
  - 正则化：帮助优化（更好条件的损失）

- **协方差（成对）**：
  - 识别哪些特征相关
  - 高协方差=冗余=多重共线性问题
  - 解决方案：PCA或特征选择以去相关

- **协方差矩阵**（完整结构）：
  - Cov(X) = X^T · X / n是(n_features, n_features)
  - 对称半正定
  - Cov的特征分解= PCA
  - 用于高斯分布、协方差估计

- **训练的归一化**：
  - 标准化：`(X - mean) / std` → 零均值、单位方差
  - 白化：`X · Q · Λ^(-1/2)`（通过特征分解去相关）
  - 批归一化：在小批中归一化

- **概率模型中的协方差**：
  - 高斯：由均值和协方差参数化
  - 高斯过程：数据点之间的协方差函数
  - 多变量正态：采样需要协方差矩阵

**为什么重要**：协方差结构决定了优化难度和特征冗余。理解协方差是预处理和模型诊断的关键。

---
