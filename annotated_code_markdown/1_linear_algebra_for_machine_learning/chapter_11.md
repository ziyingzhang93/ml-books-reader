# 线性代数与机器学习 / Linear Algebra for Machine Learning
## Chapter 11

---

### Triangular Matrix



---

### Diagonal Matrix

# 02 — Diagonal Matrices / 对角矩阵

**Chapter 11 — File 2 of 4 / 第11章 — 第2个文件（共4个）**

## Summary / 总结

Learn to extract diagonal elements from a matrix and create a diagonal matrix from a vector.

学习如何从矩阵中提取对角元素，以及从向量创建对角矩阵。

## Core Formula / 核心公式

A diagonal matrix has non-zero elements only on the main diagonal:
$$\mathbf{D} = \begin{bmatrix} d_1 & 0 & \cdots & 0 \\ 0 & d_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & d_n \end{bmatrix}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Required Functions / 导入所需函数

We import the `array` function and the `diag` function from NumPy.

```python
# Import array to create matrices
# 导入 array 来创建矩阵
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Import diag to extract or create diagonal matrices
# 导入 diag 来提取或创建对角矩阵
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import diag
```

## Step 2 — Create a Matrix / 创建矩阵

We create a square matrix.

```python
# Create a 3x3 matrix M
# 创建一个 3x3 矩阵 M
M = array([[1, 2, 3],[1, 2, 3],[1, 2, 3]])

# Print the original matrix
# 打印原始矩阵
# 打印输出 / Print output
print(M)
```

## Step 3 — Extract Diagonal Elements / 提取对角元素

We use `diag()` on a matrix to extract the diagonal elements as a vector.

```python
# Extract diagonal elements from matrix M
# Result: [1, 2, 3] (the diagonal elements)
# 从矩阵 M 中提取对角元素
# 结果：[1, 2, 3]（对角元素）
d = diag(M)

# Print the diagonal vector
# 打印对角向量
# 打印输出 / Print output
print(d)
```

## Step 4 — Create Diagonal Matrix from Vector / 从向量创建对角矩阵

We use `diag()` on a vector to create a diagonal matrix.

```python
# Create a diagonal matrix from the diagonal vector d
# Result: 3x3 matrix with d on the diagonal and 0s elsewhere
# 从对角向量 d 创建对角矩阵
# 结果：3x3 矩阵，对角线上为 d，其他地方为 0
D = diag(d)

# Print the diagonal matrix
# 打印对角矩阵
# 打印输出 / Print output
print(D)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Diagonal matrices have non-zero elements only on the main diagonal. They are the simplest form of matrices and have efficient computation for multiplication.
- **ML Application / 机器学习应用**: Diagonal matrices appear in eigenvalue decomposition, covariance matrices, and scaling operations. They represent independent operations along each dimension.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `03_identity_matrix.ipynb` — Create and use identity matrices

## Complete Code / 完整代码一览

```python
# --- Diagonal Matrices / 对角矩阵 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import diag
M = array([[1, 2, 3],[1, 2, 3],[1, 2, 3]])
# 打印输出 / Print output
print(M)
d = diag(M)
# 打印输出 / Print output
print(d)
D = diag(d)
# 打印输出 / Print output
print(D)
```

---

### Identity Matrix

# 03 — Identity Matrix / 单位矩阵

**Chapter 11 — File 3 of 4 / 第11章 — 第3个文件（共4个）**

## Summary / 总结

Learn to create the identity matrix, a special diagonal matrix with 1s on the diagonal. It's fundamental to matrix operations and matrix inversion.

学习创建单位矩阵，这是一个特殊的对角矩阵，对角线上全为 1。它是矩阵运算和矩阵求逆的基础。

## Core Formula / 核心公式

The identity matrix $\mathbf{I}$ is a diagonal matrix with all 1s:
$$\mathbf{I}_n = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Required Functions / 导入所需函数

We import the `identity` function from NumPy.

```python
# Import identity function to create identity matrices
# 导入 identity 函数来创建单位矩阵
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import identity
```

## Step 2 — Create a 3x3 Identity Matrix / 创建 3x3 单位矩阵

We create a 3x3 identity matrix by passing the size to the `identity()` function.

```python
# Create a 3x3 identity matrix
# Diagonal elements are 1, all others are 0
# 创建一个 3x3 单位矩阵
# 对角线元素为 1，其他所有元素为 0
I = identity(3)

# Print the identity matrix
# 打印单位矩阵
# 打印输出 / Print output
print(I)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: The identity matrix is the multiplicative identity for matrices. Multiplying any matrix by I gives the same matrix: $A \times I = A$ and $I \times A = A$. It represents "no transformation."
- **ML Application / 机器学习应用**: The identity matrix is fundamental in computing matrix inverses, as the goal of solving $Ax = b$ is often to find $x = A^{-1}b$. It also appears in regularization and constraint optimization.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `04_orthogonal_matrix.ipynb` — Work with orthogonal matrices

## Complete Code / 完整代码一览

```python
# --- Identity Matrix / 单位矩阵 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import identity
I = identity(3)
# 打印输出 / Print output
print(I)
```

---

### Orthogonal Matrix

# 04 — Orthogonal Matrix / 正交矩阵

**Chapter 11 — File 4 of 4 / 第11章 — 第4个文件（共4个）**

## Summary / 总结

Learn about orthogonal matrices, where the transpose equals the inverse. These matrices preserve distances and angles, making them crucial in numerical computations.

学习正交矩阵，其中转置等于逆矩阵。这些矩阵保持距离和角度，使其在数值计算中至关重要。

## Core Formula / 核心公式

For an orthogonal matrix $\mathbf{Q}$:
$$\mathbf{Q}^T \times \mathbf{Q} = \mathbf{I}$$
$$\mathbf{Q}^{-1} = \mathbf{Q}^T$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Required Functions / 导入所需函数

We import the `array` function and the `inv` function from `numpy.linalg`.

```python
# Import array to create matrices
# 导入 array 来创建矩阵
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Import inv to compute matrix inverse
# 导入 inv 来计算矩阵逆
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import inv
```

## Step 2 — Create an Orthogonal Matrix / 创建正交矩阵

We create a simple 2x2 orthogonal matrix (reflection matrix).

```python
# Create a simple 2x2 orthogonal matrix Q (reflection across x-axis)
# This is an orthogonal matrix because Q^T * Q = I
# 创建一个简单的 2x2 正交矩阵 Q（关于 x 轴的反射）
# 这是正交矩阵，因为 Q^T * Q = I
Q = array([[1, 0],[0, -1]])

# Print the orthogonal matrix
# 打印正交矩阵
# 打印输出 / Print output
print(Q)
```

## Step 3 — Compute the Inverse / 计算逆矩阵

We compute the matrix inverse using `inv()`.

```python
# Compute the inverse of Q
# 计算 Q 的逆矩阵
V = inv(Q)

# Print the inverse
# 打印逆矩阵
# 打印输出 / Print output
print(V)
```

## Step 4 — Compute the Transpose / 计算转置

We compute the transpose of Q using the `.T` attribute.

```python
# Compute the transpose of Q using .T
# For orthogonal matrices, Q^T should equal Q^-1
# 使用 .T 计算 Q 的转置
# 对于正交矩阵，Q^T 应该等于 Q^-1
# 打印输出 / Print output
print(Q.T)
```

## Step 5 — Verify Orthogonality / 验证正交性

We verify that $Q^T \times Q = I$ by computing the product.

```python
# Verify orthogonality: Q^T * Q should equal identity matrix I
# 验证正交性：Q^T * Q 应该等于单位矩阵 I
I = Q.dot(Q.T)

# Print the result (should be identity matrix)
# 打印结果（应该是单位矩阵）
# 打印输出 / Print output
print(I)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Orthogonal matrices have the property that their transpose equals their inverse. They preserve lengths and angles, making them numerically stable and efficient for computations.
- **ML Application / 机器学习应用**: Orthogonal matrices appear in QR decomposition, singular value decomposition (SVD), and principal component analysis (PCA). They are essential for stable numerical algorithms.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: End of Chapter 11 - You have completed all special matrix operations!

## Complete Code / 完整代码一览

```python
# --- Orthogonal Matrix / 正交矩阵 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import inv
Q = array([[1, 0],[0, -1]])
# 打印输出 / Print output
print(Q)
V = inv(Q)
# 打印输出 / Print output
print(Q.T)
# 打印输出 / Print output
print(V)
I = Q.dot(Q.T)
# 打印输出 / Print output
print(I)
```

---

### Chapter Summary / 章节总结

# Chapter 11 Summary / 第11章总结：Special Matrices

## Theme / 主题

Some matrices have special structure that makes them computationally efficient or mathematically elegant. Triangular, diagonal, identity, and orthogonal matrices each have unique properties. Understanding these structures is key to efficient numerical algorithms and deep learning optimization.

某些矩阵有特殊的结构，使它们在计算上更高效或数学上更优雅。三角形、对角线、单位和正交矩阵各有独特的属性。理解这些结构是高效数值算法和深度学习优化的关键。

## Evolution / 演化路线

```
01_triangular_matrices.ipynb
    └─ Upper and lower triangular (上三角和下三角矩阵)
    
02_diagonal_matrices.ipynb
    └─ Diagonal: zeros everywhere except main diagonal (只有对角线有值)
    
03_identity_matrix.ipynb
    └─ Identity: diagonal 1s, multiplication is no-op (对角线为1，乘法不改变)
    
04_orthogonal_matrices.ipynb
    └─ Orthogonal: Q^T · Q = I, preserves length and angles (保持长度和角度)
```

## Progression Logic / 进度逻辑

Special matrices are presented by **increasing geometric constraint**:

1. **Triangular**: Only half the matrix has nonzero entries
2. **Diagonal**: Only the main diagonal has entries
3. **Identity**: Diagonal matrix with all 1s — "do nothing"
4. **Orthogonal**: Rows/columns are orthonormal — preserves geometry perfectly

Each adds more structure, leading to stronger properties and faster computations. Identity is the "neutral element" of matrix multiplication, while orthogonal matrices preserve distances and angles (isometries).

特殊矩阵通过**增加几何约束**呈现：

1. **三角形**：只有一半的矩阵有非零条目
2. **对角线**：只有主对角线有条目
3. **单位**：对角线矩阵，所有1 — "什么都不做"
4. **正交**：行/列是正交归一的 — 完美保持几何

每个添加更多结构，导致更强的属性和更快的计算。单位是矩阵乘法的"中立元素"，而正交矩阵保持距离和角度（等距）。

## ML Relevance / 机器学习相关性

In machine learning:
- **Triangular matrices**:
  - Forward/backward substitution in LU decomposition (efficient solving)
  - Recurrent networks can be structured as triangular

- **Diagonal matrices**:
  - Scaling features: diagonal matrices scale each feature
  - Scaling in attention: position-aware attention uses diagonal masking
  - Efficient storage for sparse diagonal matrices

- **Identity matrix**:
  - Initialization: start with identity-like structure for stability
  - Residual connections: `output = input + transform(input)` uses identity implicitly
  - Regularization baseline: compare model to identity function

- **Orthogonal matrices**:
  - QR decomposition: numerically stable least squares
  - Orthogonal initialization of weights: preserve activation magnitudes
  - Rotation invariance: some tasks are rotationally symmetric
  - Whitening data: orthogonal transformation to decorrelate features

Orthogonal matrices are especially important: they preserve norms and angles, making them numerically stable and geometrically meaningful.

在机器学习中：
- **三角形矩阵**：
  - LU分解中的前向/后向替换（高效求解）
  - 递归网络可以结构化为三角形

- **对角矩阵**：
  - 缩放特征：对角矩阵缩放每个特征
  - 注意力中的缩放：位置感知注意力使用对角掩码
  - 稀疏对角矩阵的高效存储

- **单位矩阵**：
  - 初始化：从类似单位的结构开始以获得稳定性
  - 残差连接：`output = input + transform(input)`隐式使用单位
  - 正则化基线：将模型与单位函数比较

- **正交矩阵**：
  - QR分解：数值稳定的最小二乘
  - 权重的正交初始化：保持激活幅度
  - 旋转不变性：某些任务在旋转对称
  - 白化数据：正交变换以去相关特征

正交矩阵特别重要：它们保持范数和角度，使它们在数值上稳定且几何上有意义。

---
