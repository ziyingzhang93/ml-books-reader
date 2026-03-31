# 线性代数与机器学习 / Linear Algebra for Machine Learning
## Chapter 12

---

### Transpose

# Matrix Transpose / 矩阵转置

**Chapter 12 — File 1 of 6 / 第12章 — 第1个文件（共6个）**

## Summary / 总结

Matrix transpose flips a matrix along its diagonal, converting rows to columns and columns to rows. For a matrix $A$ of shape $(m, n)$, the transpose $A^T$ has shape $(n, m)$. This is a fundamental operation in linear algebra used in dot products, solving systems of equations, and many ML algorithms.

矩阵转置是沿对角线翻转矩阵的操作，将行变为列，列变为行。对于形状为 $(m, n)$ 的矩阵 $A$，其转置 $A^T$ 的形状为 $(n, m)$。这是线性代数中的基本操作，用于点积、求解方程组和许多机器学习算法中。

## Core Formula / 核心公式

$$A^T_{ij} = A_{ji}$$

If $A = \begin{bmatrix} a & b \\ c & d \\ e & f \end{bmatrix}$, then $A^T = \begin{bmatrix} a & c & e \\ b & d & f \end{bmatrix}$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create a matrix / 创建矩阵

We create a 3×2 matrix (3 rows, 2 columns). This demonstrates how transpose will convert it to a 2×3 matrix.

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Create a 3x2 matrix: 3 rows, 2 columns
# 创建一个 3x2 矩阵：3行，2列
A = array([[1, 2],
           [3, 4],
           [5, 6]])

# 打印输出 / Print output
print("Original matrix A (shape: 3x2) / 原始矩阵 A（形状：3x2）:")
# 打印输出 / Print output
print(A)
```

## Step 2 — Perform transpose / 执行转置

Use the `.T` attribute to transpose the matrix. Rows become columns and columns become rows.

```python
# Transpose the matrix using .T attribute
# 使用 .T 属性对矩阵进行转置
C = A.T

# 打印输出 / Print output
print("Transposed matrix C = A^T (shape: 2x3) / 转置矩阵 C = A^T（形状：2x3）:")
# 打印输出 / Print output
print(C)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"\nOriginal shape: {A.shape}, Transposed shape: {C.shape}")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"原始形状：{A.shape}，转置后形状：{C.shape}")
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Transpose is an involution ($A^{TT} = A$) that swaps the row and column indices. It's linear: $(A+B)^T = A^T + B^T$ and $(kA)^T = k A^T$.

- **ML Application / 机器学习应用**: Transpose is essential in matrix multiplication (computing $X^T X$ for covariance matrices), computing gradients in backpropagation, and transforming feature matrices for neural networks.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `02_inverse.ipynb` — Compute matrix inverse and verify with identity matrix

## Complete Code / 完整代码一览

```python
# --- Matrix Transpose / 矩阵转置 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Step 1: Create a 3x2 matrix
# 步骤 1：创建一个 3x2 矩阵
A = array([[1, 2],
           [3, 4],
           [5, 6]])

# 打印输出 / Print output
print("Original matrix A (shape: 3x2) / 原始矩阵 A（形状：3x2）:")
# 打印输出 / Print output
print(A)

# Step 2: Transpose using .T attribute
# 步骤 2：使用 .T 属性进行转置
C = A.T

# 打印输出 / Print output
print("\nTransposed matrix C = A^T (shape: 2x3) / 转置矩阵 C = A^T（形状：2x3）:")
# 打印输出 / Print output
print(C)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"\nOriginal shape: {A.shape}, Transposed shape: {C.shape}")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"原始形状：{A.shape}，转置后形状：{C.shape}")
```

---

### Inverse

# Matrix Inverse / 矩阵求逆

**Chapter 12 — File 2 of 6 / 第12章 — 第2个文件（共6个）**

## Summary / 总结

The matrix inverse is analogous to division in scalar arithmetic. For a square matrix $A$, the inverse $A^{-1}$ satisfies $A \cdot A^{-1} = I$ (identity matrix). Not all matrices have inverses; a matrix must be square and have full rank. The inverse is crucial for solving linear systems $Ax = b$ by computing $x = A^{-1}b$.

矩阵逆是标量算术中除法的类似物。对于方阵 $A$，其逆 $A^{-1}$ 满足 $A \cdot A^{-1} = I$（单位矩阵）。不是所有矩阵都有逆；矩阵必须是方阵且满秩。逆矩阵对于求解线性系统 $Ax = b$ 是至关重要的，通过计算 $x = A^{-1}b$ 来求解。

## Core Formula / 核心公式

$$A \cdot A^{-1} = A^{-1} \cdot A = I$$

For a 2×2 matrix: $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, $A^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create a square matrix / 创建方阵

We create a 2×2 matrix with non-zero determinant, which ensures it has an inverse.

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Create a 2x2 matrix with floating point values
# 创建一个带有浮点值的 2x2 矩阵
A = array([[1.0, 2.0],
           [3.0, 4.0]])

# 打印输出 / Print output
print("Original matrix A / 原始矩阵 A:")
# 打印输出 / Print output
print(A)
```

## Step 2 — Compute the inverse / 计算逆矩阵

Use `numpy.linalg.inv()` to compute the matrix inverse.

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import inv

# Compute the inverse using numpy.linalg.inv()
# 使用 numpy.linalg.inv() 计算逆矩阵
B = inv(A)

# 打印输出 / Print output
print("Inverse matrix B = A^(-1) / 逆矩阵 B = A^(-1):")
# 打印输出 / Print output
print(B)
```

## Step 3 — Verify with identity matrix / 用单位矩阵验证

Multiply $A \cdot A^{-1}$ to verify we get the identity matrix (approximately, due to floating point precision).

```python
# Verify: A * A^(-1) should equal the identity matrix I
# 验证：A * A^(-1) 应该等于单位矩阵 I
I = A.dot(B)

# 打印输出 / Print output
print("Product A * B = A * A^(-1) / 乘积 A * B = A * A^(-1):")
# 打印输出 / Print output
print(I)
# 打印输出 / Print output
print("\nThis should be approximately the identity matrix [1 0; 0 1]")
# 打印输出 / Print output
print("这应该近似为单位矩阵 [1 0; 0 1]")
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Matrix inverse only exists for square, full-rank matrices (non-singular). The inverse satisfies the property $(A^{-1})^{-1} = A$ and $(AB)^{-1} = B^{-1}A^{-1}$.

- **ML Application / 机器学习应用**: Computing matrix inverse is fundamental in linear regression (Normal Equation: $\theta = (X^T X)^{-1} X^T y$), least squares optimization, and Gaussian elimination algorithms.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `03_trace.ipynb` — Compute the trace (sum of diagonal elements) of a matrix

## Complete Code / 完整代码一览

```python
# --- Matrix Inverse / 矩阵求逆 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import inv

# Step 1: Create a 2x2 matrix
# 步骤 1：创建一个 2x2 矩阵
A = array([[1.0, 2.0],
           [3.0, 4.0]])

# 打印输出 / Print output
print("Original matrix A / 原始矩阵 A:")
# 打印输出 / Print output
print(A)

# Step 2: Compute the inverse
# 步骤 2：计算逆矩阵
B = inv(A)

# 打印输出 / Print output
print("\nInverse matrix B = A^(-1) / 逆矩阵 B = A^(-1):")
# 打印输出 / Print output
print(B)

# Step 3: Verify with identity matrix
# 步骤 3：用单位矩阵验证
I = A.dot(B)

# 打印输出 / Print output
print("\nProduct A * B = A * A^(-1) / 乘积 A * B = A * A^(-1):")
# 打印输出 / Print output
print(I)
# 打印输出 / Print output
print("\nThis should be approximately the identity matrix [1 0; 0 1]")
# 打印输出 / Print output
print("这应该近似为单位矩阵 [1 0; 0 1]")
```

---

### Trace



---

### Determinant



---

### Vector Rank

# Vector Rank / 向量的秩

**Chapter 12 — File 5 of 6 / 第12章 — 第5个文件（共6个）**

## Summary / 总结

The rank of a vector is typically 0 (for the zero vector) or 1 (for any non-zero vector). In linear algebra context, rank represents the dimension of the vector space spanned by the vector. When computing rank of a single vector as a row/column, numpy's `matrix_rank()` returns 0 for zero vectors and 1 for any non-zero vector. Understanding vector rank is fundamental to understanding matrix rank and linear independence.

向量的秩通常为 0（对于零向量）或 1（对于任何非零向量）。在线性代数背景下，秩表示由向量张成的向量空间的维数。当计算单个向量作为行/列的秩时，numpy 的 `matrix_rank()` 对零向量返回 0，对任何非零向量返回 1。理解向量秩对于理解矩阵秩和线性独立性至关重要。

## Core Formula / 核心公式

$$\text{rank}(\mathbf{v}) = \begin{cases} 0 & \text{if } \mathbf{v} = \mathbf{0} \\ 1 & \text{if } \mathbf{v} \neq \mathbf{0} \end{cases}$$

For $\mathbf{v} = [1, 2, 3]$, rank = 1. For $\mathbf{v} = [0, 0, 0]$, rank = 0.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Compute rank of a non-zero vector / 计算非零向量的秩

Create a non-zero vector and compute its rank.

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import matrix_rank

# Create a non-zero vector
# 创建一个非零向量
v1 = array([1, 2, 3])

# 打印输出 / Print output
print("Non-zero vector v1 / 非零向量 v1:")
# 打印输出 / Print output
print(v1)

# Compute rank of the vector
# 计算向量的秩
vr1 = matrix_rank(v1)

# 打印输出 / Print output
print(f"\nRank of v1: rank(v1) = {vr1}")
# 打印输出 / Print output
print(f"v1 的秩：rank(v1) = {vr1}")
# 打印输出 / Print output
print(f"\nExplanation: Any non-zero vector has rank 1 (spans a 1D space)")
# 打印输出 / Print output
print(f"解释：任何非零向量的秩为 1（张成一个 1D 空间）")
```

## Step 2 — Compute rank of the zero vector / 计算零向量的秩

Create a zero vector and compute its rank.

```python
# Create a zero vector
# 创建一个零向量
v2 = array([0, 0, 0, 0, 0])

# 打印输出 / Print output
print("Zero vector v2 / 零向量 v2:")
# 打印输出 / Print output
print(v2)

# Compute rank of the zero vector
# 计算零向量的秩
vr2 = matrix_rank(v2)

# 打印输出 / Print output
print(f"\nRank of v2: rank(v2) = {vr2}")
# 打印输出 / Print output
print(f"v2 的秩：rank(v2) = {vr2}")
# 打印输出 / Print output
print(f"\nExplanation: The zero vector has rank 0 (spans only the zero element)")
# 打印输出 / Print output
print(f"解释：零向量的秩为 0（仅张成零元素）")
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: A vector's rank is the dimension of the subspace it spans. Non-zero vectors have rank 1 (span a line through origin). Zero vector has rank 0 (spans only origin). This foundation extends to understanding matrix rank as maximum number of linearly independent rows or columns.

- **ML Application / 机器学习应用**: Understanding vector and matrix rank is crucial for: detecting feature redundancy (collinearity), dimensionality reduction (PCA, SVD), solving underdetermined/overdetermined systems, and analyzing model capacity in neural networks.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `06_matrix_rank.ipynb` — Compute matrix rank and understand linear independence

## Complete Code / 完整代码一览

```python
# --- Vector Rank / 向量的秩 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import matrix_rank

# Step 1: Rank of non-zero vector
# 步骤 1：非零向量的秩
v1 = array([1, 2, 3])

# 打印输出 / Print output
print("Non-zero vector v1 / 非零向量 v1:")
# 打印输出 / Print output
print(v1)

vr1 = matrix_rank(v1)
# 打印输出 / Print output
print(f"\nRank of v1: rank(v1) = {vr1}")
# 打印输出 / Print output
print(f"v1 的秩：rank(v1) = {vr1}")

# Step 2: Rank of zero vector
# 步骤 2：零向量的秩
v2 = array([0, 0, 0, 0, 0])

# 打印输出 / Print output
print("\nZero vector v2 / 零向量 v2:")
# 打印输出 / Print output
print(v2)

vr2 = matrix_rank(v2)
# 打印输出 / Print output
print(f"\nRank of v2: rank(v2) = {vr2}")
# 打印输出 / Print output
print(f"v2 的秩：rank(v2) = {vr2}")
```

---

### Matrix Rank

# Matrix Rank / 矩阵的秩

**Chapter 12 — File 6 of 6 / 第12章 — 第6个文件（共6个）**

## Summary / 总结

The rank of a matrix is the dimension of its row space (or column space, they're equal). It equals the number of linearly independent rows or columns. For an $m \times n$ matrix, rank ≤ min(m, n). Full rank means rank equals min(m, n); rank-deficient means rank < min(m, n). Rank is crucial for understanding solvability of linear systems and invertibility of matrices.

矩阵的秩是其行空间（或列空间，两者相等）的维数。它等于线性独立的行或列的数量。对于 $m \times n$ 矩阵，秩 ≤ min(m, n)。满秩意味着秩等于 min(m, n)；秩亏矩阵意味着秩 < min(m, n)。秩对于理解线性系统的可解性和矩阵的可逆性至关重要。

## Core Formula / 核心公式

$$0 \leq \text{rank}(A) \leq \min(m, n)$$

For full rank: $\text{rank}(A) = \min(m, n)$

For rank-deficient: $\text{rank}(A) < \min(m, n)$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Rank of zero matrix / 零矩阵的秩

Compute the rank of a zero matrix (all elements are zero).

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import matrix_rank

# Create a zero matrix
# 创建一个零矩阵
M0 = array([[0, 0],
            [0, 0]])

# 打印输出 / Print output
print("Zero matrix M0 / 零矩阵 M0:")
# 打印输出 / Print output
print(M0)

# Compute rank
# 计算秩
mr0 = matrix_rank(M0)

# 打印输出 / Print output
print(f"\nRank of M0: rank(M0) = {mr0}")
# 打印输出 / Print output
print(f"M0 的秩：rank(M0) = {mr0}")
# 打印输出 / Print output
print(f"\nExplanation: Zero matrix has rank 0 (no linearly independent rows/columns)")
# 打印输出 / Print output
print(f"解释：零矩阵的秩为 0（没有线性独立的行/列）")
```

## Step 2 — Rank of rank-deficient matrix / 秩亏矩阵的秩

Compute the rank of a rank-deficient matrix (linearly dependent rows).

```python
# Create a rank-deficient matrix (row 2 = row 1)
# 创建一个秩亏矩阵（第2行 = 第1行）
M1 = array([[1, 2],
            [1, 2]])

# 打印输出 / Print output
print("Rank-deficient matrix M1 / 秩亏矩阵 M1:")
# 打印输出 / Print output
print(M1)

# Compute rank
# 计算秩
mr1 = matrix_rank(M1)

# 打印输出 / Print output
print(f"\nRank of M1: rank(M1) = {mr1}")
# 打印输出 / Print output
print(f"M1 的秩：rank(M1) = {mr1}")
# 打印输出 / Print output
print(f"\nExplanation: Row 2 equals Row 1, so only 1 linearly independent row")
# 打印输出 / Print output
print(f"解释：第2行等于第1行，所以只有 1 个线性独立的行")
# 打印输出 / Print output
print(f"Maximum possible rank: min(2, 2) = 2, but actual rank = 1 (rank-deficient)")
# 打印输出 / Print output
print(f"最大可能的秩：min(2, 2) = 2，但实际秩 = 1（秩亏）")
```

## Step 3 — Rank of full-rank matrix / 满秩矩阵的秩

Compute the rank of a full-rank matrix (linearly independent rows and columns).

```python
# Create a full-rank matrix (all rows linearly independent)
# 创建一个满秩矩阵（所有行线性独立）
M2 = array([[1, 2],
            [3, 4]])

# 打印输出 / Print output
print("Full-rank matrix M2 / 满秩矩阵 M2:")
# 打印输出 / Print output
print(M2)

# Compute rank
# 计算秩
mr2 = matrix_rank(M2)

# 打印输出 / Print output
print(f"\nRank of M2: rank(M2) = {mr2}")
# 打印输出 / Print output
print(f"M2 的秩：rank(M2) = {mr2}")
# 打印输出 / Print output
print(f"\nExplanation: All rows are linearly independent")
# 打印输出 / Print output
print(f"解释：所有行都线性独立")
# 打印输出 / Print output
print(f"Maximum possible rank: min(2, 2) = 2, actual rank = 2 (full-rank matrix)")
# 打印输出 / Print output
print(f"最大可能的秩：min(2, 2) = 2，实际秩 = 2（满秩矩阵）")
# 打印输出 / Print output
print(f"\nFull-rank matrices are invertible and have non-zero determinant")
# 打印输出 / Print output
print(f"满秩矩阵是可逆的，且有非零行列式")
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Matrix rank is the dimension of its column (or row) space. Key property: rank is invariant under row/column operations. For $A \in \mathbb{R}^{m \times n}$: rank(A) = number of non-zero singular values = number of non-zero eigenvalues (for square A).

- **ML Application / 机器学习应用**: Matrix rank is essential for: determining if linear systems have solutions, checking feature collinearity in regression, singular value decomposition (SVD), rank-constrained optimization, and analyzing neural network expressivity and generalization.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `../chapter_13/01_sparse.ipynb` — Explore sparse matrix representations for efficient computation

## Complete Code / 完整代码一览

```python
# --- Matrix Rank / 矩阵的秩 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import matrix_rank

# Step 1: Zero matrix
# 步骤 1：零矩阵
M0 = array([[0, 0],
            [0, 0]])

# 打印输出 / Print output
print("Zero matrix M0 / 零矩阵 M0:")
# 打印输出 / Print output
print(M0)

mr0 = matrix_rank(M0)
# 打印输出 / Print output
print(f"Rank of M0: rank(M0) = {mr0}\n")

# Step 2: Rank-deficient matrix
# 步骤 2：秩亏矩阵
M1 = array([[1, 2],
            [1, 2]])

# 打印输出 / Print output
print("Rank-deficient matrix M1 / 秩亏矩阵 M1:")
# 打印输出 / Print output
print(M1)

mr1 = matrix_rank(M1)
# 打印输出 / Print output
print(f"Rank of M1: rank(M1) = {mr1}\n")

# Step 3: Full-rank matrix
# 步骤 3：满秩矩阵
M2 = array([[1, 2],
            [3, 4]])

# 打印输出 / Print output
print("Full-rank matrix M2 / 满秩矩阵 M2:")
# 打印输出 / Print output
print(M2)

mr2 = matrix_rank(M2)
# 打印输出 / Print output
print(f"Rank of M2: rank(M2) = {mr2}")
# 打印输出 / Print output
print(f"M2 is invertible and has non-zero determinant")
# 打印输出 / Print output
print(f"M2 可逆且行列式非零")
```

---

### Chapter Summary / 章节总结

# Chapter 12 Summary / 第12章总结：Matrix Properties

## Theme / 主题

Matrices have intrinsic properties that reveal their structure and behavior. This chapter covers operations that extract or compute these properties: transpose (flip rows/columns), inverse (undo multiplication), trace (sum of diagonal), determinant (volume scaling), and rank (dimensionality). These properties answer fundamental questions: Is the matrix invertible? What's its dimension? How much does it scale volume?

矩阵具有显示其结构和行为的内在属性。本章涵盖提取或计算这些属性的操作：转置（翻转行/列）、逆（撤销乘法）、迹（对角线和）、行列式（体积缩放）和秩（维数）。这些属性回答基本问题：矩阵是否可逆？其维数是什么？它缩放体积多少？

## Evolution / 演化路线

```
01_transpose.ipynb
    └─ Flip rows and columns: A^T (行列互换)
    
02_inverse.ipynb
    └─ Find A^(-1) such that A · A^(-1) = I (矩阵求逆)
    
03_trace.ipynb
    └─ Sum of diagonal elements (对角线元素和)
    
04_determinant.ipynb
    └─ Volume scaling factor (体积缩放因子)
    
05_vector_rank.ipynb
    └─ Linear independence of vectors (向量的线性独立性)
    
06_matrix_rank.ipynb
    └─ Number of independent rows/columns (独立行/列数)
```

## Progression Logic / 进度逻辑

Matrix properties progress from **simple to structural**:

1. **Transpose**: Mechanical operation, flips dimensions
2. **Inverse**: Requires square matrix, non-singular (det ≠ 0)
3. **Trace**: Single number, related to eigenvalue sum
4. **Determinant**: Single number, zero means singular (non-invertible)
5. **Rank**: Number from 0 to min(m, n), measures true dimensionality

The key insight: **rank tells you the true dimension**. A (100, 100) matrix with rank 50 really only spans 50 dimensions. This is critical for understanding data structure and when least squares problems are ill-posed.

矩阵属性从**简单到结构性**进行：

1. **转置**：机械操作，翻转维度
2. **逆**：需要方阵，非奇异（det ≠ 0）
3. **迹**：单个数字，与特征值和相关
4. **行列式**：单个数字，零表示奇异（不可逆）
5. **秩**：从0到min(m, n)的数字，测量真实维数

关键见解：**秩告诉你真实的维度**。秩为50的(100, 100)矩阵实际上只跨越50个维度。这对于理解数据结构和何时最小二乘问题不适定至关重要。

## ML Relevance / 机器学习相关性

In machine learning:
- **Transpose**:
  - Data shapes: "transpose data to match input shape"
  - Gradient computation: backprop uses A^T extensively
  - Covariance: `Cov = X^T · X / n`

- **Inverse**:
  - Normal equation: `w = (X^T · X)^(-1) · X^T · y`
  - Numerical stability: direct inversion is often avoided (use QR, SVD instead)
  - Gaussian elimination with back-substitution

- **Trace**:
  - Loss function: trace of covariance in PCA
  - Implicit in optimization: related to eigenvalues

- **Determinant**:
  - Tells you if inverse exists: det(A) = 0 ⟹ A is singular
  - Volume interpretation: how much does linear transformation scale volume?

- **Rank**:
  - Data dimensionality: rank(X) tells true feature dimension
  - Degeneracy detection: rank < n_features means some features are redundant
  - Underdetermined systems: if rank(X) < n, least squares has infinite solutions
  - Overdetermined systems: if rank(X) = n < m, least squares has unique solution

Rank is especially critical: it determines whether a least squares problem has a unique solution, no solution, or infinite solutions.

在机器学习中：
- **转置**：
  - 数据形状："转置数据以匹配输入形状"
  - 梯度计算：反向传播大量使用A^T
  - 协方差：`Cov = X^T · X / n`

- **逆**：
  - 正规方程：`w = (X^T · X)^(-1) · X^T · y`
  - 数值稳定性：通常避免直接求逆（改用QR、SVD）
  - 高斯消元法与回代

- **迹**：
  - 损失函数：PCA中协方差的迹
  - 隐式在优化中：与特征值相关

- **行列式**：
  - 告诉您逆是否存在：det(A) = 0 ⟹ A是奇异的
  - 体积解释：线性变换缩放体积多少？

- **秩**：
  - 数据维数：rank(X)告诉真实特征维度
  - 退化检测：rank < n_features意味着某些特征是冗余的
  - 欠定系统：如果rank(X) < n，最小二乘有无限解
  - 超定系统：如果rank(X) = n < m，最小二乘有唯一解

秩特别关键：它确定最小二乘问题是否有唯一解、无解或无限解。

---
