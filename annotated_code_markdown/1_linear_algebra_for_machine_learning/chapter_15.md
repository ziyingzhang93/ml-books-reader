# 线性代数与机器学习
## Chapter 15

---

### Qr Decomposition

# 02 — QR Decomposition / QR 分解

**Chapter 15 — File 2 of 3 / 第15章 — 第2个文件（兦3个）**

---

## Summary / 总结

This script demonstrates **QR decomposition** of a rectangular matrix using NumPy, and verifies the result by reconstructing the original matrix from its factors $Q$ and $R$.

本脚本演示使用 NumPy 对矩形矩阵进行 **QR 分解**，并通过 $Q \cdot R$ 重构原矩阵来验证分解结果。

### Core Formula / 核心公式

$$A = Q \cdot R$$

| Symbol / 符号 | Name / 名称 | Description / 描述 |
|:---:|:---|:---|
| $Q$ | Orthogonal matrix / 正交矩阵 | Columns are orthonormal / 列是正交正规的 |
| $R$ | Upper triangular / 上三角矩阵 | All entries below diagonal are 0 / 对角线下方为0 |

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Import Libraries / 导入渪

```python
from numpy import array          # NumPy 数组工具 / NumPy array utility
from numpy.linalg import qr      # QR 分解函数 / QR decomposition function
```

---
## Step 2 — Define a Rectangular Matrix / 定义矩形矩阵

We create a $3 \times 2$ rectangular matrix. QR decomposition works for **any** rectangular matrix (unlike LU which requires square matrices).

创建一个 $3 \times 2$ 矩形矩阵。QR 分解適用于**任何**矩形矩阵（不像 LU 需要方阵）。

```python
A = array([[1, 2],
           [3, 4],
           [5, 6]])

print("A (shape: 3x2) =")
print(A)
```

---
## Step 3 — Perform QR Decomposition / 执行 QR 分解

`numpy.linalg.qr()` with `mode='complete'` returns:

- **Q** — Orthogonal matrix: columns form an orthonormal basis / 正交矩阵：列为正交正规唢
- **R** — Upper triangular: encodes the scaling and rotation / 上三角矩阵：编码缩放和旋转

```python
Q, R = qr(A, 'complete')     # QR 分解：完全模式 / QR decomposition: complete mode

print("Q (orthogonal / 正交矩阵) =")
print(Q)
print()
print("R (upper triangular / 上三角矩阵) =")
print(R)
```

---
## Step 4 — Reconstruct Original Matrix / 重构原矩阵

If the decomposition is correct, $Q \cdot R$ should give back the original matrix $A$.

如果分解正确，$Q \cdot R$ 应该返回原矩阵 $A$。

```python
B = Q.dot(R)        # Q·R should equal A / Q·R 应等于 A

print("Reconstructed B = Q · R =")
print(B)
```

---
## Learning Notes / 学习笔记

- **数学本质**: QR 分解通过 Gram-Schmidt 正交化提取矩阵列空间的正交基，数值永久性优于 LU。  
  *QR decomposes via Gram-Schmidt orthogonalization — numerically more stable than LU.*

- **ML 应用**: 常用于最小二乘法问题、正规化和根据关系的数值稳定性告伟。  
  *Used in least squares, regularization, and provides numerical stability for condition numbers.*

---

➡️ **Next / 下一步**: `03_cholesky_decomposition.ipynb` — Cholesky 分解，专门量正定矩阵。  
*Cholesky decomposition — for positive definite matrices.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一覽

Below is the full annotated script in one block for quick reference after step-by-step learning.

以下是完整的注释版代码，供逐步学习后快速回顾。

```python
# ===============================
# QR Decomposition / QR 分解
# Complete Code / 完整代码
# ===============================

# --- Import libraries / 导入渪 ---
from numpy import array          # NumPy 数组工具 / NumPy array utility
from numpy.linalg import qr      # QR 分解函数 / QR decomposition function

# --- Define a rectangular matrix / 定义矩形矩阵 ---
A = array([[1, 2],
           [3, 4],
           [5, 6]])
print("A (shape: 3x2) =")
print(A)

# --- QR decomposition / QR 分解 ---
# A = Q · R
#   Q: orthogonal matrix   / 正交矩阵
#   R: upper triangular    / 上三角矩阵
Q, R = qr(A, 'complete')
print("\nQ (orthogonal / 正交矩阵) =")
print(Q)
print("\nR (upper triangular / 上三角矩阵) =")
print(R)

# --- Reconstruct original matrix / 重构原矩阵 ---
B = Q.dot(R)        # Q·R should equal A / Q·R 应等于 A
print("\nReconstructed B = Q · R =")
print(B)
```

---

### Cholesky Decomposition

# 03 — Cholesky Decomposition / Cholesky 分解

**Chapter 15 — File 3 of 3 / 第15章 — 第3个文件（兦3个）**

---

## Summary / 总结

This script demonstrates **Cholesky decomposition** of a symmetric positive-definite matrix using NumPy, and verifies the result by reconstructing the original matrix from its factor $L$.

本脚本演示使用 NumPy 对对称正定矩阵进行 **Cholesky 分解**，并通过 $L \cdot L^T$ 重构原矩阵来验证分解结果。

### Core Formula / 核心公式

$$A = L \cdot L^T$$

| Symbol / 符号 | Name / 名称 | Description / 描述 |
|:---:|:---|:---|
| $L$ | Lower triangular / 下三角矩阵 | All entries above diagonal are 0 / 对角线上方为0 |
| $L^T$ | Transpose / 转置 | Same as upper triangular / 即上三角矩阵 |
| $A$ | Symmetric positive-definite / 对称正定 | $A = A^T$ and all eigenvalues > 0 / $A = A^T$ 且所有特征值 > 0 |

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Import Libraries / 导入渪

```python
from numpy import array              # NumPy 数组工具 / NumPy array utility
from numpy.linalg import cholesky    # Cholesky 分解函数 / Cholesky decomposition function
```

---
## Step 2 — Define a Symmetric Positive-Definite Matrix / 定义对称正定矩阵

Cholesky decomposition **requires** a symmetric positive-definite matrix:
- **Symmetric**: $A = A^T$ / $A = A^T$
- **Positive-definite**: All eigenvalues > 0 / 所有特征值 > 0

Cholesky 分解**仅技于**对称正定矩阵。

```python
A = array([[2, 1, 1],
           [1, 2, 1],
           [1, 1, 2]])

print("A (symmetric positive-definite / 对称正定) =")
print(A)
```

---
## Step 3 — Perform Cholesky Decomposition / 执行 Cholesky 分解

`numpy.linalg.cholesky()` returns:

- **L** — Lower triangular matrix: the "square root" of the matrix / 下三角矩阵：矩阵的「平方根」

```python
L = cholesky(A)     # Cholesky 分解 / Cholesky decomposition

print("L (lower triangular / 下三角矩阵) =")
print(L)
```

---
## Step 4 — Reconstruct Original Matrix / 重构原矩阵

If the decomposition is correct, $L \cdot L^T$ should give back the original matrix $A$.

如果分解正确，$L \cdot L^T$ 应该返回原矩阵 $A$。

```python
B = L.dot(L.T)      # L·L^T should equal A / L·L^T 应等于 A

print("Reconstructed B = L · L^T =")
print(B)
```

---
## Learning Notes / 学习笔记

- **数学本质**: Cholesky 分解是正定矩阵的「平方根」的矩阵算法，需要矩阵对称正定。  
  *Cholesky is the matrix "square root" for symmetric positive-definite matrices.*

- **ML 应用**: 常用于高效解决正定系统、方差矩阵分解、以及取样方法中的不确定性量化。  
  *Used for efficient solving of positive-definite systems, covariance decomposition, and uncertainty quantification.*

---

➡️ **Next / 下一步**: `../chapter_16/01_eigendecomposition.ipynb` — 特征整不。  
*Eigendecomposition — the foundation of many ML algorithms.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一覽

Below is the full annotated script in one block for quick reference after step-by-step learning.

以下是完整的注释版代码，供逐步学习后快速回顾。

```python
# ===============================
# Cholesky Decomposition / Cholesky 分解
# Complete Code / 完整代码
# ===============================

# --- Import libraries / 导入渪 ---
from numpy import array              # NumPy 数组工具 / NumPy array utility
from numpy.linalg import cholesky    # Cholesky 分解函数 / Cholesky decomposition function

# --- Define a symmetric positive-definite matrix / 定义对称正定矩阵 ---
A = array([[2, 1, 1],
           [1, 2, 1],
           [1, 1, 2]])
print("A (symmetric positive-definite / 对称正定) =")
print(A)

# --- Cholesky decomposition / Cholesky 分解 ---
# A = L · L^T
#   L: lower triangular / 下三角矩阵
L = cholesky(A)
print("\nL (lower triangular / 下三角矩阵) =")
print(L)

# --- Reconstruct original matrix / 重构原矩阵 ---
B = L.dot(L.T)      # L·L^T should equal A / L·L^T 应等于 A
print("\nReconstructed B = L · L^T =")
print(B)
```

---

### Chapter Summary

# Chapter 15 Summary / 第15章总结：Matrix Factorization

## Theme / 主题

Matrix factorization decomposes a matrix A into a product of simpler matrices: A = LU, A = QR, A = LL^T. Each decomposition reveals different structure and has different computational properties. These are the foundation of numerical linear algebra—they enable efficient solving of linear systems, least squares, and eigenvalue problems.

矩阵因子分解将矩阵A分解为简单矩阵的乘积：A = LU, A = QR, A = LL^T。每个分解揭示不同的结构并具有不同的计算属性。这些是数值线性代数的基础——它们使线性系统、最小二乘和特征值问题的高效求解成为可能。

## Evolution / 演化路线

```
01_lu_decomposition.ipynb
    └─ LU: A = L · U (一般矩阵，最基础)
       Lower triangular (L) · Upper triangular (U)
    
02_qr_decomposition.ipynb
    └─ QR: A = Q · R (数值稳定)
       Orthogonal (Q) · Upper triangular (R)
    
03_cholesky_decomposition.ipynb
    └─ Cholesky: A = L · L^T (最快，对称正定矩阵)
       Lower triangular (L) squared
```

## Progression Logic / 进度逻辑

Matrix factorizations progress by **increasing constraints and efficiency**:

1. **LU decomposition**: Works for any square matrix, fastest, but numerically less stable
2. **QR decomposition**: More stable due to orthogonal Q, works for rectangular matrices, medium cost
3. **Cholesky decomposition**: Fastest and most stable, but requires symmetric positive definite matrix

Each adds more structure (L → LU → QR → LL^T), leading to better numerical properties and faster computation. This is a classic trade-off: more constraints = better performance.

矩阵因子分解通过**增加约束和效率**进行：

1. **LU分解**：适用于任何方阵，最快，但数值上不太稳定
2. **QR分解**：由于正交Q更稳定，适用于矩形矩阵，中等成本
3. **Cholesky分解**：最快最稳定，但需要对称正定矩阵

每个添加更多结构（L → LU → QR → LL^T），导致更好的数值属性和更快的计算。这是一个经典的权衡：更多约束=更好的性能。

## ML Relevance / 机器学习相关性

In machine learning:
- **LU decomposition**:
  - Solving linear systems: `LU · x = b` via forward elimination
  - Gaussian elimination: solve A·w = y for regression
  - Determinant: det(A) = product of L and U diagonals (fast computation)

- **QR decomposition**:
  - Least squares solution: `w = R^(-1) · Q^T · y` (more stable than normal equation)
  - Numerical stability: orthogonal Q preserves condition number
  - When data is poorly conditioned (high condition number), QR is more accurate
  - Linear regression via QR (alternative to SVD, faster when m >> n)

- **Cholesky decomposition**:
  - Covariance matrix operations: positive definite covariance matrices
  - Gaussian process regression: Cholesky for efficient inference
  - Sampling from multivariate normal: generate samples via L
  - Kalman filter: Cholesky for numerical stability

- **When to use which**:
  - LU: Fast, general, but unstable (academic setting)
  - QR: Standard for least squares (A is tall/rectangular)
  - Cholesky: When you know A is symmetric positive definite (covariance, Gram matrices)

QR is the standard choice for least squares in production systems because it's more numerically stable than directly using the normal equation. Cholesky is crucial for probabilistic models (Gaussians, Gaussian processes).

在机器学习中：
- **LU分解**：
  - 求解线性系统：通过前向消除`LU · x = b`
  - 高斯消除法：求解回归的A·w = y
  - 行列式：det(A) = L和U对角线的乘积（快速计算）

- **QR分解**：
  - 最小二乘解：`w = R^(-1) · Q^T · y`（比正规方程更稳定）
  - 数值稳定性：正交Q保持条件数
  - 当数据条件不好（高条件数）时，QR更准确
  - 通过QR的线性回归（当m >> n时比SVD更快的替代方案）

- **Cholesky分解**：
  - 协方差矩阵操作：正定协方差矩阵
  - 高斯过程回归：Cholesky用于高效推理
  - 从多变量正态采样：通过L生成样本
  - 卡尔曼滤波器：Cholesky用于数值稳定性

- **何时使用哪个**：
  - LU：快速、通用，但不稳定（学术设置）
  - QR：最小二乘的标准（A是高/矩形）
  - Cholesky：当您知道A是对称正定时（协方差、Gram矩阵）

QR是生产系统中最小二乘的标准选择，因为它在数值上比直接使用正规方程更稳定。Cholesky对概率模型（高斯、高斯过程）至关重要。

---
