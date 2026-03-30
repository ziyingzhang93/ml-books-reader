# 线性代数与机器学习
## Chapter 16

---

### Confirm Eigenvector

# 02 — Confirm Eigenvector / 确认特征变数

**Chapter 16 — File 2 of 3 / 第16章 — 第2个文件（兦3个）**

---

## Summary / 总结

This script verifies the fundamental eigenvalue equation $A \mathbf{v} = \lambda \mathbf{v}$ by computing both sides and confirming they are equal (within numerical precision).

本脚本于计算 $A \mathbf{v} = \lambda \mathbf{v}$ 的两端来验证特征值-特征变数关系。

### Core Formula / 核心公式

For eigenvector $\mathbf{v}$ and eigenvalue $\lambda$:

$$A \mathbf{v} = \lambda \mathbf{v}$$

**Verification**: Compute $A \mathbf{v}$ and $\lambda \mathbf{v}$ separately and check equality.

**验证**: 分别计算 $A \mathbf{v}$ 和 $\lambda \mathbf{v}$，然后检查是否相等。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Import Libraries / 导入渪

```python
from numpy import array          # NumPy 数组工具 / NumPy array utility
from numpy.linalg import eig     # 特征整函数 / Eigendecomposition function
```

---
## Step 2 — Define Matrix and Compute Eigenvalues/Vectors / 定义矩阵并计算特征值

Use the same matrix and eigenvector computation as before.

使用前一个脚本的矩阵和特征值特征变数。

```python
A = array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

values, vectors = eig(A)     # eigendecomposition / 特征整

print("A =")
print(A)
print("\nEigenvalues / 特征值:")
print(values)
print("\nEigenvectors / 特征变数:")
print(vectors)
```

---
## Step 3 — Verify: Compute $A \mathbf{v}$ / 验证：计算 $A \mathbf{v}$

Extract the first eigenvector (first column) and multiply by $A$.

每一列是一个特征变数。提取第一个特征变数，然后与 $A$ 相乘。

```python
# Extract first eigenvector (first column) / 提取第一个特征变数
B = A.dot(vectors[:, 0])     # A · v / A 乘上 v

print("B = A · v (first eigenvector) / A 乘上第一个特征变数:")
print(B)
```

---
## Step 4 — Verify: Compute $\lambda \mathbf{v}$ / 验证：计算 $\lambda \mathbf{v}$

Multiply the first eigenvector by its corresponding eigenvalue. This should equal $B$.

第一个特征变数乘以第一个特征值。应该等于 $B$。

```python
# Compute λ · v / 计算 λ 乘以 v
C = vectors[:, 0] * values[0]     # λ · v / λ 乘以 v

print("C = λ · v (eigenvalue times eigenvector) / 特征值乘以特征变数:")
print(C)
print()
print("Are B and C equal? / B 和 C 是否相等？")
print(f"B - C =")
print(B - C)
```

---
## Learning Notes / 学习笔记

- **数学本质**: 特征值-特征变数关系是效棨旨是：$A \mathbf{v}$ 仅是将 $\mathbf{v}$ 缩放了 $\lambda$ 倍，不改变方向。  
  *Eigenvectors are "magic" directions: applying $A$ just scales them.*

- **ML 应用**: 帮助我们理解数据的主要変化方向、耗时稷首性质、实现的底层原理。  
  *Reveals principal directions, time-stepping behavior, and algorithm fundamentals.*

---

➡️ **Next / 下一步**: `03_reconstruct_matrix.ipynb` — 使用特征值重构矩阵。  
*Reconstruct the matrix using eigendecomposition.*

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
# Confirm Eigenvector / 确认特征变数
# Complete Code / 完整代码
# ===============================

# --- Import libraries / 导入渪 ---
from numpy import array          # NumPy 数组工具 / NumPy array utility
from numpy.linalg import eig     # 特征整函数 / Eigendecomposition function

# --- Define matrix and compute eigendecomposition / 定义矩阵并究个特征整 ---
A = array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

values, vectors = eig(A)
print("A =")
print(A)

# --- Verification / 验证 ---
# The fundamental eigenvalue equation: A·v = λ·v
# 特征值方程： A·v = λ·v

# Left side: A·v / 左边：A 乘以 v
B = A.dot(vectors[:, 0])
print("\nB = A · v[0]:")
print(B)

# Right side: λ·v / 右边：λ 乘以 v
C = vectors[:, 0] * values[0]
print("\nC = λ[0] · v[0]:")
print(C)

# Check equality / 检查是否相等
print("\nDifference (should be ~0) / 差值（应为约 0）:")
print(B - C)
```

---

### Reconstruct Matrix

# 03 — Reconstruct Matrix / 重构矩阵

**Chapter 16 — File 3 of 3 / 第16章 — 第3个文件（兦3个）**

---

## Summary / 总结

This script demonstrates how to **reconstruct a matrix** from its eigendecomposition. If $A = Q \Lambda Q^{-1}$, we can recover $A$ by multiplying these components back together.

本脚本演示如何从特征整重构矩阵。如果 $A = Q \Lambda Q^{-1}$，我们可以将这些成分相乘来恢复原矩阵。

### Core Formula / 核心公式

$$A = Q \Lambda Q^{-1}$$

| Symbol / 符号 | Name / 名称 | Description / 描述 |
|:---:|:---|:---|
| $Q$ | Eigenvector matrix / 特征变数矩阵 | Columns are eigenvectors / 列是特征变数 |
| $\Lambda$ | Eigenvalue matrix / 特征值对角矩阵 | Diagonal with eigenvalues / 对角为特征值 |
| $Q^{-1}$ | Inverse of $Q$ / $Q$ 的逻永业逆 | Inverse of eigenvector matrix / 特征变数矩阵的逻永业逆 |

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Import Libraries / 导入渪

```python
from numpy import array          # NumPy 数组工具 / NumPy array utility
from numpy import diag           # Create diagonal matrix / 创建对角矩阵
from numpy.linalg import inv     # Matrix inverse / 矩阵逻永业逆
from numpy.linalg import eig     # Eigendecomposition / 特征整
```

---
## Step 2 — Define Matrix and Compute Eigendecomposition / 定义矩阵并计算特征整

Compute the eigenvalues and eigenvectors of our test matrix.

计算矩阵的特征值和特征变数。

```python
A = array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

print("A =")
print(A)

values, vectors = eig(A)     # eigendecomposition / 特征整

print("\nEigenvalues / 特征值:")
print(values)
print("\nEigenvectors / 特征变数:")
print(vectors)
```

---
## Step 3 — Construct Eigenvalue Diagonal Matrix / 构建特征值对角矩阵

Create a diagonal matrix $\Lambda$ with eigenvalues on the diagonal.

创建对角矩阵 $\Lambda$，其对角为特征值。

```python
Q = vectors          # Q: matrix of eigenvectors / Q: 特征变数矩阵
R = inv(Q)           # R: inverse of Q / R: Q 的逻永业逆
L = diag(values)     # L: diagonal eigenvalue matrix / L: 对角特征值矩阵

print("Q (eigenvector matrix / 特征变数矩阵) =")
print(Q)
print("\nL (eigenvalue diagonal matrix / 对角特征值矩阵) =")
print(L)
print("\nR = Q^{-1} (inverse of Q / Q 的逻永业逆) =")
print(R)
```

---
## Step 4 — Reconstruct Original Matrix / 重构原矩阵

Multiply back: $A = Q \cdot \Lambda \cdot Q^{-1}$

相乘回来：$A = Q \cdot \Lambda \cdot Q^{-1}$

```python
B = Q.dot(L).dot(R)     # A = Q · Λ · Q^{-1} / A = Q 乘以 Λ 乘以 Q^{-1}

print("Reconstructed B = Q · L · R =")
print(B)
print()
print("Original A =")
print(A)
print()
print("Difference (should be ~0) / 差值（应为约 0）:")
print(B - A)
```

---
## Learning Notes / 学习笔记

- **数学本质**: 特征整是矩阵的“树丁分解”。它托会矩阵的主要处处是止、旋转、旨位移动。  
  *Eigendecomposition is the "standard form" of a matrix — it reveals scaling, rotation, and shift behaviors.*

- **ML 应用**: 是 PCA、主成分分析、スペクトル罱箱、介患提修批次和介患過源批次等歜算法的算基。  
  *Foundation of PCA, spectral clustering, power iteration, Lanczos, and Krylov subspace methods.*

---

➡️ **Next / 下一步**: `../chapter_17/01_svd.ipynb` — 特異值分解 (SVD)，更一般的矩阵因子分解。  
*Singular Value Decomposition — the most general matrix factorization.*

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
# Reconstruct Matrix / 重构矩阵
# Complete Code / 完整代码
# ===============================

# --- Import libraries / 导入渪 ---
from numpy import array          # NumPy 数组工具 / NumPy array utility
from numpy import diag           # Create diagonal matrix / 创建对角矩阵
from numpy.linalg import inv     # Matrix inverse / 矩阵逻永业逆
from numpy.linalg import eig     # Eigendecomposition / 特征整

# --- Define matrix and compute eigendecomposition / 定义矩阵并计算特征整 ---
A = array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
print("A =")
print(A)

values, vectors = eig(A)

# --- Construct factorization matrices / 构建因子整矩阵 ---
# A = Q · Λ · Q^{-1}
#   Q: eigenvector matrix / 特征变数矩阵
#   Λ: eigenvalue diagonal matrix / 特征值对角矩阵
#   Q^{-1}: inverse of Q / Q 的逻永业逆
Q = vectors
L = diag(values)
R = inv(Q)

print("\nQ (eigenvector matrix) =")
print(Q)
print("\nΛ (eigenvalue diagonal matrix) =")
print(L)
print("\nQ^{-1} (inverse) =")
print(R)

# --- Reconstruct original matrix / 重构原矩阵 ---
B = Q.dot(L).dot(R)
print("\nReconstructed B = Q · Λ · Q^{-1} =")
print(B)
print("\nDifference (should be ~0) / 差值（应为约 0）:")
print(B - A)
```

---

### Chapter Summary

# Chapter 16 Summary / 第16章总结：Eigendecomposition

## Theme / 主题

Eigendecomposition factors a square matrix into eigenvectors and eigenvalues: A = Q·Λ·Q^(-1). Eigenvectors are directions that don't change direction under multiplication by A, eigenvalues are the scaling factors. This decomposition reveals the "intrinsic directions" of a transformation and is fundamental to PCA, Google's PageRank, and neural network analysis.

特征分解将方阵分解为特征向量和特征值：A = Q·Λ·Q^(-1)。特征向量是在与A相乘后方向不改变的方向，特征值是缩放因子。该分解揭示了变换的"固有方向"，是PCA、Google的PageRank和神经网络分析的基础。

## Evolution / 演化路线

```
01_compute_eigen.ipynb
    └─ Compute eigenvalues and eigenvectors (计算特征值和特征向量)
    
02_verify_eigen.ipynb
    └─ Verify: A · v = λ · v (验证特征分解)
    
03_reconstruct_matrix.ipynb
    └─ Reconstruct: A = Q · Λ · Q^(-1) (从特征向量和特征值重建)
```

## Progression Logic / 进度逻辑

Eigendecomposition is learned through **compute → verify → reconstruct**:

1. **Compute**: Use `np.linalg.eig()` to find eigenvalues and eigenvectors
2. **Verify**: Check that A·v = λ·v (fundamental eigenvalue equation)
3. **Reconstruct**: Build A back from Q, Λ, Q^(-1) to understand decomposition completeness

This progression teaches that eigendecomposition is not magic—it's a concrete factorization you can verify and reconstruct. The insight: **Q contains eigenvectors as columns, Λ is diagonal with eigenvalues**.

特征分解通过**计算→验证→重建**学习：

1. **计算**：使用`np.linalg.eig()`查找特征值和特征向量
2. **验证**：检查A·v = λ·v（基本特征值方程）
3. **重建**：从Q、Λ、Q^(-1)构建A以理解分解的完整性

这个进度教导特征分解不是魔法——它是一个具体的因子分解，您可以验证和重建。见解：**Q将特征向量作为列包含，Λ是具有特征值的对角矩阵**。

## ML Relevance / 机器学习相关性

In machine learning:
- **PCA (Principal Component Analysis)**:
  - Eigendecomposition of covariance matrix: Cov = X^T · X / n
  - Eigenvectors = principal components (directions of max variance)
  - Eigenvalues = variance along each component
  - Dimensionality reduction: keep top-k eigenvectors, discard rest

- **PageRank (Google's algorithm)**:
  - Web as directed graph: link matrix L
  - Pagerank = eigenvector of L (the "steady state")
  - Eigenvalue 1: stationary distribution of random walk

- **Neural network analysis**:
  - Loss landscape: Hessian eigenvalues tell about curvature
  - Large eigenvalues: sharp valleys (unstable)
  - Small eigenvalues: flat regions (plateaus in training)
  - Spectral analysis: understand training dynamics

- **Data decorrelation**:
  - Whitening: eigendecomposition of covariance
  - Transform data to uncorrelated axes: `X_white = X · Q · Λ^(-1/2)`

- **Graph neural networks**:
  - Spectral GNNs: operate on eigenvalues/eigenvectors of adjacency matrix
  - Laplacian eigenvectors: smoothness of functions on graphs

**Why it matters**: Eigendecomposition reveals the "intrinsic structure" of a matrix. The eigenvectors are the "special directions" where the transformation is just scaling. PCA is essentially eigendecomposition of covariance, making it a fundamental ML technique.

在机器学习中：
- **PCA（主成分分析）**：
  - 协方差矩阵的特征分解：Cov = X^T · X / n
  - 特征向量=主成分（最大方差的方向）
  - 特征值=沿每个分量的方差
  - 降维：保留前k个特征向量，丢弃其余的

- **PageRank（Google的算法）**：
  - Web作为有向图：链接矩阵L
  - Pagerank = L的特征向量（"稳态"）
  - 特征值1：随机游走的平稳分布

- **神经网络分析**：
  - 损失景观：Hessian特征值告诉关于曲率
  - 大特征值：锐谷（不稳定）
  - 小特征值：平坦区域（训练中的平台）
  - 谱分析：理解训练动态

- **数据去相关**：
  - 白化：协方差的特征分解
  - 将数据变换为不相关的轴：`X_white = X · Q · Λ^(-1/2)`

- **图神经网络**：
  - 谱GNN：在邻接矩阵的特征值/特征向量上操作
  - 拉普拉斯特征向量：图上函数的平滑性

**为什么重要**：特征分解揭示了矩阵的"固有结构"。特征向量是"特殊方向"，其中变换只是缩放。PCA本质上是协方差的特征分解，使其成为一个基础ML技术。

---
