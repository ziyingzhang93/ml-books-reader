# 线性代数与机器学习 / Linear Algebra for Machine Learning
## Chapter 17

---

### Svd

# 01 — Singular Value Decomposition (SVD) / 特異值分解

**Chapter 17 — File 1 of 7 / 第17章 — 第1个文件（兦7个）**

---

## Summary / 总结

This script demonstrates **Singular Value Decomposition (SVD)**, the most general matrix factorization that works for **any** rectangular or square matrix (even rank-deficient ones).

本脚本演示 **特異值分解 (SVD)**，最需简的矩阵因子分解，適用于**任何**矩形矩阵。

### Core Formula / 核心公式

$$A = U \Sigma V^T$$

| Symbol / 符号 | Name / 名称 | Description / 描述 |
|:---:|:---|:---|
| $U$ | Left singular vectors / 左特異向量 | $m \times m$ orthogonal / $m \times m$ 正交 |
| $\Sigma$ | Singular values / 特異值 | $m \times n$ diagonal (sorted $\sigma_1 \geq \sigma_2 \geq \ldots$) / 对角矩阵 |
| $V^T$ | Right singular vectors (transposed) / 右特異向量转置 | $n \times n$ orthogonal / $n \times n$ 正交 |

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Import Libraries / 导入渪

```python
from numpy import array          # NumPy 数组工具 / NumPy array utility
from scipy.linalg import svd     # SVD 分解函数 / SVD decomposition function
```

---
## Step 2 — Define a Rectangular Matrix / 定义矩形矩阵

SVD works with **any** rectangular or square matrix, even those with more rows than columns.

SVD 適用于**任何**矩形矩阵，包括不是方阵的矩阵。

```python
A = array([[1, 2],
           [3, 4],
           [5, 6]])

# 打印输出 / Print output
print("A (3x2 matrix / 3×2 矩阵) =")
# 打印输出 / Print output
print(A)
```

---
## Step 3 — Perform SVD / 执行 SVD 分解

`scipy.linalg.svd()` returns three components:

- **U** — Left singular vectors (orthogonal): shape ($m$, $m$) / 左特異向量
- **s** — Singular values (sorted, $\geq 0$): 1D array / 特異值（一维数组）
- **VT** — Right singular vectors transposed (orthogonal): shape ($n$, $n$) / 右特異向量转置

```python
U, s, VT = svd(A)     # SVD 分解 / SVD decomposition

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("U (left singular vectors / 左特異向量, shape", U.shape, ") =")
# 打印输出 / Print output
print(U)
# 打印输出 / Print output
print()
# 打印输出 / Print output
print("s (singular values / 特異值) =")
# 打印输出 / Print output
print(s)
# 打印输出 / Print output
print()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("VT (right singular vectors transposed / 右特異向量转置, shape", VT.shape, ") =")
# 打印输出 / Print output
print(VT)
```

---
## Learning Notes / 学习笔记

- **数学本质**: SVD 是 **最一般** 的矩阵因子分解，適用于任何矩阵（不必是方阵）。会提取数据最重要的主成分。  
  *SVD is the most general decomposition — works on any matrix and reveals the most important components.*

- **ML 应用**: PCA, 數據精伐（低稷表不）、可推茶系统、条件數估計、区倧伏因数分解。  
  *PCA, image compression, recommender systems, condition number estimation, and noise reduction.*

---

➡️ **Next / 下一步**: `02_reconstruct_rectangular_matrix.ipynb` — 用 SVD 重构矩形矩阵。  
*Reconstruct rectangular matrices from SVD.*

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
# Singular Value Decomposition / 特異值分解
# Complete Code / 完整代码
# ===============================

# --- Import libraries / 导入渪 ---
from numpy import array          # NumPy 数组工具 / NumPy array utility
from scipy.linalg import svd     # SVD 分解函数 / SVD decomposition function

# --- Define a rectangular matrix / 定义矩形矩阵 ---
A = array([[1, 2],
           [3, 4],
           [5, 6]])
# 打印输出 / Print output
print("A (3x2) =")
# 打印输出 / Print output
print(A)

# --- SVD decomposition / SVD 分解 ---
# A = U · Σ · V^T
#   U: left singular vectors (mxm orthogonal) / 左特異向量
#   s: singular values (sorted, ≥ 0) / 特異值
#   VT: right singular vectors transposed (nxn orthogonal) / 右特異向量转置
U, s, VT = svd(A)

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("\nU (left singular vectors, shape", U.shape, ") =")
# 打印输出 / Print output
print(U)
# 打印输出 / Print output
print("\ns (singular values) =")
# 打印输出 / Print output
print(s)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("\nVT (right singular vectors transposed, shape", VT.shape, ") =")
# 打印输出 / Print output
print(VT)
```

---

### Reconstruct Rectangular Matrix

# 02 — Reconstruct Rectangular Matrix / 重构矩形矩阵

**Chapter 17 — File 2 of 7 / 第17章 — 第2个文件（兦7个）**

---

## Summary / 总结

This script demonstrates how to **reconstruct a rectangular matrix** from its SVD factors. The key challenge is handling the shape mismatch: $\Sigma$ must be a $m \times n$ matrix (not square).

本脚本演示如何从 SVD 重构矩形矩阵。键不是处理形状不匹配：$\Sigma$ 必须是 $m \times n$ 矩阵。

### Core Formula / 核心公式

$$A = U \Sigma V^T$$

where $\Sigma$ is constructed as:
- Create a $m \times n$ zero matrix
- Fill the diagonal with singular values from `s`

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Import Libraries / 导入渪

```python
from numpy import array          # NumPy 数组工具 / NumPy array utility
from numpy import diag           # Create diagonal matrix / 创建对角矩阵
from numpy import zeros          # Create zero matrix / 创建零矩阵
from scipy.linalg import svd     # SVD 分解函数 / SVD decomposition function
```

---
## Step 2 — Define Rectangular Matrix and Compute SVD / 定义矩形矩阵并计算 SVD

```python
A = array([[1, 2],
           [3, 4],
           [5, 6]])

# 打印输出 / Print output
print("A (3x2 rectangular matrix) =")
# 打印输出 / Print output
print(A)

U, s, VT = svd(A)     # SVD 分解 / SVD decomposition

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("\nU shape:", U.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("s shape:", s.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("VT shape:", VT.shape)
```

---
## Step 3 — Construct $\Sigma$ with Correct Shape / 构建正确形状的 $\Sigma$ 矩阵

Since `svd()` returns `s` as a 1D array (not a matrix), we must construct $\Sigma$ manually:
1. Create a $m \times n$ zero matrix (same shape as $A$)
2. Place singular values on the diagonal

SVD 返回 `s` 是一维数组，我们需要手工构建 $\Sigma$：
1. 创建 $m \times n$ 零矩阵
2. 将特異值填入对角

```python
# Create m x n Sigma matrix / 创建 m x n Sigma 矩阵
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
Sigma = zeros((A.shape[0], A.shape[1]))

# Fill diagonal with singular values / 将特異值填入对角
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
Sigma[:A.shape[1], :A.shape[1]] = diag(s)

# 打印输出 / Print output
print("Sigma (3x2 diagonal matrix / 对角矩阵) =")
# 打印输出 / Print output
print(Sigma)
```

---
## Step 4 — Reconstruct Original Matrix / 重构原矩阵

Multiply back: $A = U \cdot \Sigma \cdot V^T$

相乘回来：$A = U \cdot \Sigma \cdot V^T$

```python
B = U.dot(Sigma.dot(VT))     # U · Sigma · VT should equal A

# 打印输出 / Print output
print("Reconstructed B = U · Sigma · V^T =")
# 打印输出 / Print output
print(B)
# 打印输出 / Print output
print()
# 打印输出 / Print output
print("Original A =")
# 打印输出 / Print output
print(A)
```

---
## Learning Notes / 学习笔记

- **数学本质**: 矩形矩阵 SVD 最带力的一点是：U 拾 U 是正交，V 拾 V 是正交，特異值透漏矩阵的秘密。  
  *SVD's power: $U$ and $V$ are orthogonal, $\Sigma$ encodes all scale information in a diagonal matrix.*

- **ML 应用**: 矩形矩阵报例：推茶泻需の擬行推茶、图像可際、暱溘措沙。  
  *Image compression, low-rank approximation, recommender systems.*

---

➡️ **Next / 下一步**: `03_reconstruct_square_matrix.ipynb` — 特次方矩阵的 SVD 重构。  
*SVD reconstruction for square matrices.*

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
# Reconstruct Rectangular Matrix / 重构矩形矩阵
# Complete Code / 完整代码
# ===============================

# --- Import libraries / 导入渪 ---
from numpy import array          # NumPy 数组工具 / NumPy array utility
from numpy import diag           # Create diagonal matrix / 创建对角矩阵
from numpy import zeros          # Create zero matrix / 创建零矩阵
from scipy.linalg import svd     # SVD 分解函数 / SVD decomposition function

# --- Define rectangular matrix and compute SVD / 定义矩形矩阵并计算 SVD ---
A = array([[1, 2],
           [3, 4],
           [5, 6]])
# 打印输出 / Print output
print("A (3x2 matrix) =")
# 打印输出 / Print output
print(A)

U, s, VT = svd(A)

# --- Construct Sigma with correct shape / 构建正确形状的 Sigma 矩阵 ---
# Create m x n zero matrix / 创建 m x n 零矩阵
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
Sigma = zeros((A.shape[0], A.shape[1]))
# Fill diagonal with singular values / 将特異值填入对角
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
# 打印输出 / Print output
print("\nSigma (3x2 diagonal matrix) =")
# 打印输出 / Print output
print(Sigma)

# --- Reconstruct original matrix / 重构原矩阵 ---
# A = U · Sigma · V^T
B = U.dot(Sigma.dot(VT))
# 打印输出 / Print output
print("\nReconstructed B = U · Sigma · V^T =")
# 打印输出 / Print output
print(B)
```

---

### Reconstruct Square Matrix

# 03 — Reconstruct Square Matrix / 重构方阵

**Chapter 17 — File 3 of 7 / 第17章 — 第3个文件（兦7个）**

---

## Summary / 总结

This script demonstrates **SVD reconstruction for square matrices**. The difference from the rectangular case: $\Sigma$ is now a true square diagonal matrix, making reconstruction simpler.

本脚本演示下方矩阵的 SVD 重构。较之矩形矩阵案例，Sigma 此时是真正的对角矩阵。

### Core Formula / 核心公式

$$A = U \Sigma V^T$$

For square matrices, $\Sigma$ is a true diagonal matrix (all dimensions equal).

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Import Libraries / 导入渪

```python
from numpy import array          # NumPy 数组工具 / NumPy array utility
from numpy import diag           # Create diagonal matrix / 创建对角矩阵
from scipy.linalg import svd     # SVD 分解函数 / SVD decomposition function
```

---
## Step 2 — Define Square Matrix and Compute SVD / 定义方阵并计算 SVD

```python
A = array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

# 打印输出 / Print output
print("A (3x3 square matrix / 3×3 方阵) =")
# 打印输出 / Print output
print(A)

U, s, VT = svd(A)     # SVD 分解 / SVD decomposition

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("\nU shape:", U.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("s shape:", s.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("VT shape:", VT.shape)
```

---
## Step 3 — Construct Diagonal $\Sigma$ Matrix / 构建对角 $\Sigma$ 矩阵

For square matrices, we can use `numpy.diag()` directly since $\Sigma$ is now square.

下方矩阵情下，Sigma 是另方形的，我们可以直接使用 `numpy.diag()`。

```python
Sigma = diag(s)     # Create diagonal matrix from singular values / 从特異值创建对角矩阵

# 打印输出 / Print output
print("Sigma (3x3 diagonal matrix / 对角矩阵) =")
# 打印输出 / Print output
print(Sigma)
```

---
## Step 4 — Reconstruct Original Matrix / 重构原矩阵

Multiply back: $A = U \cdot \Sigma \cdot V^T$

相乘回来：$A = U \cdot \Sigma \cdot V^T$

```python
B = U.dot(Sigma.dot(VT))     # Reconstruct: A = U · Sigma · V^T / 重构：A = U 乘以 Sigma 乘以 VT

# 打印输出 / Print output
print("Reconstructed B = U · Sigma · V^T =")
# 打印输出 / Print output
print(B)
# 打印输出 / Print output
print()
# 打印输出 / Print output
print("Original A =")
# 打印输出 / Print output
print(A)
```

---
## Learning Notes / 学习笔记

- **数学本质**: SVD 有一整套理论：任何矩阵都可以分解为正交矩阵 × 对角矩阵 × 正交矩阵。  
  *SVD is the universal factorization: any matrix = orthogonal × diagonal × orthogonal.*

- **ML 应用**: 是最实病的被伯基：PCA、嚱分算法、条件數、單位根提、整梯度碣练算法。  
  *Basis for PCA, clustering, condition estimation, pseudoinverse, and dimensionality reduction.*

---

➡️ **Next / 下一步**: `04_pseudoinverse.ipynb` — 使用哺伺曉总租绋贫帮譩。  
*Computing the pseudoinverse.*

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
# Reconstruct Square Matrix / 重构方阵
# Complete Code / 完整代码
# ===============================

# --- Import libraries / 导入渪 ---
from numpy import array          # NumPy 数组工具 / NumPy array utility
from numpy import diag           # Create diagonal matrix / 创建对角矩阵
from scipy.linalg import svd     # SVD 分解函数 / SVD decomposition function

# --- Define square matrix and compute SVD / 定义方阵并计算 SVD ---
A = array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
# 打印输出 / Print output
print("A (3x3 square matrix) =")
# 打印输出 / Print output
print(A)

U, s, VT = svd(A)

# --- Construct diagonal Sigma matrix / 构建对角 Sigma 矩阵 ---
# For square matrices, Sigma is a true square diagonal matrix
# 下方矩阵情下，Sigma 是真正的对角矩阵
Sigma = diag(s)
# 打印输出 / Print output
print("\nSigma (3x3 diagonal matrix) =")
# 打印输出 / Print output
print(Sigma)

# --- Reconstruct original matrix / 重构原矩阵 ---
# A = U · Sigma · V^T
B = U.dot(Sigma.dot(VT))
# 打印输出 / Print output
print("\nReconstructed B = U · Sigma · V^T =")
# 打印输出 / Print output
print(B)
```

---

### Pseudoinverse

# 04 — Pseudoinverse / 哺伺曉总租绋贫

**Chapter 17 — File 4 of 7 / 第17章 — 第4个文件（兦7个）**

---

## Summary / 总结

This script demonstrates the **pseudoinverse** (Moore-Penrose inverse) of a non-square matrix using NumPy. The pseudoinverse allows us to solve least-squares problems and generalize matrix inversion to rectangular matrices.

本脚本演示下方矩阵的 **哺伺曉总租绋贫** (Moore-Penrose 总租绋贫)。哺伺曉总租绋贫允许我们解决最小二乘问题。

### Core Formula / 核心公式

For a matrix $A$ with shape $m \times n$:

$$A^+ = V \Sigma^+ U^T$$

where $\Sigma^+$ is the pseudoinverse of $\Sigma$ (reciprocals of nonzero singular values).

| Symbol / 符号 | Name / 名称 | Description / 描述 |
|:---:|:---|:---|
| $A^+$ | Pseudoinverse / 哺伺曉总租绋贫 | Generalized inverse for non-square matrices / 验次方矩阵 |
| Shape | $n \times m$ | Transpose of input shape / 输入形状的转置 |

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Import Libraries / 导入渪

```python
from numpy import array              # NumPy 数组工具 / NumPy array utility
from numpy.linalg import pinv        # Pseudoinverse / 哺伺曉总租绋贫
```

---
## Step 2 — Define a Rectangular Matrix / 定义矩形矩阵

Pseudoinverse is typically used for rectangular matrices (more rows than columns). In this case, we have a $4 \times 2$ matrix.

哺伺曉总租绋贫通常用于矩形矩阵。这里是一个 $4 \times 2$ 矩阵。

```python
A = array([[0.1, 0.2],
           [0.3, 0.4],
           [0.5, 0.6],
           [0.7, 0.8]])

# 打印输出 / Print output
print("A (4x2 rectangular matrix) =")
# 打印输出 / Print output
print(A)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Shape:", A.shape)
```

---
## Step 3 — Compute Pseudoinverse / 计算哺伺曉总租绋贫

`numpy.linalg.pinv()` computes the Moore-Penrose pseudoinverse. For a $m \times n$ matrix, it returns a $n \times m$ matrix.

`numpy.linalg.pinv()` 计算 Moore-Penrose 哺伺曉总租绋贫。输出形状是 $n \times m$。

```python
B = pinv(A)     # Compute pseudoinverse / 计算哺伺曉总租绋贫

# 打印输出 / Print output
print("Pseudoinverse B = A^+ (2x4 matrix) =")
# 打印输出 / Print output
print(B)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Shape:", B.shape)
```

---
## Learning Notes / 学习笔记

- **数学本质**: 哺伺曉总租绋贫是正常总租绋贫的需飞尽量推握：对于靮余矩阵 $(A^T A)^{-1} A^T$。  
  *Pseudoinverse generalizes matrix inversion to non-square matrices: solves least-squares as $A^+ b$.*

- **ML 应用**: 是最小二乘法、线性回帰、条件数估計、介患勾連。  
  *Least-squares regression, solving overdetermined systems, condition number analysis, robust fitting.*

---

➡️ **Next / 下一步**: `05_svd_pseudoinverse.ipynb` — 使用 SVD 计算哺伺曉总租绋贫。  
*Computing pseudoinverse via SVD.*

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
# Pseudoinverse / 哺伺曉总租绋贫
# Complete Code / 完整代码
# ===============================

# --- Import libraries / 导入渪 ---
from numpy import array              # NumPy 数组工具 / NumPy array utility
from numpy.linalg import pinv        # Pseudoinverse / 哺伺曉总租绋贫

# --- Define rectangular matrix / 定义矩形矩阵 ---
A = array([[0.1, 0.2],
           [0.3, 0.4],
           [0.5, 0.6],
           [0.7, 0.8]])
# 打印输出 / Print output
print("A (4x2 rectangular matrix) =")
# 打印输出 / Print output
print(A)

# --- Compute pseudoinverse / 计算哺伺曉总租绋贫 ---
# A^+ = V Σ^+ U^T (computed internally via SVD)
# The pseudoinverse shape is transposed: (2x4)
# 哺伺曉总租绋贫的形状是转置的：(2x4)
B = pinv(A)
# 打印输出 / Print output
print("\nPseudoinverse B = A^+ (2x4 matrix) =")
# 打印输出 / Print output
print(B)
```

---

### Svd Pseudoinverse

# 05 — SVD Pseudoinverse / 依 SVD 计算哺伺曉总租绋贫

**Chapter 17 — File 5 of 7 / 第17章 — 第5个文件（兦7个）**

---

## Summary / 总结

This script demonstrates how to **compute the pseudoinverse from SVD components**. Instead of using the direct `pinv()` function, we compute it manually using the SVD factors $U$, $\Sigma$, and $V^T$.

本脚本演示如何供 SVD 整另计算哺伺曉总租绋贫。查看 SVD 的原理是介患哺伺曉总租绋贫的竪核。

### Core Formula / 核心公式

From SVD $A = U \Sigma V^T$, the pseudoinverse is:

$$A^+ = V \Sigma^+ U^T$$

where $\Sigma^+$ is computed by taking reciprocals of nonzero singular values (and transposing).

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Import Libraries / 导入渪

```python
from numpy import array          # NumPy 数组工具 / NumPy array utility
from numpy.linalg import svd     # SVD 分解函数 / SVD decomposition function
from numpy import zeros          # Create zero matrix / 创建零矩阵
from numpy import diag           # Create diagonal matrix / 创建对角矩阵
```

---
## Step 2 — Define Matrix and Compute SVD / 定义矩阵并计算 SVD

```python
A = array([[0.1, 0.2],
           [0.3, 0.4],
           [0.5, 0.6],
           [0.7, 0.8]])

# 打印输出 / Print output
print("A (4x2 rectangular matrix) =")
# 打印输出 / Print output
print(A)

U, s, VT = svd(A)     # SVD 分解 / SVD decomposition

# 打印输出 / Print output
print("\nSingular values / 特異值:")
# 打印输出 / Print output
print(s)
```

---
## Step 3 — Construct $\Sigma^+$ (Pseudoinverse of Singular Values) / 构建 $\Sigma^+$ 

The key step: take reciprocals of singular values and transpose to get the right shape.

键步：初断特異值的值数的你狗线（输出形状是转置的）。

```python
# Compute reciprocals of singular values / 计算特異值的值数的你狗
d = 1.0 / s     # Element-wise reciprocal / 按元素沈整敗0

# 打印输出 / Print output
print("Reciprocals of singular values / 特異值的值数的你狗:")
# 打印输出 / Print output
print(d)

# Create m x n matrix (note the transposed shape!) / 创建 n x m 矩阵（注意形状）
D = zeros(A.shape)     # Create zero matrix with shape of A / 创建与 A 同形状的零矩阵

# Fill the diagonal (transposed) / 将值数的你狗填入对角
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
D[:A.shape[1], :A.shape[1]] = diag(d)

# 打印输出 / Print output
print("\nD = Sigma^+ (2x4 pseudoinverse of singular values) =")
# 打印输出 / Print output
print(D)
```

---
## Step 4 — Compute Pseudoinverse: $A^+ = V \Sigma^+ U^T$ / 计算哺伺曉总租绋贫

Apply the SVD pseudoinverse formula.

// 供 SVD 哺伺曉总租绋贫公式

```python
# A^+ = V · D^T · U^T / A^+ = V 乘以 D^T 乘以 U^T
B = VT.T.dot(D.T).dot(U.T)     # Pseudoinverse via SVD / 供 SVD 算算哺伺曉总租绋贫

# 打印输出 / Print output
print("Pseudoinverse B = V^T · D^T · U^T =")
# 打印输出 / Print output
print(B)
```

---
## Learning Notes / 学习笔记

- **数学本质**: 哺伺曉总租绋贫有好欺候的性质：三另事你最母、最小二乘、公理向量消除、另带棋燥暳。  
  *Pseudoinverse has beautiful properties: minimal norm, least-squares solution, orthogonal projection.*

- **ML 应用**: 是最小二乘法、验证根捎、条件數、介患质林尽量、嚱分算法。  
  *Least-squares, ridge regression, condition numbers, regularization, and noise reduction.*

---

➡️ **Next / 下一步**: `06_svd_data_reduction.ipynb` — 使用 SVD 的数据技什。  
*Data reduction via SVD.*

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
# SVD Pseudoinverse / 依 SVD 计算哺伺曉总租绋贫
# Complete Code / 完整代码
# ===============================

# --- Import libraries / 导入渪 ---
from numpy import array          # NumPy 数组工具 / NumPy array utility
from numpy.linalg import svd     # SVD 分解函数 / SVD decomposition function
from numpy import zeros          # Create zero matrix / 创建零矩阵
from numpy import diag           # Create diagonal matrix / 创建对角矩阵

# --- Define rectangular matrix and compute SVD / 定义矩形矩阵并计算 SVD ---
A = array([[0.1, 0.2],
           [0.3, 0.4],
           [0.5, 0.6],
           [0.7, 0.8]])
# 打印输出 / Print output
print("A (4x2 rectangular matrix) =")
# 打印输出 / Print output
print(A)

U, s, VT = svd(A)
# 打印输出 / Print output
print("\nSingular values / 特異值:")
# 打印输出 / Print output
print(s)

# --- Construct Sigma^+ / 构建 Sigma^+ ---
# Compute reciprocals of singular values / 计算特異值的值数的你狗
d = 1.0 / s
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
D = zeros(A.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
D[:A.shape[1], :A.shape[1]] = diag(d)
# 打印输出 / Print output
print("\nSigma^+ (pseudoinverse of singular values) =")
# 打印输出 / Print output
print(D)

# --- Compute pseudoinverse / 计算哺伺曉总租绋贫 ---
# A^+ = V Σ^+ U^T
B = VT.T.dot(D.T).dot(U.T)
# 打印输出 / Print output
print("\nPseudoinverse B = V^T · Sigma^+ · U^T =")
# 打印输出 / Print output
print(B)
```

---

### Svd Data Reduction



---

### Sklearn Data Reduction

# 07 — Scikit-learn Data Reduction / 使用 scikit-learn 的数据压缩

**Chapter 17 — File 7 of 7 / 第17章 — 第7个文件（兦7个）**

---

## Summary / 总结

This script demonstrates how to perform **data reduction using scikit-learn's `TruncatedSVD`**. This is the practical, high-level API for SVD-based dimensionality reduction, which abstracts away the manual SVD manipulation from the previous lesson.

本脚本演示使用 scikit-learn 的 `TruncatedSVD` 进行数据压缩。这是修改的、项知源的 API，抵時推舶了摊千事悲幰的 SVD 上手力。

### Core Concept / 核心概念

**TruncatedSVD** in scikit-learn:
- Automatically handles SVD computation and truncation
- Provides `.fit()` and `.transform()` interface
- Returns the low-dimensional representation directly

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 训练模型 / Train the model


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏋️ 训练模型 / Train Model
```

---
## Step 1 — Import Libraries / 导入渪

```python
from numpy import array                          # NumPy 数组工具 / NumPy array utility
from sklearn.decomposition import TruncatedSVD  # TruncatedSVD from scikit-learn / TruncatedSVD 来自 scikit-learn
```

---
## Step 2 — Define Data Matrix / 定义数据矩阵

```python
A = array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
           [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
           [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]])

# 打印输出 / Print output
print("A (original 3x10 matrix) =")
# 打印输出 / Print output
print(A)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Shape:", A.shape)
```

---
## Step 3 — Create TruncatedSVD Model / 创建 TruncatedSVD 模式

Initialize with the desired number of components (dimensions to keep).

使用想要保留的维度数量进行初始化。

```python
# Create TruncatedSVD with 2 components / 使用 2 个维度剋成 TruncatedSVD
svd = TruncatedSVD(n_components=2)     # Keep 2 components / 保留 2 个维度

# 打印输出 / Print output
print("TruncatedSVD model created with n_components=2 / 创建了 n_components=2 的模式")
```

---
## Step 4 — Fit the Model and Transform Data / 拓儗模式并转换数据

Use the `.fit()` and `.transform()` interface to perform SVD and return the compressed data.

使用 `.fit()` 后由 `.transform()` 执行 SVD 分解，然后转换数据。

```python
# Fit and transform in one step / 拓儗并转换数据
svd.fit(A)
result = svd.transform(A)     # Transform to lower dimensions / 转换为低维数据

# 打印输出 / Print output
print("\nTransformed data (3x2 low-rank representation) =")
# 打印输出 / Print output
print(result)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Shape:", result.shape)
```

---
## Learning Notes / 学习笔记

- **数学本质**: TruncatedSVD 是扊其业剋成的 SVD 分解子鱼：只计算最旨桀的特異值和特異向量，不计算整个 SVD 需海。  
  *TruncatedSVD computes only the top $k$ singular values (efficient for large data).*

- **ML 应用**: 是 PCA 旨作、特高叨攱、推舶系统、网一网算法中最漫一模形所用的量化。  
  *Fast data reduction for large matrices, dimensionality reduction, feature extraction, recommender systems.*

---

✨ **End of Chapter 17** / ✨ **第17章结束**  
*You have completed all SVD lessons! Congratulations!*

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
# Scikit-learn Data Reduction / 使用 scikit-learn 的数据压缩
# Complete Code / 完整代码
# ===============================

# --- Import libraries / 导入渪 ---
from numpy import array                          # NumPy 数组工具 / NumPy array utility
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import TruncatedSVD  # TruncatedSVD from scikit-learn

# --- Define data matrix / 定义数据矩阵 ---
A = array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
           [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
           [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]])
# 打印输出 / Print output
print("A (original 3x10 matrix) =")
# 打印输出 / Print output
print(A)

# --- Create and fit TruncatedSVD model / 创建和拓儗 TruncatedSVD 模式 ---
# n_components: number of dimensions to keep / n_components: 保留的维度数量
svd = TruncatedSVD(n_components=2)
svd.fit(A)

# --- Transform data to lower dimensions / 转换为低维表示 ---
# 用已拟合的模型转换数据 / Transform data with fitted model
result = svd.transform(A)

# 打印输出 / Print output
print("\nTransformed data (low-dimensional representation) =")
# 打印输出 / Print output
print(result)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print("Original shape: (3, 10) -> Transformed shape:", result.shape)
```

---

### Chapter Summary / 章节总结

# Chapter 17 Summary / 第17章总结：SVD (Singular Value Decomposition)

## Theme / 主题

SVD is the most powerful matrix decomposition: A = U·Σ·V^T. Unlike eigendecomposition (square matrices only), SVD works on any rectangular matrix. It reveals the rank structure, enables matrix inversion via pseudoinverse, and is the foundation for dimensionality reduction and low-rank approximation. SVD is the algorithm behind PCA, recommender systems, and image compression.

SVD是最强大的矩阵分解：A = U·Σ·V^T。与特征分解（仅方阵）不同，SVD适用于任何矩形矩阵。它揭示秩结构，通过伪逆实现矩阵反演，是降维和低秩逼近的基础。SVD是PCA、推荐系统和图像压缩背后的算法。

## Evolution / 演化路线

```
01_basic_svd.ipynb
    └─ Compute U, Σ, V^T (基础SVD分解)
    
02_reconstruct_rectangular.ipynb
    └─ Reconstruct rectangular matrices (重建矩形矩阵)
    
03_reconstruct_square.ipynb
    └─ Reconstruct square matrices (重建方阵)
    
04_pseudoinverse.ipynb
    └─ Compute pseudoinverse A^+ = V · Σ^+ · U^T (伪逆计算)
    
05_lstsq_via_svd.ipynb
    └─ Least squares via SVD (通过SVD的最小二乘)
    
06_data_reduction.ipynb
    └─ Keep top-k singular values for compression (保留前k个奇异值进行压缩)
    
07_sklearn_svd.ipynb
    └─ Use scikit-learn's TruncatedSVD (使用scikit-learn的截断SVD)
```

## Progression Logic / 进度逻辑

SVD is learned through **computing → reconstructing → applying**:

1. **Basic SVD**: Compute full U, Σ, V^T
2. **Reconstruction**: Verify A = U·Σ·V^T works for rectangular and square
3. **Pseudoinverse**: A^+ = V·Σ^+·U^T solves least squares when A is singular
4. **Dimensionality reduction**: Keep top-k singular values, discard rest → compress data
5. **Production API**: Use scikit-learn's TruncatedSVD for practical use

Key insight: **singular values are sorted by importance**. Keep top-k = keep most important directions. This is why SVD is perfect for low-rank approximation and dimensionality reduction.

SVD通过**计算→重建→应用**学习：

1. **基本SVD**：计算完整U、Σ、V^T
2. **重建**：验证A = U·Σ·V^T对矩形和方阵的工作
3. **伪逆**：A^+ = V·Σ^+·U^T当A是奇异时求解最小二乘
4. **降维**：保留前k个奇异值，丢弃其余的→压缩数据
5. **生产API**：使用scikit-learn的截断SVD用于实际使用

关键见解：**奇异值按重要性排序**。保留前k = 保留最重要的方向。这就是为什么SVD对低秩逼近和降维完美的原因。

## ML Relevance / 机器学习相关性

In machine learning:
- **PCA via SVD** (alternative to eigendecomposition):
  - Data matrix X (centered), compute SVD: X = U·Σ·V^T
  - Principal components = columns of V
  - Explained variance = σ_i^2 / sum(σ^2)
  - Faster and more stable than eigendecomposition

- **Least squares via SVD**:
  - Normal equation: w = (X^T·X)^(-1)·X^T·y is numerically unstable
  - SVD approach: w = V·Σ^+·U^T·y is stable (handles rank-deficiency)
  - Pseudoinverse: A^+ = V·Σ^+·U^T automatically handles singular matrices

- **Recommender systems**:
  - User-item matrix A (sparse)
  - SVD: A ≈ U·Σ·V^T (low-rank approximation)
  - U = user embeddings, V = item embeddings
  - Predictions: reconstruct A with top-k factors

- **Dimensionality reduction**:
  - TruncatedSVD: keep top-k singular vectors
  - Reduces from n_features to k dimensions
  - Captures ~90% of variance with k << n

- **Image compression**:
  - Image = matrix M (pixel values)
  - Keep top-k singular values → compressed image
  - Trade-off: reconstruction error vs. compression ratio

- **Data denoising**:
  - Noise concentrates in small singular values
  - Truncate small σ_i → denoise data

- **Low-rank approximation**:
  - Original matrix A is m×n
  - A_k = U_k·Σ_k·V_k^T is best rank-k approximation (Eckart-Young theorem)
  - Used in matrix completion, recommendation, collaborative filtering

**Why SVD is king**: It's the most versatile decomposition. Unlike QR (only rectangular), unlike eigendecomposition (only square), SVD works on ANY matrix. It's the universal tool.

在机器学习中：
- **通过SVD的PCA**（特征分解的替代方案）：
  - 数据矩阵X（中心化），计算SVD：X = U·Σ·V^T
  - 主成分=V的列
  - 解释方差= σ_i^2 / sum(σ^2)
  - 比特征分解更快和更稳定

- **通过SVD的最小二乘**：
  - 正规方程：w = (X^T·X)^(-1)·X^T·y在数值上不稳定
  - SVD方法：w = V·Σ^+·U^T·y稳定（处理秩不足）
  - 伪逆：A^+ = V·Σ^+·U^T自动处理奇异矩阵

- **推荐系统**：
  - 用户-项目矩阵A（稀疏）
  - SVD：A ≈ U·Σ·V^T（低秩逼近）
  - U =用户嵌入，V =项目嵌入
  - 预测：用前k个因子重建A

- **降维**：
  - 截断SVD：保留前k个奇异向量
  - 从n_features减少到k维
  - 用k << n捕获~90%的方差

- **图像压缩**：
  - 图像=矩阵M（像素值）
  - 保留前k个奇异值→压缩的图像
  - 权衡：重建误差vs.压缩比率

- **数据去噪**：
  - 噪声集中在小奇异值中
  - 截断小的σ_i→去噪数据

- **低秩逼近**：
  - 原始矩阵A是m×n
  - A_k = U_k·Σ_k·V_k^T是最佳秩k逼近（Eckart-Young定理）
  - 用于矩阵完成、推荐、协作过滤

**为什么SVD是王者**：它是最通用的分解。与QR不同（仅矩形），与特征分解不同（仅方阵），SVD适用于任何矩阵。它是通用工具。

---
