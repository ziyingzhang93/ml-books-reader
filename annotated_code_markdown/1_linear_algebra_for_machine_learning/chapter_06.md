# 线性代数与机器学习
## Chapter 06

---

### Scalar 2D

## File 2 of 4 / 第3章 — 第2个文件（共4个）

### Summary / 总结
Learn how NumPy broadcasts scalar values with 2D arrays (matrices). Broadcasting extends to higher dimensions seamlessly.

学习NumPy如何将标量值与二维数组（矩阵）广播。广播无缝扩展到更高维度。

### Core Concept / 核心概念
**Broadcasting Scalar to 2D**: A scalar is broadcast to all elements in a 2D array. Example: [[1,2,3],[1,2,3]] + 2 = [[3,4,5],[3,4,5]].

**将标量广播到二维**: 标量广播到二维数组中的所有元素。示例：[[1,2,3],[1,2,3]] + 2 = [[3,4,5],[3,4,5]]。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 2D Array and Scalar / 创建二维数组和标量

We create a 2×3 matrix and a scalar value 2. The scalar will be added to every element in the matrix.

我们创建一个2×3矩阵和标量值2。标量将被添加到矩阵中的每个元素。

```python
# 导入array函数 / Import array function
from numpy import array

# 创建二维数组和标量 / Create 2D array and scalar
A = array([[1, 2, 3], [1, 2, 3]])
b = 2

print("Matrix A:")
print(A)
print("\nScalar b:")
print(b)
```

## Step 2 — Add Scalar to 2D Array / 将标量添加到二维数组

We add the scalar 2 to the entire matrix. NumPy broadcasts 2 to all 6 elements (2 rows × 3 columns).

我们将标量2添加到整个矩阵。NumPy将2广播到所有6个元素（2行×3列）。

```python
# 使用广播进行加法 / Add using broadcasting
C = A + b
print("\nResult of A + b (scalar broadcast to all elements):")
print(C)
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Broadcasting a scalar to a matrix applies the operation element-wise. For matrix **A** ∈ ℝ^(m×n) and scalar b, **A** + b adds b to each element A_{ij}.

将标量广播到矩阵会逐元素应用操作。对于矩阵**A** ∈ ℝ^(m×n)和标量b，**A** + b将b添加到每个元素A_{ij}。

• **ML Application / ML application**: In ML, broadcasting scalars to matrices is common: adding a bias vector to predictions, normalizing by subtracting mean, or scaling all features by a global factor. This makes code concise and efficient.

在ML中，将标量广播到矩阵很常见：向预测添加偏差向量、通过减去平均值来规范化或按全局因子缩放所有特征。这使代码简洁高效。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 03_1d_to_2d.ipynb** — Learn how broadcasting works between 1D and 2D arrays.

## Complete Code / 完整代码一览

```python
# --- Broadcasting Scalar to 2D Array / 标量广播到二维数组 ---
from numpy import array

# Create 2D array and scalar / 创建二维数组和标量
A = array([[1, 2, 3], [1, 2, 3]])
b = 2

print("Matrix A:")
print(A)
print("Scalar b: {}".format(b))

# Add scalar to 2D array / 将标量添加到二维数组
C = A + b
print("\nResult of A + b (scalar broadcast):")
print(C)
```

---

### 1D To 2D

## File 3 of 4 / 第3章 — 第3个文件（共4个）

### Summary / 总结
Learn how NumPy broadcasts 1D arrays with 2D arrays. This is a fundamental broadcasting pattern for applying operations like adding a bias vector to every row.

学习NumPy如何将一维数组与二维数组广播。这是用于应用操作（如向每行添加偏差向量）的基本广播模式。

### Core Concept / 核心概念
**1D to 2D Broadcasting**: A 1D array with shape (n,) is broadcast across rows of a (m, n) matrix. Each row receives the 1D array added element-wise.

**一维到二维广播**: 形状为(n,)的一维数组跨矩阵(m, n)的行广播。每行接收逐元素添加的一维数组。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create 2D Array and 1D Array / 创建二维和一维数组

We create a 2×3 matrix and a 1D array with 3 elements. The 1D array will be broadcast to each row.

我们创建一个2×3矩阵和一个有3个元素的一维数组。一维数组将广播到每一行。

```python
# 导入array函数 / Import array function
from numpy import array

# 创建二维数组和一维数组 / Create 2D and 1D arrays
A = array([[1, 2, 3], [1, 2, 3]])
b = array([1, 2, 3])

print("Matrix A (shape {}):".format(A.shape))
print(A)
print("\n1D array b (shape {}):".format(b.shape))
print(b)
```

## Step 2 — Add 1D Array to 2D Array / 将一维数组添加到二维数组

We add the 1D array [1, 2, 3] to the 2D matrix. NumPy broadcasts the vector to each row, adding it element-wise.

我们将一维数组[1, 2, 3]添加到二维矩阵。NumPy将向量广播到每一行，逐元素添加它。

```python
# 使用广播进行加法 / Add using broadcasting
C = A + b
print("\nResult of A + b (1D array broadcast to each row):")
print(C)
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: When broadcasting a 1D vector **b** to a matrix **A**, the operation applies **b** to each row of **A**. Mathematically: if **A** ∈ ℝ^(m×n) and **b** ∈ ℝ^n, then (**A** + **b**)_i = A_i + **b** for each row i.

当将一维向量**b**广播到矩阵**A**时，操作将**b**应用于**A**的每一行。数学上：如果**A** ∈ ℝ^(m×n)和**b** ∈ ℝ^n，则(**A** + **b**)_i = A_i + **b**对于每一行i。

• **ML Application / ML application**: This broadcasting pattern is essential in ML: adding bias vectors in neural networks (adding bias to predictions for each training example), feature normalization (subtracting per-feature means), or applying row-wise scaling.

这种广播模式在ML中至关重要：在神经网络中添加偏差向量（为每个训练示例的预测添加偏差）、特征规范化（减去每个特征的平均值）或应用按行缩放。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next: 04_broadcast_error.ipynb** — Learn about broadcasting failures and shape incompatibilities.

## Complete Code / 完整代码一览

```python
# --- Broadcasting 1D to 2D Array / 将一维数组广播到二维数组 ---
from numpy import array

# Create 2D and 1D arrays / 创建二维和一维数组
A = array([[1, 2, 3], [1, 2, 3]])
b = array([1, 2, 3])

print("Matrix A (shape {}):".format(A.shape))
print(A)
print("\n1D array b (shape {}):".format(b.shape))
print(b)

# Add 1D to 2D using broadcasting / 使用广播将一维添加到二维
C = A + b
print("\nResult of A + b (1D broadcast to each row):")
print(C)
```

---

### Broadcast Error

## File 4 of 4 / 第3章 — 第4个文件（共4个）

### Summary / 总结
Learn about broadcasting failures and shape incompatibilities. Understanding when broadcasting fails is critical for debugging shape-related errors.

学习广播失败和形状不兼容。理解广播何时失败对于调试与形状相关的错误至关重要。

### Core Concept / 核心概念
**Broadcasting Rules**: Arrays broadcast only if dimensions are compatible: (m, n) + (n,) works, but (m, n) + (k,) fails if k ≠ n. Incompatible dimensions raise ValueError.

**广播规则**: 只有在维度兼容时数组才能广播：(m, n) + (n,)有效，但如果k ≠ n，(m, n) + (k,)会失败。不兼容的维度会引发ValueError。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create Incompatible Arrays / 创建不兼容的数组

We create a 2×3 matrix and a 1D array with 2 elements. These have incompatible shapes: (2, 3) cannot broadcast with (2,).

我们创建一个2×3矩阵和一个有2个元素的一维数组。这些具有不兼容的形状：(2, 3)无法与(2,)广播。

```python
# 导入array函数 / Import array function
from numpy import array

# 创建不兼容的数组 / Create incompatible arrays
A = array([[1, 2, 3], [1, 2, 3]])
print("Matrix A (shape {}):".format(A.shape))
print(A)

b = array([1, 2])
print("\n1D array b (shape {}):".format(b.shape))
print(b)
print("\nAttempting to add A + b...")
```

## Step 2 — Attempt Broadcasting / 尝试广播

We try to add b (shape 2,) to A (shape 2, 3). This fails because 2 ≠ 3, and broadcasting rules don't allow this operation.

我们尝试将b（形状2,）添加到A（形状2, 3）。这会失败，因为2 ≠ 3，广播规则不允许此操作。

```python
# 尝试不兼容的广播 / Try incompatible broadcasting
try:
    C = A + b
    print("Result:")
    print(C)
except ValueError as e:
    print("Error caught!")
    print("ValueError: {}".format(e))
    print("\nReason: Shapes (2,3) and (2,) are incompatible for broadcasting")
```

## Learning Notes / 学习笔记

• **Math Essence / 数学本质**: Broadcasting has strict rules. Two arrays are compatible if dimensions match from right to left, or one dimension is 1. For example: (2, 3) + (3,) works (broadcast second to all rows), but (2, 3) + (2,) fails (incompatible middle dimension).

广播有严格的规则。如果从右到左匹配维度，或一个维度为1，则两个数组兼容。例如：(2, 3) + (3,)有效（将第二个广播到所有行），但(2, 3) + (2,)失败（中间维度不兼容）。

• **ML Application / ML application**: Broadcasting errors are common debugging points. When a model fails with "operands could not be broadcast together", always check shapes. Use .shape attribute to verify before operations: if X.shape is (100, 20) and b.shape is (30,), addition will fail.

广播错误是常见的调试点。当模型因"操作数无法一起广播"而失败时，始终检查形状。在操作前使用.shape属性验证：如果X.shape是(100, 20)，b.shape是(30,)，则添加会失败。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

🏁 **End of Chapter 6** — You've mastered NumPy broadcasting!

## Complete Code / 完整代码一览

```python
# --- Broadcasting Error Example / 广播错误示例 ---
from numpy import array

# Create incompatible arrays / 创建不兼容的数组
A = array([[1, 2, 3], [1, 2, 3]])
print("Matrix A (shape {}):".format(A.shape))
print(A)

b = array([1, 2])
print("\n1D array b (shape {}):".format(b.shape))
print(b)

# Try incompatible broadcasting / 尝试不兼容的广播
try:
    C = A + b
    print("Result:")
    print(C)
except ValueError as e:
    print("\nError: {}".format(e))
    print("Shapes (2, 3) and (2,) are incompatible for broadcasting!")
```

---
