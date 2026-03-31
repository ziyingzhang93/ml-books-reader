# 线性代数与机器学习 / Linear Algebra for Machine Learning
## Chapter 10

---

### Create Matrix



---

### Matrix Addition

# 02 — Matrix Addition / 矩阵加法

**Chapter 10 — File 2 of 8 / 第10章 — 第2个文件（共8个）**

## Summary / 总结

Master matrix addition, where corresponding elements of two matrices are added element-wise.

掌握矩阵加法，其中两个矩阵的对应元素按元素逐个相加。

## Core Formula / 核心公式

Matrix addition is element-wise:
$$\mathbf{C} = \mathbf{A} + \mathbf{B} \text{ where } c_{ij} = a_{ij} + b_{ij}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy / 导入 NumPy

We import the array function from NumPy.

```python
# Import the array function from NumPy
# 从 NumPy 导入数组函数
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
```

## Step 2 — Create Two Matrices / 创建两个矩阵

We create two matrices with the same dimensions.

```python
# Create matrix A (2x3)
# 创建矩阵 A (2x3)
A = array([[1, 2, 3],[4, 5, 6]])

# Create matrix B (2x3) with same dimensions as A
# 创建矩阵 B (2x3)，与 A 具有相同维度
B = array([[1, 2, 3],[4, 5, 6]])
```

## Step 3 — Add the Matrices / 矩阵相加

We add the two matrices using the `+` operator. NumPy automatically performs element-wise addition.

```python
# Add matrices A and B element-wise
# Result: [[1+1, 2+2, 3+3], [4+4, 5+5, 6+6]] = [[2, 4, 6], [8, 10, 12]]
# 矩阵 A 和 B 按元素相加
# 结果：[[1+1, 2+2, 3+3], [4+4, 5+5, 6+6]] = [[2, 4, 6], [8, 10, 12]]
C = A + B

# Print the result
# 打印结果
# 打印输出 / Print output
print(C)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Matrix addition adds corresponding elements and produces a matrix of the same dimensions. Both matrices must have identical dimensions.
- **ML Application / 机器学习应用**: Matrix addition is used in bias addition to neural network activations and in combining multiple transformations or predictions.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `03_matrix_subtraction.ipynb` — Subtract one matrix from another

## Complete Code / 完整代码一览

```python
# --- Matrix Addition / 矩阵加法 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
A = array([[1, 2, 3],[4, 5, 6]])
B = array([[1, 2, 3],[4, 5, 6]])
C = A + B
# 打印输出 / Print output
print(C)
```

---

### Matrix Subtraction

# 03 — Matrix Subtraction / 矩阵减法

**Chapter 10 — File 3 of 8 / 第10章 — 第3个文件（共8个）**

## Summary / 总结

Learn matrix subtraction, where elements of one matrix are subtracted from another element-wise.

学习矩阵减法，其中一个矩阵的元素从另一个矩阵的对应元素中按元素减去。

## Core Formula / 核心公式

Matrix subtraction is element-wise:
$$\mathbf{C} = \mathbf{A} - \mathbf{B} \text{ where } c_{ij} = a_{ij} - b_{ij}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy / 导入 NumPy

We import the array function from NumPy.

```python
# Import the array function from NumPy
# 从 NumPy 导入数组函数
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
```

## Step 2 — Create Two Matrices / 创建两个矩阵

We create two matrices with the same dimensions.

```python
# Create matrix A (2x3)
# 创建矩阵 A (2x3)
A = array([[1, 2, 3],[4, 5, 6]])

# Create matrix B (2x3) with smaller values
# 创建矩阵 B (2x3)，具有较小的值
B = array([[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]])
```

## Step 3 — Subtract the Matrices / 矩阵相减

We subtract matrix B from matrix A using the `-` operator.

```python
# Subtract matrix B from matrix A element-wise
# Result: [[1-0.5, 2-0.5, 3-0.5], [4-0.5, 5-0.5, 6-0.5]]
# 矩阵 A 减去矩阵 B 按元素进行
# 结果：[[1-0.5, 2-0.5, 3-0.5], [4-0.5, 5-0.5, 6-0.5]]
C = A - B

# Print the result
# 打印结果
# 打印输出 / Print output
print(C)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Matrix subtraction subtracts corresponding elements and produces a matrix of the same dimensions. Both matrices must have identical dimensions.
- **ML Application / 机器学习应用**: Matrix subtraction is used to compute residuals (predicted minus actual values) and in error analysis during model evaluation.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `04_matrix_hadamard_product.ipynb` — Element-wise multiplication (Hadamard product)

## Complete Code / 完整代码一览

```python
# --- Matrix Subtraction / 矩阵减法 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
A = array([[1, 2, 3],[4, 5, 6]])
B = array([[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]])
C = A - B
# 打印输出 / Print output
print(C)
```

---

### Matrix Hadamard Product

# 04 — Matrix Hadamard Product / 矩阵 Hadamard 积

**Chapter 10 — File 4 of 8 / 第10章 — 第4个文件（共8个）**

## Summary / 总结

Learn the Hadamard product (element-wise multiplication), where corresponding elements of two matrices are multiplied together.

学习 Hadamard 积（按元素乘法），其中两个矩阵的对应元素相互相乘。

## Core Formula / 核心公式

The Hadamard product (element-wise multiplication):
$$\mathbf{C} = \mathbf{A} \odot \mathbf{B} \text{ where } c_{ij} = a_{ij} \times b_{ij}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy / 导入 NumPy

We import the array function from NumPy.

```python
# Import the array function from NumPy
# 从 NumPy 导入数组函数
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
```

## Step 2 — Create Two Matrices / 创建两个矩阵

We create two matrices with the same dimensions.

```python
# Create matrix A (2x3)
# 创建矩阵 A (2x3)
A = array([[1, 2, 3],[4, 5, 6]])

# Create matrix B (2x3) with same dimensions as A
# 创建矩阵 B (2x3)，与 A 具有相同维度
B = array([[1, 2, 3],[4, 5, 6]])
```

## Step 3 — Compute Hadamard Product / 计算 Hadamard 积

We compute the Hadamard product (element-wise multiplication) using the `*` operator. Note: This is NOT matrix multiplication.

```python
# Hadamard product: element-wise multiplication
# Result: [[1*1, 2*2, 3*3], [4*4, 5*5, 6*6]] = [[1, 4, 9], [16, 25, 36]]
# Hadamard 积：按元素乘法
# 结果：[[1*1, 2*2, 3*3], [4*4, 5*5, 6*6]] = [[1, 4, 9], [16, 25, 36]]
C = A * B

# Print the result (not the same as matrix multiplication)
# 打印结果（与矩阵乘法不同）
# 打印输出 / Print output
print(C)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: The Hadamard product multiplies corresponding elements to produce a matrix of the same dimensions. It is different from matrix multiplication, which involves dot products.
- **ML Application / 机器学习应用**: The Hadamard product is used in applying element-wise masks, computing attention weights, and gating mechanisms in neural networks (e.g., LSTM gates).

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `05_matrix_division.ipynb` — Element-wise division of matrices

## Complete Code / 完整代码一览

```python
# --- Matrix Hadamard Product / 矩阵 Hadamard 积 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
A = array([[1, 2, 3],[4, 5, 6]])
B = array([[1, 2, 3],[4, 5, 6]])
C = A * B
# 打印输出 / Print output
print(C)
```

---

### Matrix Division

# 05 — Matrix Element-wise Division / 矩阵按元素除法

**Chapter 10 — File 5 of 8 / 第10章 — 第5个文件（共8个）**

## Summary / 总结

Learn element-wise division, where elements of one matrix are divided by corresponding elements of another matrix.

学习按元素除法，其中一个矩阵的元素除以另一个矩阵的对应元素。

## Core Formula / 核心公式

Element-wise division:
$$\mathbf{C} = \mathbf{A} \div \mathbf{B} \text{ where } c_{ij} = \frac{a_{ij}}{b_{ij}}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy / 导入 NumPy

We import the array function from NumPy.

```python
# Import the array function from NumPy
# 从 NumPy 导入数组函数
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
```

## Step 2 — Create Two Matrices / 创建两个矩阵

We create two matrices with the same dimensions.

```python
# Create matrix A (2x3)
# 创建矩阵 A (2x3)
A = array([[1, 2, 3],[4, 5, 6]])

# Create matrix B (2x3) with same dimensions as A
# 创建矩阵 B (2x3)，与 A 具有相同维度
B = array([[1, 2, 3],[4, 5, 6]])
```

## Step 3 — Divide the Matrices / 矩阵相除

We divide matrix A by matrix B element-wise using the `/` operator.

```python
# Element-wise divide matrix A by matrix B
# Result: [[1/1, 2/2, 3/3], [4/4, 5/5, 6/6]] = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
# 矩阵 A 除以矩阵 B 按元素进行
# 结果：[[1/1, 2/2, 3/3], [4/4, 5/5, 6/6]] = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
C = A / B

# Print the result
# 打印结果
# 打印输出 / Print output
print(C)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Element-wise division divides corresponding elements. Care must be taken to avoid division by zero, which can produce infinity or NaN values.
- **ML Application / 机器学习应用**: Element-wise division is used for normalizing data, computing ratios between matrices, and in some optimization algorithms for adaptive step sizes.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `06_matrix_dot_product.ipynb` — Matrix multiplication (dot product)

## Complete Code / 完整代码一览

```python
# --- Matrix Element-wise Division / 矩阵按元素除法 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
A = array([[1, 2, 3],[4, 5, 6]])
B = array([[1, 2, 3],[4, 5, 6]])
C = A / B
# 打印输出 / Print output
print(C)
```

---

### Matrix Dot Product

# 06 — Matrix Multiplication (Dot Product) / 矩阵乘法（点积）

**Chapter 10 — File 6 of 8 / 第10章 — 第6个文件（共8个）**

## Summary / 总结

Master matrix multiplication, where the dot product of rows and columns produces a new matrix. This is fundamental to linear algebra and neural networks.

掌握矩阵乘法，其中行和列的点积产生一个新矩阵。这是线性代数和神经网络的基础。

## Core Formula / 核心公式

Matrix multiplication (for $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n \times p}$):
$$\mathbf{C} = \mathbf{A} \times \mathbf{B} \text{ where } c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy / 导入 NumPy

We import the array function from NumPy.

```python
# Import the array function from NumPy
# 从 NumPy 导入数组函数
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
```

## Step 2 — Create Two Matrices / 创建两个矩阵

We create two matrices with compatible dimensions for multiplication (A is 3x2, B is 2x2).

```python
# Create matrix A (3x2)
# 创建矩阵 A (3x2)
A = array([[1, 2],[3, 4],[5, 6]])

# Create matrix B (2x2)
# Column count of A (2) must equal row count of B (2) for multiplication
# 创建矩阵 B (2x2)
# A 的列数（2）必须等于 B 的行数（2）才能进行乘法
B = array([[1, 2],[3, 4]])
```

## Step 3 — Compute Matrix Multiplication Using .dot() / 使用 .dot() 计算矩阵乘法

We compute the matrix product using the `.dot()` method.

```python
# Matrix multiplication using .dot() method
# Result: A(3x2) @ B(2x2) = C(3x2)
# 使用 .dot() 方法进行矩阵乘法
# 结果：A(3x2) @ B(2x2) = C(3x2)
C = A.dot(B)

# Print the result
# 打印结果
# 打印输出 / Print output
print(C)
```

## Step 4 — Compute Matrix Multiplication Using @ Operator / 使用 @ 运算符计算矩阵乘法

We can also use the `@` operator (modern Python way).

```python
# Matrix multiplication using @ operator (modern Python way)
# 使用 @ 运算符进行矩阵乘法（现代 Python 方法）
D = A @ B

# Print the result (same as C)
# 打印结果（与 C 相同）
# 打印输出 / Print output
print(D)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Matrix multiplication combines rows of the first matrix with columns of the second. The inner dimensions must match, producing a result with the outer dimensions.
- **ML Application / 机器学习应用**: Matrix multiplication is core to forward propagation in neural networks, linear regression, and all linear transformations in machine learning.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `07_matrix_vector_multiplication.ipynb` — Multiply a matrix by a vector

## Complete Code / 完整代码一览

```python
# --- Matrix Multiplication / 矩阵乘法 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
A = array([[1, 2],[3, 4],[5, 6]])
B = array([[1, 2],[3, 4]])

# Using .dot() method
# 使用 .dot() 方法
C = A.dot(B)
# 打印输出 / Print output
print(C)

# Using @ operator
# 使用 @ 运算符
D = A @ B
# 打印输出 / Print output
print(D)
```

---

### Matrix Vector Multiplication

# 07 — Matrix-Vector Multiplication / 矩阵向量乘法

**Chapter 10 — File 7 of 8 / 第10章 — 第7个文件（共8个）**

## Summary / 总结

Learn matrix-vector multiplication, where a matrix is multiplied by a vector to produce a vector. This is essential for neural network computations.

学习矩阵向量乘法，其中矩阵与向量相乘以产生向量。这对神经网络计算至关重要。

## Core Formula / 核心公式

Matrix-vector multiplication (for $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{v} \in \mathbb{R}^n$):
$$\mathbf{c} = \mathbf{A} \times \mathbf{v} \text{ where } c_i = \sum_{j=1}^{n} a_{ij} v_j$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy / 导入 NumPy

We import the array function from NumPy.

```python
# Import the array function from NumPy
# 从 NumPy 导入数组函数
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
```

## Step 2 — Create a Matrix and a Vector / 创建矩阵和向量

We create a matrix and a vector with compatible dimensions (matrix is 3x2, vector has 2 elements).

```python
# Create matrix A (3x2)
# 创建矩阵 A (3x2)
A = array([[1, 2],[3, 4],[5, 6]])

# Create vector B with 2 elements
# Vector size (2) must equal column count of matrix A (2) for multiplication
# 创建包含 2 个元素的向量 B
# 向量大小（2）必须等于矩阵 A 的列数（2）才能进行乘法
B = array([0.5, 0.5])
```

## Step 3 — Compute Matrix-Vector Multiplication Using .dot() / 使用 .dot() 计算矩阵向量乘法

We compute the matrix-vector product using the `.dot()` method.

```python
# Matrix-vector multiplication using .dot() method
# Result: A(3x2) @ B(2,) = C(3,)
# 使用 .dot() 方法进行矩阵向量乘法
# 结果：A(3x2) @ B(2,) = C(3,)
C = A.dot(B)

# Print the result (vector with 3 elements)
# 打印结果（包含 3 个元素的向量）
# 打印输出 / Print output
print(C)
```

## Step 4 — Compute Matrix-Vector Multiplication Using @ Operator / 使用 @ 运算符计算矩阵向量乘法

We can also use the `@` operator.

```python
# Matrix-vector multiplication using @ operator
# 使用 @ 运算符进行矩阵向量乘法
D = A @ B

# Print the result (same as C)
# 打印结果（与 C 相同）
# 打印输出 / Print output
print(D)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Matrix-vector multiplication applies linear transformation to a vector, producing a new vector. Each element of the result is a dot product of a matrix row with the vector.
- **ML Application / 机器学习应用**: This operation is fundamental in neural networks for computing layer outputs (Wx + b), in linear regression predictions, and in all linear transformations.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `08_matrix_scalar_multiplication.ipynb` — Multiply a matrix by a scalar

## Complete Code / 完整代码一览

```python
# --- Matrix-Vector Multiplication / 矩阵向量乘法 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
A = array([[1, 2],[3, 4],[5, 6]])
B = array([0.5, 0.5])

# Using .dot() method
# 使用 .dot() 方法
C = A.dot(B)
# 打印输出 / Print output
print(C)

# Using @ operator
# 使用 @ 运算符
D = A @ B
# 打印输出 / Print output
print(D)
```

---

### Matrix Scalar Multiplication

# 08 — Matrix Scalar Multiplication / 矩阵标量乘法

**Chapter 10 — File 8 of 8 / 第10章 — 第8个文件（共8个）**

## Summary / 总结

Learn scalar multiplication for matrices, where every element of a matrix is multiplied by a scalar (single number).

学习矩阵的标量乘法，其中矩阵的每个元素都乘以一个标量（单个数字）。

## Core Formula / 核心公式

Matrix scalar multiplication scales all elements:
$$s \cdot \mathbf{A} = \begin{bmatrix} s \cdot a_{11} & s \cdot a_{12} & \cdots \\ s \cdot a_{21} & s \cdot a_{22} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy / 导入 NumPy

We import the array function from NumPy.

```python
# Import the array function from NumPy
# 从 NumPy 导入数组函数
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
```

## Step 2 — Create a Matrix and Define a Scalar / 创建矩阵和定义标量

We create a matrix and define a scalar value.

```python
# Create matrix A (3x2)
# 创建矩阵 A (3x2)
A = array([[1, 2], [3, 4], [5, 6]])

# Define a scalar value b = 0.5
# 定义标量值 b = 0.5
b = 0.5
```

## Step 3 — Multiply Matrix by Scalar / 矩阵乘以标量

We multiply the matrix by the scalar using the `*` operator.

```python
# Multiply scalar b by matrix A
# Result: [[0.5*1, 0.5*2], [0.5*3, 0.5*4], [0.5*5, 0.5*6]]
# 标量 b 乘以矩阵 A
# 结果：[[0.5*1, 0.5*2], [0.5*3, 0.5*4], [0.5*5, 0.5*6]]
C = A * b

# Print the result (each element is scaled by b)
# 打印结果（每个元素都被 b 缩放）
# 打印输出 / Print output
print(C)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Scalar multiplication scales every element of the matrix uniformly, stretching or shrinking the matrix by the same factor in all directions.
- **ML Application / 机器学习应用**: Scalar multiplication is used for weight updates in gradient descent, learning rate adjustments, regularization (weight decay), and scaling data during preprocessing.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `../chapter_11/01_triangular_matrix.ipynb` — Work with triangular matrices

## Complete Code / 完整代码一览

```python
# --- Matrix Scalar Multiplication / 矩阵标量乘法 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
A = array([[1, 2], [3, 4], [5, 6]])
b = 0.5
C = A * b
# 打印输出 / Print output
print(C)
```

---

### Chapter Summary / 章节总结

# Chapter 10 Summary / 第10章总结：Matrix Operations

## Theme / 主题

Matrices are 2D arrays, and this chapter extends vector operations to matrices. The key concepts are: element-wise operations (preserve shape), matrix multiplication (true dot product), and how operations interact between different matrix types. This is where linear algebra becomes essential to ML.

矩阵是二维数组，本章将向量操作扩展到矩阵。关键概念是：逐元素操作（保持形状）、矩阵乘法（真正的点积）以及操作如何在不同矩阵类型之间交互。这是线性代数对ML变得必不可少的地方。

## Evolution / 演化路线

```
01_create_matrices.ipynb
    └─ Create 2D arrays (创建矩阵)
    
02_matrix_add.ipynb
    └─ Element-wise addition (逐元素加法)
    
03_matrix_subtract.ipynb
    └─ Element-wise subtraction (逐元素减法)
    
04_matrix_hadamard.ipynb
    └─ Hadamard product / element-wise multiply (哈达玛积)
    
05_matrix_divide.ipynb
    └─ Element-wise division (逐元素除法)
    
06_matrix_dot.ipynb
    └─ Matrix multiplication (矩阵乘法，真正的点积)
    
07_matrix_vector.ipynb
    └─ Matrix-vector multiplication (矩阵-向量乘法)
    
08_scalar_multiply.ipynb
    └─ Scale matrix by scalar (用标量缩放矩阵)
```

## Progression Logic / 进度逻辑

Matrix operations progress from **element-wise to true matrix multiplication**:

1. **Element-wise operations**: Add, subtract, Hadamard, divide → shape stays same
2. **Matrix-vector multiply**: (m, n) · (n,) → (m,) — applies transformation
3. **Matrix-matrix multiply**: (m, n) · (n, p) → (m, p) — composes transformations
4. **Scalar multiply**: Scale everything

The critical insight: **element-wise operations are simple**, but **matrix multiplication combines rows and columns** — this is the operation that actually computes predictions.

矩阵操作从**逐元素到真正的矩阵乘法**进行：

1. **逐元素操作**：加、减、哈达玛、除 → 形状保持相同
2. **矩阵-向量乘法**：(m, n) · (n,) → (m,) — 应用变换
3. **矩阵-矩阵乘法**：(m, n) · (n, p) → (m, p) — 组合变换
4. **标量乘法**：缩放所有内容

关键见解：**逐元素操作很简单**，但**矩阵乘法组合行和列** — 这是实际计算预测的操作。

## ML Relevance / 机器学习相关性

In machine learning:
- **Element-wise operations**:
  - Activation functions: ReLU, sigmoid, tanh are element-wise
  - Hadamard product: attention mechanisms, gating
  - Batch normalization uses element-wise operations

- **Matrix multiplication** (the critical one):
  - Linear regression: `y = X · w` where X is (n_samples, n_features), w is (n_features,)
  - Neural network forward pass: `z = X · W + b` where W is (n_features, n_hidden)
  - Composition: `output = act(input · W1) · W2` chains transformations
  - Attention: `attention = softmax(Q · K^T) · V` (query-key matching + value weighting)

- **Matrix-vector multiply**:
  - Inference on single sample: one data point through all layers
  - Computing gradients: backprop uses matrix-vector products

Matrix multiplication is the heartbeat of deep learning. Understanding it—especially dimensions—is essential.

在机器学习中：
- **逐元素操作**：
  - 激活函数：ReLU、sigmoid、tanh是逐元素的
  - 哈达玛积：注意力机制、门控
  - 批归一化使用逐元素操作

- **矩阵乘法**（关键的一个）：
  - 线性回归：`y = X · w`，其中X是(n_samples, n_features)，w是(n_features,)
  - 神经网络前向传递：`z = X · W + b`，其中W是(n_features, n_hidden)
  - 组合：`output = act(input · W1) · W2`链接变换
  - 注意力：`attention = softmax(Q · K^T) · V`（查询-键匹配+值加权）

- **矩阵-向量乘法**：
  - 单个样本的推理：一个数据点通过所有层
  - 计算梯度：反向传播使用矩阵-向量积

矩阵乘法是深度学习的心跳。理解它——特别是维度——是必不可少的。

---
