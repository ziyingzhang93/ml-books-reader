# 线性代数与机器学习
## Chapter 08

---

### Create Vector

# 01 — Create Vector / 创建向量

**Chapter 08 — File 1 of 7 / 第08章 — 第1个文件（共7个）**

## Summary / 总结

Learn how to create vectors using NumPy, the fundamental building block for linear algebra operations in machine learning.

学习如何使用 NumPy 创建向量，这是机器学习中线性代数运算的基础。

## Core Formula / 核心公式

A vector $\mathbf{v} \in \mathbb{R}^n$ is an ordered list of $n$ numbers:
$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy / 导入 NumPy

First, we import the `array` function from NumPy, which is used to create numerical arrays representing vectors.

```python
# Import the array function from NumPy
# 从 NumPy 导入数组函数
from numpy import array
```

## Step 2 — Create a Vector / 创建向量

We create a vector by passing a list of numbers to the `array()` function. This creates a 1-dimensional NumPy array.

```python
# Create a vector with 3 elements: [1, 2, 3]
# 创建一个包含3个元素的向量：[1, 2, 3]
v = array([1, 2, 3])

# Print the vector to see its representation
# 打印向量以查看其表示
print(v)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: A vector is a sequence of numbers that can represent points in space, directions, or features in machine learning models.
- **ML Application / 机器学习应用**: Feature vectors in machine learning represent each data point as a vector where each element corresponds to a feature (e.g., pixel intensities, word counts, sensor readings).

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `02_vector_addition.ipynb` — Add two vectors element-wise

## Complete Code / 完整代码一览

```python
# --- Vector Creation / 向量创建 ---
from numpy import array
v = array([1, 2, 3])
print(v)
```

---

### Vector Addition

# 02 — Vector Addition / 向量加法

**Chapter 08 — File 2 of 7 / 第08章 — 第2个文件（共7个）**

## Summary / 总结

Master vector addition, where corresponding elements of two vectors are added together element-wise.

掌握向量加法，其中两个向量的对应元素按元素逐个相加。

## Core Formula / 核心公式

Vector addition is element-wise:
$$\mathbf{c} = \mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy / 导入 NumPy

We import NumPy's array function to work with vectors.

```python
# Import the array function from NumPy
# 从 NumPy 导入数组函数
from numpy import array
```

## Step 2 — Create Two Vectors / 创建两个向量

We create two vectors with the same dimension (3 elements each).

```python
# Create the first vector a = [1, 2, 3]
# 创建第一个向量 a = [1, 2, 3]
a = array([1, 2, 3])

# Create the second vector b = [1, 2, 3]
# 创建第二个向量 b = [1, 2, 3]
b = array([1, 2, 3])
```

## Step 3 — Add the Vectors / 向量相加

We add the two vectors using the `+` operator. NumPy automatically performs element-wise addition.

```python
# Add vectors a and b element-wise
# Result: [1+1, 2+2, 3+3] = [2, 4, 6]
# 向量 a 和 b 按元素相加
# 结果：[1+1, 2+2, 3+3] = [2, 4, 6]
c = a + b

# Print the result
# 打印结果
print(c)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Vector addition combines two vectors by adding corresponding components. The result is a new vector in the same vector space.
- **ML Application / 机器学习应用**: Vector addition is fundamental in gradient descent, where we update model weights by adding the negative gradient to the current weights.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `03_vector_subtraction.ipynb` — Subtract one vector from another

## Complete Code / 完整代码一览

```python
# --- Vector Addition / 向量加法 ---
from numpy import array
a = array([1, 2, 3])
b = array([1, 2, 3])
c = a + b
print(c)
```

---

### Vector Subtraction

# 03 — Vector Subtraction / 向量减法

**Chapter 08 — File 3 of 7 / 第08章 — 第3个文件（共7个）**

## Summary / 总结

Learn vector subtraction, where elements of one vector are subtracted from another element-wise.

学习向量减法，其中一个向量的元素从另一个向量的对应元素中按元素减去。

## Core Formula / 核心公式

Vector subtraction is element-wise:
$$\mathbf{c} = \mathbf{a} - \mathbf{b} = \begin{bmatrix} a_1 - b_1 \\ a_2 - b_2 \\ \vdots \\ a_n - b_n \end{bmatrix}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy / 导入 NumPy

We import the array function from NumPy.

```python
# Import the array function from NumPy
# 从 NumPy 导入数组函数
from numpy import array
```

## Step 2 — Create Two Vectors / 创建两个向量

We create two vectors with the same dimension.

```python
# Create the first vector a = [1, 2, 3]
# 创建第一个向量 a = [1, 2, 3]
a = array([1, 2, 3])

# Create the second vector b = [0.5, 0.5, 0.5]
# 创建第二个向量 b = [0.5, 0.5, 0.5]
b = array([0.5, 0.5, 0.5])
```

## Step 3 — Subtract the Vectors / 向量相减

We subtract vector b from vector a using the `-` operator.

```python
# Subtract vector b from vector a element-wise
# Result: [1-0.5, 2-0.5, 3-0.5] = [0.5, 1.5, 2.5]
# 向量 a 减去向量 b 按元素进行
# 结果：[1-0.5, 2-0.5, 3-0.5] = [0.5, 1.5, 2.5]
c = a - b

# Print the result
# 打印结果
print(c)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Vector subtraction is the opposite of addition, combining two vectors by subtracting corresponding components. It can also be viewed as adding a negated vector.
- **ML Application / 机器学习应用**: Vector subtraction is used to compute differences between predictions and actual values (residuals), which is essential in loss functions and error analysis.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `04_vector_multiplication.ipynb` — Element-wise multiplication of vectors

## Complete Code / 完整代码一览

```python
# --- Vector Subtraction / 向量减法 ---
from numpy import array
a = array([1, 2, 3])
b = array([0.5, 0.5, 0.5])
c = a - b
print(c)
```

---

### Vector Multiplication

# 04 — Vector Element-wise Multiplication / 向量按元素乘法

**Chapter 08 — File 4 of 7 / 第08章 — 第4个文件（共7个）**

## Summary / 总结

Learn element-wise (Hadamard) multiplication, where corresponding elements of two vectors are multiplied together.

学习按元素（Hadamard）乘法，其中两个向量的对应元素相互相乘。

## Core Formula / 核心公式

Element-wise multiplication (Hadamard product):
$$\mathbf{c} = \mathbf{a} \odot \mathbf{b} = \begin{bmatrix} a_1 \cdot b_1 \\ a_2 \cdot b_2 \\ \vdots \\ a_n \cdot b_n \end{bmatrix}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy / 导入 NumPy

We import the array function from NumPy.

```python
# Import the array function from NumPy
# 从 NumPy 导入数组函数
from numpy import array
```

## Step 2 — Create Two Vectors / 创建两个向量

We create two vectors with the same dimension.

```python
# Create the first vector a = [1, 2, 3]
# 创建第一个向量 a = [1, 2, 3]
a = array([1, 2, 3])

# Create the second vector b = [1, 2, 3]
# 创建第二个向量 b = [1, 2, 3]
b = array([1, 2, 3])
```

## Step 3 — Element-wise Multiply / 按元素相乘

We multiply the vectors element-wise using the `*` operator. Note: This is NOT dot product (which is different).

```python
# Element-wise multiply vectors a and b
# Result: [1*1, 2*2, 3*3] = [1, 4, 9]
# 向量 a 和 b 按元素相乘
# 结果：[1*1, 2*2, 3*3] = [1, 4, 9]
c = a * b

# Print the result (note: this is NOT the dot product)
# 打印结果（注意：这不是点积）
print(c)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Element-wise multiplication (Hadamard product) multiplies corresponding components. It is different from the dot product, which produces a scalar.
- **ML Application / 机器学习应用**: Element-wise multiplication is used in neural networks for applying element-wise masks, gating mechanisms (like in LSTMs), and feature scaling.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `05_vector_division.ipynb` — Element-wise division of vectors

## Complete Code / 完整代码一览

```python
# --- Vector Element-wise Multiplication / 向量按元素乘法 ---
from numpy import array
a = array([1, 2, 3])
b = array([1, 2, 3])
c = a * b
print(c)
```

---

### Vector Division

# 05 — Vector Element-wise Division / 向量按元素除法

**Chapter 08 — File 5 of 7 / 第08章 — 第5个文件（共7个）**

## Summary / 总结

Learn element-wise division, where elements of one vector are divided by corresponding elements of another vector.

学习按元素除法，其中一个向量的元素除以另一个向量的对应元素。

## Core Formula / 核心公式

Element-wise division:
$$\mathbf{c} = \mathbf{a} \div \mathbf{b} = \begin{bmatrix} a_1 / b_1 \\ a_2 / b_2 \\ \vdots \\ a_n / b_n \end{bmatrix}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy / 导入 NumPy

We import the array function from NumPy.

```python
# Import the array function from NumPy
# 从 NumPy 导入数组函数
from numpy import array
```

## Step 2 — Create Two Vectors / 创建两个向量

We create two vectors with the same dimension.

```python
# Create the first vector a = [1, 2, 3]
# 创建第一个向量 a = [1, 2, 3]
a = array([1, 2, 3])

# Create the second vector b = [1, 2, 3]
# 创建第二个向量 b = [1, 2, 3]
b = array([1, 2, 3])
```

## Step 3 — Element-wise Divide / 按元素相除

We divide vector a by vector b element-wise using the `/` operator.

```python
# Element-wise divide vector a by vector b
# Result: [1/1, 2/2, 3/3] = [1.0, 1.0, 1.0]
# 向量 a 除以向量 b 按元素进行
# 结果：[1/1, 2/2, 3/3] = [1.0, 1.0, 1.0]
c = a / b

# Print the result
# 打印结果
print(c)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Element-wise division divides corresponding components of two vectors. It must be used carefully to avoid division by zero.
- **ML Application / 机器学习应用**: Element-wise division is used for normalization (dividing by standard deviation), scaling features by class weights, or adaptive learning rates in optimization algorithms.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `06_vector_dot_product.ipynb` — Compute the dot product of two vectors

## Complete Code / 完整代码一览

```python
# --- Vector Element-wise Division / 向量按元素除法 ---
from numpy import array
a = array([1, 2, 3])
b = array([1, 2, 3])
c = a / b
print(c)
```

---

### Vector Dot Product

# 06 — Vector Dot Product / 向量点积

**Chapter 08 — File 6 of 7 / 第08章 — 第6个文件（共7个）**

## Summary / 总结

Master the dot product (inner product), a fundamental operation that produces a scalar from two vectors. This is crucial for many ML algorithms.

掌握点积（内积），这是一个基本运算，从两个向量产生一个标量。这对许多机器学习算法至关重要。

## Core Formula / 核心公式

The dot product produces a scalar:
$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy / 导入 NumPy

We import the array function from NumPy.

```python
# Import the array function from NumPy
# 从 NumPy 导入数组函数
from numpy import array
```

## Step 2 — Create Two Vectors / 创建两个向量

We create two vectors with the same dimension.

```python
# Create the first vector a = [1, 2, 3]
# 创建第一个向量 a = [1, 2, 3]
a = array([1, 2, 3])

# Create the second vector b = [1, 2, 3]
# 创建第二个向量 b = [1, 2, 3]
b = array([1, 2, 3])
```

## Step 3 — Compute Dot Product Using .dot() / 使用 .dot() 计算点积

We compute the dot product using the `.dot()` method, which returns a scalar.

```python
# Compute dot product using .dot() method
# Result: 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
# 使用 .dot() 方法计算点积
# 结果：1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
c = a.dot(b)

# Print the scalar result
# 打印标量结果
print(c)
```

## Step 4 — Compute Dot Product Using @ Operator / 使用 @ 运算符计算点积

We can also use the `@` operator (matrix multiplication operator) to compute the dot product.

```python
# Compute dot product using @ operator (modern Python way)
# 使用 @ 运算符计算点积（现代 Python 方法）
d = a @ b

# Print the result (same as c)
# 打印结果（与 c 相同）
print(d)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: The dot product measures the projection of one vector onto another. It is the sum of element-wise products, resulting in a scalar that reflects the "alignment" between two vectors.
- **ML Application / 机器学习应用**: Dot products are central to neural networks (computing layer outputs), similarity measures (cosine similarity), and linear regression predictions.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `07_vector_scalar_multiplication.ipynb` — Multiply a vector by a scalar

## Complete Code / 完整代码一览

```python
# --- Vector Dot Product / 向量点积 ---
from numpy import array
a = array([1, 2, 3])
b = array([1, 2, 3])

# Using .dot() method
# 使用 .dot() 方法
c = a.dot(b)
print(c)

# Using @ operator
# 使用 @ 运算符
d = a @ b
print(d)
```

---

### Vector Scalar Multiplication

# 07 — Vector Scalar Multiplication / 向量标量乘法

**Chapter 08 — File 7 of 7 / 第08章 — 第7个文件（共7个）**

## Summary / 总结

Learn scalar multiplication, where every element of a vector is multiplied by a scalar (single number).

学习标量乘法，其中向量的每个元素都乘以一个标量（单个数字）。

## Core Formula / 核心公式

Scalar multiplication scales all elements:
$$s \cdot \mathbf{v} = \begin{bmatrix} s \cdot v_1 \\ s \cdot v_2 \\ \vdots \\ s \cdot v_n \end{bmatrix}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import NumPy / 导入 NumPy

We import the array function from NumPy.

```python
# Import the array function from NumPy
# 从 NumPy 导入数组函数
from numpy import array
```

## Step 2 — Create a Vector and a Scalar / 创建向量和标量

We create one vector and define a scalar value.

```python
# Create the vector a = [1, 2, 3]
# 创建向量 a = [1, 2, 3]
a = array([1, 2, 3])

# Define a scalar value s = 0.5
# 定义标量值 s = 0.5
s = 0.5
```

## Step 3 — Multiply Vector by Scalar / 向量乘以标量

We multiply the vector by the scalar using the `*` operator.

```python
# Multiply scalar s by vector a
# Result: [0.5*1, 0.5*2, 0.5*3] = [0.5, 1.0, 1.5]
# 标量 s 乘以向量 a
# 结果：[0.5*1, 0.5*2, 0.5*3] = [0.5, 1.0, 1.5]
c = s * a

# Print the result (each element is scaled by s)
# 打印结果（每个元素都被 s 缩放）
print(c)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Scalar multiplication scales a vector by stretching or shrinking it in all directions. The direction remains the same unless the scalar is negative (which reverses direction).
- **ML Application / 机器学习应用**: Scalar multiplication is used for learning rate adjustments, regularization (weight decay), and gradient updates in optimization algorithms like SGD.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `../chapter_09/01_l1_norm.ipynb` — Compute the L1 norm of a vector

## Complete Code / 完整代码一览

```python
# --- Vector Scalar Multiplication / 向量标量乘法 ---
from numpy import array
a = array([1, 2, 3])
s = 0.5
c = s * a
print(c)
```

---

### Chapter Summary

# Chapter 08 Summary / 第08章总结：Vector Operations

## Theme / 主题

Vectors are 1D arrays, and this chapter covers fundamental operations on them: element-wise ops (add, subtract, multiply, divide), the dot product (inner product), and scalar multiplication. These operations are the building blocks for all machine learning algorithms.

向量是一维数组，本章涵盖了对它们的基本操作：元素级操作（加、减、乘、除）、点积（内积）和标量乘法。这些操作是所有机器学习算法的构建块。

## Evolution / 演化路线

```
01_create_vectors.ipynb
    └─ Create 1D arrays / vectors (创建一维数组/向量)
    
02_vector_add.ipynb
    └─ Element-wise addition (逐元素加法)
    
03_vector_subtract.ipynb
    └─ Element-wise subtraction (逐元素减法)
    
04_vector_multiply.ipynb
    └─ Element-wise multiplication / Hadamard product (逐元素乘法)
    
05_vector_divide.ipynb
    └─ Element-wise division (逐元素除法)
    
06_dot_product.ipynb
    └─ Inner product / dot product (内积/点积)
    
07_scalar_multiply.ipynb
    └─ Scale vector by scalar (用标量缩放向量)
```

## Progression Logic / 进度逻辑

Vector operations progress from **element-wise to inner product to scaling**:

1. **Element-wise operations**: Add, subtract, multiply, divide → result is same shape as input
2. **Dot product**: Special operation → combines two vectors into a scalar
3. **Scalar multiplication**: Scale the entire vector

This teaches a fundamental distinction in linear algebra: **element-wise operations preserve shape**, while **dot product reduces to scalar**. This distinction is critical for understanding how neural networks work (element-wise ReLU vs. dot product for classification).

向量操作从**逐元素到内积到缩放**进行：

1. **逐元素操作**：加、减、乘、除 → 结果与输入形状相同
2. **点积**：特殊操作 → 将两个向量组合为标量
3. **标量乘法**：缩放整个向量

这教导线性代数中的基本区分：**逐元素操作保持形状**，而**点积减少到标量**。这个区分对于理解神经网络如何工作至关重要（ReLU逐元素vs.分类的点积）。

## ML Relevance / 机器学习相关性

In machine learning:
- **Element-wise operations**: 
  - Add: Bias addition in neural networks
  - Multiply: Element-wise feature interactions, attention mechanisms
  - Divide: Normalization, gradient updates

- **Dot product**: 
  - The fundamental operation in linear regression: `y = X · w`
  - Forward pass in neural networks: `z = X · W + b`
  - Similarity computations: cosine similarity is normalized dot product

- **Scalar multiplication**: 
  - Gradient descent: `w = w - learning_rate * gradient`
  - Learning rate scheduling: adjust step size

Dot product is arguably the most important operation in all of ML. Understanding it deeply unlocks understanding of neural networks.

在机器学习中：
- **逐元素操作**：
  - 加：神经网络中的偏置添加
  - 乘：逐元素特征交互、注意力机制
  - 除：归一化、梯度更新

- **点积**：
  - 线性回归中的基本操作：`y = X · w`
  - 神经网络中的前向传递：`z = X · W + b`
  - 相似性计算：余弦相似性是规范化的点积

- **标量乘法**：
  - 梯度下降：`w = w - learning_rate * gradient`
  - 学习率调度：调整步长

点积可能是所有ML中最重要的操作。深刻理解它可以解锁对神经网络的理解。

---
