# 线性代数与机器学习
## Chapter 14

---

### Create Tensor

# Create Tensor / 创建张量

**Chapter 14 — File 1 of 6 / 第14章 — 第1个文件（共6个）**

## Summary / 总结

Tensors are multi-dimensional arrays that generalize scalars (0D), vectors (1D), and matrices (2D) to arbitrary dimensions. A 3D tensor can be thought of as a stack of matrices. In ML, tensors are fundamental data structures: images are 3D tensors (height, width, channels), videos are 4D tensors (frames, height, width, channels), and neural network activations are multi-dimensional tensors. NumPy's `array()` function can create tensors of any dimension.

张量是多维数组，将标量（0D）、向量（1D）和矩阵（2D）推广到任意维度。3D 张量可以看作矩阵的堆栈。在机器学习中，张量是基础数据结构：图像是 3D 张量（高度、宽度、通道），视频是 4D 张量（帧、高度、宽度、通道），神经网络激活是多维张量。NumPy 的 `array()` 函数可以创建任何维度的张量。

## Core Concept / 核心概念

- Scalar: shape ()
- Vector: shape (n,)
- Matrix: shape (m, n)
- Tensor: shape (d₁, d₂, ..., dₖ) where k ≥ 3

A 3D tensor T with shape (3, 3, 3) contains 27 elements arranged in 3 matrices of 3×3.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create a 3D tensor / 创建 3D 张量

Create a 3×3×3 tensor (3D array) to introduce tensor data structure.

```python
from numpy import array

# Create a 3D tensor (3x3x3 array)
# 创建一个 3D 张量（3x3x3 数组）
T = array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]]
])

print("Tensor T shape / 张量 T 的形状:")
print(f"T.shape = {T.shape}")
print(f"T 的维度 = {T.ndim}")

print("\nTensor T / 张量 T:")
print(T)
```

## Step 2 — Understand tensor indexing / 理解张量索引

Access elements and sub-tensors of the 3D tensor.

```python
# Access different parts of the tensor
# 访问张量的不同部分

# Get the first matrix (2D slice)
# 获取第一个矩阵（2D 切片）
print("First matrix T[0] / 第一个矩阵 T[0]:")
print(T[0])

# Get a specific row from the first matrix
# 从第一个矩阵获取特定行
print("\nFirst row of first matrix T[0,0] / 第一个矩阵的第一行 T[0,0]:")
print(T[0,0])

# Get a single element
# 获取单个元素
print("\nSingle element T[0,0,0] / 单个元素 T[0,0,0]:")
print(T[0,0,0])
```

## Step 3 — Tensor properties / 张量的属性

Explore key properties of the tensor.

```python
# Tensor properties
# 张量的属性

print(f"Shape (dimensions): {T.shape}")
print(f"形状（维度）：{T.shape}")

print(f"\nNumber of dimensions (ndim): {T.ndim}")
print(f"维度数（ndim）：{T.ndim}")

print(f"\nTotal elements (size): {T.size}")
print(f"总元素数（size）：{T.size}")

print(f"\nData type (dtype): {T.dtype}")
print(f"数据类型（dtype）：{T.dtype}")

print(f"\nExplanation: A 3×3×3 tensor contains {T.size} elements")
print(f"解释：一个 3×3×3 张量包含 {T.size} 个元素")
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Tensors are multilinear maps. A tensor of rank k has order k (k indices). Key operations: contraction (summing over indices), slicing (fixing some indices), reshaping, and element-wise operations. Tensors provide natural representations for high-dimensional data.

- **ML Application / 机器学习应用**: Tensors are fundamental in: images (3D: H×W×C), videos (4D: T×H×W×C), batch processing (adding batch dimension), convolutional neural networks (kernels are 4D tensors), and deep learning frameworks (PyTorch, TensorFlow).

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `02_tensor_addition.ipynb` — Perform addition operations on tensors

## Complete Code / 完整代码一览

```python
# --- Create Tensor / 创建张量 ---
from numpy import array

# Step 1: Create a 3D tensor
# 步骤 1：创建一个 3D 张量
T = array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]]
])

print("Tensor T shape / 张量 T 的形状:")
print(f"T.shape = {T.shape}")
print(f"T.ndim = {T.ndim}")
print(f"T.size = {T.size}")

# Step 2: Tensor indexing and slicing
# 步骤 2：张量索引和切片
print("\nFirst matrix T[0] / 第一个矩阵 T[0]:")
print(T[0])

print("\nFirst row of first matrix T[0,0] / 第一个矩阵的第一行 T[0,0]:")
print(T[0,0])

print("\nSingle element T[0,0,0] / 单个元素 T[0,0,0]:")
print(T[0,0,0])

print("\nFull tensor / 完整张量:")
print(T)
```

---

### Tensor Addition

# Tensor Addition / 张量加法

**Chapter 14 — File 2 of 6 / 第14章 — 第2个文件（共6个）**

## Summary / 总结

Tensor addition is element-wise addition of two tensors with the same shape. The result is a new tensor where each element is the sum of corresponding elements from the input tensors. Tensor addition is associative and commutative: $A + B = B + A$ and $(A + B) + C = A + (B + C)$. This operation is fundamental in neural networks (combining feature maps, gradient accumulation) and data processing.

张量加法是两个形状相同的张量的逐元素加法。结果是一个新张量，其中每个元素是输入张量相应元素的和。张量加法是结合律和交换律的：$A + B = B + A$ 和 $(A + B) + C = A + (B + C)$。这个操作在神经网络（组合特征图、梯度累积）和数据处理中是基础的。

## Core Formula / 核心公式

$$C = A + B \implies C_{ijk...} = A_{ijk...} + B_{ijk...}$$

Element-wise addition requires both tensors to have identical shapes. Broadcasting can extend this for compatible shapes.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create two tensors / 创建两个张量

Create two 3×3×3 tensors to demonstrate addition.

```python
from numpy import array

# Create first 3D tensor
# 创建第一个 3D 张量
A = array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]]
])

print("Tensor A / 张量 A:")
print(f"Shape: {A.shape}")
print(A)

# Create second 3D tensor (same shape)
# 创建第二个 3D 张量（相同形状）
B = array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]]
])

print("\nTensor B / 张量 B:")
print(f"Shape: {B.shape}")
print(B)
```

## Step 2 — Perform tensor addition / 执行张量加法

Add the two tensors element-wise using the `+` operator.

```python
# Perform element-wise addition
# 执行逐元素加法
C = A + B

print("Result C = A + B / 结果 C = A + B:")
print(f"Shape: {C.shape}")
print(C)

print("\nVerification (sample elements) / 验证（样本元素）:")
print(f"A[0,0,0] + B[0,0,0] = {A[0,0,0]} + {B[0,0,0]} = {C[0,0,0]}")
print(f"A[1,1,1] + B[1,1,1] = {A[1,1,1]} + {B[1,1,1]} = {C[1,1,1]}")
print(f"A[2,2,2] + B[2,2,2] = {A[2,2,2]} + {B[2,2,2]} = {C[2,2,2]}")
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Tensor addition is commutative ($A + B = B + A$) and associative ($(A+B)+C = A+(B+C)$). It forms an Abelian group under the operation of addition. The zero tensor (all elements 0) is the additive identity.

- **ML Application / 机器学习应用**: Tensor addition is used in: combining feature maps in deep learning, gradient accumulation in backpropagation, residual connections in neural networks, and feature fusion in multi-modal learning.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `03_tensor_subtraction.ipynb` — Perform subtraction operations on tensors

## Complete Code / 完整代码一览

```python
# --- Tensor Addition / 张量加法 ---
from numpy import array

# Step 1: Create two 3D tensors
# 步骤 1：创建两个 3D 张量
A = array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]]
])

B = array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]]
])

print("Tensor A / 张量 A (shape: 3x3x3):")
print(A)

print("\nTensor B / 张量 B (shape: 3x3x3):")
print(B)

# Step 2: Perform tensor addition
# 步骤 2：执行张量加法
C = A + B

print("\nResult C = A + B / 结果 C = A + B (shape: 3x3x3):")
print(C)
```

---

### Tensor Hadamard Product

# Tensor Hadamard Product / 张量 Hadamard 积

**Chapter 14 — File 4 of 6 / 第14章 — 第4个文件（共6个）**

## Summary / 总结

The Hadamard product (element-wise multiplication, ⊙) multiplies corresponding elements of two tensors with identical shape. Unlike matrix multiplication, Hadamard product preserves shape and requires no index contractions. It's commutative and associative. The Hadamard product is ubiquitous in neural networks (activation masking, gating mechanisms) and is computationally very efficient. For tensors $A$ and $B$: $(A \odot B)_{ijk...} = A_{ijk...} \times B_{ijk...}$.

Hadamard 积（逐元素乘法，⊙）将两个形状相同的张量的相应元素相乘。与矩阵乘法不同，Hadamard 积保持形状，无需指标收缩。它是可交换和结合的。Hadamard 积在神经网络中无处不在（激活掩蔽、门控机制），计算效率很高。对于张量 $A$ 和 $B$：$(A \odot B)_{ijk...} = A_{ijk...} \times B_{ijk...}$。

## Core Formula / 核心公式

$$C = A \odot B \implies C_{ijk...} = A_{ijk...} \cdot B_{ijk...}$$

Properties: Commutative ($A \odot B = B \odot A$), Associative ($(A \odot B) \odot C = A \odot (B \odot C)$)

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create two tensors / 创建两个张量

Create two 3×3×3 tensors for Hadamard product demonstration.

```python
from numpy import array

# Create first 3D tensor
# 创建第一个 3D 张量
A = array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]]
])

print("Tensor A / 张量 A:")
print(f"Shape: {A.shape}")
print(A)

# Create second 3D tensor (same shape)
# 创建第二个 3D 张量（相同形状）
B = array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]]
])

print("\nTensor B / 张量 B:")
print(f"Shape: {B.shape}")
print(B)
```

## Step 2 — Perform Hadamard product / 执行 Hadamard 积

Multiply tensors element-wise using the `*` operator (in NumPy, `*` is Hadamard product for arrays).

```python
# Perform element-wise multiplication (Hadamard product)
# 执行逐元素乘法（Hadamard 积）
C = A * B

print("Result C = A ⊙ B (Hadamard product) / 结果 C = A ⊙ B（Hadamard 积）:")
print(f"Shape: {C.shape}")
print(C)

print("\nVerification (sample elements) / 验证（样本元素）:")
print(f"A[0,0,0] * B[0,0,0] = {A[0,0,0]} * {B[0,0,0]} = {C[0,0,0]}")
print(f"A[1,1,1] * B[1,1,1] = {A[1,1,1]} * {B[1,1,1]} = {C[1,1,1]}")
print(f"A[2,2,2] * B[2,2,2] = {A[2,2,2]} * {B[2,2,2]} = {C[2,2,2]}")
```

## Step 3 — Hadamard product properties / Hadamard 积的性质

Demonstrate key properties of the Hadamard product.

```python
# Test commutativity: A ⊙ B == B ⊙ A
# 测试可交换性：A ⊙ B == B ⊙ A
C1 = A * B
C2 = B * A

print("Commutativity property / 可交换性质:")
print(f"A ⊙ B == B ⊙ A: {(C1 == C2).all()}")

# Test with a zero tensor
# 用零张量测试
Zero = array([[[0]*3]*3]*3)
C3 = A * Zero

print(f"\nA ⊙ 0 = 0: {(C3 == Zero).all()}")

# Test with an identity tensor (all ones)
# 用恒等张量（全为 1）测试
Ones = array([[[1]*3]*3]*3)
C4 = A * Ones

print(f"A ⊙ 1 = A: {(C4 == A).all()}")
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Hadamard product is element-wise multiplication, distinct from tensor contraction. It's commutative, associative, and distributive over addition: $A \odot (B + C) = (A \odot B) + (A \odot C)$. Identity element is the all-ones tensor. Computationally O(n) with no index contraction needed.

- **ML Application / 机器学习应用**: Hadamard product is essential for: element-wise gates in LSTM/GRU, attention masking, dropout regularization (multiplying by binary masks), feature scaling, and activation function application in neural networks.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `05_tensor_division.ipynb` — Perform element-wise division of tensors

## Complete Code / 完整代码一览

```python
# --- Tensor Hadamard Product / 张量 Hadamard 积 ---
from numpy import array

# Step 1: Create two 3D tensors
# 步骤 1：创建两个 3D 张量
A = array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]]
])

B = array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]]
])

print("Tensor A / 张量 A (shape: 3x3x3):")
print(A)

print("\nTensor B / 张量 B (shape: 3x3x3):")
print(B)

# Step 2: Perform Hadamard product (element-wise multiplication)
# 步骤 2：执行 Hadamard 积（逐元素乘法）
C = A * B

print("\nResult C = A ⊙ B (Hadamard product) / 结果 C = A ⊙ B（Hadamard 积）:")
print(C)

# Step 3: Verify commutativity
# 步骤 3：验证可交换性
C1 = A * B
C2 = B * A
print(f"\nCommutativity: A ⊙ B == B ⊙ A: {(C1 == C2).all()}")
```

---

### Tensor Division

# Tensor Division / 张量除法

**Chapter 14 — File 5 of 6 / 第14章 — 第5个文件（共6个）**

## Summary / 总结

Tensor division is element-wise division where each element of the result equals the corresponding element of the first tensor divided by the corresponding element of the second tensor. Division is not commutative ($A / B \neq B / A$) and requires care to avoid division by zero. In ML applications, division commonly appears in normalization, scaling, and averaging operations. Element-wise division is denoted $A \oslash B$ or $A / B$ for element-wise operations.

张量除法是逐元素除法，其中结果的每个元素等于第一个张量的相应元素除以第二个张量的相应元素。除法不是可交换的（$A / B \neq B / A$），需要小心避免被零除。在机器学习应用中，除法常见于归一化、缩放和平均操作。逐元素除法记为 $A \oslash B$ 或 $A / B$。

## Core Formula / 核心公式

$$C = A \oslash B \implies C_{ijk...} = \frac{A_{ijk...}}{B_{ijk...}}$$

Care required: $B_{ijk...} \neq 0$ for all elements. Division by zero results in `inf` or `nan` in NumPy.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create two tensors / 创建两个张量

Create two 3×3×3 tensors for division demonstration.

```python
from numpy import array

# Create first 3D tensor
# 创建第一个 3D 张量
A = array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]]
], dtype=float)

print("Tensor A / 张量 A:")
print(f"Shape: {A.shape}")
print(A)

# Create second 3D tensor (same shape, all positive)
# 创建第二个 3D 张量（相同形状，全为正数）
B = array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]]
], dtype=float)

print("\nTensor B / 张量 B:")
print(f"Shape: {B.shape}")
print(B)
```

## Step 2 — Perform tensor division / 执行张量除法

Divide tensor A by tensor B element-wise using the `/` operator.

```python
# Perform element-wise division
# 执行逐元素除法
C = A / B

print("Result C = A / B (Element-wise division) / 结果 C = A / B（逐元素除法）:")
print(f"Shape: {C.shape}")
print(C)

print("\nVerification (sample elements) / 验证（样本元素）:")
print(f"A[0,0,0] / B[0,0,0] = {A[0,0,0]} / {B[0,0,0]} = {C[0,0,0]}")
print(f"A[1,1,1] / B[1,1,1] = {A[1,1,1]} / {B[1,1,1]} = {C[1,1,1]}")
print(f"A[2,2,2] / B[2,2,2] = {A[2,2,2]} / {B[2,2,2]} = {C[2,2,2]}")
```

## Step 3 — Non-commutativity and division by zero / 非交换性和被零除

Demonstrate that division is not commutative and show handling of division by zero.

```python
# Show non-commutativity
# 显示非交换性
C1 = A / B
C2 = B / A

print("Non-commutativity property / 非交换性质:")
print(f"A / B == B / A: {(C1 == C2).all()}")
print(f"\nA/B at [0,0,0]: {C1[0,0,0]}")
print(f"B/A at [0,0,0]: {C2[0,0,0]}")

# Handle division by zero
# 处理被零除
import numpy as np
B_with_zero = B.copy()
B_with_zero[0,0,0] = 0  # Set one element to zero

print("\nDivision by zero handling / 被零除处理:")
C3 = A / B_with_zero
print(f"A[0,0,0] / 0 = {C3[0,0,0]}")
print(f"Result is inf (infinity) / 结果是 inf（无穷大）")

# Check for NaN or inf
# 检查 NaN 或 inf
print(f"\nContains inf: {np.isinf(C3).any()}")
print(f"Contains nan: {np.isnan(C3).any()}")
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Tensor division is element-wise, non-commutative, non-associative. It's related to multiplication via $A / B = A \odot B^{-1}$ where $B^{-1}$ is element-wise reciprocal. Division by zero produces inf or nan, requiring careful handling (use masking or conditional logic).

- **ML Application / 机器学习应用**: Tensor division appears in: computing averages and means, normalizing data, learning rate scaling, batch normalization, computing error rates, and gradient clipping. Essential for numerical stability and regularization techniques.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `06_tensor_product.ipynb` — Compute tensor product (outer product) using tensordot

## Complete Code / 完整代码一览

```python
# --- Tensor Division / 张量除法 ---
from numpy import array
import numpy as np

# Step 1: Create two 3D tensors
# 步骤 1：创建两个 3D 张量
A = array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]]
], dtype=float)

B = array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]]
], dtype=float)

print("Tensor A / 张量 A (shape: 3x3x3):")
print(A)

print("\nTensor B / 张量 B (shape: 3x3x3):")
print(B)

# Step 2: Perform element-wise division
# 步骤 2：执行逐元素除法
C = A / B

print("\nResult C = A / B (Element-wise division) / 结果 C = A / B（逐元素除法）:")
print(C)

# Step 3: Test non-commutativity
# 步骤 3：测试非交换性
C1 = A / B
C2 = B / A
print(f"\nA / B == B / A: {(C1 == C2).all()}")
print(f"Division is NOT commutative / 除法不是可交换的")
```

---

### Tensor Product

# Tensor Product / 张量积

**Chapter 14 — File 6 of 6 / 第14章 — 第6个文件（共6个）**

## Summary / 总结

The tensor product (outer product) of two tensors is a fundamental operation that combines them into a higher-order tensor. For vectors $\mathbf{a}$ and $\mathbf{b}$, the tensor product creates a matrix/2D tensor. NumPy's `tensordot()` with `axes=0` computes the generalized tensor product. The resulting tensor has shape equal to the concatenation of input shapes. Tensor products appear in quantum mechanics, feature interactions, and higher-order learning architectures.

两个张量的张量积（外积）是一个基本操作，将它们组合成更高阶的张量。对于向量 $\mathbf{a}$ 和 $\mathbf{b}$，张量积创建矩阵/2D 张量。NumPy 的 `tensordot()` 带有 `axes=0` 计算广义张量积。生成的张量的形状等于输入形状的连接。张量积出现在量子力学、特征交互和高阶学习架构中。

## Core Formula / 核心公式

For vectors $\mathbf{a} \in \mathbb{R}^m$ and $\mathbf{b} \in \mathbb{R}^n$:
$$\mathbf{a} \otimes \mathbf{b} = \begin{bmatrix} a_1 b_1 & a_1 b_2 & \cdots & a_1 b_n \\ a_2 b_1 & \cdots & a_2 b_n \\ \vdots & \ddots & \vdots \\ a_m b_1 & \cdots & a_m b_n \end{bmatrix} \in \mathbb{R}^{m \times n}$$

Result shape: (m, n) for vectors of lengths m and n

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create two vectors / 创建两个向量

Create two 1D vectors (arrays) for tensor product computation.

```python
from numpy import array
from numpy import tensordot

# Create first vector
# 创建第一个向量
A = array([1, 2])

print("Vector A / 向量 A:")
print(f"Shape: {A.shape}")
print(A)

# Create second vector
# 创建第二个向量
B = array([3, 4])

print("\nVector B / 向量 B:")
print(f"Shape: {B.shape}")
print(B)
```

## Step 2 — Compute tensor product / 计算张量积

Use `tensordot()` with `axes=0` to compute the outer product (tensor product).

```python
# Compute tensor product using tensordot(A, B, axes=0)
# 使用 tensordot(A, B, axes=0) 计算张量积
C = tensordot(A, B, axes=0)

print("Tensor Product C = A ⊗ B / 张量积 C = A ⊗ B:")
print(f"Result shape: {C.shape}")
print(f"结果形状：{C.shape}")
print(C)
```

## Step 3 — Verify tensor product computation / 验证张量积计算

Verify the tensor product by comparing with manual element-wise computation.

```python
# Manually compute outer product to verify
# 手动计算外积以验证
print("Manual verification / 手动验证:")
print(f"\nA[0] * B = {A[0]} * {B} = {A[0] * B}")
print(f"A[1] * B = {A[1]} * {B} = {A[1] * B}")

print(f"\nTensor product elements / 张量积元素:")
print(f"C[0,0] = A[0] * B[0] = {A[0]} * {B[0]} = {C[0,0]}")
print(f"C[0,1] = A[0] * B[1] = {A[0]} * {B[1]} = {C[0,1]}")
print(f"C[1,0] = A[1] * B[0] = {A[1]} * {B[0]} = {C[1,0]}")
print(f"C[1,1] = A[1] * B[1] = {A[1]} * {B[1]} = {C[1,1]}")

# Compare with outer product using numpy's outer
# 使用 numpy 的 outer 与外积比较
import numpy as np
C_outer = np.outer(A, B)
print(f"\nComparison with np.outer / 与 np.outer 的比较:")
print(f"Results are equal: {np.allclose(C, C_outer)}")
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Tensor product generalizes outer product to arbitrary-order tensors. For vectors: $\mathbf{a} \otimes \mathbf{b}$ creates a rank-1 matrix. For higher-order tensors with `axes=0`: result shape = tuple(shape_A + shape_B). Tensor product is bilinear and distributes over addition.

- **ML Application / 机器学习应用**: Tensor products are used in: feature interaction modeling (combining features from different sources), higher-order learning architectures, attention mechanisms (combining queries and keys), and quantum machine learning. Also appears in computing outer products for metric learning.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

✅ **Completed / 完成**: All chapters covered! / 所有章节都已涵盖！

## Complete Code / 完整代码一览

```python
# --- Tensor Product / 张量积 ---
from numpy import array
from numpy import tensordot
import numpy as np

# Step 1: Create two vectors
# 步骤 1：创建两个向量
A = array([1, 2])
B = array([3, 4])

print("Vector A / 向量 A:")
print(f"Shape: {A.shape}")
print(A)

print("\nVector B / 向量 B:")
print(f"Shape: {B.shape}")
print(B)

# Step 2: Compute tensor product
# 步骤 2：计算张量积
C = tensordot(A, B, axes=0)

print("\nTensor Product C = A ⊗ B / 张量积 C = A ⊗ B:")
print(f"Result shape: {C.shape}")
print(C)

# Step 3: Verify with numpy.outer
# 步骤 3：用 numpy.outer 验证
C_outer = np.outer(A, B)
print(f"\nVerification with np.outer / 用 np.outer 验证:")
print(f"Results are equal: {np.allclose(C, C_outer)}")
print(C_outer)
```

---

### Chapter Summary

# Chapter 14 Summary / 第14章总结：Tensors

## Theme / 主题

Tensors are n-dimensional arrays (3D+). In machine learning, tensors are everywhere: batches of images (batch_size, height, width, channels), sequences of embeddings (sequence_length, embedding_dim), transformer outputs (batch, seq_len, hidden_dim). This chapter extends 2D matrix operations to arbitrary dimensions, enabling computation on modern deep learning data structures.

张量是n维数组（3D+）。在机器学习中，张量无处不在：图像批次(batch_size, height, width, channels)、嵌入序列(sequence_length, embedding_dim)、变换器输出(batch, seq_len, hidden_dim)。本章将2D矩阵操作扩展到任意维度，使现代深度学习数据结构的计算成为可能。

## Evolution / 演化路线

```
01_create_tensors.ipynb
    └─ Create 3D+ arrays (创建三维或更高维数组)
    
02_tensor_add.ipynb
    └─ Element-wise addition (逐元素加法)
    
03_tensor_subtract.ipynb
    └─ Element-wise subtraction (逐元素减法)
    
04_tensor_hadamard.ipynb
    └─ Element-wise multiplication (逐元素乘法)
    
05_tensor_divide.ipynb
    └─ Element-wise division (逐元素除法)
    
06_tensor_product.ipynb
    └─ Outer and tensor products (外积和张量积)
```

## Progression Logic / 进度逻辑

Tensor operations progress **directly from matrix operations**:

1. **Creation**: 3D arrays using zeros, ones, random
2. **Element-wise operations**: Add, subtract, multiply, divide — broadcast across all dimensions
3. **Tensor product**: Extend dot product to arbitrary dimensions

The key insight: **all element-wise operations work exactly the same on 3D as on 2D**. Broadcasting rules extend naturally. Only tensor multiplication (contraction) requires new thinking.

张量操作**直接从矩阵操作**进行：

1. **创建**：使用zeros、ones、random的3D数组
2. **逐元素操作**：加、减、乘、除 — 跨所有维度广播
3. **张量积**：将点积扩展到任意维度

关键见解：**所有逐元素操作在3D上的工作方式与2D完全相同**。广播规则自然扩展。只有张量乘法（缩并）需要新的思考。

## ML Relevance / 机器学习相关性

In machine learning:
- **Batched operations**:
  - Image batch: (batch_size, 224, 224, 3) — standard for CNN training
  - Element-wise activation: apply ReLU element-wise across entire batch
  - Batch norm: compute statistics along batch axis (axis=0)

- **Sequence models**:
  - Embeddings: (batch_size, seq_len, embedding_dim)
  - Attention: softmax(Q @ K^T) where K^T contracts batch and seq dimensions
  - RNN/LSTM: process sequences with 3D tensor inputs

- **Tensor contractions**:
  - Einsum: `einsum('bij,bjk->bik', A, B)` for batch matrix multiplication
  - Multi-head attention: separate batch and head dimensions for parallel computation
  - Convolutions: 4D tensor operations (batch, height, width, channels)

- **Modern architectures**:
  - Transformers: 3D tensors throughout (batch, seq_len, hidden_dim)
  - Vision transformers: 4D patch embeddings
  - CNNs: 4D convolution kernels

Tensors are the native data structure for modern deep learning frameworks (PyTorch, TensorFlow). Mastering tensor operations is essential for understanding and optimizing neural networks.

在机器学习中：
- **批处理操作**：
  - 图像批次：(batch_size, 224, 224, 3) — CNN训练的标准
  - 逐元素激活：在整个批次中逐元素应用ReLU
  - 批归一化：沿批轴计算统计量（axis=0）

- **序列模型**：
  - 嵌入：(batch_size, seq_len, embedding_dim)
  - 注意力：softmax(Q @ K^T)其中K^T缩并批和序列维度
  - RNN/LSTM：使用3D张量输入处理序列

- **张量缩并**：
  - Einsum：`einsum('bij,bjk->bik', A, B)`用于批矩阵乘法
  - 多头注意力：分离批和头维度以进行并行计算
  - 卷积：4D张量操作（batch、height、width、channels）

- **现代架构**：
  - 变换器：始终使用3D张量（batch、seq_len、hidden_dim）
  - 视觉变换器：4D补丁嵌入
  - CNN：4D卷积核

张量是现代深度学习框架（PyTorch、TensorFlow）的原生数据结构。掌握张量操作对于理解和优化神经网络至关重要。

---
