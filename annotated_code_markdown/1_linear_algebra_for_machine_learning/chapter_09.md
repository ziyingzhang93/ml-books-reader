# 线性代数与机器学习 / Linear Algebra for Machine Learning
## Chapter 09

---

### L1 Norm

# 01 — L1 Norm / L1 范数

**Chapter 09 — File 1 of 3 / 第09章 — 第1个文件（共3个）**

## Summary / 总结

Learn the L1 norm (Manhattan distance), which sums the absolute values of all vector elements. It measures the "taxicab" distance.

学习 L1 范数（曼哈顿距离），它将向量所有元素的绝对值相加。它测量的是"出租车"距离。

## Core Formula / 核心公式

The L1 norm (Manhattan or taxicab norm):
$$\|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i| = |v_1| + |v_2| + \cdots + |v_n|$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Required Functions / 导入所需函数

We import the `array` function from NumPy and the `norm` function from `numpy.linalg`.

```python
# Import array to create vectors
# 导入 array 来创建向量
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Import norm function to compute vector norms
# 导入 norm 函数来计算向量范数
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import norm
```

## Step 2 — Create a Vector / 创建向量

We create a vector with some values.

```python
# Create the vector a = [1, 2, 3]
# 创建向量 a = [1, 2, 3]
a = array([1, 2, 3])
```

## Step 3 — Compute L1 Norm / 计算 L1 范数

We compute the L1 norm using `norm(a, 1)`, where the second argument specifies the order (1 for L1).

```python
# Compute L1 norm: |1| + |2| + |3| = 6
# The second argument 1 specifies L1 norm
# 计算 L1 范数：|1| + |2| + |3| = 6
# 第二个参数 1 指定 L1 范数
l1 = norm(a, 1)

# Print the L1 norm
# 打印 L1 范数
# 打印输出 / Print output
print(l1)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: The L1 norm measures the "Manhattan distance" - the sum of absolute distances along each dimension. It is robust to outliers compared to L2 norm.
- **ML Application / 机器学习应用**: L1 regularization (Lasso) in machine learning promotes sparsity (many zero coefficients), making it useful for feature selection and interpretable models.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `02_l2_norm.ipynb` — Compute the L2 norm (Euclidean distance)

## Complete Code / 完整代码一览

```python
# --- L1 Norm / L1 范数 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import norm
a = array([1, 2, 3])
l1 = norm(a, 1)
# 打印输出 / Print output
print(l1)
```

---

### L2 Norm

# 02 — L2 Norm / L2 范数

**Chapter 09 — File 2 of 3 / 第09章 — 第2个文件（共3个）**

## Summary / 总结

Learn the L2 norm (Euclidean distance), the square root of the sum of squared elements. This is the most common distance metric.

学习 L2 范数（欧几里得距离），即平方元素之和的平方根。这是最常见的距离度量。

## Core Formula / 核心公式

The L2 norm (Euclidean norm):
$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Required Functions / 导入所需函数

We import the `array` function from NumPy and the `norm` function from `numpy.linalg`.

```python
# Import array to create vectors
# 导入 array 来创建向量
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Import norm function to compute vector norms
# 导入 norm 函数来计算向量范数
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import norm
```

## Step 2 — Create a Vector / 创建向量

We create a vector with some values.

```python
# Create the vector a = [1, 2, 3]
# 创建向量 a = [1, 2, 3]
a = array([1, 2, 3])
```

## Step 3 — Compute L2 Norm / 计算 L2 范数

We compute the L2 norm using `norm(a)` (no second argument defaults to L2 norm).

```python
# Compute L2 norm: sqrt(1^2 + 2^2 + 3^2) = sqrt(14) ≈ 3.742
# Default behavior of norm() is L2 norm
# 计算 L2 范数：sqrt(1^2 + 2^2 + 3^2) = sqrt(14) ≈ 3.742
# norm() 的默认行为是 L2 范数
l2 = norm(a)

# Print the L2 norm
# 打印 L2 范数
# 打印输出 / Print output
print(l2)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: The L2 norm is the Euclidean distance, representing the straight-line distance in n-dimensional space. It is rotationally invariant and the most intuitive distance measure.
- **ML Application / 机器学习应用**: L2 regularization (Ridge regression) prevents large weights, improving model generalization. L2 norm is also used in similarity metrics (e.g., in KNN and clustering algorithms).

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `03_max_norm.ipynb` — Compute the max norm

## Complete Code / 完整代码一览

```python
# --- L2 Norm / L2 范数 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import norm
a = array([1, 2, 3])
l2 = norm(a)
# 打印输出 / Print output
print(l2)
```

---

### Max Norm

# 03 — Max Norm / 最大范数

**Chapter 09 — File 3 of 3 / 第09章 — 第3个文件（共3个）**

## Summary / 总结

Learn the max norm (infinity norm), which returns the absolute value of the largest element in a vector.

学习最大范数（无穷范数），它返回向量中最大元素的绝对值。

## Core Formula / 核心公式

The max norm (infinity norm):
$$\|\mathbf{v}\|_{\infty} = \max_{i} |v_i|$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Required Functions / 导入所需函数

We import the `array` function from NumPy, the `inf` constant, and the `norm` function from `numpy.linalg`.

```python
# Import infinity constant for max norm
# 导入用于最大范数的无穷常数
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import inf

# Import array to create vectors
# 导入 array 来创建向量
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# Import norm function to compute vector norms
# 导入 norm 函数来计算向量范数
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import norm
```

## Step 2 — Create a Vector / 创建向量

We create a vector with some values.

```python
# Create the vector a = [1, 2, 3]
# 创建向量 a = [1, 2, 3]
a = array([1, 2, 3])
```

## Step 3 — Compute Max Norm / 计算最大范数

We compute the max norm using `norm(a, inf)`, where `inf` (infinity) specifies the maximum norm.

```python
# Compute max norm: max(|1|, |2|, |3|) = 3
# Pass inf as the second argument to get max norm
# 计算最大范数：max(|1|, |2|, |3|) = 3
# 将 inf 作为第二个参数以获取最大范数
maxnorm = norm(a, inf)

# Print the max norm
# 打印最大范数
# 打印输出 / Print output
print(maxnorm)
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: The max norm returns the largest absolute value in the vector. It is useful for understanding the maximum magnitude of any component in the vector.
- **ML Application / 机器学习应用**: Max norm is used in regularization techniques (max norm constraint) to prevent any single weight from growing too large, which is important for model stability and preventing overfitting.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `../chapter_10/01_create_matrix.ipynb` — Create a matrix

## Complete Code / 完整代码一览

```python
# --- Max Norm / 最大范数 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import inf
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import norm
a = array([1, 2, 3])
maxnorm = norm(a, inf)
# 打印输出 / Print output
print(maxnorm)
```

---

### Chapter Summary / 章节总结

# Chapter 09 Summary / 第09章总结：Vector Norms

## Theme / 主题

Vector norms measure the "size" or "magnitude" of a vector. Different norms capture different properties: L1 encourages sparsity, L2 measures Euclidean distance, and max norm provides a bound. Norms are fundamental to regularization, distance metrics, and optimization.

向量范数测量向量的"大小"或"幅度"。不同的范数捕获不同的属性：L1鼓励稀疏性，L2测量欧几里得距离，最大范数提供界限。范数对于正则化、距离度量和优化至关重要。

## Evolution / 演化路线

```
01_l1_norm.ipynb
    └─ L1 norm: sum of absolute values (曼哈顿距离，鼓励稀疏)
    
02_l2_norm.ipynb
    └─ L2 norm: sqrt(sum of squares) (欧几里得距离，最常见)
    
03_max_norm.ipynb
    └─ Max norm: maximum absolute value (上界，用于梯度剪裁)
```

## Progression Logic / 进度逻辑

Norms are presented by **increasing sophistication of use cases**:

1. **L1 norm**: Sum absolute values → sparse solutions (some weights become zero)
2. **L2 norm**: Euclidean distance → most common, smooth geometry
3. **Max norm**: Clipping constraint → controls largest weight

Each norm has different regularization effects. L1 pushes weights to zero (sparsity), L2 shrinks all weights equally, max norm limits explosion of any single weight.

范数通过**增加用例的复杂性**呈现：

1. **L1范数**：求和绝对值 → 稀疏解（某些权重变为零）
2. **L2范数**：欧几里得距离 → 最常见，光滑几何
3. **最大范数**：剪裁约束 → 控制最大权重

每个范数有不同的正则化效果。L1将权重推向零（稀疏），L2均匀收缩所有权重，最大范数限制任何单个权重的爆炸。

## ML Relevance / 机器学习相关性

In machine learning:
- **L1 norm (regularization)**: 
  - Lasso regression: `loss + λ * ||w||_1` → encourages zero weights
  - Feature selection: zeros out less important features automatically
  - Sparse models for interpretability

- **L2 norm (regularization)**:
  - Ridge regression: `loss + λ * ||w||_2` → shrinks all weights
  - Weight decay in neural networks
  - Euclidean distance for similarity: "how close are two samples?"

- **Max norm**:
  - Gradient clipping: prevent exploding gradients in RNNs
  - Weight constraints in neural networks

Regularization via norms is one of the most important tools for preventing overfitting. L2 is standard, but understanding L1 helps in feature selection.

在机器学习中：
- **L1范数(正则化)**：
  - Lasso回归：`loss + λ * ||w||_1` → 鼓励零权重
  - 特征选择：自动将不重要的特征清零
  - 用于可解释性的稀疏模型

- **L2范数(正则化)**：
  - Ridge回归：`loss + λ * ||w||_2` → 收缩所有权重
  - 神经网络中的权重衰减
  - 用于相似性的欧几里得距离："两个样本有多接近？"

- **最大范数**：
  - 梯度剪裁：防止RNN中的梯度爆炸
  - 神经网络中的权重约束

通过范数进行正则化是防止过拟合的最重要工具之一。L2是标准的，但理解L1有助于特征选择。

---
