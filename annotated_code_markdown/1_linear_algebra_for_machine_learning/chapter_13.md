# 线性代数与机器学习
## Chapter 13

---

### Sparse

# Sparse Matrices / 稀疏矩阵

**Chapter 13 — File 1 of 2 / 第13章 — 第1个文件（共2个）**

## Summary / 总结

Sparse matrices are matrices where most elements are zero. Storing them in dense format (standard arrays) wastes memory. Instead, sparse formats (CSR, CSC, COO, etc.) only store non-zero values and their positions. This dramatically reduces memory usage and improves computational efficiency. Sparse matrices are ubiquitous in ML applications like NLP (TF-IDF), recommendation systems, and large-scale linear systems.

稀疏矩阵是大多数元素为零的矩阵。以密集格式（标准数组）存储会浪费内存。相反，稀疏格式（CSR、CSC、COO 等）只存储非零值及其位置。这大大减少了内存使用量并提高了计算效率。稀疏矩阵在机器学习应用中无处不在，如自然语言处理（TF-IDF）、推荐系统和大规模线性系统。

## Core Formats / 核心格式

- **CSR (Compressed Sparse Row)**: Efficient for row operations and arithmetic
- **CSC (Compressed Sparse Column)**: Efficient for column operations
- **COO (Coordinate)**: Simple, flexible format for construction

Memory savings: For a matrix with 99% zeros, sparse format uses ~1% of dense storage.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create a sparse matrix / 创建稀疏矩阵

Start with a dense matrix that is mostly zeros, then convert to CSR (Compressed Sparse Row) format.

```python
from numpy import array
from scipy.sparse import csr_matrix

# Create a dense matrix (3x6) with mostly zeros
# 创建一个大多是零的密集矩阵（3x6）
A = array([[1, 0, 0, 1, 0, 0],
           [0, 0, 2, 0, 0, 1],
           [0, 0, 0, 2, 0, 0]])

print("Dense matrix A / 密集矩阵 A:")
print(A)
print(f"\nDense matrix shape: {A.shape}")
print(f"Total elements: {A.size}")
print(f"Non-zero elements: {(A != 0).sum()}")
print(f"Sparsity: {(A == 0).sum() / A.size * 100:.1f}%")
```

## Step 2 — Convert to sparse format / 转换为稀疏格式

Convert the dense matrix to CSR (Compressed Sparse Row) format for efficient storage.

```python
# Convert dense array to CSR (Compressed Sparse Row) format
# 将密集数组转换为 CSR（压缩稀疏行）格式
S = csr_matrix(A)

print("Sparse matrix S (CSR format) / 稀疏矩阵 S（CSR 格式）:")
print(S)
print(f"\nSparse format stores:")
print(f"  - data: {S.data}")
print(f"  - indices: {S.indices}")
print(f"  - indptr: {S.indptr}")
print(f"\n稀疏格式存储：")
print(f"  - 非零值（data）：{S.data}")
print(f"  - 列索引（indices）：{S.indices}")
print(f"  - 行指针（indptr）：{S.indptr}")
```

## Step 3 — Convert back to dense / 转回密集格式

Verify that we can convert the sparse matrix back to dense format without loss of information.

```python
# Convert sparse matrix back to dense format
# 将稀疏矩阵转回密集格式
B = S.todense()

print("Converted back to dense format / 转回密集格式:")
print(B)
print(f"\nOriginal matrix A == Reconstructed matrix B: {(A == B).all()}")
print(f"原始矩阵 A == 重构矩阵 B：{(A == B).all()}")
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Sparse matrix storage only preserves non-zero elements. CSR format uses three arrays: data (values), indices (column positions), and indptr (row pointers), reducing storage from O(mn) to O(nnz) where nnz is non-zero count.

- **ML Application / 机器学习应用**: Sparse matrices are essential for: text vectorization (TF-IDF with millions of features), user-item matrices in recommendation systems, graph adjacency matrices, and large-scale distributed ML where memory is constrained.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `02_calculate_sparsity.ipynb` — Calculate sparsity metric and analyze sparse matrix properties

## Complete Code / 完整代码一览

```python
# --- Sparse Matrices / 稀疏矩阵 ---
from numpy import array
from scipy.sparse import csr_matrix

# Step 1: Create a dense matrix with mostly zeros
# 步骤 1：创建一个大多是零的密集矩阵
A = array([[1, 0, 0, 1, 0, 0],
           [0, 0, 2, 0, 0, 1],
           [0, 0, 0, 2, 0, 0]])

print("Dense matrix A / 密集矩阵 A:")
print(A)
print(f"\nNon-zero elements: {(A != 0).sum()}/{A.size}")
print(f"Sparsity: {(A == 0).sum() / A.size * 100:.1f}%")

# Step 2: Convert to CSR (Compressed Sparse Row) format
# 步骤 2：转换为 CSR（压缩稀疏行）格式
S = csr_matrix(A)

print("\nSparse matrix S (CSR format) / 稀疏矩阵 S（CSR 格式）:")
print(S)

# Step 3: Convert back to dense
# 步骤 3：转回密集格式
B = S.todense()

print("\nConverted back to dense / 转回密集格式:")
print(B)
print(f"\nReconstruction successful: {(A == B).all()}")
print(f"重构成功：{(A == B).all()}")
```

---

### Calculate Sparsity

# Calculate Sparsity / 计算稀疏度

**Chapter 13 — File 2 of 2 / 第13章 — 第2个文件（共2个）**

## Summary / 总结

Sparsity is a measure of how many zeros a matrix contains, expressed as a percentage or fraction. Sparsity = (number of zeros) / (total elements). A sparsity close to 1.0 indicates a very sparse matrix where storing in dense format is wasteful. Computing sparsity helps decide whether to use sparse data structures and estimate memory savings. Understanding sparsity is critical for choosing appropriate algorithms and data structures in ML.

稀疏度是衡量矩阵包含多少零的指标，表示为百分比或分数。稀疏度 = 零数 / 总元素数。接近 1.0 的稀疏度表示以密集格式存储非常浪费。计算稀疏度有助于决定是否使用稀疏数据结构并估计内存节省。理解稀疏度对于在机器学习中选择适当的算法和数据结构至关重要。

## Core Formula / 核心公式

$$\text{Sparsity} = \frac{\text{count of zeros}}{\text{total elements}} = \frac{m \cdot n - \text{nnz}}{m \cdot n}$$

$$\text{Sparsity} = 1 - \frac{\text{non-zero elements}}{\text{total elements}}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create a sparse matrix / 创建稀疏矩阵

Create the same matrix from the previous notebook.

```python
from numpy import array

# Create a sparse matrix (3x6) with mostly zeros
# 创建一个大多是零的稀疏矩阵（3x6）
A = array([[1, 0, 0, 1, 0, 0],
           [0, 0, 2, 0, 0, 1],
           [0, 0, 0, 2, 0, 0]])

print("Matrix A / 矩阵 A:")
print(A)
```

## Step 2 — Calculate sparsity / 计算稀疏度

Use `count_nonzero()` to count non-zero elements, then calculate sparsity.

```python
from numpy import count_nonzero

# Calculate sparsity
# 计算稀疏度
sparsity = 1.0 - count_nonzero(A) / A.size

print(f"Matrix dimensions: {A.shape}")
print(f"矩阵维度：{A.shape}")

print(f"\nTotal elements: {A.size}")
print(f"总元素数：{A.size}")

print(f"\nNon-zero elements: {count_nonzero(A)}")
print(f"非零元素数：{count_nonzero(A)}")

print(f"\nZero elements: {A.size - count_nonzero(A)}")
print(f"零元素数：{A.size - count_nonzero(A)}")

print(f"\nSparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
print(f"稀疏度：{sparsity:.4f} ({sparsity*100:.2f}%)")

print(f"\nMemory savings with sparse format: ~{sparsity*100:.1f}% reduction")
print(f"使用稀疏格式的内存节省：~{sparsity*100:.1f}% 的减少")
```

## Step 3 — Interpret sparsity / 解释稀疏度

Demonstrate how sparsity interpretation guides algorithm and data structure choice.

```python
# Interpretation based on sparsity level
# 根据稀疏度水平的解释
if sparsity > 0.95:
    interpretation = "Extremely sparse - Always use sparse formats (CSR, CSC, COO)"
    interpretation_zh = "极其稀疏 - 始终使用稀疏格式（CSR、CSC、COO）"
elif sparsity > 0.80:
    interpretation = "Very sparse - Strongly recommend sparse formats"
    interpretation_zh = "非常稀疏 - 强烈推荐使用稀疏格式"
elif sparsity > 0.50:
    interpretation = "Moderately sparse - Consider sparse formats for large matrices"
    interpretation_zh = "中等稀疏 - 对于大矩阵考虑使用稀疏格式"
else:
    interpretation = "Low sparsity - Dense format may be acceptable"
    interpretation_zh = "低稀疏度 - 密集格式可能是可接受的"

print(f"Sparsity interpretation / 稀疏度解释:")
print(f"{interpretation}")
print(f"{interpretation_zh}")

# Calculate theoretical memory savings
# 计算理论内存节省
print(f"\nTheoretical memory usage comparison (assuming 8 bytes per float64):")
print(f"理论内存使用情况比较（假设每个 float64 8 字节）：")
dense_memory = A.size * 8
sparse_nnz = count_nonzero(A)
sparse_memory = (sparse_nnz * 2 + A.shape[0] + 1) * 8  # Simplified estimate
print(f"  Dense format: {dense_memory} bytes")
print(f"  Sparse format: ~{sparse_memory} bytes")
print(f"  Savings: {(1 - sparse_memory/dense_memory)*100:.1f}%")
```

## Learning Notes / 学习笔记

- **Math Essence / 数学本质**: Sparsity is a fundamental property affecting both memory and computation complexity. It determines whether specialized sparse linear algebra algorithms are beneficial. Sparse algorithms can achieve O(nnz) complexity instead of O(mn) where nnz << mn.

- **ML Application / 机器学习应用**: Sparsity analysis is crucial for: efficient feature representation in NLP and computer vision, dimensionality analysis in high-dimensional data, memory-efficient recommendation systems, and choosing between dense/sparse algorithms in optimization and deep learning.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `../chapter_14/01_create_tensor.ipynb` — Explore tensors as multi-dimensional arrays and fundamental ML data structures

## Complete Code / 完整代码一览

```python
# --- Calculate Sparsity / 计算稀疏度 ---
from numpy import array
from numpy import count_nonzero

# Step 1: Create a sparse matrix
# 步骤 1：创建一个稀疏矩阵
A = array([[1, 0, 0, 1, 0, 0],
           [0, 0, 2, 0, 0, 1],
           [0, 0, 0, 2, 0, 0]])

print("Matrix A / 矩阵 A:")
print(A)

# Step 2: Calculate sparsity
# 步骤 2：计算稀疏度
sparsity = 1.0 - count_nonzero(A) / A.size

print(f"\nMatrix shape: {A.shape}")
print(f"Total elements: {A.size}")
print(f"Non-zero elements: {count_nonzero(A)}")
print(f"Zero elements: {A.size - count_nonzero(A)}")
print(f"\nSparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
print(f"Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
```

---

### Chapter Summary

# Chapter 13 Summary / 第13章总结：Sparse Matrices

## Theme / 主题

Most real-world matrices are mostly zeros. Storing and computing with dense matrices wastes memory and time. Sparse matrices represent only the nonzero entries, enabling efficient operations on large datasets with sparse structure. This is critical for NLP (text), recommendation systems, and graph neural networks.

大多数现实矩阵主要是零。存储和使用密集矩阵浪费内存和时间。稀疏矩阵只表示非零条目，使得在具有稀疏结构的大型数据集上进行高效操作成为可能。这对于NLP（文本）、推荐系统和图神经网络至关重要。

## Evolution / 演化路线

```
01_sparse_conversion.ipynb
    └─ Convert dense → sparse (密集转稀疏，节省内存)
    
02_sparsity_calculation.ipynb
    └─ Measure % nonzero entries (测量非零条目的百分比)
```

## Progression Logic / 进度逻辑

Sparse matrices are learned through **practical conversion and measurement**:

1. **Conversion**: Create dense matrix → convert to sparse format
2. **Measurement**: Calculate sparsity (% nonzero) to understand structure

The key insight: sparse matrices use **different storage formats** (COO, CSR, CSC) to only store nonzero values. This reduces memory from O(m·n) to O(nnz) where nnz = number of nonzeros.

稀疏矩阵通过**实际转换和测量**来学习：

1. **转换**：创建密集矩阵→转换为稀疏格式
2. **测量**：计算稀疏度（%非零）以了解结构

关键见解：稀疏矩阵使用**不同的存储格式**（COO、CSR、CSC）只存储非零值。这将内存从O(m·n)减少到O(nnz)，其中nnz =非零数量。

## ML Relevance / 机器学习相关性

In machine learning:
- **NLP/Text**:
  - Bag-of-words matrices: (n_documents, n_vocabulary) is typically 99.9% sparse
  - TF-IDF vectors: sparse representation of document content
  - Without sparse matrices: impossible to process real-world text datasets

- **Recommendation systems**:
  - User-item matrix: (n_users, n_items) is mostly zeros (users rate few items)
  - Factorization: SVD on sparse user-item matrices
  - Memory savings: dense representation would require TB of RAM

- **Graph neural networks**:
  - Adjacency matrices: (n_nodes, n_nodes) is sparse for most graphs
  - Message passing: aggregate only from neighbors (sparse operation)

- **Dimensionality reduction**:
  - PCA on sparse data: special algorithms preserve sparsity
  - Feature hashing: create high-dimensional sparse representations

Sparse matrices are not optional in modern ML. They're essential for scaling to realistic datasets. Understanding sparsity patterns is key to efficient algorithms.

在机器学习中：
- **NLP/文本**：
  - 词袋矩阵：(n_documents, n_vocabulary)通常99.9%稀疏
  - TF-IDF向量：文档内容的稀疏表示
  - 没有稀疏矩阵：无法处理现实文本数据集

- **推荐系统**：
  - 用户-项目矩阵：(n_users, n_items)大多是零（用户评价很少项目）
  - 因子分解：对稀疏用户-项目矩阵的SVD
  - 内存节省：密集表示需要TB级RAM

- **图神经网络**：
  - 邻接矩阵：(n_nodes, n_nodes)对于大多数图是稀疏的
  - 消息传递：仅从邻居聚合（稀疏操作）

- **降维**：
  - 稀疏数据上的PCA：特殊算法保留稀疏性
  - 特征哈希：创建高维稀疏表示

稀疏矩阵在现代ML中不是可选的。它们对于扩展到现实数据集至关重要。理解稀疏性模式是高效算法的关键。

---
