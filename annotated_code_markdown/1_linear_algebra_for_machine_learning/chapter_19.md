# 线性代数与机器学习 / Linear Algebra for Machine Learning
## Chapter 19

---

### Pca

# 19.1 — Principal Component Analysis (PCA) / 主成分分析

**Chapter 19 — File 1 of 2 / 第19章 — 第1个文件（共2个）**

## Summary / 总结

Implement PCA from scratch using NumPy. PCA finds the principal directions of maximum variance in data through eigendecomposition of the covariance matrix.

使用NumPy从头实现PCA。PCA通过协方差矩阵的特征分解找到数据中最大方差的主方向。

## Core Formula / 核心公式

1. **Center data**: $C = X - \mu$
2. **Covariance matrix**: $V = \frac{1}{n-1}C^T C$
3. **Eigendecomposition**: $V = U \Lambda U^T$ where $\Lambda$ are eigenvalues
4. **Project**: $P = U^T C^T$ (transform to PC space)

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import NumPy linear algebra functions
# 导入NumPy线性代数函数
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import cov
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import eig
```

## Step 2 — Create Sample Data / 创建样本数据

```python
# Define a simple 3x2 dataset (3 observations, 2 features)
# 定义一个简单的3x2数据集
A = array([[1, 2],
           [3, 4],
           [5, 6]])
# 打印输出 / Print output
print("Original data A:")
# 打印输出 / Print output
print(A)
```

## Step 3 — Center the Data / 中心化数据

```python
# Compute mean of each feature (along features dimension)
# 计算每个特征的均值
M = mean(A.T, axis=1)
# 打印输出 / Print output
print(f"Mean of each feature: {M}")

# Center the data by subtracting the mean
# 通过减去均值来中心化数据
C = A - M
# 打印输出 / Print output
print(f"\nCentered data C:")
# 打印输出 / Print output
print(C)
```

## Step 4 — Compute Covariance Matrix / 计算协方差矩阵

```python
# Compute covariance matrix from centered data
# 从中心化数据计算协方差矩阵
V = cov(C.T)
# 打印输出 / Print output
print("Covariance matrix V:")
# 打印输出 / Print output
print(V)
```

## Step 5 — Eigendecomposition / 特征分解

```python
# Perform eigendecomposition: V = U * Lambda * U^T
# eig() returns (eigenvalues, eigenvectors)
# 执行特征分解
# eig()返回（特征值，特征向量）
values, vectors = eig(V)
# 打印输出 / Print output
print("Eigenvalues (variance explained by each PC):")
# 打印输出 / Print output
print(values)
# 打印输出 / Print output
print("\nEigenvectors (principal component directions):")
# 打印输出 / Print output
print(vectors)
```

## Step 6 — Project Data to PC Space / 投影数据到PC空间

```python
# Project centered data onto principal components
# P = vectors^T * C^T (project observations onto PCs)
# 将中心化数据投影到主成分上
P = vectors.T.dot(C.T)
# 打印输出 / Print output
print("Projected data (PCs):")
# 打印输出 / Print output
print(P.T)  # Transpose back to show as (observations, components)
```

## Learning Notes / 学习笔记

- **Math Essence**: PCA finds orthogonal directions (eigenvectors) that maximize variance in the data. The eigenvalues represent the amount of variance explained along each principal component. The larger the eigenvalue, the more important that component.
  
  **数学本质**：PCA找到正交方向（特征向量），最大化数据中的方差。特征值代表每个主成分解释的方差量。特征值越大，该成分越重要。

- **ML Application**: PCA is used for dimensionality reduction (keeping top-k PCs), visualization (projecting to 2D/3D), noise reduction, and feature extraction. It reveals hidden structure in high-dimensional data without using labels.
  
  **ML应用**：PCA用于降维、可视化、噪声减少和特征提取。它在不使用标签的情况下揭示高维数据中的隐藏结构。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `02_pca_scikit_learn.ipynb` — Using scikit-learn for PCA

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import cov
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.linalg import eig

# --- Manual PCA Implementation / 手动PCA实现 ---
A = array([[1, 2],
           [3, 4],
           [5, 6]])
# 打印输出 / Print output
print(A)

# --- Step 1: Center Data / 步骤1：中心化数据 ---
M = mean(A.T, axis=1)
C = A - M

# --- Step 2: Covariance Matrix / 步骤2：协方差矩阵 ---
V = cov(C.T)

# --- Step 3: Eigendecomposition / 步骤3：特征分解 ---
values, vectors = eig(V)
# 打印输出 / Print output
print(vectors)
# 打印输出 / Print output
print(values)

# --- Step 4: Project Data / 步骤4：投影数据 ---
P = vectors.T.dot(C.T)
# 打印输出 / Print output
print(P.T)
```

---

### Pca Scikit Learn



---

### Chapter Summary / 章节总结



---
