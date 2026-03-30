# 线性代数与机器学习 / Linear Algebra for Machine Learning

## Book Summary / 全书总结

# Book Summary / 全书总结
## Linear Algebra for Machine Learning
**21 Chapters · ~100 Code Examples · 4 Parts**

## Part 1: NumPy Foundations / NumPy 基础 (Chapters 04-07)
Master the tools for matrix manipulation / 掌握操作矩阵的工具

| Chapter | Topic / 主题 | Files | Key Concept / 核心概念 |
|---------|-------------|-------|----------------------|
| 04 | Array Creation / 数组创建 | 6 | array, zeros, ones, vstack, hstack |
| 05 | Index, Slice, Reshape / 索引切片重塑 | 16 | Indexing, slicing, train-test split, reshape |
| 06 | Broadcasting / 广播机制 | 4 | Automatic dimension stretching |
| 07 | Aggregate Functions / 聚合函数 | 6 | axis=None vs axis=0 vs axis=1 |

## Part 2: Core Linear Algebra Objects / 核心线性代数对象 (Chapters 08-14)
Understand the building blocks / 理解线性代数的砖块

| Chapter | Topic | Files | Key Concept |
|---------|-------|-------|-------------|
| 08 | Vector Operations / 向量运算 | 7 | +, -, *, /, dot, scalar multiply |
| 09 | Vector Norms / 向量范数 | 3 | L1 (稀疏), L2 (距离), Max (边界) |
| 10 | Matrix Operations / 矩阵运算 | 8 | Hadamard vs dot product, matrix-vector multiply |
| 11 | Special Matrices / 特殊矩阵 | 4 | Triangular, diagonal, identity, orthogonal |
| 12 | Matrix Properties / 矩阵属性 | 6 | Transpose, inverse, trace, determinant, rank |
| 13 | Sparse Matrices / 稀疏矩阵 | 2 | CSR format, sparsity calculation |
| 14 | Tensors / 张量 | 6 | 3D arrays, tensor product (deep learning foundation) |

## Part 3: Matrix Decomposition / 矩阵分解 (Chapters 15-17)
Learn to break matrices apart / 学会拆解矩阵的武器

| Chapter | Topic | Files | Key Formula |
|---------|-------|-------|-------------|
| 15 | Matrix Factorization / 矩阵分解 | 3 | LU: A=PLU, QR: A=QR, Cholesky: A=LLᵀ |
| 16 | Eigendecomposition / 特征分解 | 3 | A = QΛQ⁻¹ |
| 17 | SVD / 奇异值分解 | 7 | A = UΣVᵀ, pseudoinverse, dimensionality reduction |

Progression: General factorization → Eigenvalues → SVD (most powerful)
递进逻辑: 通用分解 → 特征值分解 → SVD（最强大）

## Part 4: Statistics & Applications / 统计与应用 (Chapters 18-24)
Bridge from math to ML / 打通从数学到机器学习的落地

| Chapter | Topic | Files | Application |
|---------|-------|-------|-------------|
| 18 | Statistics / 统计量 | 8 | Mean, variance, covariance, correlation |
| 19 | PCA / 主成分分析 | 2 | Manual PCA + scikit-learn |
| 20 | Linear Regression / 线性回归 | 5 | 4 methods: direct, QR, SVD, lstsq |
| 21 | Visualization / 可视化 | 5 | Wine, Digits, Iris with PCA |
| 22 | Country Comparison / 国家比较 | 2 | Euclidean & cosine distance |
| 23 | Recommender / 推荐系统 | 2 | SVD-based collaborative filtering |
| 24 | Eigenfaces / 特征脸 | 1 | PCA face recognition |

## Reading Path / 阅读路径

```
NumPy 基础 (04-07)          核心对象 (08-14)           分解技术 (15-17)         实战应用 (18-24)
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Arrays          │    │ Vectors              │    │ LU / QR /       │    │ Statistics       │
│ Indexing        │───▶│ Norms                │───▶│ Cholesky        │───▶│ PCA              │
│ Broadcasting    │    │ Matrices             │    │ Eigen           │    │ Linear Regression│
│ Aggregation     │    │ Sparse / Tensors     │    │ SVD             │    │ Real-world Apps  │
└─────────────────┘    └──────────────────────┘    └─────────────────┘    └──────────────────┘
   Tools / 工具          Building Blocks / 砖块     Weapons / 武器        Applications / 应用
```

## Key Takeaways / 核心收获

1. **NumPy is the language of linear algebra in Python** — every ML library builds on it.
   NumPy 是 Python 线性代数的语言——所有 ML 库都建立在它之上。n
2. **Decomposition is the core technique** — LU, QR, Eigen, SVD are not just theory; they power regression, PCA, recommender systems, and face recognition.
   矩阵分解是核心技术——LU、QR、特征分解、SVD 不只是理论，它们驱动着回归、PCA、推荐系统和人脸识别。

3. **The path from math to ML is short** — Chapter 15's SVD directly enables Chapter 23's recommender; Chapter 16's eigendecomposition directly enables Chapter 24's eigenfaces.
   从数学到 ML 的路径很短——第15章的 SVD 直接支撑第23章的推荐系统；第16章的特征分解直接支撑第24章的人脸识别。
