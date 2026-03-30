# 线性代数与机器学习
## Chapter 20

---

### Dataset

# 20.1 — Linear Regression Dataset / 线性回归数据集

**Chapter 20 — File 1 of 5 / 第20章 — 第1个文件（共5个）**

## Summary / 总结

Load and visualize a simple linear regression dataset. This is the starting point for learning different methods to solve linear regression.

加载和可视化一个简单的线性回归数据集。这是学习不同线性回归求解方法的起点。

## Core Concept / 核心概念

Linear regression seeks to find a line $y = mx + b$ that best fits the data by minimizing the sum of squared errors:
$$\text{minimize } \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import array operations and plotting
# 导入数组操作和绘图
from numpy import array
from matplotlib import pyplot
```

## Step 2 — Define Dataset / 定义数据集

```python
# Define a simple 2D dataset: (X, y) pairs
# Each row is [x_value, y_value]
# 定义一个简单的2D数据集：(X, y)对
# 每行是[x值, y值]
data = array([
    [0.05, 0.12],
    [0.18, 0.22],
    [0.31, 0.35],
    [0.42, 0.38],
    [0.5, 0.49]
])
print("Dataset:")
print(data)
```

## Step 3 — Separate Features and Target / 分离特征和目标

```python
# Extract X (features) from first column
# Extract y (target) from second column
# 从第一列提取X（特征）
# 从第二列提取y（目标）
X, y = data[:, 0], data[:, 1]
print(f"X (features): {X}")
print(f"y (target): {y}")
```

## Step 4 — Reshape Features for Algorithms / 重塑特征以适应算法

```python
# Linear regression algorithms expect X as a 2D array (n_samples, n_features)
# Currently X is 1D, so reshape to (5, 1)
# 线性回归算法期望X为2D数组
# 当前X是1D，所以重塑为(5, 1)
X = X.reshape((len(X), 1))
print(f"X shape: {X.shape}")
print(f"X reshaped:")
print(X)
```

## Step 5 — Visualize the Data / 可视化数据

```python
# Create a scatter plot of the data
# 创建数据的散点图
pyplot.scatter(X, y)
pyplot.xlabel('X')
pyplot.ylabel('y')
pyplot.title('Linear Regression Dataset')
pyplot.show()

print("The data shows a clear positive linear relationship")
```

## Learning Notes / 学习笔记

- **Math Essence**: Linear regression solves the least squares problem: find $b$ minimizing $||Xb - y||^2$. This has an analytical solution $b = (X^T X)^{-1} X^T y$ (normal equation) or can be solved via matrix decompositions (QR, SVD).
  
  **数学本质**：线性回归解决最小二乘问题：找到最小化 $||Xb - y||^2$ 的 $b$。这有解析解 $b = (X^T X)^{-1} X^T y$（正规方程）或可通过矩阵分解求解（QR、SVD）。

- **ML Application**: Linear regression is the foundation of supervised learning. Despite its simplicity, it's effective for high-dimensional data and serves as a baseline for more complex models. Key steps: (1) Load data, (2) Fit model (next steps), (3) Evaluate predictions.
  
  **ML应用**：线性回归是监督学习的基础。尽管很简单，但对高维数据很有效，并为更复杂的模型服务。关键步骤：(1) 加载数据，(2) 拟合模型（后续步骤），(3) 评估预测。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next / 下一步**: `02_direct_solution.ipynb` — Solving linear regression using the normal equation

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
from numpy import array
from matplotlib import pyplot

# --- Load and Prepare Data / 加载和准备数据 ---
data = array([[0.05, 0.12],
              [0.18, 0.22],
              [0.31, 0.35],
              [0.42, 0.38],
              [0.5, 0.49]])
X, y = data[:, 0], data[:, 1]
X = X.reshape((len(X), 1))

# --- Visualize / 可视化 ---
pyplot.scatter(X, y)
pyplot.show()
```

---

### Qr Decomposition Solution

# 20.3 — QR Decomposition Solution / QR分解求解

**Chapter 20 — File 3 of 5 / 第20章 — 第3个文件（共5个）**

## Summary / 总结

Solve linear regression using QR decomposition. This method is more numerically stable than the normal equation because it avoids computing $X^T X$ explicitly.

使用QR分解求解线性回归。这种方法比正规方程更数值稳定，因为它避免显式计算 $X^T X$。

## Core Formula / 核心公式

1. **QR Decomposition**: $X = QR$ where Q is orthogonal, R is upper triangular
2. **Solve**: $Rb = Q^T y$ (back substitution is stable)

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import QR decomposition and matrix inversion
# 导入QR分解和矩阵求逆
from numpy import array
from numpy.linalg import inv
from numpy.linalg import qr
from matplotlib import pyplot
```

## Step 2 — Load Dataset / 加载数据集

```python
# Load the regression dataset
# 加载回归数据集
data = array([[0.05, 0.12],
              [0.18, 0.22],
              [0.31, 0.35],
              [0.42, 0.38],
              [0.5, 0.49]])
X, y = data[:, 0], data[:, 1]
X = X.reshape((len(X), 1))
print(f"X shape: {X.shape}, y shape: {y.shape}")
```

## Step 3 — Perform QR Decomposition / 执行QR分解

```python
# Decompose X = Q * R
# Q: orthogonal matrix (Q^T Q = I)
# R: upper triangular matrix
# 分解 X = Q * R
# Q：正交矩阵 (Q^T Q = I)
# R：上三角矩阵
Q, R = qr(X)
print(f"Q shape: {Q.shape}")
print(f"R shape: {R.shape}")
print(f"\nQ =\n{Q}")
print(f"\nR =\n{R}")
```

## Step 4 — Verify Decomposition / 验证分解

```python
# Verify that Q*R = X
# 验证 Q*R = X
X_reconstructed = Q.dot(R)
print("Original X:")
print(X)
print("\nReconstructed X (Q*R):")
print(X_reconstructed)
print(f"\nReconstruction error: {max((X - X_reconstructed).flatten())}")
```

## Step 5 — Solve Rb = Q^T y / 求解 Rb = Q^T y

```python
# Use QR to solve: Rb = Q^T y
# Equivalent to: X*b = y  -->  (QR)*b = y  -->  Q*R*b = y
# Multiply both sides by Q^T: R*b = Q^T*y
# 使用QR求解
QTy = Q.T.dot(y)
print(f"Q^T y = {QTy}")

# Solve R*b = Q^T*y by inverting R
# 通过对R求逆来求解
b = inv(R).dot(QTy)
print(f"\nCoefficients b: {b}")
```

## Step 6 — Make Predictions / 进行预测

```python
# Compute predictions
# 计算预测值
yhat = X.dot(b)
print(f"Predictions: {yhat}")
print(f"Actual:      {y}")
```

## Step 7 — Visualize Results / 可视化结果

```python
# Plot original data and fitted line
# 绘制原始数据和拟合线
pyplot.scatter(X, y)
pyplot.plot(X, yhat, color='red')
pyplot.xlabel('X')
pyplot.ylabel('y')
pyplot.title('Linear Regression - QR Decomposition Solution')
pyplot.show()
```

## Learning Notes / 学习笔记

- **Math Essence**: QR decomposition avoids the condition number squared problem of normal equations. Since we solve $Rb = Q^T y$ directly (where R is upper triangular, back-substitution is stable), this is numerically superior to computing $(X^T X)^{-1}$.
  
  **数学本质**：QR分解避免了正规方程的条件数平方问题。直接求解 $Rb = Q^T y$ 更数值稳定。

- **ML Application**: QR decomposition is the preferred method for medium-sized problems. It's implemented in many statistical packages. For very large datasets, iterative methods or SGD are preferred. QR is also used in other algorithms like least squares for polynomial fitting.
  
  **ML应用**：QR分解是中等规模问题的首选方法。它在许多统计包中实现。对于大型数据集，迭代方法或SGD更好。QR也用于多项式拟合等其他算法。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

➡️ **Next / 下一步**: `04_svd_solution.ipynb` — Using Singular Value Decomposition (SVD)

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
from numpy import array
from numpy.linalg import inv
from numpy.linalg import qr
from matplotlib import pyplot

# --- Load Data / 加载数据 ---
data = array([[0.05, 0.12],
              [0.18, 0.22],
              [0.31, 0.35],
              [0.42, 0.38],
              [0.5, 0.49]])
X, y = data[:, 0], data[:, 1]
X = X.reshape((len(X), 1))

# --- QR Decomposition Solution / QR分解求解 ---
Q, R = qr(X)
b = inv(R).dot(Q.T).dot(y)
print(b)

# --- Make Predictions and Plot / 进行预测和绘图 ---
yhat = X.dot(b)
pyplot.scatter(X, y)
pyplot.plot(X, yhat, color='red')
pyplot.show()
```

---

### Linear Regression Function

# 20.5 — Linear Regression via lstsq / 使用lstsq的线性回归

**Chapter 20 — File 5 of 5 / 第20章 — 第5个文件（共5个）**

## Summary / 总结

Use NumPy's `lstsq()` function for linear regression. This is the recommended practical approach as it automatically selects the best underlying algorithm and returns useful diagnostic information.

使用NumPy的 `lstsq()` 函数进行线性回归。这是推荐的实际方法，因为它自动选择最佳的基础算法并返回有用的诊断信息。

## Core Function / 核心函数

**numpy.linalg.lstsq**: Solves $Xb = y$ for $b$ using the best numerical approach
- Returns: coefficients, residuals, rank, singular values

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import lstsq and plotting
# 导入lstsq和绘图
from numpy import array
from numpy.linalg import lstsq
from matplotlib import pyplot
```

## Step 2 — Load Dataset / 加载数据集

```python
# Load the regression dataset
# 加载回归数据集
data = array([[0.05, 0.12],
              [0.18, 0.22],
              [0.31, 0.35],
              [0.42, 0.38],
              [0.5, 0.49]])
X, y = data[:, 0], data[:, 1]
X = X.reshape((len(X), 1))
print(f"X shape: {X.shape}, y shape: {y.shape}")
```

## Step 3 — Solve Using lstsq / 使用lstsq求解

```python
# Call lstsq to solve X*b = y
# lstsq() returns a tuple with 4 elements:
# - coefficients (solution)
# - residuals (sum of squared errors)
# - rank (matrix rank)
# - singular values
# 调用lstsq求解 X*b = y
# lstsq()返回包含4个元素的元组
b, residuals, rank, s = lstsq(X, y)
print(f"Coefficients b: {b}")
```

## Step 4 — Examine Diagnostic Information / 检查诊断信息

```python
# Analyze the diagnostic information returned by lstsq
# 分析lstsq返回的诊断信息
print(f"\nRank of X: {rank}")
print(f"Expected rank (features): {X.shape[1]}")
print(f"Full rank: {rank == X.shape[1]}")

print(f"\nSingular values: {s}")
print(f"Condition number: {s[0] / s[-1]:.6f}")
print("(Condition number measures sensitivity to numerical errors)")

print(f"\nResidual sum of squares: {residuals}")
if residuals.size > 0:
    mse = residuals[0] / len(y)
    print(f"Mean squared error: {mse:.6f}")
```

## Step 5 — Make Predictions / 进行预测

```python
# Compute predictions using the fitted coefficients
# 使用拟合的系数计算预测值
yhat = X.dot(b)
print(f"Predictions: {yhat}")
print(f"Actual:      {y}")
print(f"\nResiduals (y - yhat): {y - yhat}")
```

## Step 6 — Visualize Results / 可视化结果

```python
# Plot original data and fitted line
# 绘制原始数据和拟合线
pyplot.scatter(X, y, label='Data')
pyplot.plot(X, yhat, color='red', label='Fitted line')
pyplot.xlabel('X')
pyplot.ylabel('y')
pyplot.legend()
pyplot.title('Linear Regression - lstsq Solution')
pyplot.show()
```

## Step 7 — Compare Methods / 比较方法

```python
# Compare with other methods for verification
# 与其他方法比较以验证
from numpy.linalg import inv, pinv, qr

# Method 1: Normal equation
b1 = inv(X.T.dot(X)).dot(X.T).dot(y)

# Method 2: QR decomposition
Q, R = qr(X)
b2 = inv(R).dot(Q.T).dot(y)

# Method 3: SVD (pseudo-inverse)
b3 = pinv(X).dot(y)

# Method 4: lstsq
b4 = b  # already computed

print("Comparison of methods:")
print(f"Normal equation:    {b1}")
print(f"QR decomposition:   {b2}")
print(f"SVD (pseudo-inv):   {b3}")
print(f"lstsq:              {b4}")
print("\nAll methods produce the same result (within numerical precision)")
```

## Learning Notes / 学习笔记

- **Math Essence**: `lstsq()` internally uses efficient algorithms (usually SVD-based) to solve the least squares problem. It handles all numerical edge cases and returns diagnostic information (rank, residuals, singular values) useful for model evaluation.
  
  **数学本质**：`lstsq()` 内部使用高效算法（通常基于SVD）来解决最小二乘问题。它处理所有数值边界情况并返回诊断信息。

- **ML Application**: Use `lstsq()` for practical linear regression. It's the recommended approach because: (1) Numerically stable, (2) Returns diagnostic info (rank, condition number, residuals), (3) Automatically handles overdetermined and underdetermined systems, (4) Faster than manual computation. This is what scikit-learn uses internally.
  
  **ML应用**：使用 `lstsq()` 进行实际线性回归。这是推荐的方法，因为它数值稳定，返回诊断信息，自动处理各种情况。这是scikit-learn内部使用的方法。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

➡️ **Next / 下一步**: `../chapter_21/01_load_data.ipynb` — PCA Visualization and Applications

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
from numpy import array
from numpy.linalg import lstsq
from matplotlib import pyplot

# --- Load Data / 加载数据 ---
data = array([[0.05, 0.12],
              [0.18, 0.22],
              [0.31, 0.35],
              [0.42, 0.38],
              [0.5, 0.49]])
X, y = data[:, 0], data[:, 1]
X = X.reshape((len(X), 1))

# --- Solve Using lstsq / 使用lstsq求解 ---
b, residuals, rank, s = lstsq(X, y)
print(b)

# --- Make Predictions and Plot / 进行预测和绘图 ---
yhat = X.dot(b)
pyplot.scatter(X, y)
pyplot.plot(X, yhat, color='red')
pyplot.show()
```

---

### Chapter Summary

# Chapter 20 Summary / 第20章总结：Linear Regression

## Theme / 主题

Linear regression is the canonical ML problem: given (X, y), find w such that y ≈ X·w. This chapter shows there are multiple linear algebra approaches to solve it: direct (normal equation), numerically stable (QR), robust (SVD), and convenient (lstsq). Each approach trades off speed, stability, and generality. Understanding these options is essential for production ML.

线性回归是规范的ML问题：给定(X, y)，找到w使得y ≈ X·w。本章显示有多个线性代数方法来解决它：直接（正规方程）、数值稳定（QR）、鲁棒（SVD）和方便（lstsq）。每种方法在速度、稳定性和通用性之间权衡。理解这些选项对于生产ML至关重要。

## Evolution / 演化路线

```
01_load_dataset.ipynb
    └─ Load and prepare data (加载和准备数据)
    
02_normal_equation.ipynb
    └─ Direct: w = (X^T · X)^(-1) · X^T · y (最直接)
       Fastest, but numerically unstable for ill-conditioned X
    
03_qr_solution.ipynb
    └─ QR: w = R^(-1) · Q^T · y (数值稳定)
       More stable than normal equation, works for rectangular X
    
04_svd_solution.ipynb
    └─ SVD: w = V · Σ^+ · U^T · y (最稳定)
       Handles rank-deficient X, gives pseudoinverse solution
    
05_lstsq.ipynb
    └─ Convenience: w = np.linalg.lstsq(X, y) (最方便)
       Automatic choice of best method, recommended for production
```

## Progression Logic / 进度逻辑

Linear regression solutions progress from **simple → stable → general**:

1. **Normal equation**: w = (X^T·X)^(-1)·X^T·y
   - Pros: Direct, O(n²) time
   - Cons: Requires inversion, numerically unstable if X is ill-conditioned

2. **QR solution**: w = R^(-1)·Q^T·y where X = Q·R
   - Pros: More stable (orthogonal Q), no inversion needed (triangular solve)
   - Cons: Slower than normal equation, but more reliable

3. **SVD solution**: w = V·Σ+·U^T·y where X = U·Σ·V^T
   - Pros: Most stable, handles rank-deficient X (gives minimum norm solution)
   - Cons: Slowest, but most robust

4. **lstsq**: Automatic selection
   - Automatically chooses best method
   - Recommended for production
   - Handles edge cases gracefully

The trade-off: **speed vs. stability vs. generality**. For well-conditioned X, all three are similar. For ill-conditioned or rank-deficient X, QR and SVD are essential.

线性回归解从**简单→稳定→通用**进行：

1. **正规方程**：w = (X^T·X)^(-1)·X^T·y
   - 优点：直接，O(n²)时间
   - 缺点：需要求逆，如果X条件不好则数值不稳定

2. **QR解**：w = R^(-1)·Q^T·y，其中X = Q·R
   - 优点：更稳定（正交Q），不需要求逆（三角形求解）
   - 缺点：比正规方程慢，但更可靠

3. **SVD解**：w = V·Σ+·U^T·y，其中X = U·Σ·V^T
   - 优点：最稳定，处理秩不足的X（给出最小范数解）
   - 缺点：最慢，但最鲁棒

4. **lstsq**：自动选择
   - 自动选择最佳方法
   - 推荐用于生产
   - 优雅地处理边界情况

权衡：**速度vs.稳定性vs.通用性**。对于条件好的X，所有三个都很相似。对于条件不好或秩不足的X，QR和SVD至关重要。

## ML Relevance / 机器学习相关性

In machine learning:
- **Normal equation**: Classic approach, taught in textbooks
  - Fast for small datasets
  - Fails silently on ill-conditioned data
  - Not recommended for production

- **QR decomposition**: Numerically stable standard
  - Default for most statistical software (R, MATLAB)
  - Recommended when data has multicollinearity but is full-rank
  - Works for rectangular X (m != n)

- **SVD**: Most general and robust
  - Handles rank-deficient X (underdetermined systems)
  - Gives minimum-norm solution when infinite solutions exist
  - Essential for data with redundant features
  - Used internally by scikit-learn.LinearRegression

- **lstsq**: Production-ready
  - Recommended for practical use
  - `np.linalg.lstsq(X, y)` returns [w, residuals, rank, s]
  - Returns rank and singular values (useful diagnostics)
  - Automatically handles edge cases

- **When to use each**:
  - **Well-conditioned X**: Any method works (normal equation fastest)
  - **Ill-conditioned X**: Use QR or SVD
  - **Rank-deficient X**: Must use SVD (or regularization)
  - **Production**: Always use lstsq

- **Regularization** (not in this chapter, but related):
  - Ridge (L2): adds λ·I term to normal equation
  - Lasso (L1): regularization via optimization
  - Both help when X is ill-conditioned or has many features

**Key insight**: The problem y = X·w has multiple solutions when:
- m < n (more features than samples): infinite solutions, use SVD for minimum norm
- rank(X) < n: some features are redundant, SVD gives least-norm solution
- X is ill-conditioned: numerical errors dominant, use QR or SVD

在机器学习中：
- **正规方程**：经典方法，教科书中教导
  - 对于小数据集很快
  - 在条件不好的数据上无声地失败
  - 不建议用于生产

- **QR分解**：数值稳定的标准
  - 大多数统计软件的默认（R、MATLAB）
  - 当数据有多重共线性但满秩时推荐
  - 适用于矩形X（m != n）

- **SVD**：最通用和最鲁棒
  - 处理秩不足的X（欠定系统）
  - 当存在无限解时给出最小范数解
  - 对于具有冗余特征的数据至关重要
  - 由scikit-learn.LinearRegression内部使用

- **lstsq**：生产就绪
  - 推荐用于实际使用
  - `np.linalg.lstsq(X, y)`返回[w, residuals, rank, s]
  - 返回秩和奇异值（有用的诊断）
  - 自动处理边界情况

- **何时使用每个**：
  - **条件好的X**：任何方法都有效（正规方程最快）
  - **条件不好的X**：使用QR或SVD
  - **秩不足的X**：必须使用SVD（或正则化）
  - **生产**：总是使用lstsq

- **正则化**（不在本章中，但相关）：
  - Ridge（L2）：在正规方程中添加λ·I项
  - Lasso（L1）：通过优化进行正则化
  - 两者在X条件不好或特征多时都有帮助

**关键见解**：问题y = X·w在以下情况有多个解：
- m < n（特征多于样本）：无限解，使用SVD获得最小范数
- rank(X) < n：某些特征冗余，SVD给出最小范数解
- X条件不好：数值错误主要，使用QR或SVD

---
