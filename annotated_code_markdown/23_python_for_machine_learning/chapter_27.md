# Python 机器学习 / Python for Machine Learning
## Chapter 27

---

### Plot

# 02 — Plot / 02 Plot

**Chapter 27 — File 1 of 9 / 第27章 — 第1个文件（共9个）**

---

## Summary / 总结

This script demonstrates **convert vector into 2D arrays**.

本脚本演示 **convert vector into 2D arrays**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 生成等间距数组 / Generate evenly spaced array
x = np.linspace(-1, 1, 100)
# 生成等间距数组 / Generate evenly spaced array
y = np.linspace(-2, 2, 100)
```

---
## Step 2 — convert vector into 2D arrays

```python
xx, yy = np.meshgrid(x,y)
```

---
## Step 3 — computation on matching

```python
z = np.sqrt(1 - xx**2 - (yy/2)**2)

# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_zlim([0,2])
ax.plot_surface(xx, yy, z, cmap="cividis")
ax.view_init(45, 35)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: convert vector into 2D arrays 是机器学习中的常用技术。  
  *convert vector into 2D arrays is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot / 02 Plot
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 生成等间距数组 / Generate evenly spaced array
x = np.linspace(-1, 1, 100)
# 生成等间距数组 / Generate evenly spaced array
y = np.linspace(-2, 2, 100)

# convert vector into 2D arrays
xx, yy = np.meshgrid(x,y)
# computation on matching
z = np.sqrt(1 - xx**2 - (yy/2)**2)

# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_zlim([0,2])
ax.plot_surface(xx, yy, z, cmap="cividis")
ax.view_init(45, 35)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 9

---

### Dims

# 04 — Dims / 04 Dims

**Chapter 27 — File 2 of 9 / 第27章 — 第2个文件（共9个）**

---

## Summary / 总结

This script demonstrates **image has axes 0, 1, and 2, adding axis 3**.

本脚本演示 **image has axes 0, 1, and 2, adding axis 3**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_digits
images = load_digits()["images"]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(images.shape)
```

---
## Step 2 — image has axes 0, 1, and 2, adding axis 3

```python
# 增加一个维度 / Add a dimension
images = np.expand_dims(images, 3)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(images.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: image has axes 0, 1, and 2, adding axis 3 是机器学习中的常用技术。  
  *image has axes 0, 1, and 2, adding axis 3 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dims / 04 Dims
# Complete Code / 完整代码
# ===============================

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import load_digits
images = load_digits()["images"]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(images.shape)

# image has axes 0, 1, and 2, adding axis 3
# 增加一个维度 / Add a dimension
images = np.expand_dims(images, 3)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(images.shape)
```

---

➡️ **Next / 下一步**: File 3 of 9

---

### Positive

# 06 — Positive / 06 Positive

**Chapter 27 — File 3 of 9 / 第27章 — 第3个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Positive**.

本脚本演示 **06 Positive**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 创建NumPy数组 / Create NumPy array
X = np.array([
    [ 1.299,  0.332,  0.594, -0.047,  0.834],
    [ 0.842,  0.441, -0.705, -1.086, -0.252],
    [ 0.785,  0.478, -0.665, -0.532, -0.673],
    [ 0.062,  1.228, -0.333,  0.867,  0.371]
])

y = (X > 0).all(axis=0)
# 打印输出 / Print output
print(y)
```

---
## Learning Notes / 学习笔记

- **概念**: Positive 是机器学习中的常用技术。  
  *Positive is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Positive / 06 Positive
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 创建NumPy数组 / Create NumPy array
X = np.array([
    [ 1.299,  0.332,  0.594, -0.047,  0.834],
    [ 0.842,  0.441, -0.705, -1.086, -0.252],
    [ 0.785,  0.478, -0.665, -0.532, -0.673],
    [ 0.062,  1.228, -0.333,  0.867,  0.371]
])

y = (X > 0).all(axis=0)
# 打印输出 / Print output
print(y)
```

---

➡️ **Next / 下一步**: File 4 of 9

---

### Subarray

# 07 — Subarray / 07 Subarray

**Chapter 27 — File 4 of 9 / 第27章 — 第4个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Subarray**.

本脚本演示 **07 Subarray**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 创建NumPy数组 / Create NumPy array
X = np.array([
    [ 1.299,  0.332,  0.594, -0.047,  0.834],
    [ 0.842,  0.441, -0.705, -1.086, -0.252],
    [ 0.785,  0.478, -0.665, -0.532, -0.673],
    [ 0.062,  1.228, -0.333,  0.867,  0.371]
])

y = X[:, (X > 0).all(axis=0)]
# 打印输出 / Print output
print(y)
```

---
## Learning Notes / 学习笔记

- **概念**: Subarray 是机器学习中的常用技术。  
  *Subarray is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Subarray / 07 Subarray
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 创建NumPy数组 / Create NumPy array
X = np.array([
    [ 1.299,  0.332,  0.594, -0.047,  0.834],
    [ 0.842,  0.441, -0.705, -1.086, -0.252],
    [ 0.785,  0.478, -0.665, -0.532, -0.673],
    [ 0.062,  1.228, -0.333,  0.867,  0.371]
])

y = X[:, (X > 0).all(axis=0)]
# 打印输出 / Print output
print(y)
```

---

➡️ **Next / 下一步**: File 5 of 9

---

### Fancy

# 08 — Fancy / 08 Fancy

**Chapter 27 — File 5 of 9 / 第27章 — 第5个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Fancy**.

本脚本演示 **08 Fancy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 创建NumPy数组 / Create NumPy array
X = np.array([
    [ 1.299,  0.332,  0.594, -0.047,  0.834],
    [ 0.842,  0.441, -0.705, -1.086, -0.252],
    [ 0.785,  0.478, -0.665, -0.532, -0.673],
    [ 0.062,  1.228, -0.333,  0.867,  0.371]
])

y = X[:, [0,1,1,0]]
# 打印输出 / Print output
print(y)
```

---
## Learning Notes / 学习笔记

- **概念**: Fancy 是机器学习中的常用技术。  
  *Fancy is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Fancy / 08 Fancy
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 创建NumPy数组 / Create NumPy array
X = np.array([
    [ 1.299,  0.332,  0.594, -0.047,  0.834],
    [ 0.842,  0.441, -0.705, -1.086, -0.252],
    [ 0.785,  0.478, -0.665, -0.532, -0.673],
    [ 0.062,  1.228, -0.333,  0.867,  0.371]
])

y = X[:, [0,1,1,0]]
# 打印输出 / Print output
print(y)
```

---

➡️ **Next / 下一步**: File 6 of 9

---

### Gaussian

# 09 — Gaussian / 09 Gaussian

**Chapter 27 — File 6 of 9 / 第27章 — 第6个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Gaussian**.

本脚本演示 **09 Gaussian**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
from scipy.stats import multivariate_normal
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

mean = [0, 0]             # zero mean
cov = [[1, 0.8],[0.8, 1]] # covariance matrix
# 生成随机数 / Generate random numbers
X1 = np.random.default_rng().multivariate_normal(mean, cov, 5000)
X2 = multivariate_normal.rvs(mean, cov, 5000)

# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(12,6))
# 创建子图 / Create subplot
ax = plt.subplot(121)
ax.scatter(X1[:,0], X1[:,1], s=1)
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_title("NumPy")

# 创建子图 / Create subplot
ax = plt.subplot(122)
ax.scatter(X2[:,0], X2[:,1], s=1)
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_title("SciPy")

# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Gaussian 是机器学习中的常用技术。  
  *Gaussian is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gaussian / 09 Gaussian
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
from scipy.stats import multivariate_normal
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

mean = [0, 0]             # zero mean
cov = [[1, 0.8],[0.8, 1]] # covariance matrix
# 生成随机数 / Generate random numbers
X1 = np.random.default_rng().multivariate_normal(mean, cov, 5000)
X2 = multivariate_normal.rvs(mean, cov, 5000)

# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(12,6))
# 创建子图 / Create subplot
ax = plt.subplot(121)
ax.scatter(X1[:,0], X1[:,1], s=1)
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_title("NumPy")

# 创建子图 / Create subplot
ax = plt.subplot(122)
ax.scatter(X2[:,0], X2[:,1], s=1)
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_title("SciPy")

# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 7 of 9

---

### Cdf

# 10 — Cdf / 10 Cdf

**Chapter 27 — File 7 of 9 / 第27章 — 第7个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Cdf**.

本脚本演示 **10 Cdf**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from scipy.stats import norm
n = norm.cdf([1,2,3,-1,-2,-3])
# 打印输出 / Print output
print(n)
# 打印输出 / Print output
print(n[:3] - n[-3:])
```

---
## Learning Notes / 学习笔记

- **概念**: Cdf 是机器学习中的常用技术。  
  *Cdf is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cdf / 10 Cdf
# Complete Code / 完整代码
# ===============================

from scipy.stats import norm
n = norm.cdf([1,2,3,-1,-2,-3])
# 打印输出 / Print output
print(n)
# 打印输出 / Print output
print(n[:3] - n[-3:])
```

---

➡️ **Next / 下一步**: File 8 of 9

---

### Ppf

# 12 — Ppf / 12 Ppf

**Chapter 27 — File 8 of 9 / 第27章 — 第8个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Ppf**.

本脚本演示 **12 Ppf**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from scipy.stats import norm
# 打印输出 / Print output
print(norm.ppf(0.99))
```

---
## Learning Notes / 学习笔记

- **概念**: Ppf 是机器学习中的常用技术。  
  *Ppf is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ppf / 12 Ppf
# Complete Code / 完整代码
# ===============================

from scipy.stats import norm
# 打印输出 / Print output
print(norm.ppf(0.99))
```

---

➡️ **Next / 下一步**: File 9 of 9

---

### Tsne

# 13 — Tsne / 13 Tsne

**Chapter 27 — File 9 of 9 / 第27章 — 第9个文件（共9个）**

---

## Summary / 总结

This script demonstrates **normalize input**.

本脚本演示 **normalize input**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
import datetime

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
import numba

def tSNE(X, ndims=2, perplexity=30, seed=0, max_iter=500,
         stop_lying_iter=100, mom_switch_iter=400):
    """The t-SNE algorithm

	Args:
		X: the high-dimensional coordinates
		ndims: number of dimensions in output domain
    Returns:
        Points of X in low dimension
    """
    momentum = 0.5
    final_momentum = 0.8
    eta = 200.0
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    N, _D = X.shape
    # 生成随机数 / Generate random numbers
    np.random.seed(seed)
```

---
## Step 2 — normalize input

```python
X -= X.mean(axis=0) # zero mean
    X /= np.abs(X).max() # min-max scaled
```

---
## Step 3 — compute input similarity for exact t-SNE

```python
P = computeGaussianPerplexity(X, perplexity)
```

---
## Step 4 — symmetrize and normalize input similarities

```python
P = P + P.T
    P /= P.sum()
```

---
## Step 5 — lie about the P-values

```python
P *= 12.0
```

---
## Step 6 — initialize solution

```python
# 生成随机数 / Generate random numbers
Y = np.random.randn(N, ndims) * 0.0001
```

---
## Step 7 — perform main training loop

```python
gains = np.ones_like(Y)
    uY = np.zeros_like(Y)
    # 生成整数序列 / Generate integer sequence
    for i in range(max_iter):
```

---
## Step 8 — compute gradient, update gains

```python
dY = computeExactGradient(P, Y)
        gains = np.where(np.sign(dY) != np.sign(uY), gains+0.2, gains*0.8).clip(0.1)
```

---
## Step 9 — gradient update with momentum and gains

```python
uY = momentum * uY - eta * gains * dY
        Y = Y + uY
```

---
## Step 10 — make the solution zero-mean

```python
Y -= Y.mean(axis=0)
```

---
## Step 11 — Stop lying about the P-values after a while, and switch momentum

```python
if i == stop_lying_iter:
            P /= 12.0
        if i == mom_switch_iter:
            momentum = final_momentum
```

---
## Step 12 — print progress

```python
if (i % 50) == 0:
            C = evaluateError(P, Y)
            now = datetime.datetime.now()
            # 打印输出 / Print output
            print(f"{now} - Iteration {i}: Error = {C}")
    return Y

@numba.jit(nopython=True)
def computeExactGradient(P, Y):
    """Gradient of t-SNE cost function

	Args:
        P: similarity matrix
        Y: low-dimensional coordinates
    Returns:
        dY, a numpy array of shape (N,D)
	"""
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    N, _D = Y.shape
```

---
## Step 13 — compute squared Euclidean distance matrix of Y, the Q matrix, and the
normalization sum

```python
DD = computeSquaredEuclideanDistance(Y)
    Q = 1/(1+DD)
    sum_Q = Q.sum()
```

---
## Step 14 — compute gradient

```python
mult = (P - (Q/sum_Q)) * Q
    dY = np.zeros_like(Y)
    # 生成整数序列 / Generate integer sequence
    for n in range(N):
        # 生成整数序列 / Generate integer sequence
        for m in range(N):
            if n==m: continue
            dY[n] += (Y[n] - Y[m]) * mult[n,m]
    return dY

@numba.jit(nopython=True)
def evaluateError(P, Y):
    """Evaluate t-SNE cost function

    Args:
        P: similarity matrix
        Y: low-dimensional coordinates
    Returns:
        Total t-SNE error C
    """
    DD = computeSquaredEuclideanDistance(Y)
```

---
## Step 15 — Compute Q-matrix and normalization sum

```python
Q = 1/(1+DD)
    np.fill_diagonal(Q, np.finfo(np.float32).eps)
    Q /= Q.sum()
```

---
## Step 16 — Sum t-SNE error: sum P log(P/Q)

```python
error = P * np.log( (P + np.finfo(np.float32).eps)
                       / (Q + np.finfo(np.float32).eps) )
    return error.sum()

@numba.jit(nopython=True)
def computeGaussianPerplexity(X, perplexity):
    """Compute Gaussian Perplexity

    Args:
        X: numpy array of shape (N,D)
        perplexity: double
    Returns:
        Similarity matrix P
    """
```

---
## Step 17 — Compute the squared Euclidean distance matrix

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
N, _D = X.shape
    DD = computeSquaredEuclideanDistance(X)
```

---
## Step 18 — Compute the Gaussian kernel row by row

```python
P = np.zeros_like(DD)
    # 生成整数序列 / Generate integer sequence
    for n in range(N):
        found = False
        beta = 1.0
        min_beta = -np.inf
        max_beta = np.inf
        tol = 1e-5
```

---
## Step 19 — iterate until we get a good perplexity

```python
n_iter = 0
        while not found and n_iter < 200:
```

---
## Step 20 — compute Gaussian kernel row

```python
P[n] = np.exp(-beta * DD[n])
            P[n,n] = np.finfo(np.float32).eps
```

---
## Step 21 — compute entropy of current row
Gaussians to be row-normalized to make it a probability
then H = sum_i -P[i] log(P[i])
= sum_i -P[i] (-beta * DD[n] - log(sum_P))
= sum_i P[i] * beta * DD[n] + log(sum_P)

```python
sum_P = P[n].sum()
            H = beta * (DD[n] @ P[n]) / sum_P + np.log(sum_P)
```

---
## Step 22 — Evaluate if entropy within tolerance level

```python
Hdiff = H - np.log2(perplexity)
            if -tol < Hdiff < tol:
                found = True
                break
            if Hdiff > 0:
                min_beta = beta
                if max_beta in (np.inf, -np.inf):
                    beta *= 2
                else:
                    beta = (beta + max_beta) / 2
            else:
                max_beta = beta
                if min_beta in (np.inf, -np.inf):
                    beta /= 2
                else:
                    beta = (beta + min_beta) / 2
            n_iter += 1
```

---
## Step 23 — normalize this row

```python
P[n] /= P[n].sum()
    assert not np.isnan(P).any()
    return P

@numba.jit(nopython=True)
def computeSquaredEuclideanDistance(X):
    """Compute squared distance
    Args:
        X: numpy array of shape (N,D)
    Returns:
        numpy array of shape (N,N) of squared distances
    """
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    N, _D = X.shape
    # 创建全零数组 / Create array of zeros
    DD = np.zeros((N,N))
    # 生成整数序列 / Generate integer sequence
    for i in range(N-1):
        # 生成整数序列 / Generate integer sequence
        for j in range(i+1, N):
            diff = X[i] - X[j]
            DD[j][i] = DD[i][j] = diff @ diff
    return DD

# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
```

---
## Step 24 — pick 1000 samples from the dataset

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
rows = np.random.choice(X_test.shape[0], 1000, replace=False)
# 转换数据类型 / Convert data type
X_data = X_train[rows].reshape(1000, -1).astype("float")
X_label = y_train[rows]
```

---
## Step 25 — run t-SNE to transform into 2D and visualize in scatter plot

```python
Y = tSNE(X_data, 2, 30, 0, 500, 100, 400)
# 创建画布 / Create figure canvas
plt.figure(figsize=(8,8))
# 绘制散点图 / Draw scatter plot
plt.scatter(Y[:,0], Y[:,1], c=X_label)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: normalize input 是机器学习中的常用技术。  
  *normalize input is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `matplotlib` | 绑图库 | Plotting library |
| `np.ones` | 全一数组 | Array filled with ones |
| `np.random` | 随机数生成 | Random number generation |
| `np.zeros` | 全零数组 | Array filled with zeros |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Tsne / 13 Tsne
# Complete Code / 完整代码
# ===============================

import datetime

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
import numba

def tSNE(X, ndims=2, perplexity=30, seed=0, max_iter=500,
         stop_lying_iter=100, mom_switch_iter=400):
    """The t-SNE algorithm

	Args:
		X: the high-dimensional coordinates
		ndims: number of dimensions in output domain
    Returns:
        Points of X in low dimension
    """
    momentum = 0.5
    final_momentum = 0.8
    eta = 200.0
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    N, _D = X.shape
    # 生成随机数 / Generate random numbers
    np.random.seed(seed)

    # normalize input
    X -= X.mean(axis=0) # zero mean
    X /= np.abs(X).max() # min-max scaled

    # compute input similarity for exact t-SNE
    P = computeGaussianPerplexity(X, perplexity)
    # symmetrize and normalize input similarities
    P = P + P.T
    P /= P.sum()
    # lie about the P-values
    P *= 12.0
    # initialize solution
    # 生成随机数 / Generate random numbers
    Y = np.random.randn(N, ndims) * 0.0001
    # perform main training loop
    gains = np.ones_like(Y)
    uY = np.zeros_like(Y)
    # 生成整数序列 / Generate integer sequence
    for i in range(max_iter):
        # compute gradient, update gains
        dY = computeExactGradient(P, Y)
        gains = np.where(np.sign(dY) != np.sign(uY), gains+0.2, gains*0.8).clip(0.1)
        # gradient update with momentum and gains
        uY = momentum * uY - eta * gains * dY
        Y = Y + uY
        # make the solution zero-mean
        Y -= Y.mean(axis=0)
        # Stop lying about the P-values after a while, and switch momentum
        if i == stop_lying_iter:
            P /= 12.0
        if i == mom_switch_iter:
            momentum = final_momentum
        # print progress
        if (i % 50) == 0:
            C = evaluateError(P, Y)
            now = datetime.datetime.now()
            # 打印输出 / Print output
            print(f"{now} - Iteration {i}: Error = {C}")
    return Y

@numba.jit(nopython=True)
def computeExactGradient(P, Y):
    """Gradient of t-SNE cost function

	Args:
        P: similarity matrix
        Y: low-dimensional coordinates
    Returns:
        dY, a numpy array of shape (N,D)
	"""
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    N, _D = Y.shape
    # compute squared Euclidean distance matrix of Y, the Q matrix, and the
    # normalization sum
    DD = computeSquaredEuclideanDistance(Y)
    Q = 1/(1+DD)
    sum_Q = Q.sum()
    # compute gradient
    mult = (P - (Q/sum_Q)) * Q
    dY = np.zeros_like(Y)
    # 生成整数序列 / Generate integer sequence
    for n in range(N):
        # 生成整数序列 / Generate integer sequence
        for m in range(N):
            if n==m: continue
            dY[n] += (Y[n] - Y[m]) * mult[n,m]
    return dY

@numba.jit(nopython=True)
def evaluateError(P, Y):
    """Evaluate t-SNE cost function

    Args:
        P: similarity matrix
        Y: low-dimensional coordinates
    Returns:
        Total t-SNE error C
    """
    DD = computeSquaredEuclideanDistance(Y)
    # Compute Q-matrix and normalization sum
    Q = 1/(1+DD)
    np.fill_diagonal(Q, np.finfo(np.float32).eps)
    Q /= Q.sum()
    # Sum t-SNE error: sum P log(P/Q)
    error = P * np.log( (P + np.finfo(np.float32).eps)
                       / (Q + np.finfo(np.float32).eps) )
    return error.sum()

@numba.jit(nopython=True)
def computeGaussianPerplexity(X, perplexity):
    """Compute Gaussian Perplexity

    Args:
        X: numpy array of shape (N,D)
        perplexity: double
    Returns:
        Similarity matrix P
    """
    # Compute the squared Euclidean distance matrix
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    N, _D = X.shape
    DD = computeSquaredEuclideanDistance(X)
    # Compute the Gaussian kernel row by row
    P = np.zeros_like(DD)
    # 生成整数序列 / Generate integer sequence
    for n in range(N):
        found = False
        beta = 1.0
        min_beta = -np.inf
        max_beta = np.inf
        tol = 1e-5

        # iterate until we get a good perplexity
        n_iter = 0
        while not found and n_iter < 200:
            # compute Gaussian kernel row
            P[n] = np.exp(-beta * DD[n])
            P[n,n] = np.finfo(np.float32).eps
            # compute entropy of current row
            # Gaussians to be row-normalized to make it a probability
            # then H = sum_i -P[i] log(P[i])
            #        = sum_i -P[i] (-beta * DD[n] - log(sum_P))
            #        = sum_i P[i] * beta * DD[n] + log(sum_P)
            sum_P = P[n].sum()
            H = beta * (DD[n] @ P[n]) / sum_P + np.log(sum_P)
            # Evaluate if entropy within tolerance level
            Hdiff = H - np.log2(perplexity)
            if -tol < Hdiff < tol:
                found = True
                break
            if Hdiff > 0:
                min_beta = beta
                if max_beta in (np.inf, -np.inf):
                    beta *= 2
                else:
                    beta = (beta + max_beta) / 2
            else:
                max_beta = beta
                if min_beta in (np.inf, -np.inf):
                    beta /= 2
                else:
                    beta = (beta + min_beta) / 2
            n_iter += 1
        # normalize this row
        P[n] /= P[n].sum()
    assert not np.isnan(P).any()
    return P

@numba.jit(nopython=True)
def computeSquaredEuclideanDistance(X):
    """Compute squared distance
    Args:
        X: numpy array of shape (N,D)
    Returns:
        numpy array of shape (N,N) of squared distances
    """
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    N, _D = X.shape
    # 创建全零数组 / Create array of zeros
    DD = np.zeros((N,N))
    # 生成整数序列 / Generate integer sequence
    for i in range(N-1):
        # 生成整数序列 / Generate integer sequence
        for j in range(i+1, N):
            diff = X[i] - X[j]
            DD[j][i] = DD[i][j] = diff @ diff
    return DD

# 加载数据集 / Load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# pick 1000 samples from the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
rows = np.random.choice(X_test.shape[0], 1000, replace=False)
# 转换数据类型 / Convert data type
X_data = X_train[rows].reshape(1000, -1).astype("float")
X_label = y_train[rows]
# run t-SNE to transform into 2D and visualize in scatter plot
Y = tSNE(X_data, 2, 30, 0, 500, 100, 400)
# 创建画布 / Create figure canvas
plt.figure(figsize=(8,8))
# 绘制散点图 / Draw scatter plot
plt.scatter(Y[:,0], Y[:,1], c=X_label)
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 27 Summary / 第27章总结

## Theme / 主题: Chapter 27 / Chapter 27

This chapter contains **9 code files** demonstrating chapter 27.

本章包含 **9 个代码文件**，演示Chapter 27。

---
## Evolution / 演化路线

  1. `02_plot.ipynb` — Plot
  2. `04_dims.ipynb` — Dims
  3. `06_positive.ipynb` — Positive
  4. `07_subarray.ipynb` — Subarray
  5. `08_fancy.ipynb` — Fancy
  6. `09_gaussian.ipynb` — Gaussian
  7. `10_cdf.ipynb` — Cdf
  8. `12_ppf.ipynb` — Ppf
  9. `13_tsne.ipynb` — Tsne

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 27) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 27）是机器学习流水线中的基础构建块。

---
