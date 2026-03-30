# PyTorch 深度学习 / Deep Learning with PyTorch
## Chapter 04

---

### Matrix

# 01 — Matrix / 01 Matrix

**Chapter 04 — File 1 of 9 / 第04章 — 第1个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Matrix**.

本脚本演示 **01 Matrix**。

---
## Step 1 — Step 1

```python
import torch

x = torch.tensor([1, 2, 3])
print(x)
print(x.shape)
print(x.dtype)
```

---
## Learning Notes / 学习笔记

- **概念**: Matrix 是机器学习中的常用技术。  
  *Matrix is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Matrix / 01 Matrix
# Complete Code / 完整代码
# ===============================

import torch

x = torch.tensor([1, 2, 3])
print(x)
print(x.shape)
print(x.dtype)
```

---

➡️ **Next / 下一步**: File 2 of 9

---

### Grad

# 02 — Grad / 02 Grad

**Chapter 04 — File 2 of 9 / 第04章 — 第2个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Grad**.

本脚本演示 **02 Grad**。

---
## Step 1 — Step 1

```python
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
print(x)
print(x.shape)
print(x.dtype)
```

---
## Learning Notes / 学习笔记

- **概念**: Grad 是机器学习中的常用技术。  
  *Grad is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grad / 02 Grad
# Complete Code / 完整代码
# ===============================

import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
print(x)
print(x.shape)
print(x.dtype)
```

---

➡️ **Next / 下一步**: File 3 of 9

---

### Ops

# 03 — Ops / 03 Ops

**Chapter 04 — File 3 of 9 / 第04章 — 第3个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Ops**.

本脚本演示 **03 Ops**。

---
## Step 1 — Step 1

```python
import torch

x = torch.tensor(3.6, requires_grad=True)
y = x * x
y.backward()
print("x =", x)
print("y =", y)
print("x.grad =", x.grad)
```

---
## Learning Notes / 学习笔记

- **概念**: Ops 是机器学习中的常用技术。  
  *Ops is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ops / 03 Ops
# Complete Code / 完整代码
# ===============================

import torch

x = torch.tensor(3.6, requires_grad=True)
y = x * x
y.backward()
print("x =", x)
print("y =", y)
print("x.grad =", x.grad)
```

---

➡️ **Next / 下一步**: File 4 of 9

---

### Polynomial

# 04 — Polynomial / 04 Polynomial

**Chapter 04 — File 4 of 9 / 第04章 — 第4个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Polynomial**.

本脚本演示 **04 Polynomial**。

---
## Step 1 — Step 1

```python
import numpy as np

polynomial = np.poly1d([1, 2, 3])
print(polynomial)
```

---
## Learning Notes / 学习笔记

- **概念**: Polynomial 是机器学习中的常用技术。  
  *Polynomial is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Polynomial / 04 Polynomial
# Complete Code / 完整代码
# ===============================

import numpy as np

polynomial = np.poly1d([1, 2, 3])
print(polynomial)
```

---

➡️ **Next / 下一步**: File 5 of 9

---

### Eval

# 05 — Eval / 模型评估

**Chapter 04 — File 5 of 9 / 第04章 — 第5个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Eval**.

本脚本演示 **模型评估**。

---
## Step 1 — Step 1

```python
import numpy as np

polynomial = np.poly1d([1, 2, 3])
print(polynomial(1.5))
```

---
## Learning Notes / 学习笔记

- **概念**: Eval 是机器学习中的常用技术。  
  *Eval is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Eval / 模型评估
# Complete Code / 完整代码
# ===============================

import numpy as np

polynomial = np.poly1d([1, 2, 3])
print(polynomial(1.5))
```

---

➡️ **Next / 下一步**: File 6 of 9

---

### Samples

# 06 — Samples / 06 Samples

**Chapter 04 — File 6 of 9 / 第04章 — 第6个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Generate random samples roughly between -10 to +10**.

本脚本演示 **Generate random samples roughly between -10 to +10**。

---
## Step 1 — Step 1

```python
import numpy as np

polynomial = np.poly1d([1, 2, 3])
N = 20   # number of samples
```

---
## Step 2 — Generate random samples roughly between -10 to +10

```python
X = np.random.randn(N,1) * 5
Y = polynomial(X)
print(X)
print(Y)
```

---
## Learning Notes / 学习笔记

- **概念**: Generate random samples roughly between -10 to +10 是机器学习中的常用技术。  
  *Generate random samples roughly between -10 to +10 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Samples / 06 Samples
# Complete Code / 完整代码
# ===============================

import numpy as np

polynomial = np.poly1d([1, 2, 3])
N = 20   # number of samples

# Generate random samples roughly between -10 to +10
X = np.random.randn(N,1) * 5
Y = polynomial(X)
print(X)
print(Y)
```

---

➡️ **Next / 下一步**: File 7 of 9

---

### Grad

# 07 — Grad / 07 Grad

**Chapter 04 — File 7 of 9 / 第04章 — 第7个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Generate random samples roughly between -10 to +10**.

本脚本演示 **Generate random samples roughly between -10 to +10**。

---
## Step 1 — Step 1

```python
import numpy as np
import torch

polynomial = np.poly1d([1, 2, 3])
```

---
## Step 2 — Generate random samples roughly between -10 to +10

```python
N = 20   # number of samples
X = np.random.randn(N,1) * 5
Y = polynomial(X)

XX = np.hstack([X*X, X, np.ones_like(X)])

w = torch.randn(3, 1, requires_grad=True)  # the 3 coefficients
x = torch.tensor(XX, dtype=torch.float32)  # input sample
y = torch.tensor(Y, dtype=torch.float32)   # output sample
optimizer = torch.optim.NAdam([w], lr=0.01)
print(w)

for _ in range(1000):
    y_pred = x @ w
    mse = torch.mean(torch.square(y - y_pred))
    optimizer.zero_grad()
    mse.backward()
    optimizer.step()

print(w)
```

---
## Learning Notes / 学习笔记

- **概念**: Generate random samples roughly between -10 to +10 是机器学习中的常用技术。  
  *Generate random samples roughly between -10 to +10 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grad / 07 Grad
# Complete Code / 完整代码
# ===============================

import numpy as np
import torch

polynomial = np.poly1d([1, 2, 3])

# Generate random samples roughly between -10 to +10
N = 20   # number of samples
X = np.random.randn(N,1) * 5
Y = polynomial(X)

XX = np.hstack([X*X, X, np.ones_like(X)])

w = torch.randn(3, 1, requires_grad=True)  # the 3 coefficients
x = torch.tensor(XX, dtype=torch.float32)  # input sample
y = torch.tensor(Y, dtype=torch.float32)   # output sample
optimizer = torch.optim.NAdam([w], lr=0.01)
print(w)

for _ in range(1000):
    y_pred = x @ w
    mse = torch.mean(torch.square(y - y_pred))
    optimizer.zero_grad()
    mse.backward()
    optimizer.step()

print(w)
```

---

➡️ **Next / 下一步**: File 8 of 9

---

### Fitpoly

# 08 — Fitpoly / 08 Fitpoly

**Chapter 04 — File 8 of 9 / 第04章 — 第8个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Generate random samples roughly between -10 to +10**.

本脚本演示 **Generate random samples roughly between -10 to +10**。

---
## Step 1 — Step 1

```python
import numpy as np
import torch

polynomial = np.poly1d([1, 2, 3])
N = 20   # number of samples
```

---
## Step 2 — Generate random samples roughly between -10 to +10

```python
X = np.random.randn(N,1) * 5
Y = polynomial(X)
```

---
## Step 3 — Prepare input as an array of shape (N,3)

```python
XX = np.hstack([X*X, X, np.ones_like(X)])
```

---
## Step 4 — Prepare tensors

```python
w = torch.randn(3, 1, requires_grad=True)  # the 3 coefficients
x = torch.tensor(XX, dtype=torch.float32)  # input sample
y = torch.tensor(Y, dtype=torch.float32)   # output sample
optimizer = torch.optim.NAdam([w], lr=0.01)
print(w)
```

---
## Step 5 — Run optimizer

```python
for _ in range(1000):
    optimizer.zero_grad()
    y_pred = x @ w
    mse = torch.mean(torch.square(y - y_pred))
    mse.backward()
    optimizer.step()

print(w)
```

---
## Learning Notes / 学习笔记

- **概念**: Generate random samples roughly between -10 to +10 是机器学习中的常用技术。  
  *Generate random samples roughly between -10 to +10 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Fitpoly / 08 Fitpoly
# Complete Code / 完整代码
# ===============================

import numpy as np
import torch

polynomial = np.poly1d([1, 2, 3])
N = 20   # number of samples

# Generate random samples roughly between -10 to +10
X = np.random.randn(N,1) * 5
Y = polynomial(X)

# Prepare input as an array of shape (N,3)
XX = np.hstack([X*X, X, np.ones_like(X)])

# Prepare tensors
w = torch.randn(3, 1, requires_grad=True)  # the 3 coefficients
x = torch.tensor(XX, dtype=torch.float32)  # input sample
y = torch.tensor(Y, dtype=torch.float32)   # output sample
optimizer = torch.optim.NAdam([w], lr=0.01)
print(w)

# Run optimizer
for _ in range(1000):
    optimizer.zero_grad()
    y_pred = x @ w
    mse = torch.mean(torch.square(y - y_pred))
    mse.backward()
    optimizer.step()

print(w)
```

---

➡️ **Next / 下一步**: File 9 of 9

---

### Puzzle

# 09 — Puzzle / 09 Puzzle

**Chapter 04 — File 9 of 9 / 第04章 — 第9个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Gradient descent loop**.

本脚本演示 **Gradient descent loop**。

---
## Step 1 — Step 1

```python
import random
import torch

A = torch.tensor(random.random(), requires_grad=True)
B = torch.tensor(random.random(), requires_grad=True)
C = torch.tensor(random.random(), requires_grad=True)
D = torch.tensor(random.random(), requires_grad=True)
```

---
## Step 2 — Gradient descent loop

```python
EPOCHS = 2000
optimizer = torch.optim.NAdam([A, B, C, D], lr=0.01)
for _ in range(EPOCHS):
    y1 = A + B - 9
    y2 = C - D - 1
    y3 = A + C - 8
    y4 = B - D - 2
    sqerr = y1*y1 + y2*y2 + y3*y3 + y4*y4
    optimizer.zero_grad()
    sqerr.backward()
    optimizer.step()

print(A)
print(B)
print(C)
print(D)
```

---
## Learning Notes / 学习笔记

- **概念**: Gradient descent loop 是机器学习中的常用技术。  
  *Gradient descent loop is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Puzzle / 09 Puzzle
# Complete Code / 完整代码
# ===============================

import random
import torch

A = torch.tensor(random.random(), requires_grad=True)
B = torch.tensor(random.random(), requires_grad=True)
C = torch.tensor(random.random(), requires_grad=True)
D = torch.tensor(random.random(), requires_grad=True)

# Gradient descent loop
EPOCHS = 2000
optimizer = torch.optim.NAdam([A, B, C, D], lr=0.01)
for _ in range(EPOCHS):
    y1 = A + B - 9
    y2 = C - D - 1
    y3 = A + C - 8
    y4 = B - D - 2
    sqerr = y1*y1 + y2*y2 + y3*y3 + y4*y4
    optimizer.zero_grad()
    sqerr.backward()
    optimizer.step()

print(A)
print(B)
print(C)
print(D)
```

---

### Chapter Summary / 章节总结

# Chapter 04 Summary / 第04章总结

## Theme / 主题: Chapter 04 / Chapter 04

This chapter contains **9 code files** demonstrating chapter 04.

本章包含 **9 个代码文件**，演示Chapter 04。

---
## Evolution / 演化路线

  1. `01_matrix.ipynb` — Matrix
  2. `02_grad.ipynb` — Grad
  3. `03_ops.ipynb` — Ops
  4. `04_polynomial.ipynb` — Polynomial
  5. `05_eval.ipynb` — Eval
  6. `06_samples.ipynb` — Samples
  7. `07_grad.ipynb` — Grad
  8. `08_fitpoly.ipynb` — Fitpoly
  9. `09_puzzle.ipynb` — Puzzle

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 04) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 04）是机器学习流水线中的基础构建块。

---
