# ML微积分
## Chapter 26

---

### Mvp

# 01 — Mvp / 01 Mvp

**Chapter 26 — File 1 of 2 / 第26章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Equality constraint: The result required be zero**.

本脚本演示 **Equality constraint: The result required be zero**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import numpy as np
from scipy.optimize import minimize

def objective(x):
    return 0.25*x[0]**2 + 0.1*x[1]**2 + 0.3*x[0]*x[1]

def constraint1(x):
```

---
## Step 2 — Equality constraint: The result required be zero

```python
return x[0] + x[1] - 1

def constraint2(x):
```

---
## Step 3 — Inequality constraint: The result required be non-negative

```python
return x[0]

def constraint3(x):
```

---
## Step 4 — Inequality constraint: The result required be non-negative

```python
return 1-x[0]
```

---
## Step 5 — initial guesses

```python
x0 = np.array([0, 1])
```

---
## Step 6 — optimize

```python
bounds = ((0,1), (0,1))
constraints = [
    {"type":"eq", "fun":constraint1},
    {"type":"ineq", "fun":constraint2},
    {"type":"ineq", "fun":constraint3},
]
solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
x = solution.x
```

---
## Step 7 — show solution

```python
print('Objective:', objective(x))
print('Solution:', x)
```

---
## Learning Notes / 学习笔记

- **概念**: Equality constraint: The result required be zero 是机器学习中的常用技术。  
  *Equality constraint: The result required be zero is a common technique in machine learning.*

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
# Mvp / 01 Mvp
# Complete Code / 完整代码
# ===============================

import numpy as np
from scipy.optimize import minimize

def objective(x):
    return 0.25*x[0]**2 + 0.1*x[1]**2 + 0.3*x[0]*x[1]

def constraint1(x):
    # Equality constraint: The result required be zero
    return x[0] + x[1] - 1

def constraint2(x):
    # Inequality constraint: The result required be non-negative
    return x[0]

def constraint3(x):
    # Inequality constraint: The result required be non-negative
    return 1-x[0]

# initial guesses
x0 = np.array([0, 1])

# optimize
bounds = ((0,1), (0,1))
constraints = [
    {"type":"eq", "fun":constraint1},
    {"type":"ineq", "fun":constraint2},
    {"type":"ineq", "fun":constraint3},
]
solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
x = solution.x

# show solution
print('Objective:', objective(x))
print('Solution:', x)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Water

# 02 — Water / 02 Water

**Chapter 26 — File 2 of 2 / 第26章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Equality constraint: The result required be zero**.

本脚本演示 **Equality constraint: The result required be zero**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import numpy as np
from scipy.optimize import minimize

def objective(x):
    return np.log(1+0.9*x[0]) + np.log(0.9+0.8*x[1]) + np.log(1+0.7*x[2])
```

---
## Step 2 — Equality constraint: The result required be zero

```python
def constraint1(x):
    return x[0] + x[1] + x[2] - 1
```

---
## Step 3 — Inequality constraints: The result required be non-negative

```python
def constraint2(x):
    return x[0]
def constraint3(x):
    return x[1]
def constraint4(x):
    return x[2]
```

---
## Step 4 — initial guesses

```python
x0 = np.array([0.4, 0.4, 0.4])
```

---
## Step 5 — optimize

```python
bounds = ((0,1), (0,1), (0,1))
constraints = [
    {"type":"eq", "fun":constraint1},
    {"type":"ineq", "fun":constraint2},
    {"type":"ineq", "fun":constraint3},
    {"type":"ineq", "fun":constraint4},
]
solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
x = solution.x
```

---
## Step 6 — show solution

```python
print('Objective:', objective(x))
print('Solution:', x)
```

---
## Learning Notes / 学习笔记

- **概念**: Equality constraint: The result required be zero 是机器学习中的常用技术。  
  *Equality constraint: The result required be zero is a common technique in machine learning.*

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
# Water / 02 Water
# Complete Code / 完整代码
# ===============================

import numpy as np
from scipy.optimize import minimize

def objective(x):
    return np.log(1+0.9*x[0]) + np.log(0.9+0.8*x[1]) + np.log(1+0.7*x[2])

# Equality constraint: The result required be zero
def constraint1(x):
    return x[0] + x[1] + x[2] - 1

# Inequality constraints: The result required be non-negative
def constraint2(x):
    return x[0]
def constraint3(x):
    return x[1]
def constraint4(x):
    return x[2]

# initial guesses
x0 = np.array([0.4, 0.4, 0.4])

# optimize
bounds = ((0,1), (0,1), (0,1))
constraints = [
    {"type":"eq", "fun":constraint1},
    {"type":"ineq", "fun":constraint2},
    {"type":"ineq", "fun":constraint3},
    {"type":"ineq", "fun":constraint4},
]
solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
x = solution.x

# show solution
print('Objective:', objective(x))
print('Solution:', x)
```

---

### Chapter Summary

# Chapter 26 Summary / 第26章总结

## Theme / 主题: Chapter 26 / Chapter 26

This chapter contains **2 code files** demonstrating chapter 26.

本章包含 **2 个代码文件**，演示Chapter 26。

---
## Evolution / 演化路线

  1. `01_mvp.ipynb` — Mvp
  2. `02_water.ipynb` — Water

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 26) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 26）是机器学习流水线中的基础构建块。

---
