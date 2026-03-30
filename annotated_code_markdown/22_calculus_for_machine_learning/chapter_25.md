# ML微积分
## Chapter 25

---

### Example1

# 01 — Example1 / 01 Example1

**Chapter 25 — File 1 of 2 / 第25章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **initial guesses**.

本脚本演示 **initial guesses**。

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
    return x[0]**2 + x[1]**2

def constraint(x):
    return x[0]+2*x[1]-1
```

---
## Step 2 — initial guesses

```python
x0 = np.array([3,3])
```

---
## Step 3 — optimize

```python
bounds = ((-10,10), (-10,10))
constraints = [{"type":"eq", "fun":constraint}]
solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
x = solution.x
```

---
## Step 4 — show solution

```python
print('Objective:', objective(x))
print('Solution:', x)
```

---
## Learning Notes / 学习笔记

- **概念**: initial guesses 是机器学习中的常用技术。  
  *initial guesses is a common technique in machine learning.*

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
# Example1 / 01 Example1
# Complete Code / 完整代码
# ===============================

import numpy as np
from scipy.optimize import minimize

def objective(x):
    return x[0]**2 + x[1]**2

def constraint(x):
    return x[0]+2*x[1]-1

# initial guesses
x0 = np.array([3,3])

# optimize
bounds = ((-10,10), (-10,10))
constraints = [{"type":"eq", "fun":constraint}]
solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
x = solution.x

# show solution
print('Objective:', objective(x))
print('Solution:', x)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Example2

# 02 — Example2 / 02 Example2

**Chapter 25 — File 2 of 2 / 第25章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **initial guesses**.

本脚本演示 **initial guesses**。

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
    return x[0]**2 + 4*x[1]**2

def constraint1(x):
    return x[0]+x[1]

def constraint2(x):
    return x[0]**2 + x[1]**2 - 1
```

---
## Step 2 — initial guesses

```python
x0 = np.array([3,3])
```

---
## Step 3 — optimize

```python
bounds = ((-10,10), (-10,10))
constraints = [{"type":"eq", "fun":constraint1}, {"type":"eq", "fun":constraint2}]
solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
x = solution.x
```

---
## Step 4 — show solution

```python
print('Objective:', objective(x))
print('Solution:', x)
```

---
## Learning Notes / 学习笔记

- **概念**: initial guesses 是机器学习中的常用技术。  
  *initial guesses is a common technique in machine learning.*

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
# Example2 / 02 Example2
# Complete Code / 完整代码
# ===============================

import numpy as np
from scipy.optimize import minimize

def objective(x):
    return x[0]**2 + 4*x[1]**2

def constraint1(x):
    return x[0]+x[1]

def constraint2(x):
    return x[0]**2 + x[1]**2 - 1

# initial guesses
x0 = np.array([3,3])

# optimize
bounds = ((-10,10), (-10,10))
constraints = [{"type":"eq", "fun":constraint1}, {"type":"eq", "fun":constraint2}]
solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
x = solution.x

# show solution
print('Objective:', objective(x))
print('Solution:', x)
```

---

### Chapter Summary

# Chapter 25 Summary / 第25章总结

## Theme / 主题: Chapter 25 / Chapter 25

This chapter contains **2 code files** demonstrating chapter 25.

本章包含 **2 个代码文件**，演示Chapter 25。

---
## Evolution / 演化路线

  1. `01_example1.ipynb` — Example1
  2. `02_example2.ipynb` — Example2

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 25) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 25）是机器学习流水线中的基础构建块。

---
