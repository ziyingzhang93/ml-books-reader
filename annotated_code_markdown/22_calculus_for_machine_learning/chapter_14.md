# 机器学习微积分 / Calculus for Machine Learning
## Chapter 14

---

### Reciprocal

# 01 — Reciprocal / 01 Reciprocal

**Chapter 14 — File 1 of 3 / 第14章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Reciprocal**.

本脚本演示 **01 Reciprocal**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

def f(x):
    return 1/x

epsilon = np.finfo(np.float32).eps
for x in [1, -1]:
    slope = (f(x+epsilon) - f(x))/epsilon
    y = f(x)
    c = y - slope * x
    # 打印输出 / Print output
    print("Slope at x={} is {}".format(x, slope))
    # 打印输出 / Print output
    print("Tangent line is y={:f}x{:+f}".format(slope,c))
```

---
## Learning Notes / 学习笔记

- **概念**: Reciprocal 是机器学习中的常用技术。  
  *Reciprocal is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Reciprocal / 01 Reciprocal
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

def f(x):
    return 1/x

epsilon = np.finfo(np.float32).eps
for x in [1, -1]:
    slope = (f(x+epsilon) - f(x))/epsilon
    y = f(x)
    c = y - slope * x
    # 打印输出 / Print output
    print("Slope at x={} is {}".format(x, slope))
    # 打印输出 / Print output
    print("Tangent line is y={:f}x{:+f}".format(slope,c))
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Squared

# 02 — Squared / 02 Squared

**Chapter 14 — File 2 of 3 / 第14章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Squared**.

本脚本演示 **02 Squared**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

def f(x):
    return x**2

epsilon = np.finfo(np.float32).eps
for x in [2, -2]:
    slope = (f(x+epsilon) - f(x))/epsilon
    y = f(x)
    c = y - slope * x
    # 打印输出 / Print output
    print("Slope at x={} is {}".format(x, slope))
    # 打印输出 / Print output
    print("Tangent line is y={:f}x{:+f}".format(slope,c))
```

---
## Learning Notes / 学习笔记

- **概念**: Squared 是机器学习中的常用技术。  
  *Squared is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Squared / 02 Squared
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

def f(x):
    return x**2

epsilon = np.finfo(np.float32).eps
for x in [2, -2]:
    slope = (f(x+epsilon) - f(x))/epsilon
    y = f(x)
    c = y - slope * x
    # 打印输出 / Print output
    print("Slope at x={} is {}".format(x, slope))
    # 打印输出 / Print output
    print("Tangent line is y={:f}x{:+f}".format(slope,c))
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Polynomial

# 03 — Polynomial / 03 Polynomial

**Chapter 14 — File 3 of 3 / 第14章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Polynomial**.

本脚本演示 **03 Polynomial**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

def f(x):
    return x**3 + 2*x + 1

epsilon = np.finfo(np.float32).eps
for x in [2, 0, -2]:
    slope = (f(x+epsilon) - f(x))/epsilon
    y = f(x)
    c = y - slope * x
    # 打印输出 / Print output
    print("Slope at x={} is {}".format(x, slope))
    # 打印输出 / Print output
    print("Tangent line is y={:f}x{:+f}".format(slope,c))
```

---
## Learning Notes / 学习笔记

- **概念**: Polynomial 是机器学习中的常用技术。  
  *Polynomial is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Polynomial / 03 Polynomial
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

def f(x):
    return x**3 + 2*x + 1

epsilon = np.finfo(np.float32).eps
for x in [2, 0, -2]:
    slope = (f(x+epsilon) - f(x))/epsilon
    y = f(x)
    c = y - slope * x
    # 打印输出 / Print output
    print("Slope at x={} is {}".format(x, slope))
    # 打印输出 / Print output
    print("Tangent line is y={:f}x{:+f}".format(slope,c))
```

---

### Chapter Summary / 章节总结

# Chapter 14 Summary / 第14章总结

## Theme / 主题: Chapter 14 / Chapter 14

This chapter contains **3 code files** demonstrating chapter 14.

本章包含 **3 个代码文件**，演示Chapter 14。

---
## Evolution / 演化路线

  1. `01_reciprocal.ipynb` — Reciprocal
  2. `02_squared.ipynb` — Squared
  3. `03_polynomial.ipynb` — Polynomial

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 14) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 14）是机器学习流水线中的基础构建块。

---
