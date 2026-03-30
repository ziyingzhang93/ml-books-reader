# ML微积分
## Chapter 20

---

### Deriv Gf

# 01 — Deriv Gf / 01 Deriv Gf

**Chapter 20 — File 1 of 5 / 第20章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Deriv Gf**.

本脚本演示 **01 Deriv Gf**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy.abc import x, y
from sympy import diff, sqrt, pprint

f = x**2 - 10
g = sqrt(f)
result = diff(g, x)
print("Function")
pprint(g)
print("has derivative")
pprint(result)
```

---
## Learning Notes / 学习笔记

- **概念**: Deriv Gf 是机器学习中的常用技术。  
  *Deriv Gf is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Deriv Gf / 01 Deriv Gf
# Complete Code / 完整代码
# ===============================

from sympy.abc import x, y
from sympy import diff, sqrt, pprint

f = x**2 - 10
g = sqrt(f)
result = diff(g, x)
print("Function")
pprint(g)
print("has derivative")
pprint(result)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Deriv H

# 02 — Deriv H / 02 Deriv H

**Chapter 20 — File 2 of 5 / 第20章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Deriv H**.

本脚本演示 **02 Deriv H**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy.abc import x, y
from sympy import diff, cos, pprint

u = x**3 - 1
h = cos(u)
result = diff(h, x)
print("Function")
pprint(h)
print("has derivative")
pprint(result)
```

---
## Learning Notes / 学习笔记

- **概念**: Deriv H 是机器学习中的常用技术。  
  *Deriv H is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Deriv H / 02 Deriv H
# Complete Code / 完整代码
# ===============================

from sympy.abc import x, y
from sympy import diff, cos, pprint

u = x**3 - 1
h = cos(u)
result = diff(h, x)
print("Function")
pprint(h)
print("has derivative")
pprint(result)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Deriv H

# 03 — Deriv H / 03 Deriv H

**Chapter 20 — File 3 of 5 / 第20章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Deriv H**.

本脚本演示 **03 Deriv H**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy.abc import x, y
from sympy import diff, sqrt, cos, simplify, pprint

u = x * sqrt(x**2 - 10)
h = cos(u)
result = diff(h, x)
print("Function")
pprint(h)
print("has derivative")
pprint(simplify(result))
```

---
## Learning Notes / 学习笔记

- **概念**: Deriv H 是机器学习中的常用技术。  
  *Deriv H is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Deriv H / 03 Deriv H
# Complete Code / 完整代码
# ===============================

from sympy.abc import x, y
from sympy import diff, sqrt, cos, simplify, pprint

u = x * sqrt(x**2 - 10)
h = cos(u)
result = diff(h, x)
print("Function")
pprint(h)
print("has derivative")
pprint(simplify(result))
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Chain H

# 04 — Chain H / 04 Chain H

**Chapter 20 — File 4 of 5 / 第20章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Chain H**.

本脚本演示 **04 Chain H**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy.abc import x, y
from sympy import diff, pprint

s = x*y
t = 2*x - y
h = s**2 + t**3
dhdx = diff(h, x)
dhdy = diff(h, y)
print("Function")
pprint(h)
print("Derivative with respect to x")
pprint(dhdx)
print("Derivative with respect to y")
pprint(dhdy)
```

---
## Learning Notes / 学习笔记

- **概念**: Chain H 是机器学习中的常用技术。  
  *Chain H is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Chain H / 04 Chain H
# Complete Code / 完整代码
# ===============================

from sympy.abc import x, y
from sympy import diff, pprint

s = x*y
t = 2*x - y
h = s**2 + t**3
dhdx = diff(h, x)
dhdy = diff(h, y)
print("Function")
pprint(h)
print("Derivative with respect to x")
pprint(dhdx)
print("Derivative with respect to y")
pprint(dhdy)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Chain G

# 05 — Chain G / 05 Chain G

**Chapter 20 — File 5 of 5 / 第20章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Chain G**.

本脚本演示 **05 Chain G**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy.abc import x, y
from sympy import diff, cos, exp, pprint

r = x*cos(y)
s = x*exp(y)
t = x + y
h = r**2 - r*s + t**3
dhdx = diff(h, x)
dhdy = diff(h, y)
print("Function")
pprint(h)
print("Derivative with respect to x")
pprint(dhdx)
print("Derivative with respect to y")
pprint(dhdy)
```

---
## Learning Notes / 学习笔记

- **概念**: Chain G 是机器学习中的常用技术。  
  *Chain G is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Chain G / 05 Chain G
# Complete Code / 完整代码
# ===============================

from sympy.abc import x, y
from sympy import diff, cos, exp, pprint

r = x*cos(y)
s = x*exp(y)
t = x + y
h = r**2 - r*s + t**3
dhdx = diff(h, x)
dhdy = diff(h, y)
print("Function")
pprint(h)
print("Derivative with respect to x")
pprint(dhdx)
print("Derivative with respect to y")
pprint(dhdy)
```

---

### Chapter Summary

# Chapter 20 Summary / 第20章总结

## Theme / 主题: Chapter 20 / Chapter 20

This chapter contains **5 code files** demonstrating chapter 20.

本章包含 **5 个代码文件**，演示Chapter 20。

---
## Evolution / 演化路线

  1. `01_deriv_gf.ipynb` — Deriv Gf
  2. `02_deriv_h.ipynb` — Deriv H
  3. `03_deriv_h.ipynb` — Deriv H
  4. `04_chain_h.ipynb` — Chain H
  5. `05_chain_g.ipynb` — Chain G

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 20) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 20）是机器学习流水线中的基础构建块。

---
