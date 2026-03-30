# ML微积分
## Chapter 11

---

### Verify Power Rule

# 01 — Verify Power Rule / 01 Verify Power Rule

**Chapter 11 — File 1 of 3 / 第11章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Verify Power Rule**.

本脚本演示 **01 Verify Power Rule**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy import diff, sqrt, pprint
from sympy.abc import x

expressions = [x**2, sqrt(x)]
for expression in expressions:
    result = diff(expression, x)
    print("Derivative of")
    pprint(expression)
    print("with respect to x is")
    pprint(result)
    print()
```

---
## Learning Notes / 学习笔记

- **概念**: Verify Power Rule 是机器学习中的常用技术。  
  *Verify Power Rule is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Verify Power Rule / 01 Verify Power Rule
# Complete Code / 完整代码
# ===============================

from sympy import diff, sqrt, pprint
from sympy.abc import x

expressions = [x**2, sqrt(x)]
for expression in expressions:
    result = diff(expression, x)
    print("Derivative of")
    pprint(expression)
    print("with respect to x is")
    pprint(result)
    print()
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Deriv F

# 02 — Deriv F / 02 Deriv F

**Chapter 11 — File 2 of 3 / 第11章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Deriv F**.

本脚本演示 **02 Deriv F**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy import diff, sin, pprint
from sympy.abc import x

u = x**2
v = sin(x)
f = u * v
result = diff(f, x)
print("Derivative of")
pprint(f)
print("with respect to x is")
pprint(result)
```

---
## Learning Notes / 学习笔记

- **概念**: Deriv F 是机器学习中的常用技术。  
  *Deriv F is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Deriv F / 02 Deriv F
# Complete Code / 完整代码
# ===============================

from sympy import diff, sin, pprint
from sympy.abc import x

u = x**2
v = sin(x)
f = u * v
result = diff(f, x)
print("Derivative of")
pprint(f)
print("with respect to x is")
pprint(result)
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Deriv Tan

# 03 — Deriv Tan / 03 Deriv Tan

**Chapter 11 — File 3 of 3 / 第11章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Deriv Tan**.

本脚本演示 **03 Deriv Tan**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy import diff, sin, cos, simplify, pprint
from sympy.abc import x

f = sin(x) / cos(x)
result = diff(f, x)
print("Derivative of")
pprint(f)
print("with respect to x is")
pprint(simplify(result))
```

---
## Learning Notes / 学习笔记

- **概念**: Deriv Tan 是机器学习中的常用技术。  
  *Deriv Tan is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Deriv Tan / 03 Deriv Tan
# Complete Code / 完整代码
# ===============================

from sympy import diff, sin, cos, simplify, pprint
from sympy.abc import x

f = sin(x) / cos(x)
result = diff(f, x)
print("Derivative of")
pprint(f)
print("with respect to x is")
pprint(simplify(result))
```

---

### Chapter Summary

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **3 code files** demonstrating chapter 11.

本章包含 **3 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `01_verify_power_rule.ipynb` — Verify Power Rule
  2. `02_deriv_f.ipynb` — Deriv F
  3. `03_deriv_tan.ipynb` — Deriv Tan

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---
