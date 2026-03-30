# ML微积分
## Appendix C

---

### Sympy

# 01 — Sympy / 01 Sympy

**Chapter appendix_c — File 1 of 3 / 第appendix_c章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Sympy**.

本脚本演示 **01 Sympy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy import *

x = Symbol("x")
expression = x**2 * sin(cos(x))
print(expression)
print(diff(expression))
```

---
## Learning Notes / 学习笔记

- **概念**: Sympy 是机器学习中的常用技术。  
  *Sympy is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sympy / 01 Sympy
# Complete Code / 完整代码
# ===============================

from sympy import *

x = Symbol("x")
expression = x**2 * sin(cos(x))
print(expression)
print(diff(expression))
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Print

# 02 — Print / 02 Print

**Chapter appendix_c — File 2 of 3 / 第appendix_c章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Print**.

本脚本演示 **02 Print**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy import *

w, x, b = symbols("w x b")
y = tanh(w*x + b)
print(y)
print(diff(y, w))
print(diff(y, b))
```

---
## Learning Notes / 学习笔记

- **概念**: Print 是机器学习中的常用技术。  
  *Print is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Print / 02 Print
# Complete Code / 完整代码
# ===============================

from sympy import *

w, x, b = symbols("w x b")
y = tanh(w*x + b)
print(y)
print(diff(y, w))
print(diff(y, b))
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Pprint

# 03 — Pprint / 03 Pprint

**Chapter appendix_c — File 3 of 3 / 第appendix_c章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Pprint**.

本脚本演示 **03 Pprint**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy import *
from sympy.abc import w, x, b

y = tanh(w*x + b)
pprint(y)
pprint(diff(y, w))
pprint(diff(y, b))
```

---
## Learning Notes / 学习笔记

- **概念**: Pprint 是机器学习中的常用技术。  
  *Pprint is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pprint / 03 Pprint
# Complete Code / 完整代码
# ===============================

from sympy import *
from sympy.abc import w, x, b

y = tanh(w*x + b)
pprint(y)
pprint(diff(y, w))
pprint(diff(y, b))
```

---

### Chapter Summary

# Chapter appendix_c Summary / 第appendix_c章总结

## Theme / 主题: Appendix / 附录

This chapter contains **3 code files** demonstrating appendix.

本章包含 **3 个代码文件**，演示附录。

---
## Evolution / 演化路线

  1. `01_sympy.ipynb` — Sympy
  2. `02_print.ipynb` — Print
  3. `03_pprint.ipynb` — Pprint

---
## ML Relevance / ML 关联

The techniques in this chapter (Appendix) are fundamental building blocks in machine learning pipelines.

本章技术（附录）是机器学习流水线中的基础构建块。

---
