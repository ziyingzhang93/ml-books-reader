# 机器学习微积分 / Calculus for Machine Learning
## Chapter 12

---

### Examples 1 2

# 01 — Examples 1 2 / 01 Examples 1 2

**Chapter 12 — File 1 of 2 / 第12章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Examples 1 2**.

本脚本演示 **01 Examples 1 2**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy import limit, oo, ln, simplify, pprint
from sympy.abc import x

expression = ln(x-1)/(x-2)
result = limit(expression, x, 2)
# 打印输出 / Print output
print("Limit of")
# 打印输出 / Print output
pprint(expression)
# 打印输出 / Print output
print("at x = 2 is", result)
# 打印输出 / Print output
print()

expression = ln(x)/x
result = limit(expression, x, oo)
# 打印输出 / Print output
print("Limit of")
# 打印输出 / Print output
pprint(expression)
# 打印输出 / Print output
print("at x = infinity is", result)
```

---
## Learning Notes / 学习笔记

- **概念**: Examples 1 2 是机器学习中的常用技术。  
  *Examples 1 2 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Examples 1 2 / 01 Examples 1 2
# Complete Code / 完整代码
# ===============================

from sympy import limit, oo, ln, simplify, pprint
from sympy.abc import x

expression = ln(x-1)/(x-2)
result = limit(expression, x, 2)
# 打印输出 / Print output
print("Limit of")
# 打印输出 / Print output
pprint(expression)
# 打印输出 / Print output
print("at x = 2 is", result)
# 打印输出 / Print output
print()

expression = ln(x)/x
result = limit(expression, x, oo)
# 打印输出 / Print output
print("Limit of")
# 打印输出 / Print output
pprint(expression)
# 打印输出 / Print output
print("at x = infinity is", result)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Examples 3 4 5

# 02 — Examples 3 4 5 / 02 Examples 3 4 5

**Chapter 12 — File 2 of 2 / 第12章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Examples 3 4 5**.

本脚本演示 **02 Examples 3 4 5**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from sympy import limit, oo, sin, cos, simplify, pprint
from sympy.abc import x

expression = x * sin(1/x)
result = limit(expression, x, oo)
# 打印输出 / Print output
print("Limit of")
# 打印输出 / Print output
pprint(expression)
# 打印输出 / Print output
print("at x = infinity is", result)
# 打印输出 / Print output
print()

expression = 1/(1-cos(x)) - 1/x
result = limit(expression, x, 0)
# 打印输出 / Print output
print("Limit of")
# 打印输出 / Print output
pprint(expression)
# 打印输出 / Print output
print("at x = 0 is", result)
# 打印输出 / Print output
print()

expression = (1+x)**(1/x)
result = limit(expression, x, oo)
# 打印输出 / Print output
print("Limit of")
# 打印输出 / Print output
pprint(expression)
# 打印输出 / Print output
print("at x = infinity is", result)
```

---
## Learning Notes / 学习笔记

- **概念**: Examples 3 4 5 是机器学习中的常用技术。  
  *Examples 3 4 5 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Examples 3 4 5 / 02 Examples 3 4 5
# Complete Code / 完整代码
# ===============================

from sympy import limit, oo, sin, cos, simplify, pprint
from sympy.abc import x

expression = x * sin(1/x)
result = limit(expression, x, oo)
# 打印输出 / Print output
print("Limit of")
# 打印输出 / Print output
pprint(expression)
# 打印输出 / Print output
print("at x = infinity is", result)
# 打印输出 / Print output
print()

expression = 1/(1-cos(x)) - 1/x
result = limit(expression, x, 0)
# 打印输出 / Print output
print("Limit of")
# 打印输出 / Print output
pprint(expression)
# 打印输出 / Print output
print("at x = 0 is", result)
# 打印输出 / Print output
print()

expression = (1+x)**(1/x)
result = limit(expression, x, oo)
# 打印输出 / Print output
print("Limit of")
# 打印输出 / Print output
pprint(expression)
# 打印输出 / Print output
print("at x = infinity is", result)
```

---

### Chapter Summary / 章节总结

# Chapter 12 Summary / 第12章总结

## Theme / 主题: Chapter 12 / Chapter 12

This chapter contains **2 code files** demonstrating chapter 12.

本章包含 **2 个代码文件**，演示Chapter 12。

---
## Evolution / 演化路线

  1. `01_examples_1_2.ipynb` — Examples 1 2
  2. `02_examples_3_4_5.ipynb` — Examples 3 4 5

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 12) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 12）是机器学习流水线中的基础构建块。

---
