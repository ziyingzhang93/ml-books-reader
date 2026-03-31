# Python 机器学习 / Python for Machine Learning
## Chapter 03

---

### Lookup

# 01 — Lookup / 01 Lookup

**Chapter 03 — File 1 of 11 / 第03章 — 第1个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Lookup**.

本脚本演示 **01 Lookup**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
value = 0 # This is obtained from a model

value_to_name = {0: "cat", 1: "dog"}
# 打印输出 / Print output
print("Result is %s" % value_to_name[value])
```

---
## Learning Notes / 学习笔记

- **概念**: Lookup 是机器学习中的常用技术。  
  *Lookup is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lookup / 01 Lookup
# Complete Code / 完整代码
# ===============================

value = 0 # This is obtained from a model

value_to_name = {0: "cat", 1: "dog"}
# 打印输出 / Print output
print("Result is %s" % value_to_name[value])
```

---

➡️ **Next / 下一步**: File 2 of 11

---

### Counter

# 02 — Counter / 02 Counter

**Chapter 03 — File 2 of 11 / 第03章 — 第2个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Counter**.

本脚本演示 **02 Counter**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
sentence = "Portez ce vieux whisky au juge blond qui fume"
counter = {}
for char in sentence:
    if char not in counter:
        counter[char] = 0
    counter[char] += 1

# 打印输出 / Print output
print(counter)
```

---
## Learning Notes / 学习笔记

- **概念**: Counter 是机器学习中的常用技术。  
  *Counter is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Counter / 02 Counter
# Complete Code / 完整代码
# ===============================

sentence = "Portez ce vieux whisky au juge blond qui fume"
counter = {}
for char in sentence:
    if char not in counter:
        counter[char] = 0
    counter[char] += 1

# 打印输出 / Print output
print(counter)
```

---

➡️ **Next / 下一步**: File 3 of 11

---

### Listconcat

# 03 — Listconcat / 03 Listconcat

**Chapter 03 — File 3 of 11 / 第03章 — 第3个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Listconcat**.

本脚本演示 **03 Listconcat**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
A = [1, 2, "fizz", 4, "buzz", "fizz", 7]
A += [8, "fizz", "buzz", 11, "fizz", 13, 14, "fizzbuzz"]
# 打印输出 / Print output
print(A)
```

---
## Learning Notes / 学习笔记

- **概念**: Listconcat 是机器学习中的常用技术。  
  *Listconcat is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Listconcat / 03 Listconcat
# Complete Code / 完整代码
# ===============================

A = [1, 2, "fizz", 4, "buzz", "fizz", 7]
A += [8, "fizz", "buzz", 11, "fizz", 13, 14, "fizzbuzz"]
# 打印输出 / Print output
print(A)
```

---

➡️ **Next / 下一步**: File 4 of 11

---

### List Operations

# 04 — List Operations / 04 List Operations

**Chapter 03 — File 4 of 11 / 第03章 — 第4个文件（共11个）**

---

## Summary / 总结

This script demonstrates **List Operations**.

本脚本演示 **04 List Operations**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
A = [1, 2, "fizz", 4, "buzz", "fizz", 7]
A += [8, "fizz", "buzz", 11, "fizz", 13, 14, "fizzbuzz"]
# 打印输出 / Print output
print(A)
A[2:2] = [2.1, 2.2]
# 打印输出 / Print output
print(A)
A[0:2] = []
# 打印输出 / Print output
print(A)
```

---
## Learning Notes / 学习笔记

- **概念**: List Operations 是机器学习中的常用技术。  
  *List Operations is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# List Operations / 04 List Operations
# Complete Code / 完整代码
# ===============================

A = [1, 2, "fizz", 4, "buzz", "fizz", 7]
A += [8, "fizz", "buzz", 11, "fizz", 13, 14, "fizzbuzz"]
# 打印输出 / Print output
print(A)
A[2:2] = [2.1, 2.2]
# 打印输出 / Print output
print(A)
A[0:2] = []
# 打印输出 / Print output
print(A)
```

---

➡️ **Next / 下一步**: File 5 of 11

---

### Swap

# 07 — Swap / 07 Swap

**Chapter 03 — File 5 of 11 / 第03章 — 第5个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Swap**.

本脚本演示 **07 Swap**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
a = 42
b = "foo"
# 打印输出 / Print output
print("a is %s; b is %s" % (a,b))
a, b = b, a # swap
# 打印输出 / Print output
print("After swap, a is %s; b is %s" % (a,b))
```

---
## Learning Notes / 学习笔记

- **概念**: Swap 是机器学习中的常用技术。  
  *Swap is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Swap / 07 Swap
# Complete Code / 完整代码
# ===============================

a = 42
b = "foo"
# 打印输出 / Print output
print("a is %s; b is %s" % (a,b))
a, b = b, a # swap
# 打印输出 / Print output
print("After swap, a is %s; b is %s" % (a,b))
```

---

➡️ **Next / 下一步**: File 6 of 11

---

### Template

# 08 — Template / 08 Template

**Chapter 03 — File 6 of 11 / 第03章 — 第6个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Template**.

本脚本演示 **08 Template**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
template = "Square root of %d is %.3f"
n = 10
answer = template % (n, n**0.5)
# 打印输出 / Print output
print(answer)
```

---
## Learning Notes / 学习笔记

- **概念**: Template 是机器学习中的常用技术。  
  *Template is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Template / 08 Template
# Complete Code / 完整代码
# ===============================

template = "Square root of %d is %.3f"
n = 10
answer = template % (n, n**0.5)
# 打印输出 / Print output
print(answer)
```

---

➡️ **Next / 下一步**: File 7 of 11

---

### Dontcare

# 09 — Dontcare / 09 Dontcare

**Chapter 03 — File 7 of 11 / 第03章 — 第7个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Dontcare**.

本脚本演示 **09 Dontcare**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
A = pd.DataFrame([[11,12,13],[12,13,14],[13,14,15],[15,16,17]], columns=["x","y","z"])
# 打印输出 / Print output
print(A)

for _, row in A.iterrows():
    # 打印输出 / Print output
    print(row["z"])
```

---
## Learning Notes / 学习笔记

- **概念**: Dontcare 是机器学习中的常用技术。  
  *Dontcare is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dontcare / 09 Dontcare
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
A = pd.DataFrame([[11,12,13],[12,13,14],[13,14,15],[15,16,17]], columns=["x","y","z"])
# 打印输出 / Print output
print(A)

for _, row in A.iterrows():
    # 打印输出 / Print output
    print(row["z"])
```

---

➡️ **Next / 下一步**: File 8 of 11

---

### Zip

# 10 — Zip / 10 Zip

**Chapter 03 — File 8 of 11 / 第03章 — 第8个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Zip**.

本脚本演示 **10 Zip**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
a = ["x", "y", "z"]
b = [3, 5, 7, 9]
c = [2.1, 2.5, 2.9]
# 将多个序列配对 / Pair multiple sequences
for x in zip(a, b, c):
    # 打印输出 / Print output
    print(x)
```

---
## Learning Notes / 学习笔记

- **概念**: Zip 是机器学习中的常用技术。  
  *Zip is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Zip / 10 Zip
# Complete Code / 完整代码
# ===============================

a = ["x", "y", "z"]
b = [3, 5, 7, 9]
c = [2.1, 2.5, 2.9]
# 将多个序列配对 / Pair multiple sequences
for x in zip(a, b, c):
    # 打印输出 / Print output
    print(x)
```

---

➡️ **Next / 下一步**: File 9 of 11

---

### Transpose

# 11 — Transpose / 11 Transpose

**Chapter 03 — File 9 of 11 / 第03章 — 第9个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Transpose**.

本脚本演示 **11 Transpose**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
a = [['x', 3, 2.1], ['y', 5, 2.5], ['z', 7, 2.9]]
# 将多个序列配对 / Pair multiple sequences
p,q,r = zip(*a)
# 打印输出 / Print output
print(p)
# 打印输出 / Print output
print(q)
# 打印输出 / Print output
print(r)
```

---
## Learning Notes / 学习笔记

- **概念**: Transpose 是机器学习中的常用技术。  
  *Transpose is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Transpose / 11 Transpose
# Complete Code / 完整代码
# ===============================

a = [['x', 3, 2.1], ['y', 5, 2.5], ['z', 7, 2.9]]
# 将多个序列配对 / Pair multiple sequences
p,q,r = zip(*a)
# 打印输出 / Print output
print(p)
# 打印输出 / Print output
print(q)
# 打印输出 / Print output
print(r)
```

---

➡️ **Next / 下一步**: File 10 of 11

---

### Enumerate

# 12 — Enumerate / 12 Enumerate

**Chapter 03 — File 10 of 11 / 第03章 — 第10个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Enumerate**.

本脚本演示 **12 Enumerate**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
a = ["quick", "brown", "fox", "jumps", "over"]
# 同时获取索引和值 / Get both index and value
for num, item in enumerate(a):
    # 打印输出 / Print output
    print("item %d is %s" % (num, item))
```

---
## Learning Notes / 学习笔记

- **概念**: Enumerate 是机器学习中的常用技术。  
  *Enumerate is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Enumerate / 12 Enumerate
# Complete Code / 完整代码
# ===============================

a = ["quick", "brown", "fox", "jumps", "over"]
# 同时获取索引和值 / Get both index and value
for num, item in enumerate(a):
    # 打印输出 / Print output
    print("item %d is %s" % (num, item))
```

---

➡️ **Next / 下一步**: File 11 of 11

---

### No Enumerate

# 13 — No Enumerate / 13 No Enumerate

**Chapter 03 — File 11 of 11 / 第03章 — 第11个文件（共11个）**

---

## Summary / 总结

This script demonstrates **No Enumerate**.

本脚本演示 **13 No Enumerate**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
a = ["quick", "brown", "fox", "jumps", "over"]
# 获取长度 / Get length
for num in range(len(a)):
    # 打印输出 / Print output
    print("item %d is %s" % (num, a[num]))
```

---
## Learning Notes / 学习笔记

- **概念**: No Enumerate 是机器学习中的常用技术。  
  *No Enumerate is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# No Enumerate / 13 No Enumerate
# Complete Code / 完整代码
# ===============================

a = ["quick", "brown", "fox", "jumps", "over"]
# 获取长度 / Get length
for num in range(len(a)):
    # 打印输出 / Print output
    print("item %d is %s" % (num, a[num]))
```

---

### Chapter Summary / 章节总结

# Chapter 03 Summary / 第03章总结

## Theme / 主题: Chapter 03 / Chapter 03

This chapter contains **11 code files** demonstrating chapter 03.

本章包含 **11 个代码文件**，演示Chapter 03。

---
## Evolution / 演化路线

  1. `01_lookup.ipynb` — Lookup
  2. `02_counter.ipynb` — Counter
  3. `03_listconcat.ipynb` — Listconcat
  4. `04_list_operations.ipynb` — List Operations
  5. `07_swap.ipynb` — Swap
  6. `08_template.ipynb` — Template
  7. `09_dontcare.ipynb` — Dontcare
  8. `10_zip.ipynb` — Zip
  9. `11_transpose.ipynb` — Transpose
  10. `12_enumerate.ipynb` — Enumerate
  11. `13_no_enumerate.ipynb` — No Enumerate

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 03) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 03）是机器学习流水线中的基础构建块。

---
