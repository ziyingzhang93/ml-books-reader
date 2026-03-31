# Python 机器学习 / Python for Machine Learning
## Chapter 20

---

### Add

# 01 — Add / 01 Add

**Chapter 20 — File 1 of 13 / 第20章 — 第1个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Add**.

本脚本演示 **01 Add**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def add(a, b):
    return a + b

c = add("one", "two")
```

---
## Learning Notes / 学习笔记

- **概念**: Add 是机器学习中的常用技术。  
  *Add is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Add / 01 Add
# Complete Code / 完整代码
# ===============================

def add(a, b):
    return a + b

c = add("one", "two")
```

---

➡️ **Next / 下一步**: File 2 of 13

---

### Add

# 02 — Add / 02 Add

**Chapter 20 — File 2 of 13 / 第20章 — 第2个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Add**.

本脚本演示 **02 Add**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def add(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("Input must be numbers")
    return a + b

add("one", "two")
```

---
## Learning Notes / 学习笔记

- **概念**: Add 是机器学习中的常用技术。  
  *Add is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Add / 02 Add
# Complete Code / 完整代码
# ===============================

def add(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("Input must be numbers")
    return a + b

add("one", "two")
```

---

➡️ **Next / 下一步**: File 3 of 13

---

### Add



---

### Range



---

### Add

# 05 — Add / 05 Add

**Chapter 20 — File 5 of 13 / 第20章 — 第5个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Add**.

本脚本演示 **05 Add**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def add(a, b):
    assert isinstance(a, (int, float)), "`a` must be a number"
    assert isinstance(b, (int, float)), "`b` must be a number"
    return a + b

add("one", "two")
```

---
## Learning Notes / 学习笔记

- **概念**: Add 是机器学习中的常用技术。  
  *Add is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Add / 05 Add
# Complete Code / 完整代码
# ===============================

def add(a, b):
    assert isinstance(a, (int, float)), "`a` must be a number"
    assert isinstance(b, (int, float)), "`b` must be a number"
    return a + b

add("one", "two")
```

---

➡️ **Next / 下一步**: File 6 of 13

---

### Assert

# 06 — Assert / 06 Assert

**Chapter 20 — File 6 of 13 / 第20章 — 第6个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Assert**.

本脚本演示 **06 Assert**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def evenitems(arr):
    newarr = []
    # 获取长度 / Get length
    for i in range(len(arr)):
        if i % 2 == 0:
            # 添加元素到列表末尾 / Append element to list end
            newarr.append(arr[i])
    # 获取长度 / Get length
    assert len(newarr) * 2 >= len(arr)
    return newarr
```

---
## Learning Notes / 学习笔记

- **概念**: Assert 是机器学习中的常用技术。  
  *Assert is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Assert / 06 Assert
# Complete Code / 完整代码
# ===============================

def evenitems(arr):
    newarr = []
    # 获取长度 / Get length
    for i in range(len(arr)):
        if i % 2 == 0:
            # 添加元素到列表末尾 / Append element to list end
            newarr.append(arr[i])
    # 获取长度 / Get length
    assert len(newarr) * 2 >= len(arr)
    return newarr
```

---

➡️ **Next / 下一步**: File 7 of 13

---

### Binary

# 07 — Binary / 07 Binary

**Chapter 20 — File 7 of 13 / 第20章 — 第7个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Binary**.

本脚本演示 **07 Binary**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def binary_search(array, target):
    """Binary search on array for target

    Args:
        array: sorted array
        target: the element to search for
    Returns:
        index n on the array such that array[n]==target
        if the target not found, return -1
    """
    # 获取长度 / Get length
    s,e = 0, len(array)
    while s < e:
        m = (s+e)//2
        if array[m] == target:
            return m
        elif array[m] > target:
            e = m
        elif array[m] < target:
            s = m+1
        assert m != (s+e)//2, "we didn't move our midpoint"
    return -1
```

---
## Learning Notes / 学习笔记

- **概念**: Binary 是机器学习中的常用技术。  
  *Binary is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Binary / 07 Binary
# Complete Code / 完整代码
# ===============================

def binary_search(array, target):
    """Binary search on array for target

    Args:
        array: sorted array
        target: the element to search for
    Returns:
        index n on the array such that array[n]==target
        if the target not found, return -1
    """
    # 获取长度 / Get length
    s,e = 0, len(array)
    while s < e:
        m = (s+e)//2
        if array[m] == target:
            return m
        elif array[m] > target:
            e = m
        elif array[m] < target:
            s = m+1
        assert m != (s+e)//2, "we didn't move our midpoint"
    return -1
```

---

➡️ **Next / 下一步**: File 8 of 13

---

### Notimplemented

# 08 — Notimplemented / 08 Notimplemented

**Chapter 20 — File 8 of 13 / 第20章 — 第8个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Notimplemented**.

本脚本演示 **08 Notimplemented**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入数学函数库 / Import math functions library
import math

REGISTRY = {}

def register(name):
    def _decorator(fn):
        REGISTRY[name] = fn
        return fn
    return _decorator

@register("relu")
def rectified(x):
    return x if x > 0 else 0

@register("sigmoid")
def sigmoid(x):
    return 1/(1 + math.exp(-x))

def activate(x, funcname):
    if funcname not in REGISTRY:
        raise NotImplementedError(f"Function {funcname} is not implemented")
    else:
        func = REGISTRY[funcname]
        return func(x)

# 打印输出 / Print output
print(activate(1.23, "relu"))
# 打印输出 / Print output
print(activate(1.23, "sigmoid"))
# 打印输出 / Print output
print(activate(1.23, "tanh"))
```

---
## Learning Notes / 学习笔记

- **概念**: Notimplemented 是机器学习中的常用技术。  
  *Notimplemented is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Notimplemented / 08 Notimplemented
# Complete Code / 完整代码
# ===============================

# 导入数学函数库 / Import math functions library
import math

REGISTRY = {}

def register(name):
    def _decorator(fn):
        REGISTRY[name] = fn
        return fn
    return _decorator

@register("relu")
def rectified(x):
    return x if x > 0 else 0

@register("sigmoid")
def sigmoid(x):
    return 1/(1 + math.exp(-x))

def activate(x, funcname):
    if funcname not in REGISTRY:
        raise NotImplementedError(f"Function {funcname} is not implemented")
    else:
        func = REGISTRY[funcname]
        return func(x)

# 打印输出 / Print output
print(activate(1.23, "relu"))
# 打印输出 / Print output
print(activate(1.23, "sigmoid"))
# 打印输出 / Print output
print(activate(1.23, "tanh"))
```

---

➡️ **Next / 下一步**: File 9 of 13

---

### Nestedloop

# 12 — Nestedloop / 12 Nestedloop

**Chapter 20 — File 9 of 13 / 第20章 — 第9个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Nestedloop**.

本脚本演示 **12 Nestedloop**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def neg_in_upper_tri(matrix):
    # 获取长度 / Get length
    n_rows = len(matrix)
    # 获取长度 / Get length
    n_cols = len(matrix[0])
    # 生成整数序列 / Generate integer sequence
    for i in range(n_rows):
        # 生成整数序列 / Generate integer sequence
        for j in range(n_cols):
            if i > j:
                continue  # we are not in upper triangular
            if matrix[i][j] < 0:
                return True
    return False
```

---
## Learning Notes / 学习笔记

- **概念**: Nestedloop 是机器学习中的常用技术。  
  *Nestedloop is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nestedloop / 12 Nestedloop
# Complete Code / 完整代码
# ===============================

def neg_in_upper_tri(matrix):
    # 获取长度 / Get length
    n_rows = len(matrix)
    # 获取长度 / Get length
    n_cols = len(matrix[0])
    # 生成整数序列 / Generate integer sequence
    for i in range(n_rows):
        # 生成整数序列 / Generate integer sequence
        for j in range(n_cols):
            if i > j:
                continue  # we are not in upper triangular
            if matrix[i][j] < 0:
                return True
    return False
```

---

➡️ **Next / 下一步**: File 10 of 13

---

### Generator

# 13 — Generator / 13 Generator

**Chapter 20 — File 10 of 13 / 第20章 — 第10个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Generator**.

本脚本演示 **13 Generator**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def get_upper_tri(matrix):
    # 获取长度 / Get length
    n_rows = len(matrix)
    # 获取长度 / Get length
    n_cols = len(matrix[0])
    # 生成整数序列 / Generate integer sequence
    for i in range(n_rows):
        # 生成整数序列 / Generate integer sequence
        for j in range(n_cols):
            if i > j:
                continue  # we are not in upper triangular
            yield matrix[i][j]

def neg_in_upper_tri(matrix):
    for element in get_upper_tri(matrix):
        if element[i][j] < 0:
            return True
    return False
```

---
## Learning Notes / 学习笔记

- **概念**: Generator 是机器学习中的常用技术。  
  *Generator is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Generator / 13 Generator
# Complete Code / 完整代码
# ===============================

def get_upper_tri(matrix):
    # 获取长度 / Get length
    n_rows = len(matrix)
    # 获取长度 / Get length
    n_cols = len(matrix[0])
    # 生成整数序列 / Generate integer sequence
    for i in range(n_rows):
        # 生成整数序列 / Generate integer sequence
        for j in range(n_cols):
            if i > j:
                continue  # we are not in upper triangular
            yield matrix[i][j]

def neg_in_upper_tri(matrix):
    for element in get_upper_tri(matrix):
        if element[i][j] < 0:
            return True
    return False
```

---

➡️ **Next / 下一步**: File 11 of 13

---

### Checkfloat

# 14 — Checkfloat / 14 Checkfloat

**Chapter 20 — File 11 of 13 / 第20章 — 第11个文件（共13个）**

---

## Summary / 总结

This script demonstrates **Checkfloat**.

本脚本演示 **14 Checkfloat**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def isfloat(floatstring):
    if not isinstance(floatstring, str):
        raise ValueError("Expects a string input")
    seen_integer = False
    seen_dot = False
    seen_decimal = False
    for char in floatstring:
        if char.isdigit():
            if not seen_integer:
                seen_integer = True
            elif seen_dot and not seen_decimal:
                seen_decimal = True
        elif char == ".":
            if not seen_integer:
                return False  # e.g., ".3456"
            elif not seen_dot:
                seen_dot = True
            else:
                return False  # e.g., "1..23"
        else:
            return False  # e.g. "foo"
    if not seen_integer:
        return False   # e.g., ""
    if seen_dot and not seen_decimal:
        return False  # e.g., "2."
    return True


# 打印输出 / Print output
print(isfloat("foo"))       # False
# 打印输出 / Print output
print(isfloat(".3456"))     # False
# 打印输出 / Print output
print(isfloat("1.23"))      # True
# 打印输出 / Print output
print(isfloat("1..23"))     # False
# 打印输出 / Print output
print(isfloat("2"))         # True
# 打印输出 / Print output
print(isfloat("2."))        # False
# 打印输出 / Print output
print(isfloat("2,345.67"))  # False
```

---
## Learning Notes / 学习笔记

- **概念**: Checkfloat 是机器学习中的常用技术。  
  *Checkfloat is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Checkfloat / 14 Checkfloat
# Complete Code / 完整代码
# ===============================

def isfloat(floatstring):
    if not isinstance(floatstring, str):
        raise ValueError("Expects a string input")
    seen_integer = False
    seen_dot = False
    seen_decimal = False
    for char in floatstring:
        if char.isdigit():
            if not seen_integer:
                seen_integer = True
            elif seen_dot and not seen_decimal:
                seen_decimal = True
        elif char == ".":
            if not seen_integer:
                return False  # e.g., ".3456"
            elif not seen_dot:
                seen_dot = True
            else:
                return False  # e.g., "1..23"
        else:
            return False  # e.g. "foo"
    if not seen_integer:
        return False   # e.g., ""
    if seen_dot and not seen_decimal:
        return False  # e.g., "2."
    return True


# 打印输出 / Print output
print(isfloat("foo"))       # False
# 打印输出 / Print output
print(isfloat(".3456"))     # False
# 打印输出 / Print output
print(isfloat("1.23"))      # True
# 打印输出 / Print output
print(isfloat("1..23"))     # False
# 打印输出 / Print output
print(isfloat("2"))         # True
# 打印输出 / Print output
print(isfloat("2."))        # False
# 打印输出 / Print output
print(isfloat("2,345.67"))  # False
```

---

➡️ **Next / 下一步**: File 12 of 13

---

### Statemachine

# 15 — Statemachine / 15 Statemachine

**Chapter 20 — File 12 of 13 / 第20章 — 第12个文件（共13个）**

---

## Summary / 总结

This script demonstrates **States: "start", "integer", "dot", "decimal"**.

本脚本演示 **States: "start", "integer", "dot", "decimal"**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def isfloat(floatstring):
    if not isinstance(floatstring, str):
        raise ValueError("Expects a string input")
```

---
## Step 2 — States: "start", "integer", "dot", "decimal"

```python
state = "start"
    for char in floatstring:
        if state == "start":
            if char.isdigit():
                state = "integer"
            else:
                return False  # bad transition, can't continue
        elif state == "integer":
            if char.isdigit():
                pass  # stay in the same state
            elif char == ".":
                state = "dot"
            else:
                return False  # bad transition, can't continue
        elif state == "dot":
            if char.isdigit():
                state = "decimal"
            else:
                return False  # bad transition, can't continue
        elif state == "decimal":
            if not char.isdigit():
                return False  # bad transition, can't continue
    if state in ["integer", "decimal"]:
        return True
    else:
        return False

# 打印输出 / Print output
print(isfloat("foo"))       # False
# 打印输出 / Print output
print(isfloat(".3456"))     # False
# 打印输出 / Print output
print(isfloat("1.23"))      # True
# 打印输出 / Print output
print(isfloat("1..23"))     # False
# 打印输出 / Print output
print(isfloat("2"))         # True
# 打印输出 / Print output
print(isfloat("2."))        # False
# 打印输出 / Print output
print(isfloat("2,345.67"))  # False
```

---
## Learning Notes / 学习笔记

- **概念**: States: "start", "integer", "dot", "decimal" 是机器学习中的常用技术。  
  *States: "start", "integer", "dot", "decimal" is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Statemachine / 15 Statemachine
# Complete Code / 完整代码
# ===============================

def isfloat(floatstring):
    if not isinstance(floatstring, str):
        raise ValueError("Expects a string input")
    # States: "start", "integer", "dot", "decimal"
    state = "start"
    for char in floatstring:
        if state == "start":
            if char.isdigit():
                state = "integer"
            else:
                return False  # bad transition, can't continue
        elif state == "integer":
            if char.isdigit():
                pass  # stay in the same state
            elif char == ".":
                state = "dot"
            else:
                return False  # bad transition, can't continue
        elif state == "dot":
            if char.isdigit():
                state = "decimal"
            else:
                return False  # bad transition, can't continue
        elif state == "decimal":
            if not char.isdigit():
                return False  # bad transition, can't continue
    if state in ["integer", "decimal"]:
        return True
    else:
        return False

# 打印输出 / Print output
print(isfloat("foo"))       # False
# 打印输出 / Print output
print(isfloat(".3456"))     # False
# 打印输出 / Print output
print(isfloat("1.23"))      # True
# 打印输出 / Print output
print(isfloat("1..23"))     # False
# 打印输出 / Print output
print(isfloat("2"))         # True
# 打印输出 / Print output
print(isfloat("2."))        # False
# 打印输出 / Print output
print(isfloat("2,345.67"))  # False
```

---

➡️ **Next / 下一步**: File 13 of 13

---

### Regex



---

### Chapter Summary / 章节总结



---
