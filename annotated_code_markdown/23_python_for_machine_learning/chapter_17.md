# Python ML
## Chapter 17

---

### Decorator

# 05 — Decorator / 05 Decorator

**Chapter 17 — File 5 of 9 / 第17章 — 第5个文件（共9个）**

---

## Summary / 总结

This script demonstrates **repeat_decorator should return a function that's a decorator**.

本脚本演示 **repeat_decorator should return a function that's a decorator**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def repeat_decorator(num_repeats=2):
```

---
## Step 2 — repeat_decorator should return a function that's a decorator

```python
def inner_decorator(fn):
        def decorated_fn():
            for i in range(num_repeats):
                fn()
```

---
## Step 3 — return the new function

```python
return decorated_fn
```

---
## Step 4 — return the decorator that actually takes the function in as the input

```python
return inner_decorator
```

---
## Step 5 — use the decorator with num_repeats argument set as 5 to repeat the function call 5 times

```python
@repeat_decorator(5)
def hello_world():
    print("Hello world!")
```

---
## Step 6 — call the function

```python
hello_world()
```

---
## Learning Notes / 学习笔记

- **概念**: repeat_decorator should return a function that's a decorator 是机器学习中的常用技术。  
  *repeat_decorator should return a function that's a decorator is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Decorator / 05 Decorator
# Complete Code / 完整代码
# ===============================

def repeat_decorator(num_repeats=2):
    # repeat_decorator should return a function that's a decorator
    def inner_decorator(fn):
        def decorated_fn():
            for i in range(num_repeats):
                fn()
        # return the new function
        return decorated_fn
    # return the decorator that actually takes the function in as the input
    return inner_decorator

# use the decorator with num_repeats argument set as 5 to repeat the function call 5 times
@repeat_decorator(5)
def hello_world():
    print("Hello world!")

# call the function
hello_world()
```

---

➡️ **Next / 下一步**: File 6 of 9

---

### Modifies

# 08 — Modifies / 08 Modifies

**Chapter 17 — File 6 of 9 / 第17章 — 第6个文件（共9个）**

---

## Summary / 总结

This script demonstrates **function decorator to ensure numpy input**.

本脚本演示 **function decorator to ensure numpy input**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
import numpy as np
import pandas as pd
```

---
## Step 2 — function decorator to ensure numpy input
and round off output to 4 decimal places

```python
def ensure_numpy(fn):
    def decorated_function(data):
        array = np.asarray(data)
        output = fn(array)
        return np.around(output, 4)
    return decorated_function

@ensure_numpy
def numpysum(array):
    return array.sum()

x = np.random.randn(10,3)
y = pd.DataFrame(x, columns=["A", "B", "C"])
```

---
## Step 3 — output of numpy .sum() function

```python
print("x.sum():", x.sum())
print()
```

---
## Step 4 — output of pandas .sum() funuction

```python
print("y.sum():", y.sum())
print(y.sum())
print()
```

---
## Step 5 — calling decorated numpysum function

```python
print("numpysum(x):", numpysum(x))
print("numpysum(y):", numpysum(y))
```

---
## Learning Notes / 学习笔记

- **概念**: function decorator to ensure numpy input 是机器学习中的常用技术。  
  *function decorator to ensure numpy input is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Modifies / 08 Modifies
# Complete Code / 完整代码
# ===============================

import numpy as np
import pandas as pd

# function decorator to ensure numpy input
# and round off output to 4 decimal places
def ensure_numpy(fn):
    def decorated_function(data):
        array = np.asarray(data)
        output = fn(array)
        return np.around(output, 4)
    return decorated_function

@ensure_numpy
def numpysum(array):
    return array.sum()

x = np.random.randn(10,3)
y = pd.DataFrame(x, columns=["A", "B", "C"])

# output of numpy .sum() function
print("x.sum():", x.sum())
print()

# output of pandas .sum() funuction
print("y.sum():", y.sum())
print(y.sum())
print()

# calling decorated numpysum function
print("numpysum(x):", numpysum(x))
print("numpysum(y):", numpysum(y))
```

---

➡️ **Next / 下一步**: File 7 of 9

---

### Memoize

# 09 — Memoize / 09 Memoize

**Chapter 17 — File 7 of 9 / 第17章 — 第7个文件（共9个）**

---

## Summary / 总结

This script demonstrates **pickle the function arguments and obtain hash as the store keys**.

本脚本演示 **pickle the function arguments and obtain hash as the store keys**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import pickle
import hashlib


MEMO = {} # To remember the function input and output

def memoize(fn):
    def _deco(*args, **kwargs):
```

---
## Step 2 — pickle the function arguments and obtain hash as the store keys

```python
key = (fn.__name__, hashlib.md5(pickle.dumps((args, kwargs), 4)).hexdigest())
```

---
## Step 3 — check if the key exists

```python
if key in MEMO:
            ret = pickle.loads(MEMO[key])
        else:
            ret = fn(*args, **kwargs)
            MEMO[key] = pickle.dumps(ret)
        return ret
    return _deco

@memoize
def fibonacci(n):
    if n in [0, 1]:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(40))
print(MEMO)
```

---
## Learning Notes / 学习笔记

- **概念**: pickle the function arguments and obtain hash as the store keys 是机器学习中的常用技术。  
  *pickle the function arguments and obtain hash as the store keys is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Memoize / 09 Memoize
# Complete Code / 完整代码
# ===============================

import pickle
import hashlib


MEMO = {} # To remember the function input and output

def memoize(fn):
    def _deco(*args, **kwargs):
        # pickle the function arguments and obtain hash as the store keys
        key = (fn.__name__, hashlib.md5(pickle.dumps((args, kwargs), 4)).hexdigest())
        # check if the key exists
        if key in MEMO:
            ret = pickle.loads(MEMO[key])
        else:
            ret = fn(*args, **kwargs)
            MEMO[key] = pickle.dumps(ret)
        return ret
    return _deco

@memoize
def fibonacci(n):
    if n in [0, 1]:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(40))
print(MEMO)
```

---

➡️ **Next / 下一步**: File 8 of 9

---
