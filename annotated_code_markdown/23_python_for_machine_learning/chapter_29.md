# Python 机器学习 / Python for Machine Learning
## Chapter 29

---

### Multiproc

# 04 — Multiproc / 04 Multiproc

**Chapter 29 — File 1 of 7 / 第29章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Creates two processes**.

本脚本演示 **Creates two processes**。

---
## Step 1 — Step 1

```python
import multiprocessing
import time

def task():
    print('Sleeping for 0.5 seconds')
    time.sleep(0.5)
    print('Finished sleeping')

if __name__ == "__main__":
    start_time = time.perf_counter()
```

---
## Step 2 — Creates two processes

```python
p1 = multiprocessing.Process(target=task)
    p2 = multiprocessing.Process(target=task)
```

---
## Step 3 — Starts both processes

```python
p1.start()
    p2.start()

    finish_time = time.perf_counter()

    print(f"Program finished in {finish_time-start_time} seconds")
```

---
## Learning Notes / 学习笔记

- **概念**: Creates two processes 是机器学习中的常用技术。  
  *Creates two processes is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Multiproc / 04 Multiproc
# Complete Code / 完整代码
# ===============================

import multiprocessing
import time

def task():
    print('Sleeping for 0.5 seconds')
    time.sleep(0.5)
    print('Finished sleeping')

if __name__ == "__main__":
    start_time = time.perf_counter()

    # Creates two processes
    p1 = multiprocessing.Process(target=task)
    p2 = multiprocessing.Process(target=task)

    # Starts both processes
    p1.start()
    p2.start()

    finish_time = time.perf_counter()

    print(f"Program finished in {finish_time-start_time} seconds")
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### 10Processes

# 06 — 10Processes / 06 10Processes

**Chapter 29 — File 2 of 7 / 第29章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Creates 10 processes then starts them**.

本脚本演示 **Creates 10 processes then starts them**。

---
## Step 1 — Step 1

```python
import multiprocessing
import time

def task():
    print('Sleeping for 0.5 seconds')
    time.sleep(0.5)
    print('Finished sleeping')

if __name__ == "__main__":
    start_time = time.perf_counter()
    processes = []
```

---
## Step 2 — Creates 10 processes then starts them

```python
for i in range(10):
        p = multiprocessing.Process(target = task)
        p.start()
        processes.append(p)
```

---
## Step 3 — Joins all the processes

```python
for p in processes:
        p.join()

    finish_time = time.perf_counter()

    print(f"Program finished in {finish_time-start_time} seconds")
```

---
## Learning Notes / 学习笔记

- **概念**: Creates 10 processes then starts them 是机器学习中的常用技术。  
  *Creates 10 processes then starts them is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# 10Processes / 06 10Processes
# Complete Code / 完整代码
# ===============================

import multiprocessing
import time

def task():
    print('Sleeping for 0.5 seconds')
    time.sleep(0.5)
    print('Finished sleeping')

if __name__ == "__main__":
    start_time = time.perf_counter()
    processes = []

    # Creates 10 processes then starts them
    for i in range(10):
        p = multiprocessing.Process(target = task)
        p.start()
        processes.append(p)

    # Joins all the processes
    for p in processes:
        p.join()

    finish_time = time.perf_counter()

    print(f"Program finished in {finish_time-start_time} seconds")
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Parallelcube

# 08 — Parallelcube / 08 Parallelcube

**Chapter 29 — File 3 of 7 / 第29章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **this does not work**.

本脚本演示 **this does not work**。

---
## Step 1 — Step 1

```python
import multiprocessing

def cube(x):
    return x**3

if __name__ == "__main__":
```

---
## Step 2 — this does not work

```python
processes = [multiprocessing.Process(target=cube, args=(x,)) for x in range(1,1000)]
    [p.start() for p in processes]
    result = [p.join() for p in processes]
    print(result)
```

---
## Learning Notes / 学习笔记

- **概念**: this does not work 是机器学习中的常用技术。  
  *this does not work is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Parallelcube / 08 Parallelcube
# Complete Code / 完整代码
# ===============================

import multiprocessing

def cube(x):
    return x**3

if __name__ == "__main__":
    # this does not work
    processes = [multiprocessing.Process(target=cube, args=(x,)) for x in range(1,1000)]
    [p.start() for p in processes]
    result = [p.join() for p in processes]
    print(result)
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Pool

# 09 — Pool / 09 Pool

**Chapter 29 — File 4 of 7 / 第29章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Pool**.

本脚本演示 **09 Pool**。

---
## Step 1 — Step 1

```python
import multiprocessing
import time

def cube(x):
    return x**3

if __name__ == "__main__":
    pool = multiprocessing.Pool(3)
    start_time = time.perf_counter()
    processes = [pool.apply_async(cube, args=(x,)) for x in range(1,1000)]
    result = [p.get() for p in processes]
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    print(result)
```

---
## Learning Notes / 学习笔记

- **概念**: Pool 是机器学习中的常用技术。  
  *Pool is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pool / 09 Pool
# Complete Code / 完整代码
# ===============================

import multiprocessing
import time

def cube(x):
    return x**3

if __name__ == "__main__":
    pool = multiprocessing.Pool(3)
    start_time = time.perf_counter()
    processes = [pool.apply_async(cube, args=(x,)) for x in range(1,1000)]
    result = [p.get() for p in processes]
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    print(result)
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Map

# 10 — Map / 10 Map

**Chapter 29 — File 5 of 7 / 第29章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Map**.

本脚本演示 **10 Map**。

---
## Step 1 — Step 1

```python
import multiprocessing
import time

def cube(x):
    return x**3

if __name__ == "__main__":
    pool = multiprocessing.Pool(3)
    start_time = time.perf_counter()
    result = pool.map(cube, range(1,1000))
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    print(result)
```

---
## Learning Notes / 学习笔记

- **概念**: Map 是机器学习中的常用技术。  
  *Map is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Map / 10 Map
# Complete Code / 完整代码
# ===============================

import multiprocessing
import time

def cube(x):
    return x**3

if __name__ == "__main__":
    pool = multiprocessing.Pool(3)
    start_time = time.perf_counter()
    result = pool.map(cube, range(1,1000))
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    print(result)
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Futures

# 12 — Futures / 12 Futures

**Chapter 29 — File 6 of 7 / 第29章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Futures**.

本脚本演示 **12 Futures**。

---
## Step 1 — Step 1

```python
import concurrent.futures
import time

def cube(x):
    return x**3

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(3) as executor:
        start_time = time.perf_counter()
        result = list(executor.map(cube, range(1,1000)))
        finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    print(result)
```

---
## Learning Notes / 学习笔记

- **概念**: Futures 是机器学习中的常用技术。  
  *Futures is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Futures / 12 Futures
# Complete Code / 完整代码
# ===============================

import concurrent.futures
import time

def cube(x):
    return x**3

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(3) as executor:
        start_time = time.perf_counter()
        result = list(executor.map(cube, range(1,1000)))
        finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    print(result)
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Joblib

# 13 — Joblib / 13 Joblib

**Chapter 29 — File 7 of 7 / 第29章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Joblib**.

本脚本演示 **13 Joblib**。

---
## Step 1 — Step 1

```python
import time
from joblib import Parallel, delayed

def cube(x):
    return x**3

start_time = time.perf_counter()
result = Parallel(n_jobs=3)(delayed(cube)(i) for i in range(1,1000))
finish_time = time.perf_counter()
print(f"Program finished in {finish_time-start_time} seconds")
print(result)
```

---
## Learning Notes / 学习笔记

- **概念**: Joblib 是机器学习中的常用技术。  
  *Joblib is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Joblib / 13 Joblib
# Complete Code / 完整代码
# ===============================

import time
from joblib import Parallel, delayed

def cube(x):
    return x**3

start_time = time.perf_counter()
result = Parallel(n_jobs=3)(delayed(cube)(i) for i in range(1,1000))
finish_time = time.perf_counter()
print(f"Program finished in {finish_time-start_time} seconds")
print(result)
```

---

### Chapter Summary / 章节总结

# Chapter 29 Summary / 第29章总结

## Theme / 主题: Chapter 29 / Chapter 29

This chapter contains **7 code files** demonstrating chapter 29.

本章包含 **7 个代码文件**，演示Chapter 29。

---
## Evolution / 演化路线

  1. `04_multiproc.ipynb` — Multiproc
  2. `06_10processes.ipynb` — 10Processes
  3. `08_parallelcube.ipynb` — Parallelcube
  4. `09_pool.ipynb` — Pool
  5. `10_map.ipynb` — Map
  6. `12_futures.ipynb` — Futures
  7. `13_joblib.ipynb` — Joblib

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 29) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 29）是机器学习流水线中的基础构建块。

---
