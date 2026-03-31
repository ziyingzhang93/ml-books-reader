# Python 机器学习 / Python for Machine Learning
## Chapter 29

---

### Multiproc



---

### 10Processes

# 06 — 10Processes / 06 10Processes

**Chapter 29 — File 2 of 7 / 第29章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Creates 10 processes then starts them**.

本脚本演示 **Creates 10 processes then starts them**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import multiprocessing
# 导入时间处理模块 / Import time module
import time

def task():
    # 打印输出 / Print output
    print('Sleeping for 0.5 seconds')
    time.sleep(0.5)
    # 打印输出 / Print output
    print('Finished sleeping')

if __name__ == "__main__":
    start_time = time.perf_counter()
    processes = []
```

---
## Step 2 — Creates 10 processes then starts them

```python
# 生成整数序列 / Generate integer sequence
for i in range(10):
        p = multiprocessing.Process(target = task)
        p.start()
        # 添加元素到列表末尾 / Append element to list end
        processes.append(p)
```

---
## Step 3 — Joins all the processes

```python
for p in processes:
        p.join()

    finish_time = time.perf_counter()

    # 打印输出 / Print output
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
# 导入时间处理模块 / Import time module
import time

def task():
    # 打印输出 / Print output
    print('Sleeping for 0.5 seconds')
    time.sleep(0.5)
    # 打印输出 / Print output
    print('Finished sleeping')

if __name__ == "__main__":
    start_time = time.perf_counter()
    processes = []

    # Creates 10 processes then starts them
    # 生成整数序列 / Generate integer sequence
    for i in range(10):
        p = multiprocessing.Process(target = task)
        p.start()
        # 添加元素到列表末尾 / Append element to list end
        processes.append(p)

    # Joins all the processes
    for p in processes:
        p.join()

    finish_time = time.perf_counter()

    # 打印输出 / Print output
    print(f"Program finished in {finish_time-start_time} seconds")
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Parallelcube



---

### Pool



---

### Map



---

### Futures



---

### Joblib



---

### Chapter Summary / 章节总结



---
