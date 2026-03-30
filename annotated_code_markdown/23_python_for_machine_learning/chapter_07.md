# Python ML
## Chapter 07

---

### Indentprint

# 01 — Indentprint / 01 Indentprint

**Chapter 07 — File 1 of 5 / 第07章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Indentprint**.

本脚本演示 **01 Indentprint**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def indentprint(x, indent=0, prefix="", suffix=""):
    if isinstance(x, dict):
        printdict(x, indent, prefix, suffix)
    elif isinstance(x, list):
        printlist(x, indent, prefix, suffix)
    elif isinstance(x, str):
        printstring(x, indent, prefix, suffix)
    else:
        printnumber(x, indent, prefix, suffix)

def printdict(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + "{")
    for n, key in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(x[key], indent+2, str(key)+": ", comma)
    print(spaces + "}" + suffix)

def printlist(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + "[")
    for n, item in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(item, indent+2, "", comma)
    print(spaces + "]" + suffix)

def printstring(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + '"' + str(x) + '"' + suffix)

def printnumber(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + str(x) + suffix)

data = {
    "a": [{
        "p": 3, "q": 4,
        "r": [3,4,5],
    },{
        "f": "foo", "g": 2.71
    },{
        "u": None, "v": "bar"
    }],
    "c": {
        "s": ["fizz", 2, 1.1],
        "t": []
    },
}

indentprint(data)
```

---
## Learning Notes / 学习笔记

- **概念**: Indentprint 是机器学习中的常用技术。  
  *Indentprint is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Indentprint / 01 Indentprint
# Complete Code / 完整代码
# ===============================

def indentprint(x, indent=0, prefix="", suffix=""):
    if isinstance(x, dict):
        printdict(x, indent, prefix, suffix)
    elif isinstance(x, list):
        printlist(x, indent, prefix, suffix)
    elif isinstance(x, str):
        printstring(x, indent, prefix, suffix)
    else:
        printnumber(x, indent, prefix, suffix)

def printdict(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + "{")
    for n, key in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(x[key], indent+2, str(key)+": ", comma)
    print(spaces + "}" + suffix)

def printlist(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + "[")
    for n, item in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(item, indent+2, "", comma)
    print(spaces + "]" + suffix)

def printstring(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + '"' + str(x) + '"' + suffix)

def printnumber(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + str(x) + suffix)

data = {
    "a": [{
        "p": 3, "q": 4,
        "r": [3,4,5],
    },{
        "f": "foo", "g": 2.71
    },{
        "u": None, "v": "bar"
    }],
    "c": {
        "s": ["fizz", 2, 1.1],
        "t": []
    },
}

indentprint(data)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Indentprint

# 02 — Indentprint / 02 Indentprint

**Chapter 07 — File 2 of 5 / 第07章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Indentprint**.

本脚本演示 **02 Indentprint**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def indentprint(x, indent=0, prefix="", suffix=""):
    print(f'indentprint(x, {indent}, "{prefix}", "{suffix}")')
    if isinstance(x, dict):
        printdict(x, indent, prefix, suffix)
    elif isinstance(x, list):
        printlist(x, indent, prefix, suffix)
    elif isinstance(x, str):
        printstring(x, indent, prefix, suffix)
    else:
        printnumber(x, indent, prefix, suffix)

def printdict(x, indent, prefix, suffix):
    print(f'printdict(x, {indent}, "{prefix}", "{suffix}")')
    spaces = " " * indent
    print(spaces + prefix + "{")
    for n, key in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(x[key], indent+2, str(key)+": ", comma)
    print(spaces + "}" + suffix)

def printlist(x, indent, prefix, suffix):
    print(f'printlist(x, {indent}, "{prefix}", "{suffix}")')
    spaces = " " * indent
    print(spaces + prefix + "[")
    for n, item in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(item, indent+2, "", comma)
    print(spaces + "]" + suffix)

def printstring(x, indent, prefix, suffix):
    print(f'printstring(x, {indent}, "{prefix}", "{suffix}")')
    spaces = " " * indent
    print(spaces + prefix + '"' + str(x) + '"' + suffix)

def printnumber(x, indent, prefix, suffix):
    print(f'printnumber(x, {indent}, "{prefix}", "{suffix}")')
    spaces = " " * indent
    print(spaces + prefix + str(x) + suffix)

data = {
    "a": [{
        "p": 3, "q": 4,
        "r": [3,4,5],
    },{
        "f": "foo", "g": 2.71
    },{
        "u": None, "v": "bar"
    }],
    "c": {
        "s": ["fizz", 2, 1.1],
        "t": []
    },
}

indentprint(data)
```

---
## Learning Notes / 学习笔记

- **概念**: Indentprint 是机器学习中的常用技术。  
  *Indentprint is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Indentprint / 02 Indentprint
# Complete Code / 完整代码
# ===============================

def indentprint(x, indent=0, prefix="", suffix=""):
    print(f'indentprint(x, {indent}, "{prefix}", "{suffix}")')
    if isinstance(x, dict):
        printdict(x, indent, prefix, suffix)
    elif isinstance(x, list):
        printlist(x, indent, prefix, suffix)
    elif isinstance(x, str):
        printstring(x, indent, prefix, suffix)
    else:
        printnumber(x, indent, prefix, suffix)

def printdict(x, indent, prefix, suffix):
    print(f'printdict(x, {indent}, "{prefix}", "{suffix}")')
    spaces = " " * indent
    print(spaces + prefix + "{")
    for n, key in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(x[key], indent+2, str(key)+": ", comma)
    print(spaces + "}" + suffix)

def printlist(x, indent, prefix, suffix):
    print(f'printlist(x, {indent}, "{prefix}", "{suffix}")')
    spaces = " " * indent
    print(spaces + prefix + "[")
    for n, item in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(item, indent+2, "", comma)
    print(spaces + "]" + suffix)

def printstring(x, indent, prefix, suffix):
    print(f'printstring(x, {indent}, "{prefix}", "{suffix}")')
    spaces = " " * indent
    print(spaces + prefix + '"' + str(x) + '"' + suffix)

def printnumber(x, indent, prefix, suffix):
    print(f'printnumber(x, {indent}, "{prefix}", "{suffix}")')
    spaces = " " * indent
    print(spaces + prefix + str(x) + suffix)

data = {
    "a": [{
        "p": 3, "q": 4,
        "r": [3,4,5],
    },{
        "f": "foo", "g": 2.71
    },{
        "u": None, "v": "bar"
    }],
    "c": {
        "s": ["fizz", 2, 1.1],
        "t": []
    },
}

indentprint(data)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Printstack

# 05 — Printstack / 堆叠方法

**Chapter 07 — File 3 of 5 / 第07章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Printstack**.

本脚本演示 **堆叠方法**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import traceback
import random

def compute():
    n = random.randint(0, 10)
    m = random.randint(0, 10)
    return n/m

def compute_many(n_times):
    try:
        for _ in range(n_times):
            x = compute()
        print(f"Completed {n_times} times")
    except:
        print("Something wrong")
        traceback.print_exc()

compute_many(100)
```

---
## Learning Notes / 学习笔记

- **概念**: Printstack 是机器学习中的常用技术。  
  *Printstack is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Printstack / 堆叠方法
# Complete Code / 完整代码
# ===============================

import traceback
import random

def compute():
    n = random.randint(0, 10)
    m = random.randint(0, 10)
    return n/m

def compute_many(n_times):
    try:
        for _ in range(n_times):
            x = compute()
        print(f"Completed {n_times} times")
    except:
        print("Something wrong")
        traceback.print_exc()

compute_many(100)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Printstack

# 06 — Printstack / 堆叠方法

**Chapter 07 — File 4 of 5 / 第07章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **print the function name and line number of the script**.

本脚本演示 **print the function name and line number of the script**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import traceback
import random

def print_tb_with_local():
    """Print stack trace with local variables. This does not need to be in
    exception. Print is using the system's print() function to stderr.
    """
    import traceback, sys
    tb = sys.exc_info()[2]
    stack = []
    while tb:
        stack.append(tb.tb_frame)
        tb = tb.tb_next
    traceback.print_exc()
    print("Locals by frame, most recent call first", file=sys.stderr)
    for frame in stack:
```

---
## Step 2 — print the function name and line number of the script

```python
print("Frame {0} in {1} at line {2}".format(
            frame.f_code.co_name,
            frame.f_code.co_filename,
            frame.f_lineno), file=sys.stderr)
```

---
## Step 3 — print each variable defined inside each function

```python
for key, value in frame.f_locals.items():
            print(f"\t{key} = ", file=sys.stderr)
            try:
                if '__repr__' in dir(value):
                    print(value.__repr__(), file=sys.stderr)
                elif '__str__' in dir(value):
                    print(value.__str__(), file=sys.stderr)
                else:
                    print(value, file=sys.stderr)
            except:
                print("", file=sys.stderr)

def compute():
    n = random.randint(0, 10)
    m = random.randint(0, 10)
    return n/m

def compute_many(n_times):
    try:
        for _ in range(n_times):
            x = compute()
        print(f"Completed {n_times} times")
    except:
        print("Something wrong")
        print_tb_with_local()

compute_many(100)
```

---
## Learning Notes / 学习笔记

- **概念**: print the function name and line number of the script 是机器学习中的常用技术。  
  *print the function name and line number of the script is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Printstack / 堆叠方法
# Complete Code / 完整代码
# ===============================

import traceback
import random

def print_tb_with_local():
    """Print stack trace with local variables. This does not need to be in
    exception. Print is using the system's print() function to stderr.
    """
    import traceback, sys
    tb = sys.exc_info()[2]
    stack = []
    while tb:
        stack.append(tb.tb_frame)
        tb = tb.tb_next
    traceback.print_exc()
    print("Locals by frame, most recent call first", file=sys.stderr)
    for frame in stack:
        # print the function name and line number of the script
        print("Frame {0} in {1} at line {2}".format(
            frame.f_code.co_name,
            frame.f_code.co_filename,
            frame.f_lineno), file=sys.stderr)
        # print each variable defined inside each function
        for key, value in frame.f_locals.items():
            print(f"\t{key} = ", file=sys.stderr)
            try:
                if '__repr__' in dir(value):
                    print(value.__repr__(), file=sys.stderr)
                elif '__str__' in dir(value):
                    print(value.__str__(), file=sys.stderr)
                else:
                    print(value, file=sys.stderr)
            except:
                print("", file=sys.stderr)

def compute():
    n = random.randint(0, 10)
    m = random.randint(0, 10)
    return n/m

def compute_many(n_times):
    try:
        for _ in range(n_times):
            x = compute()
        print(f"Completed {n_times} times")
    except:
        print("Something wrong")
        print_tb_with_local()

compute_many(100)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Training

# 07 — Training / 07 Training

**Chapter 07 — File 5 of 5 / 第07章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **define model**.

本脚本演示 **define model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model


---
## Step 1 — Step 1

```python
import numpy as np

sequence = np.arange(0.1, 1.0, 0.1)  # 0.1 to 0.9
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))
```

---
## Step 2 — define model

```python
import tensorflow as tf
from tensorflow.keras.layers import LSTM, RepeatVector, Dense, TimeDistributed, Input
from tensorflow.keras import Sequential, Model

model = Sequential([
    LSTM(100, activation="relu", input_shape=(n_in+1, 1)),
    RepeatVector(n_in),
    LSTM(100, activation="relu", return_sequences=True),
    TimeDistributed(Dense(1))
])
model.compile(optimizer="adam", loss="mse")

model.fit(sequence, sequence, epochs=300, verbose=0)
```

---
## Learning Notes / 学习笔记

- **概念**: define model 是机器学习中的常用技术。  
  *define model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Training / 07 Training
# Complete Code / 完整代码
# ===============================

import numpy as np

sequence = np.arange(0.1, 1.0, 0.1)  # 0.1 to 0.9
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))

# define model
import tensorflow as tf
from tensorflow.keras.layers import LSTM, RepeatVector, Dense, TimeDistributed, Input
from tensorflow.keras import Sequential, Model

model = Sequential([
    LSTM(100, activation="relu", input_shape=(n_in+1, 1)),
    RepeatVector(n_in),
    LSTM(100, activation="relu", return_sequences=True),
    TimeDistributed(Dense(1))
])
model.compile(optimizer="adam", loss="mse")

model.fit(sequence, sequence, epochs=300, verbose=0)
```

---

### Chapter Summary

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **5 code files** demonstrating chapter 07.

本章包含 **5 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `01_indentprint.ipynb` — Indentprint
  2. `02_indentprint.ipynb` — Indentprint
  3. `05_printstack.ipynb` — Printstack
  4. `06_printstack.ipynb` — Printstack
  5. `07_training.ipynb` — Training

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---
