# Python ML
## Chapter 12

---

### Badcomment

# 01 — Badcomment / 01 Badcomment

**Chapter 12 — File 1 of 10 / 第12章 — 第1个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Badcomment**.

本脚本演示 **01 Badcomment**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import datetime

timestamp = datetime.datetime.now()  # Get the current date and time
x = 0    # initialize x to zero
```

---
## Learning Notes / 学习笔记

- **概念**: Badcomment 是机器学习中的常用技术。  
  *Badcomment is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Badcomment / 01 Badcomment
# Complete Code / 完整代码
# ===============================

import datetime

timestamp = datetime.datetime.now()  # Get the current date and time
x = 0    # initialize x to zero
```

---

➡️ **Next / 下一步**: File 2 of 10

---

### Commenting

# 02 — Commenting / 02 Commenting

**Chapter 12 — File 2 of 10 / 第12章 — 第2个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Commenting**.

本脚本演示 **02 Commenting**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import scipy.stats

z_alpha = scipy.stats.norm.ppf(0.975)  # Call the inverse CDF of standard normal
```

---
## Learning Notes / 学习笔记

- **概念**: Commenting 是机器学习中的常用技术。  
  *Commenting is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Commenting / 02 Commenting
# Complete Code / 完整代码
# ===============================

import scipy.stats

z_alpha = scipy.stats.norm.ppf(0.975)  # Call the inverse CDF of standard normal
```

---

➡️ **Next / 下一步**: File 3 of 10

---

### Adadelta

# 03 — Adadelta / 03 Adadelta

**Chapter 12 — File 3 of 10 / 第12章 — 第3个文件（共10个）**

---

## Summary / 总结

This script demonstrates **generate an initial point**.

本脚本演示 **generate an initial point**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def adadelta(objective, derivative, bounds, n_iter, rho, ep=1e-3):
```

---
## Step 2 — generate an initial point

```python
solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
```

---
## Step 3 — lists to hold the average square gradients for each variable and
average parameter updates

```python
sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
    sq_para_avg = [0.0 for _ in range(bounds.shape[0])]
```

---
## Step 4 — run the gradient descent

```python
for it in range(n_iter):
        gradient = derivative(solution[0], solution[1])
```

---
## Step 5 — update the moving average of the squared partial derivatives

```python
for i in range(gradient.shape[0]):
            sg = gradient[i]**2.0
            sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (sg * (1.0-rho))
```

---
## Step 6 — build a solution one variable at a time

```python
new_solution = list()
        for i in range(solution.shape[0]):
```

---
## Step 7 — calculate the step size for this variable

```python
alpha = (ep + sqrt(sq_para_avg[i])) / (ep + sqrt(sq_grad_avg[i]))
```

---
## Step 8 — calculate the change and update the moving average of the squared change

```python
change = alpha * gradient[i]
            sq_para_avg[i] = (sq_para_avg[i] * rho) + (change**2.0 * (1.0-rho))
```

---
## Step 9 — calculate the new position in this variable and store as new solution

```python
value = solution[i] - change
            new_solution.append(value)
```

---
## Step 10 — evaluate candidate point

```python
solution = asarray(new_solution)
        solution_eval = objective(solution[0], solution[1])
```

---
## Step 11 — report progress

```python
print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
    return [solution, solution_eval]
```

---
## Learning Notes / 学习笔记

- **概念**: generate an initial point 是机器学习中的常用技术。  
  *generate an initial point is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Adadelta / 03 Adadelta
# Complete Code / 完整代码
# ===============================

def adadelta(objective, derivative, bounds, n_iter, rho, ep=1e-3):
    # generate an initial point
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # lists to hold the average square gradients for each variable and
    # average parameter updates
    sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
    sq_para_avg = [0.0 for _ in range(bounds.shape[0])]
    # run the gradient descent
    for it in range(n_iter):
        gradient = derivative(solution[0], solution[1])
        # update the moving average of the squared partial derivatives
        for i in range(gradient.shape[0]):
            sg = gradient[i]**2.0
            sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (sg * (1.0-rho))
        # build a solution one variable at a time
        new_solution = list()
        for i in range(solution.shape[0]):
            # calculate the step size for this variable
            alpha = (ep + sqrt(sq_para_avg[i])) / (ep + sqrt(sq_grad_avg[i]))
            # calculate the change and update the moving average of the squared change
            change = alpha * gradient[i]
            sq_para_avg[i] = (sq_para_avg[i] * rho) + (change**2.0 * (1.0-rho))
            # calculate the new position in this variable and store as new solution
            value = solution[i] - change
            new_solution.append(value)
        # evaluate candidate point
        solution = asarray(new_solution)
        solution_eval = objective(solution[0], solution[1])
        # report progress
        print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
    return [solution, solution_eval]
```

---

➡️ **Next / 下一步**: File 4 of 10

---

### Docstring

# 09 — Docstring / 09 Docstring

**Chapter 12 — File 5 of 10 / 第12章 — 第5个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Docstring**.

本脚本演示 **09 Docstring**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def square(x):
    """Just to compute the square of a value

    Args:
        x (int or float): A numerical value

    Returns:
        int or float: The square of x
    """
    return x * x

print("Function name:", square.__name__)
print("Docstring:", square.__doc__)
```

---
## Learning Notes / 学习笔记

- **概念**: Docstring 是机器学习中的常用技术。  
  *Docstring is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Docstring / 09 Docstring
# Complete Code / 完整代码
# ===============================

def square(x):
    """Just to compute the square of a value

    Args:
        x (int or float): A numerical value

    Returns:
        int or float: The square of x
    """
    return x * x

print("Function name:", square.__name__)
print("Docstring:", square.__doc__)
```

---

➡️ **Next / 下一步**: File 6 of 10

---

### Typehint

# 11 — Typehint / 11 Typehint

**Chapter 12 — File 6 of 10 / 第12章 — 第6个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Typehint**.

本脚本演示 **11 Typehint**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def square(x: int) -> int:
    return x * x
```

---
## Learning Notes / 学习笔记

- **概念**: Typehint 是机器学习中的常用技术。  
  *Typehint is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Typehint / 11 Typehint
# Complete Code / 完整代码
# ===============================

def square(x: int) -> int:
    return x * x
```

---

➡️ **Next / 下一步**: File 7 of 10

---

### Typehint

# 12 — Typehint / 12 Typehint

**Chapter 12 — File 7 of 10 / 第12章 — 第7个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Typehint**.

本脚本演示 **12 Typehint**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
def square(x: int) -> int:
    value: int = x * x
    return value
```

---
## Learning Notes / 学习笔记

- **概念**: Typehint 是机器学习中的常用技术。  
  *Typehint is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Typehint / 12 Typehint
# Complete Code / 完整代码
# ===============================

def square(x: int) -> int:
    value: int = x * x
    return value
```

---

➡️ **Next / 下一步**: File 8 of 10

---

### Typehint

# 13 — Typehint / 13 Typehint

**Chapter 12 — File 8 of 10 / 第12章 — 第8个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Typehint**.

本脚本演示 **13 Typehint**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from typing import Any, Union, List

def square(x: Union[int, float]) -> Union[int, float]:
    return x * x

def append(x: List[Any], y: Any) -> None:
    x.append(y)
```

---
## Learning Notes / 学习笔记

- **概念**: Typehint 是机器学习中的常用技术。  
  *Typehint is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Typehint / 13 Typehint
# Complete Code / 完整代码
# ===============================

from typing import Any, Union, List

def square(x: Union[int, float]) -> Union[int, float]:
    return x * x

def append(x: List[Any], y: Any) -> None:
    x.append(y)
```

---

➡️ **Next / 下一步**: File 9 of 10

---

### Typehint

# 14 — Typehint / 14 Typehint

**Chapter 12 — File 9 of 10 / 第12章 — 第9个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Typehint**.

本脚本演示 **14 Typehint**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
n: int = 3.5
n = "assign a string"
```

---
## Learning Notes / 学习笔记

- **概念**: Typehint 是机器学习中的常用技术。  
  *Typehint is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Typehint / 14 Typehint
# Complete Code / 完整代码
# ===============================

n: int = 3.5
n = "assign a string"
```

---

➡️ **Next / 下一步**: File 10 of 10

---

### Typehintexample

# 15 — Typehintexample / 15 Typehintexample

**Chapter 12 — File 10 of 10 / 第12章 — 第10个文件（共10个）**

---

## Summary / 总结

This script demonstrates **pick one start time and security**.

本脚本演示 **pick one start time and security**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
from typing import List, Tuple, Generator
import pandas as pd
import numpy as np

TrainingSampleGenerator = Generator[Tuple[np.ndarray,np.ndarray], None, None]

def lstm_gen(data: pd.DataFrame,
             timesteps: int,
             batch_size: int) -> TrainingSampleGenerator:
    """Generator to produce random samples for LSTM training

    Args:
        data: DataFrame of data with datetime index in chronological order,
              samples are drawn from this
        timesteps: Number of time steps for each sample, data will be
                   produced from a window of such length
        batch_size: Number of samples in each batch

    Yields:
        ndarray, ndarray: The (X,Y) training samples drawn on a random window
        from the input data
    """
    input_columns = [c for c in data.columns if c != "target"]
    batch: List[Tuple[pd.DataFrame, pd.Series]] = []
    while True:
```

---
## Step 2 — pick one start time and security

```python
while True:
```

---
## Step 3 — Start from a random point from the data and clip a window

```python
row = data["target"].sample()
            starttime = row.index[0]
            window: pd.DataFrame = data[starttime:].iloc[:timesteps]
```

---
## Step 4 — If we are at the end of the DataFrame, we can't get a full
window and we must start over

```python
if len(window) == timesteps:
                break
```

---
## Step 5 — Extract the input and output

```python
y = window["target"]
        X = window[input_columns]
        batch.append((X, y))
```

---
## Step 6 — If accumulated enough for one batch, dispatch

```python
if len(batch) == batch_size:
            X, y = zip(*batch)
            yield np.array(X).astype("float32"), np.array(y).astype("float32")
            batch = []
```

---
## Learning Notes / 学习笔记

- **概念**: pick one start time and security 是机器学习中的常用技术。  
  *pick one start time and security is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Typehintexample / 15 Typehintexample
# Complete Code / 完整代码
# ===============================

from typing import List, Tuple, Generator
import pandas as pd
import numpy as np

TrainingSampleGenerator = Generator[Tuple[np.ndarray,np.ndarray], None, None]

def lstm_gen(data: pd.DataFrame,
             timesteps: int,
             batch_size: int) -> TrainingSampleGenerator:
    """Generator to produce random samples for LSTM training

    Args:
        data: DataFrame of data with datetime index in chronological order,
              samples are drawn from this
        timesteps: Number of time steps for each sample, data will be
                   produced from a window of such length
        batch_size: Number of samples in each batch

    Yields:
        ndarray, ndarray: The (X,Y) training samples drawn on a random window
        from the input data
    """
    input_columns = [c for c in data.columns if c != "target"]
    batch: List[Tuple[pd.DataFrame, pd.Series]] = []
    while True:
        # pick one start time and security
        while True:
            # Start from a random point from the data and clip a window
            row = data["target"].sample()
            starttime = row.index[0]
            window: pd.DataFrame = data[starttime:].iloc[:timesteps]
            # If we are at the end of the DataFrame, we can't get a full
            # window and we must start over
            if len(window) == timesteps:
                break
        # Extract the input and output
        y = window["target"]
        X = window[input_columns]
        batch.append((X, y))
        # If accumulated enough for one batch, dispatch
        if len(batch) == batch_size:
            X, y = zip(*batch)
            yield np.array(X).astype("float32"), np.array(y).astype("float32")
            batch = []
```

---
