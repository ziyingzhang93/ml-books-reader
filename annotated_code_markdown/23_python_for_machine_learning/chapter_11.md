# Python 机器学习 / Python for Machine Learning
## Chapter 11

---

### Logging

# 01 — Logging / 01 Logging

**Chapter 11 — File 1 of 11 / 第11章 — 第1个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Logging**.

本脚本演示 **01 Logging**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import logging

logging.debug('Debug message')
# 显示数据类型和缺失值信息 / Show data types and missing value info
logging.info('Info message')
logging.warning('Warning message')
logging.error('Error message')
logging.critical('Critical message')
```

---
## Learning Notes / 学习笔记

- **概念**: Logging 是机器学习中的常用技术。  
  *Logging is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Logging / 01 Logging
# Complete Code / 完整代码
# ===============================

import logging

logging.debug('Debug message')
# 显示数据类型和缺失值信息 / Show data types and missing value info
logging.info('Info message')
logging.warning('Warning message')
logging.error('Error message')
logging.critical('Critical message')
```

---

➡️ **Next / 下一步**: File 2 of 11

---

### Logging

# 02 — Logging / 02 Logging

**Chapter 11 — File 2 of 11 / 第11章 — 第2个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Logging**.

本脚本演示 **02 Logging**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import logging

logging.basicConfig(filename = 'file.log',
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logging.debug('Debug message')
# 显示数据类型和缺失值信息 / Show data types and missing value info
logging.info('Info message')
logging.warning('Warning message')
logging.error('Error message')
logging.critical('Critical message')
```

---
## Learning Notes / 学习笔记

- **概念**: Logging 是机器学习中的常用技术。  
  *Logging is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Logging / 02 Logging
# Complete Code / 完整代码
# ===============================

import logging

logging.basicConfig(filename = 'file.log',
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logging.debug('Debug message')
# 显示数据类型和缺失值信息 / Show data types and missing value info
logging.info('Info message')
logging.warning('Warning message')
logging.error('Error message')
logging.critical('Critical message')
```

---

➡️ **Next / 下一步**: File 3 of 11

---

### Logging

# 03 — Logging / 03 Logging

**Chapter 11 — File 3 of 11 / 第11章 — 第3个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Create `parent.child` logger**.

本脚本演示 **Create `parent.child` logger**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import logging
```

---
## Step 2 — Create `parent.child` logger

```python
logger = logging.getLogger("parent.child")
```

---
## Step 3 — Emit a log message of level INFO, by default this is not print to the screen

```python
# 显示数据类型和缺失值信息 / Show data types and missing value info
logger.info("this is info level")
```

---
## Step 4 — Create `parent` logger

```python
parentlogger = logging.getLogger("parent")
```

---
## Step 5 — Set parent's level to INFO and assign a new handler

```python
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
parentlogger.setLevel(logging.INFO)
parentlogger.addHandler(handler)
```

---
## Step 6 — Let child logger emit a log message again

```python
# 显示数据类型和缺失值信息 / Show data types and missing value info
logger.info("this is info level again")
```

---
## Learning Notes / 学习笔记

- **概念**: Create `parent.child` logger 是机器学习中的常用技术。  
  *Create `parent.child` logger is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Logging / 03 Logging
# Complete Code / 完整代码
# ===============================

import logging

# Create `parent.child` logger
logger = logging.getLogger("parent.child")

# Emit a log message of level INFO, by default this is not print to the screen
# 显示数据类型和缺失值信息 / Show data types and missing value info
logger.info("this is info level")

# Create `parent` logger
parentlogger = logging.getLogger("parent")

# Set parent's level to INFO and assign a new handler
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
parentlogger.setLevel(logging.INFO)
parentlogger.addHandler(handler)

# Let child logger emit a log message again
# 显示数据类型和缺失值信息 / Show data types and missing value info
logger.info("this is info level again")
```

---

➡️ **Next / 下一步**: File 4 of 11

---

### Streamhandler

# 05 — Streamhandler / 05 Streamhandler

**Chapter 11 — File 4 of 11 / 第11章 — 第4个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Set up root logger, and add a file handler to root logger**.

本脚本演示 **Set up root logger, and add a file handler to root logger**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import logging
```

---
## Step 2 — Set up root logger, and add a file handler to root logger

```python
logging.basicConfig(filename = 'file.log',
                    level = logging.WARNING,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
```

---
## Step 3 — Create logger, set level, and add stream handler

```python
parent_logger = logging.getLogger("parent")
parent_logger.setLevel(logging.INFO)
parent_shandler = logging.StreamHandler()
parent_logger.addHandler(parent_shandler)
```

---
## Step 4 — Log message of severity INFO or above will be handled

```python
parent_logger.debug('Debug message')
# 显示数据类型和缺失值信息 / Show data types and missing value info
parent_logger.info('Info message')
parent_logger.warning('Warning message')
parent_logger.error('Error message')
parent_logger.critical('Critical message')
```

---
## Learning Notes / 学习笔记

- **概念**: Set up root logger, and add a file handler to root logger 是机器学习中的常用技术。  
  *Set up root logger, and add a file handler to root logger is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Streamhandler / 05 Streamhandler
# Complete Code / 完整代码
# ===============================

import logging

# Set up root logger, and add a file handler to root logger
logging.basicConfig(filename = 'file.log',
                    level = logging.WARNING,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

# Create logger, set level, and add stream handler
parent_logger = logging.getLogger("parent")
parent_logger.setLevel(logging.INFO)
parent_shandler = logging.StreamHandler()
parent_logger.addHandler(parent_shandler)

# Log message of severity INFO or above will be handled
parent_logger.debug('Debug message')
# 显示数据类型和缺失值信息 / Show data types and missing value info
parent_logger.info('Info message')
parent_logger.warning('Warning message')
parent_logger.error('Error message')
parent_logger.critical('Critical message')
```

---

➡️ **Next / 下一步**: File 5 of 11

---

### Filehandler

# 06 — Filehandler / 06 Filehandler

**Chapter 11 — File 5 of 11 / 第11章 — 第5个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Set up root logger, and add a file handler to root logger**.

本脚本演示 **Set up root logger, and add a file handler to root logger**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import logging
```

---
## Step 2 — Set up root logger, and add a file handler to root logger

```python
logging.basicConfig(filename = 'file.log',
                    level = logging.WARNING,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
```

---
## Step 3 — Create logger, set level, and add stream handler

```python
parent_logger = logging.getLogger("parent")
parent_logger.setLevel(logging.INFO)
parent_fhandler = logging.FileHandler('parent.log')
parent_fhandler.setLevel(logging.WARNING)
parent_logger.addHandler(parent_fhandler)
```

---
## Step 4 — Log message of severity INFO or above will be handled

```python
parent_logger.debug('Debug message')
# 显示数据类型和缺失值信息 / Show data types and missing value info
parent_logger.info('Info message')
parent_logger.warning('Warning message')
parent_logger.error('Error message')
parent_logger.critical('Critical message')
```

---
## Learning Notes / 学习笔记

- **概念**: Set up root logger, and add a file handler to root logger 是机器学习中的常用技术。  
  *Set up root logger, and add a file handler to root logger is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Filehandler / 06 Filehandler
# Complete Code / 完整代码
# ===============================

import logging

# Set up root logger, and add a file handler to root logger
logging.basicConfig(filename = 'file.log',
                    level = logging.WARNING,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

# Create logger, set level, and add stream handler
parent_logger = logging.getLogger("parent")
parent_logger.setLevel(logging.INFO)
parent_fhandler = logging.FileHandler('parent.log')
parent_fhandler.setLevel(logging.WARNING)
parent_logger.addHandler(parent_fhandler)

# Log message of severity INFO or above will be handled
parent_logger.debug('Debug message')
# 显示数据类型和缺失值信息 / Show data types and missing value info
parent_logger.info('Info message')
parent_logger.warning('Warning message')
parent_logger.error('Error message')
parent_logger.critical('Critical message')
```

---

➡️ **Next / 下一步**: File 6 of 11

---

### Formatter

# 07 — Formatter / 07 Formatter

**Chapter 11 — File 6 of 11 / 第11章 — 第6个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Set up root logger, and add a file handler to root logger**.

本脚本演示 **Set up root logger, and add a file handler to root logger**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import logging
```

---
## Step 2 — Set up root logger, and add a file handler to root logger

```python
logging.basicConfig(filename = 'file.log',
                    level = logging.WARNING,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
```

---
## Step 3 — Create logger, set level, and add stream handler

```python
parent_logger = logging.getLogger("parent")
parent_logger.setLevel(logging.INFO)
parent_fhandler = logging.FileHandler('parent.log')
parent_fhandler.setLevel(logging.WARNING)
parent_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
parent_fhandler.setFormatter(parent_formatter)
parent_logger.addHandler(parent_fhandler)
```

---
## Step 4 — Log message of severity INFO or above will be handled

```python
parent_logger.debug('Debug message')
# 显示数据类型和缺失值信息 / Show data types and missing value info
parent_logger.info('Info message')
parent_logger.warning('Warning message')
parent_logger.error('Error message')
parent_logger.critical('Critical message')
```

---
## Learning Notes / 学习笔记

- **概念**: Set up root logger, and add a file handler to root logger 是机器学习中的常用技术。  
  *Set up root logger, and add a file handler to root logger is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Formatter / 07 Formatter
# Complete Code / 完整代码
# ===============================

import logging

# Set up root logger, and add a file handler to root logger
logging.basicConfig(filename = 'file.log',
                    level = logging.WARNING,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

# Create logger, set level, and add stream handler
parent_logger = logging.getLogger("parent")
parent_logger.setLevel(logging.INFO)
parent_fhandler = logging.FileHandler('parent.log')
parent_fhandler.setLevel(logging.WARNING)
parent_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
parent_fhandler.setFormatter(parent_formatter)
parent_logger.addHandler(parent_fhandler)

# Log message of severity INFO or above will be handled
parent_logger.debug('Debug message')
# 显示数据类型和缺失值信息 / Show data types and missing value info
parent_logger.info('Info message')
parent_logger.warning('Warning message')
parent_logger.error('Error message')
parent_logger.critical('Critical message')
```

---

➡️ **Next / 下一步**: File 7 of 11

---

### Nadam

# 08 — Nadam / 08 Nadam

**Chapter 11 — File 7 of 11 / 第11章 — 第7个文件（共11个）**

---

## Summary / 总结

This script demonstrates **gradient descent optimization with nadam for a two-dimensional test function**.

本脚本演示 **gradient descent optimization with nadam for a two-dimensional test function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — gradient descent optimization with nadam for a two-dimensional test function

```python
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
```

---
## Step 2 — objective function

```python
def objective(x, y):
	return x**2.0 + y**2.0
```

---
## Step 3 — derivative of objective function

```python
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])
```

---
## Step 4 — gradient descent algorithm with nadam

```python
def nadam(objective, derivative, bounds, n_iter, alpha, mu, nu, eps=1e-8):
```

---
## Step 5 — generate an initial point

```python
# 获取长度 / Get length
x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	score = objective(x[0], x[1])
```

---
## Step 6 — initialize decaying moving averages

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
m = [0.0 for _ in range(bounds.shape[0])]
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	n = [0.0 for _ in range(bounds.shape[0])]
```

---
## Step 7 — run the gradient descent

```python
# 生成整数序列 / Generate integer sequence
for t in range(n_iter):
```

---
## Step 8 — calculate gradient g(t)

```python
g = derivative(x[0], x[1])
```

---
## Step 9 — build a solution one variable at a time

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for i in range(bounds.shape[0]):
```

---
## Step 10 — m(t) = mu * m(t-1) + (1 - mu) * g(t)

```python
m[i] = mu * m[i] + (1.0 - mu) * g[i]
```

---
## Step 11 — n(t) = nu * n(t-1) + (1 - nu) * g(t)^2

```python
n[i] = nu * n[i] + (1.0 - nu) * g[i]**2
```

---
## Step 12 — mhat = (mu * m(t) / (1 - mu)) + ((1 - mu) * g(t) / (1 - mu))

```python
mhat = (mu * m[i] / (1.0 - mu)) + ((1 - mu) * g[i] / (1.0 - mu))
```

---
## Step 13 — nhat = nu * n(t) / (1 - nu)

```python
nhat = nu * n[i] / (1.0 - nu)
```

---
## Step 14 — x(t) = x(t-1) - alpha / (sqrt(nhat) + eps) * mhat

```python
x[i] = x[i] - alpha / (sqrt(nhat) + eps) * mhat
```

---
## Step 15 — evaluate candidate point

```python
score = objective(x[0], x[1])
```

---
## Step 16 — report progress

```python
# 打印输出 / Print output
print('>%d f(%s) = %.5f' % (t, x, score))
	return [x, score]
```

---
## Step 17 — seed the pseudo random number generator

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 18 — define range for input

```python
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
```

---
## Step 19 — define the total iterations

```python
n_iter = 50
```

---
## Step 20 — steps size

```python
alpha = 0.02
```

---
## Step 21 — factor for average gradient

```python
mu = 0.8
```

---
## Step 22 — factor for average squared gradient

```python
nu = 0.999
```

---
## Step 23 — perform the gradient descent search with nadam

```python
best, score = nadam(objective, derivative, bounds, n_iter, alpha, mu, nu)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
```

---
## Learning Notes / 学习笔记

- **概念**: gradient descent optimization with nadam for a two-dimensional test function 是机器学习中的常用技术。  
  *gradient descent optimization with nadam for a two-dimensional test function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nadam / 08 Nadam
# Complete Code / 完整代码
# ===============================

# gradient descent optimization with nadam for a two-dimensional test function
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# derivative of objective function
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])

# gradient descent algorithm with nadam
def nadam(objective, derivative, bounds, n_iter, alpha, mu, nu, eps=1e-8):
	# generate an initial point
 # 获取长度 / Get length
	x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	score = objective(x[0], x[1])
	# initialize decaying moving averages
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	m = [0.0 for _ in range(bounds.shape[0])]
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	n = [0.0 for _ in range(bounds.shape[0])]
	# run the gradient descent
 # 生成整数序列 / Generate integer sequence
	for t in range(n_iter):
		# calculate gradient g(t)
		g = derivative(x[0], x[1])
		# build a solution one variable at a time
  # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
		for i in range(bounds.shape[0]):
			# m(t) = mu * m(t-1) + (1 - mu) * g(t)
			m[i] = mu * m[i] + (1.0 - mu) * g[i]
			# n(t) = nu * n(t-1) + (1 - nu) * g(t)^2
			n[i] = nu * n[i] + (1.0 - nu) * g[i]**2
			# mhat = (mu * m(t) / (1 - mu)) + ((1 - mu) * g(t) / (1 - mu))
			mhat = (mu * m[i] / (1.0 - mu)) + ((1 - mu) * g[i] / (1.0 - mu))
			# nhat = nu * n(t) / (1 - nu)
			nhat = nu * n[i] / (1.0 - nu)
			# x(t) = x(t-1) - alpha / (sqrt(nhat) + eps) * mhat
			x[i] = x[i] - alpha / (sqrt(nhat) + eps) * mhat
		# evaluate candidate point
		score = objective(x[0], x[1])
		# report progress
  # 打印输出 / Print output
		print('>%d f(%s) = %.5f' % (t, x, score))
	return [x, score]

# seed the pseudo random number generator
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# steps size
alpha = 0.02
# factor for average gradient
mu = 0.8
# factor for average squared gradient
nu = 0.999
# perform the gradient descent search with nadam
best, score = nadam(objective, derivative, bounds, n_iter, alpha, mu, nu)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
```

---

➡️ **Next / 下一步**: File 8 of 11

---

### Nadam



---

### Nadam

# 12 — Nadam / 12 Nadam

**Chapter 11 — File 9 of 11 / 第11章 — 第9个文件（共11个）**

---

## Summary / 总结

This script demonstrates **gradient descent optimization with nadam for a two-dimensional test function**.

本脚本演示 **gradient descent optimization with nadam for a two-dimensional test function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — gradient descent optimization with nadam for a two-dimensional test function

```python
import logging
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
```

---
## Step 2 — A Python decorator to log the function call and return value

```python
def loggingdecorator(name):
    logger = logging.getLogger(name)
    def _decor(fn):
        function_name = fn.__name__
        def _fn(*args, **kwargs):
            ret = fn(*args, **kwargs)
            argstr = [str(x) for x in args]
            # 获取字典的键值对 / Get dict key-value pairs
            argstr += [key+"="+str(val) for key,val in kwargs.items()]
            logger.debug("%s(%s) -> %s", function_name, ", ".join(argstr), ret)
            return ret
        return _fn
    return _decor
```

---
## Step 3 — objective function

```python
@loggingdecorator("nadam.function")
def objective(x, y):
    return x**2.0 + y**2.0
```

---
## Step 4 — derivative of objective function

```python
@loggingdecorator("nadam.function")
def derivative(x, y):
    return asarray([x * 2.0, y * 2.0])
```

---
## Step 5 — gradient descent algorithm with nadam

```python
def nadam(objective, derivative, bounds, n_iter, alpha, mu, nu, eps=1e-8):
    logger = logging.getLogger("nadam")
```

---
## Step 6 — generate an initial point

```python
# 获取长度 / Get length
x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
```

---
## Step 7 — initialize decaying moving averages

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
m = [0.0 for _ in range(bounds.shape[0])]
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    n = [0.0 for _ in range(bounds.shape[0])]
```

---
## Step 8 — run the gradient descent

```python
# 生成整数序列 / Generate integer sequence
for t in range(n_iter):
        iterlogger = logging.getLogger("nadam.iter")
```

---
## Step 9 — calculate gradient g(t)

```python
g = derivative(x[0], x[1])
```

---
## Step 10 — build a solution one variable at a time

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for i in range(bounds.shape[0]):
```

---
## Step 11 — m(t) = mu * m(t-1) + (1 - mu) * g(t)

```python
m[i] = mu * m[i] + (1.0 - mu) * g[i]
```

---
## Step 12 — n(t) = nu * n(t-1) + (1 - nu) * g(t)^2

```python
n[i] = nu * n[i] + (1.0 - nu) * g[i]**2
```

---
## Step 13 — mhat = (mu * m(t) / (1 - mu)) + ((1 - mu) * g(t) / (1 - mu))

```python
mhat = (mu * m[i] / (1.0 - mu)) + ((1 - mu) * g[i] / (1.0 - mu))
```

---
## Step 14 — nhat = nu * n(t) / (1 - nu)

```python
nhat = nu * n[i] / (1.0 - nu)
```

---
## Step 15 — x(t) = x(t-1) - alpha / (sqrt(nhat) + eps) * mhat

```python
x[i] = x[i] - alpha / (sqrt(nhat) + eps) * mhat
            # 显示数据类型和缺失值信息 / Show data types and missing value info
            iterlogger.info("Iteration %d variable %d: mhat=%f nhat=%f", t, i, mhat, nhat)
```

---
## Step 16 — evaluate candidate point

```python
score = objective(x[0], x[1])
```

---
## Step 17 — report progress

```python
# 显示数据类型和缺失值信息 / Show data types and missing value info
logger.info('>%d f(%s) = %.5f' % (t, x, score))
    return [x, score]
```

---
## Step 18 — Create logger and assign handler

```python
logger = logging.getLogger("nadam")
handler  = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("nadam.iter")
logger.setLevel(logging.INFO)
```

---
## Step 19 — seed the pseudo random number generator

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 20 — define range for input

```python
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
```

---
## Step 21 — define the total iterations

```python
n_iter = 50
```

---
## Step 22 — steps size

```python
alpha = 0.02
```

---
## Step 23 — factor for average gradient

```python
mu = 0.8
```

---
## Step 24 — factor for average squared gradient

```python
nu = 0.999
```

---
## Step 25 — perform the gradient descent search with nadam

```python
best, score = nadam(objective, derivative, bounds, n_iter, alpha, mu, nu)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
```

---
## Learning Notes / 学习笔记

- **概念**: gradient descent optimization with nadam for a two-dimensional test function 是机器学习中的常用技术。  
  *gradient descent optimization with nadam for a two-dimensional test function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nadam / 12 Nadam
# Complete Code / 完整代码
# ===============================

# gradient descent optimization with nadam for a two-dimensional test function
import logging
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed

# A Python decorator to log the function call and return value
def loggingdecorator(name):
    logger = logging.getLogger(name)
    def _decor(fn):
        function_name = fn.__name__
        def _fn(*args, **kwargs):
            ret = fn(*args, **kwargs)
            argstr = [str(x) for x in args]
            # 获取字典的键值对 / Get dict key-value pairs
            argstr += [key+"="+str(val) for key,val in kwargs.items()]
            logger.debug("%s(%s) -> %s", function_name, ", ".join(argstr), ret)
            return ret
        return _fn
    return _decor

# objective function
@loggingdecorator("nadam.function")
def objective(x, y):
    return x**2.0 + y**2.0

# derivative of objective function
@loggingdecorator("nadam.function")
def derivative(x, y):
    return asarray([x * 2.0, y * 2.0])

# gradient descent algorithm with nadam
def nadam(objective, derivative, bounds, n_iter, alpha, mu, nu, eps=1e-8):
    logger = logging.getLogger("nadam")
    # generate an initial point
    # 获取长度 / Get length
    x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
    # initialize decaying moving averages
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    m = [0.0 for _ in range(bounds.shape[0])]
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    n = [0.0 for _ in range(bounds.shape[0])]
    # run the gradient descent
    # 生成整数序列 / Generate integer sequence
    for t in range(n_iter):
        iterlogger = logging.getLogger("nadam.iter")
        # calculate gradient g(t)
        g = derivative(x[0], x[1])
        # build a solution one variable at a time
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        for i in range(bounds.shape[0]):
            # m(t) = mu * m(t-1) + (1 - mu) * g(t)
            m[i] = mu * m[i] + (1.0 - mu) * g[i]
            # n(t) = nu * n(t-1) + (1 - nu) * g(t)^2
            n[i] = nu * n[i] + (1.0 - nu) * g[i]**2
            # mhat = (mu * m(t) / (1 - mu)) + ((1 - mu) * g(t) / (1 - mu))
            mhat = (mu * m[i] / (1.0 - mu)) + ((1 - mu) * g[i] / (1.0 - mu))
            # nhat = nu * n(t) / (1 - nu)
            nhat = nu * n[i] / (1.0 - nu)
            # x(t) = x(t-1) - alpha / (sqrt(nhat) + eps) * mhat
            x[i] = x[i] - alpha / (sqrt(nhat) + eps) * mhat
            # 显示数据类型和缺失值信息 / Show data types and missing value info
            iterlogger.info("Iteration %d variable %d: mhat=%f nhat=%f", t, i, mhat, nhat)
        # evaluate candidate point
        score = objective(x[0], x[1])
        # report progress
        # 显示数据类型和缺失值信息 / Show data types and missing value info
        logger.info('>%d f(%s) = %.5f' % (t, x, score))
    return [x, score]

# Create logger and assign handler
logger = logging.getLogger("nadam")
handler  = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("nadam.iter")
logger.setLevel(logging.INFO)
# seed the pseudo random number generator
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# steps size
alpha = 0.02
# factor for average gradient
mu = 0.8
# factor for average squared gradient
nu = 0.999
# perform the gradient descent search with nadam
best, score = nadam(objective, derivative, bounds, n_iter, alpha, mu, nu)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
```

---

➡️ **Next / 下一步**: File 10 of 11

---

### Colorama

# 13 — Colorama / 13 Colorama

**Chapter 11 — File 10 of 11 / 第11章 — 第10个文件（共11个）**

---

## Summary / 总结

This script demonstrates **Initialize the terminal for color**.

本脚本演示 **Initialize the terminal for color**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import logging
import colorama
from colorama import Fore, Back, Style
```

---
## Step 2 — Initialize the terminal for color

```python
colorama.init(autoreset = True)
```

---
## Step 3 — Set up logger as usual

```python
logger = logging.getLogger("color")
logger.setLevel(logging.DEBUG)
shandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
shandler.setFormatter(formatter)
logger.addHandler(shandler)
```

---
## Step 4 — Emit log message with color

```python
logger.debug('Debug message')
# 显示数据类型和缺失值信息 / Show data types and missing value info
logger.info(Fore.GREEN + 'Info message')
logger.warning(Fore.BLUE + 'Warning message')
logger.error(Fore.YELLOW + Style.BRIGHT + 'Error message')
logger.critical(Fore.RED + Back.YELLOW + Style.BRIGHT + 'Critical message')
```

---
## Learning Notes / 学习笔记

- **概念**: Initialize the terminal for color 是机器学习中的常用技术。  
  *Initialize the terminal for color is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Colorama / 13 Colorama
# Complete Code / 完整代码
# ===============================

import logging
import colorama
from colorama import Fore, Back, Style

# Initialize the terminal for color
colorama.init(autoreset = True)

# Set up logger as usual
logger = logging.getLogger("color")
logger.setLevel(logging.DEBUG)
shandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
shandler.setFormatter(formatter)
logger.addHandler(shandler)

# Emit log message with color
logger.debug('Debug message')
# 显示数据类型和缺失值信息 / Show data types and missing value info
logger.info(Fore.GREEN + 'Info message')
logger.warning(Fore.BLUE + 'Warning message')
logger.error(Fore.YELLOW + Style.BRIGHT + 'Error message')
logger.critical(Fore.RED + Back.YELLOW + Style.BRIGHT + 'Critical message')
```

---

➡️ **Next / 下一步**: File 11 of 11

---

### Colorlog

# 15 — Colorlog / 15 Colorlog

**Chapter 11 — File 11 of 11 / 第11章 — 第11个文件（共11个）**

---

## Summary / 总结

This script demonstrates **gradient descent optimization with nadam for a two-dimensional test function**.

本脚本演示 **gradient descent optimization with nadam for a two-dimensional test function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — gradient descent optimization with nadam for a two-dimensional test function

```python
import logging
import colorama
from colorama import Fore

from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed

def loggingdecorator(name):
    logger = logging.getLogger(name)
    def _decor(fn):
        function_name = fn.__name__
        def _fn(*args, **kwargs):
            ret = fn(*args, **kwargs)
            argstr = [str(x) for x in args]
            # 获取字典的键值对 / Get dict key-value pairs
            argstr += [key+"="+str(val) for key,val in kwargs.items()]
            logger.debug("%s(%s) -> %s", function_name, ", ".join(argstr), ret)
            return ret
        return _fn
    return _decor
```

---
## Step 2 — objective function

```python
@loggingdecorator("nadam.function")
def objective(x, y):
    return x**2.0 + y**2.0
```

---
## Step 3 — derivative of objective function

```python
@loggingdecorator("nadam.function")
def derivative(x, y):
    return asarray([x * 2.0, y * 2.0])
```

---
## Step 4 — gradient descent algorithm with nadam

```python
def nadam(objective, derivative, bounds, n_iter, alpha, mu, nu, eps=1e-8):
    logger = logging.getLogger("nadam")
```

---
## Step 5 — generate an initial point

```python
# 获取长度 / Get length
x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
```

---
## Step 6 — initialize decaying moving averages

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
m = [0.0 for _ in range(bounds.shape[0])]
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    n = [0.0 for _ in range(bounds.shape[0])]
```

---
## Step 7 — run the gradient descent

```python
# 生成整数序列 / Generate integer sequence
for t in range(n_iter):
        iterlogger = logging.getLogger("nadam.iter")
```

---
## Step 8 — calculate gradient g(t)

```python
g = derivative(x[0], x[1])
```

---
## Step 9 — build a solution one variable at a time

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
for i in range(bounds.shape[0]):
```

---
## Step 10 — m(t) = mu * m(t-1) + (1 - mu) * g(t)

```python
m[i] = mu * m[i] + (1.0 - mu) * g[i]
```

---
## Step 11 — n(t) = nu * n(t-1) + (1 - nu) * g(t)^2

```python
n[i] = nu * n[i] + (1.0 - nu) * g[i]**2
```

---
## Step 12 — mhat = (mu * m(t) / (1 - mu)) + ((1 - mu) * g(t) / (1 - mu))

```python
mhat = (mu * m[i] / (1.0 - mu)) + ((1 - mu) * g[i] / (1.0 - mu))
```

---
## Step 13 — nhat = nu * n(t) / (1 - nu)

```python
nhat = nu * n[i] / (1.0 - nu)
```

---
## Step 14 — x(t) = x(t-1) - alpha / (sqrt(nhat) + eps) * mhat

```python
x[i] = x[i] - alpha / (sqrt(nhat) + eps) * mhat
            # 显示数据类型和缺失值信息 / Show data types and missing value info
            iterlogger.info("Iteration %d variable %d: mhat=%f nhat=%f",
                            t, i, mhat, nhat)
```

---
## Step 15 — evaluate candidate point

```python
score = objective(x[0], x[1])
```

---
## Step 16 — report progress

```python
logger.warning('>%d f(%s) = %.5f' % (t, x, score))
    return [x, score]
```

---
## Step 17 — Prepare the colored formatter

```python
colorama.init(autoreset = True)
colors = {"DEBUG":Fore.BLUE, "INFO":Fore.CYAN,
          "WARNING":Fore.YELLOW, "ERROR":Fore.RED, "CRITICAL":Fore.MAGENTA}
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.levelname in colors:
            msg = colors[record.levelname] + msg + Fore.RESET
        return msg
```

---
## Step 18 — Create logger and assign handler

```python
logger = logging.getLogger("nadam")
handler  = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("nadam.iter")
logger.setLevel(logging.DEBUG)
```

---
## Step 19 — seed the pseudo random number generator

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 20 — define range for input

```python
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
```

---
## Step 21 — define the total iterations

```python
n_iter = 50
```

---
## Step 22 — steps size

```python
alpha = 0.02
```

---
## Step 23 — factor for average gradient

```python
mu = 0.8
```

---
## Step 24 — factor for average squared gradient

```python
nu = 0.999
```

---
## Step 25 — perform the gradient descent search with nadam

```python
best, score = nadam(objective, derivative, bounds, n_iter, alpha, mu, nu)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
```

---
## Learning Notes / 学习笔记

- **概念**: gradient descent optimization with nadam for a two-dimensional test function 是机器学习中的常用技术。  
  *gradient descent optimization with nadam for a two-dimensional test function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Colorlog / 15 Colorlog
# Complete Code / 完整代码
# ===============================

# gradient descent optimization with nadam for a two-dimensional test function
import logging
import colorama
from colorama import Fore

from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed

def loggingdecorator(name):
    logger = logging.getLogger(name)
    def _decor(fn):
        function_name = fn.__name__
        def _fn(*args, **kwargs):
            ret = fn(*args, **kwargs)
            argstr = [str(x) for x in args]
            # 获取字典的键值对 / Get dict key-value pairs
            argstr += [key+"="+str(val) for key,val in kwargs.items()]
            logger.debug("%s(%s) -> %s", function_name, ", ".join(argstr), ret)
            return ret
        return _fn
    return _decor

# objective function
@loggingdecorator("nadam.function")
def objective(x, y):
    return x**2.0 + y**2.0

# derivative of objective function
@loggingdecorator("nadam.function")
def derivative(x, y):
    return asarray([x * 2.0, y * 2.0])

# gradient descent algorithm with nadam
def nadam(objective, derivative, bounds, n_iter, alpha, mu, nu, eps=1e-8):
    logger = logging.getLogger("nadam")
    # generate an initial point
    # 获取长度 / Get length
    x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
    # initialize decaying moving averages
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    m = [0.0 for _ in range(bounds.shape[0])]
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    n = [0.0 for _ in range(bounds.shape[0])]
    # run the gradient descent
    # 生成整数序列 / Generate integer sequence
    for t in range(n_iter):
        iterlogger = logging.getLogger("nadam.iter")
        # calculate gradient g(t)
        g = derivative(x[0], x[1])
        # build a solution one variable at a time
        # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
        for i in range(bounds.shape[0]):
            # m(t) = mu * m(t-1) + (1 - mu) * g(t)
            m[i] = mu * m[i] + (1.0 - mu) * g[i]
            # n(t) = nu * n(t-1) + (1 - nu) * g(t)^2
            n[i] = nu * n[i] + (1.0 - nu) * g[i]**2
            # mhat = (mu * m(t) / (1 - mu)) + ((1 - mu) * g(t) / (1 - mu))
            mhat = (mu * m[i] / (1.0 - mu)) + ((1 - mu) * g[i] / (1.0 - mu))
            # nhat = nu * n(t) / (1 - nu)
            nhat = nu * n[i] / (1.0 - nu)
            # x(t) = x(t-1) - alpha / (sqrt(nhat) + eps) * mhat
            x[i] = x[i] - alpha / (sqrt(nhat) + eps) * mhat
            # 显示数据类型和缺失值信息 / Show data types and missing value info
            iterlogger.info("Iteration %d variable %d: mhat=%f nhat=%f",
                            t, i, mhat, nhat)
        # evaluate candidate point
        score = objective(x[0], x[1])
        # report progress
        logger.warning('>%d f(%s) = %.5f' % (t, x, score))
    return [x, score]

# Prepare the colored formatter
colorama.init(autoreset = True)
colors = {"DEBUG":Fore.BLUE, "INFO":Fore.CYAN,
          "WARNING":Fore.YELLOW, "ERROR":Fore.RED, "CRITICAL":Fore.MAGENTA}
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.levelname in colors:
            msg = colors[record.levelname] + msg + Fore.RESET
        return msg

# Create logger and assign handler
logger = logging.getLogger("nadam")
handler  = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("nadam.iter")
logger.setLevel(logging.DEBUG)
# seed the pseudo random number generator
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# steps size
alpha = 0.02
# factor for average gradient
mu = 0.8
# factor for average squared gradient
nu = 0.999
# perform the gradient descent search with nadam
best, score = nadam(objective, derivative, bounds, n_iter, alpha, mu, nu)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best, score))
```

---

### Chapter Summary / 章节总结

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **11 code files** demonstrating chapter 11.

本章包含 **11 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `01_logging.ipynb` — Logging
  2. `02_logging.ipynb` — Logging
  3. `03_logging.ipynb` — Logging
  4. `05_streamhandler.ipynb` — Streamhandler
  5. `06_filehandler.ipynb` — Filehandler
  6. `07_formatter.ipynb` — Formatter
  7. `08_nadam.ipynb` — Nadam
  8. `11_nadam.ipynb` — Nadam
  9. `12_nadam.ipynb` — Nadam
  10. `13_colorama.ipynb` — Colorama
  11. `15_colorlog.ipynb` — Colorlog

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---
