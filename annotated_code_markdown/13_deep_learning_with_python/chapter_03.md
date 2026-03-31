# Python 深度学习 / Deep Learning with Python
## Chapter 03

---

### Matrix

# 01 — Matrix / 01 Matrix

**Chapter 03 — File 1 of 6 / 第03章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Matrix**.

本脚本演示 **01 Matrix**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf

x = tf.constant([1, 2, 3])
# 打印输出 / Print output
print(x)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(x.shape)
# 打印输出 / Print output
print(x.dtype)
```

---
## Learning Notes / 学习笔记

- **概念**: Matrix 是机器学习中的常用技术。  
  *Matrix is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Matrix / 01 Matrix
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf

x = tf.constant([1, 2, 3])
# 打印输出 / Print output
print(x)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(x.shape)
# 打印输出 / Print output
print(x.dtype)
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Variable

# 02 — Variable / 02 Variable

**Chapter 03 — File 2 of 6 / 第03章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Variable**.

本脚本演示 **02 Variable**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf

x = tf.Variable([1, 2, 3])
# 打印输出 / Print output
print(x)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(x.shape)
# 打印输出 / Print output
print(x.dtype)
```

---
## Learning Notes / 学习笔记

- **概念**: Variable 是机器学习中的常用技术。  
  *Variable is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Variable / 02 Variable
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf

x = tf.Variable([1, 2, 3])
# 打印输出 / Print output
print(x)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(x.shape)
# 打印输出 / Print output
print(x.dtype)
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Gradient

# 03 — Gradient / 梯度方法

**Chapter 03 — File 3 of 6 / 第03章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Gradient**.

本脚本演示 **梯度方法**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf

x = tf.Variable(3.6)

with tf.GradientTape() as tape:
    y = x*x

dy = tape.gradient(y, x)
# 打印输出 / Print output
print(dy)
```

---
## Learning Notes / 学习笔记

- **概念**: Gradient 是机器学习中的常用技术。  
  *Gradient is a common technique in machine learning.*

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
# Gradient / 梯度方法
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf

x = tf.Variable(3.6)

with tf.GradientTape() as tape:
    y = x*x

dy = tape.gradient(y, x)
# 打印输出 / Print output
print(dy)
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Polynomial

# 04 — Polynomial / 04 Polynomial

**Chapter 03 — File 4 of 6 / 第03章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Polynomial**.

本脚本演示 **04 Polynomial**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

polynomial = np.poly1d([1, 2, 3])
# 打印输出 / Print output
print(polynomial)
```

---
## Learning Notes / 学习笔记

- **概念**: Polynomial 是机器学习中的常用技术。  
  *Polynomial is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Polynomial / 04 Polynomial
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

polynomial = np.poly1d([1, 2, 3])
# 打印输出 / Print output
print(polynomial)
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Regression



---

### Puzzle

# 09 — Puzzle / 09 Puzzle

**Chapter 03 — File 6 of 6 / 第03章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Gradient descent loop**.

本脚本演示 **Gradient descent loop**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入随机数生成模块 / Import random number module
import random

A = tf.Variable(random.random())
B = tf.Variable(random.random())
C = tf.Variable(random.random())
D = tf.Variable(random.random())
```

---
## Step 2 — Gradient descent loop

```python
EPOCHS = 1000
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.1)
# 生成整数序列 / Generate integer sequence
for _ in range(EPOCHS):
    with tf.GradientTape() as tape:
        y1 = A + B - 8
        y2 = C - D - 6
        y3 = A + C - 13
        y4 = B + D - 8
        sqerr = y1*y1 + y2*y2 + y3*y3 + y4*y4
    gradA, gradB, gradC, gradD = tape.gradient(sqerr, [A, B, C, D])
    optimizer.apply_gradients([(gradA, A), (gradB, B), (gradC, C), (gradD, D)])

# 打印输出 / Print output
print(A)
# 打印输出 / Print output
print(B)
# 打印输出 / Print output
print(C)
# 打印输出 / Print output
print(D)
```

---
## Learning Notes / 学习笔记

- **概念**: Gradient descent loop 是机器学习中的常用技术。  
  *Gradient descent loop is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Puzzle / 09 Puzzle
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入随机数生成模块 / Import random number module
import random

A = tf.Variable(random.random())
B = tf.Variable(random.random())
C = tf.Variable(random.random())
D = tf.Variable(random.random())

# Gradient descent loop
EPOCHS = 1000
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.1)
# 生成整数序列 / Generate integer sequence
for _ in range(EPOCHS):
    with tf.GradientTape() as tape:
        y1 = A + B - 8
        y2 = C - D - 6
        y3 = A + C - 13
        y4 = B + D - 8
        sqerr = y1*y1 + y2*y2 + y3*y3 + y4*y4
    gradA, gradB, gradC, gradD = tape.gradient(sqerr, [A, B, C, D])
    optimizer.apply_gradients([(gradA, A), (gradB, B), (gradC, C), (gradD, D)])

# 打印输出 / Print output
print(A)
# 打印输出 / Print output
print(B)
# 打印输出 / Print output
print(C)
# 打印输出 / Print output
print(D)
```

---

### Chapter Summary / 章节总结

# Chapter 03 Summary / 第03章总结

## Theme / 主题: Chapter 03 / Chapter 03

This chapter contains **6 code files** demonstrating chapter 03.

本章包含 **6 个代码文件**，演示Chapter 03。

---
## Evolution / 演化路线

  1. `01_matrix.ipynb` — Matrix
  2. `02_variable.ipynb` — Variable
  3. `03_gradient.ipynb` — Gradient
  4. `04_polynomial.ipynb` — Polynomial
  5. `08_regression.ipynb` — Regression
  6. `09_puzzle.ipynb` — Puzzle

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 03) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 03）是机器学习流水线中的基础构建块。

---
