# Python深度学习
## Chapter 02

---

### Regression

# 02 — Regression / 回归

**Chapter 02 — File 2 of 3 / 第02章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3**.

本脚本演示 **Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import tensorflow as tf
import numpy as np
```

---
## Step 2 — Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3

```python
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
```

---
## Step 3 — Try to find values for W and b that compute y_data = W * x_data + b
(We know that W should be 0.1 and b 0.3, but Tensorflow will figure that out for us.)

```python
W = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.zeros([1]))
```

---
## Step 4 — A function to compute mean squared error between y_data and computed y

```python
def mse_loss():
    y = W * x_data + b
    loss = tf.reduce_mean(tf.square(y - y_data))
    return loss
```

---
## Step 5 — Minimize the mean squared errors.

```python
optimizer = tf.keras.optimizers.Adam()
for step in range(5000):
    optimizer.minimize(mse_loss, var_list=[W,b])
    if step % 500 == 0:
        print(step, W.numpy(), b.numpy())
```

---
## Learning Notes / 学习笔记

- **概念**: Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3 是机器学习中的常用技术。  
  *Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Regression / 回归
# Complete Code / 完整代码
# ===============================

import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will figure that out for us.)
W = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.zeros([1]))

# A function to compute mean squared error between y_data and computed y
def mse_loss():
    y = W * x_data + b
    loss = tf.reduce_mean(tf.square(y - y_data))
    return loss

# Minimize the mean squared errors.
optimizer = tf.keras.optimizers.Adam()
for step in range(5000):
    optimizer.minimize(mse_loss, var_list=[W,b])
    if step % 500 == 0:
        print(step, W.numpy(), b.numpy())

# Learns best fit is W: [0.1], b: [0.3]
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Chapter Summary

# Chapter 02 Summary / 第02章总结

## Theme / 主题: Chapter 02 / Chapter 02

This chapter contains **3 code files** demonstrating chapter 02.

本章包含 **3 个代码文件**，演示Chapter 02。

---
## Evolution / 演化路线

  1. `01_add.ipynb` — Add
  2. `02_regression.ipynb` — Regression
  3. `03_location.ipynb` — Location

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 02) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 02）是机器学习流水线中的基础构建块。

---
