# 注意力与Transformer / Transformer Models with Attention
## Chapter 13

---

### Encoding

# 01 — Encoding / 01 Encoding

**Chapter 13 — File 1 of 3 / 第13章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Encoding**.

本脚本演示 **01 Encoding**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

def getPositionEncoding(seq_len, d, n=10000):
    # 创建全零数组 / Create array of zeros
    P = np.zeros((seq_len, d))
    # 生成整数序列 / Generate integer sequence
    for k in range(seq_len):
        # 生成等差数组 / Generate array with step
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

P = getPositionEncoding(seq_len=4, d=4, n=100)
# 打印输出 / Print output
print(P)
```

---
## Learning Notes / 学习笔记

- **概念**: Encoding 是机器学习中的常用技术。  
  *Encoding is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.zeros` | 全零数组 | Array filled with zeros |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Encoding / 01 Encoding
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

def getPositionEncoding(seq_len, d, n=10000):
    # 创建全零数组 / Create array of zeros
    P = np.zeros((seq_len, d))
    # 生成整数序列 / Generate integer sequence
    for k in range(seq_len):
        # 生成等差数组 / Generate array with step
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

P = getPositionEncoding(seq_len=4, d=4, n=100)
# 打印输出 / Print output
print(P)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Plot

# 02 — Plot / 02 Plot

**Chapter 13 — File 2 of 3 / 第13章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Plot**.

本脚本演示 **02 Plot**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

def plotSinusoid(k, d=512, n=10000):
    # 生成等差数组 / Generate array with step
    x = np.arange(0, 100, 1)
    denominator = np.power(n, 2*x/d)
    y = np.sin(k/denominator)
    # 绘制折线图 / Draw line plot
    plt.plot(x, y)
    # 设置图表标题 / Set chart title
    plt.title('k = ' + str(k))

# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(15, 4))
# 生成整数序列 / Generate integer sequence
for i in range(4):
    # 创建子图 / Create subplot
    plt.subplot(141 + i)
    plotSinusoid(i*4)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Plot 是机器学习中的常用技术。  
  *Plot is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot / 02 Plot
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

def plotSinusoid(k, d=512, n=10000):
    # 生成等差数组 / Generate array with step
    x = np.arange(0, 100, 1)
    denominator = np.power(n, 2*x/d)
    y = np.sin(k/denominator)
    # 绘制折线图 / Draw line plot
    plt.plot(x, y)
    # 设置图表标题 / Set chart title
    plt.title('k = ' + str(k))

# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(15, 4))
# 生成整数序列 / Generate integer sequence
for i in range(4):
    # 创建子图 / Create subplot
    plt.subplot(141 + i)
    plotSinusoid(i*4)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Matrix

# 03 — Matrix / 03 Matrix

**Chapter 13 — File 3 of 3 / 第13章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Matrix**.

本脚本演示 **03 Matrix**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

def getPositionEncoding(seq_len, d, n=10000):
    # 创建全零数组 / Create array of zeros
    P = np.zeros((seq_len, d))
    # 生成整数序列 / Generate integer sequence
    for k in range(seq_len):
        # 生成等差数组 / Generate array with step
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

P = getPositionEncoding(seq_len=100, d=512, n=10000)
cax = plt.matshow(P)
plt.gcf().colorbar(cax)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Matrix 是机器学习中的常用技术。  
  *Matrix is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.zeros` | 全零数组 | Array filled with zeros |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Matrix / 03 Matrix
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

def getPositionEncoding(seq_len, d, n=10000):
    # 创建全零数组 / Create array of zeros
    P = np.zeros((seq_len, d))
    # 生成整数序列 / Generate integer sequence
    for k in range(seq_len):
        # 生成等差数组 / Generate array with step
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

P = getPositionEncoding(seq_len=100, d=512, n=10000)
cax = plt.matshow(P)
plt.gcf().colorbar(cax)
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 13 Summary / 第13章总结

## Theme / 主题: Chapter 13 / Chapter 13

This chapter contains **3 code files** demonstrating chapter 13.

本章包含 **3 个代码文件**，演示Chapter 13。

---
## Evolution / 演化路线

  1. `01_encoding.ipynb` — Encoding
  2. `02_plot.ipynb` — Plot
  3. `03_matrix.ipynb` — Matrix

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 13) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 13）是机器学习流水线中的基础构建块。

---
