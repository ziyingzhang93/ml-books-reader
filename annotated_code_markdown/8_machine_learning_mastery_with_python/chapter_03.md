# Python 机器学习实战 / ML Mastery with Python
## Chapter 03

---

### Chapter Summary / 章节总结

# Chapter 03 Summary / 第03章总结

## Theme / 主题: Chapter 03 / Chapter 03

This chapter contains **4 code files** demonstrating chapter 03.

本章包含 **4 个代码文件**，演示Chapter 03。

---
## Evolution / 演化路线

  1. `matplotlib_crash_course.ipynb` — Matplotlib Crash Course
  2. `numpy_crash_course.ipynb` — Numpy Crash Course
  3. `pandas_crash_course.ipynb` — Pandas Crash Course
  4. `python_crash_course.ipynb` — Python Crash Course

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 03) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 03）是机器学习流水线中的基础构建块。

---

### Matplotlib Crash Course

# 01 — Matplotlib Crash Course / Matplotlib Crash Course

**Chapter 03 — File 1 of 4 / 第03章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **matplotlib crash course**.

本脚本演示 **matplotlib crash course**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — matplotlib crash course
basic line plot

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
myarray = numpy.array([1, 2, 3])
# 绘制折线图 / Draw line plot
plt.plot(myarray)
# 设置X轴标签 / Set X-axis label
plt.xlabel('some x axis')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('some y axis')
# 显示图表 / Display the plot
plt.show()
```

---
## Step 2 — basic scatter plot

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
x = numpy.array([1, 2, 3])
y = numpy.array([2, 4, 6])
# 绘制散点图 / Draw scatter plot
plt.scatter(x,y)
# 设置X轴标签 / Set X-axis label
plt.xlabel('some x axis')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('some y axis')
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: matplotlib crash course 是机器学习中的常用技术。  
  *matplotlib crash course is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Matplotlib Crash Course / Matplotlib Crash Course
# Complete Code / 完整代码
# ===============================

# matplotlib crash course


# basic line plot
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
myarray = numpy.array([1, 2, 3])
# 绘制折线图 / Draw line plot
plt.plot(myarray)
# 设置X轴标签 / Set X-axis label
plt.xlabel('some x axis')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('some y axis')
# 显示图表 / Display the plot
plt.show()


# basic scatter plot
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
x = numpy.array([1, 2, 3])
y = numpy.array([2, 4, 6])
# 绘制散点图 / Draw scatter plot
plt.scatter(x,y)
# 设置X轴标签 / Set X-axis label
plt.xlabel('some x axis')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('some y axis')
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Numpy Crash Course

# 01 — Numpy Crash Course / Numpy Crash Course

**Chapter 03 — File 2 of 4 / 第03章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **numpy crash course**.

本脚本演示 **numpy crash course**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — numpy crash course
define an array

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
mylist = [1, 2, 3]
myarray = numpy.array(mylist)
# 打印输出 / Print output
print(myarray)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(myarray.shape)
```

---
## Step 2 — access values

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
mylist = [[1, 2, 3], [3, 4, 5]]
myarray = numpy.array(mylist)
# 打印输出 / Print output
print(myarray)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(myarray.shape)
# 打印输出 / Print output
print("First row: %s" % myarray[0])
# 打印输出 / Print output
print("Last row: %s" % myarray[-1])
# 打印输出 / Print output
print("Specific row and col: %s" % myarray[0, 2])
# 打印输出 / Print output
print("Whole col: %s" % myarray[:, 2])
```

---
## Step 3 — arithmetic

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
myarray1 = numpy.array([2, 2, 2])
myarray2 = numpy.array([3, 3, 3])
# 打印输出 / Print output
print("Addition: %s" % (myarray1 + myarray2))
# 打印输出 / Print output
print("Multiplication: %s" % (myarray1 * myarray2))
```

---
## Learning Notes / 学习笔记

- **概念**: numpy crash course 是机器学习中的常用技术。  
  *numpy crash course is a common technique in machine learning.*

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
# Numpy Crash Course / Numpy Crash Course
# Complete Code / 完整代码
# ===============================

# numpy crash course

# define an array
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
mylist = [1, 2, 3]
myarray = numpy.array(mylist)
# 打印输出 / Print output
print(myarray)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(myarray.shape)

# access values
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
mylist = [[1, 2, 3], [3, 4, 5]]
myarray = numpy.array(mylist)
# 打印输出 / Print output
print(myarray)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(myarray.shape)
# 打印输出 / Print output
print("First row: %s" % myarray[0])
# 打印输出 / Print output
print("Last row: %s" % myarray[-1])
# 打印输出 / Print output
print("Specific row and col: %s" % myarray[0, 2])
# 打印输出 / Print output
print("Whole col: %s" % myarray[:, 2])

# arithmetic
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
myarray1 = numpy.array([2, 2, 2])
myarray2 = numpy.array([3, 3, 3])
# 打印输出 / Print output
print("Addition: %s" % (myarray1 + myarray2))
# 打印输出 / Print output
print("Multiplication: %s" % (myarray1 * myarray2))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Pandas Crash Course

# 01 — Pandas Crash Course / Pandas Crash Course

**Chapter 03 — File 3 of 4 / 第03章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **pandas crash course**.

本脚本演示 **pandas crash course**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — pandas crash course
series

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas
myarray = numpy.array([1, 2, 3])
rownames = ['a', 'b', 'c']
myseries = pandas.Series(myarray, index=rownames)
# 打印输出 / Print output
print(myseries)

# 打印输出 / Print output
print(myseries[0])
# 打印输出 / Print output
print(myseries['a'])
```

---
## Step 2 — dataframe

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas
myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
# 打印输出 / Print output
print(mydataframe)

# 打印输出 / Print output
print("one column: %s" % mydataframe['one'])
# 打印输出 / Print output
print("one column: %s" % mydataframe.one)
```

---
## Learning Notes / 学习笔记

- **概念**: pandas crash course 是机器学习中的常用技术。  
  *pandas crash course is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pandas Crash Course / Pandas Crash Course
# Complete Code / 完整代码
# ===============================

# pandas crash course


# series
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas
myarray = numpy.array([1, 2, 3])
rownames = ['a', 'b', 'c']
myseries = pandas.Series(myarray, index=rownames)
# 打印输出 / Print output
print(myseries)

# 打印输出 / Print output
print(myseries[0])
# 打印输出 / Print output
print(myseries['a'])


# dataframe
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas
myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
# 打印输出 / Print output
print(mydataframe)

# 打印输出 / Print output
print("one column: %s" % mydataframe['one'])
# 打印输出 / Print output
print("one column: %s" % mydataframe.one)
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Python Crash Course



---
