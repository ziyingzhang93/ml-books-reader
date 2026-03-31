# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 14

---

### Plot Longley

# 05 — Plot Longley / 05 Plot Longley

**Chapter 14 — File 1 of 5 / 第14章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **plot "Population" vs "Employed"**.

本脚本演示 **plot "Population" vs "Employed"**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — plot "Population" vs "Employed"

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load the dataset

```python
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(url, header=None)
# 转换为NumPy数组 / Convert to NumPy array
data = dataframe.values
```

---
## Step 3 — choose the input and output variables

```python
x, y = data[:, 4], data[:, -1]
```

---
## Step 4 — plot input vs output

```python
pyplot.scatter(x, y)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot "Population" vs "Employed" 是机器学习中的常用技术。  
  *plot "Population" vs "Employed" is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Longley / 05 Plot Longley
# Complete Code / 完整代码
# ===============================

# plot "Population" vs "Employed"
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(url, header=None)
# 转换为NumPy数组 / Convert to NumPy array
data = dataframe.values
# choose the input and output variables
x, y = data[:, 4], data[:, -1]
# plot input vs output
pyplot.scatter(x, y)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Fit Longley

# 12 — Fit Longley / 12 Fit Longley

**Chapter 14 — File 2 of 5 / 第14章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **fit a straight line to the economic data**.

本脚本演示 **fit a straight line to the economic data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — fit a straight line to the economic data

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from scipy.optimize import curve_fit
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — define the true objective function

```python
def objective(x, a, b):
	return a * x + b
```

---
## Step 3 — load the dataset

```python
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(url, header=None)
# 转换为NumPy数组 / Convert to NumPy array
data = dataframe.values
```

---
## Step 4 — choose the input and output variables

```python
x, y = data[:, 4], data[:, -1]
```

---
## Step 5 — curve fit

```python
popt, _ = curve_fit(objective, x, y)
```

---
## Step 6 — summarize the parameter values

```python
a, b = popt
# 打印输出 / Print output
print('y = %.5f * x + %.5f' % (a, b))
```

---
## Step 7 — plot input vs output

```python
pyplot.scatter(x, y)
```

---
## Step 8 — define a sequence of inputs between the smallest and largest known inputs

```python
# 生成整数序列 / Generate integer sequence
x_line = arange(min(x), max(x), 1)
```

---
## Step 9 — calculate the output for the range

```python
y_line = objective(x_line, a, b)
```

---
## Step 10 — create a line plot for the mapping function

```python
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: fit a straight line to the economic data 是机器学习中的常用技术。  
  *fit a straight line to the economic data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Fit Longley / 12 Fit Longley
# Complete Code / 完整代码
# ===============================

# fit a straight line to the economic data
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from scipy.optimize import curve_fit
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# define the true objective function
def objective(x, a, b):
	return a * x + b

# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(url, header=None)
# 转换为NumPy数组 / Convert to NumPy array
data = dataframe.values
# choose the input and output variables
x, y = data[:, 4], data[:, -1]
# curve fit
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b = popt
# 打印输出 / Print output
print('y = %.5f * x + %.5f' % (a, b))
# plot input vs output
pyplot.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
# 生成整数序列 / Generate integer sequence
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b)
# create a line plot for the mapping function
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Second Degree

# 14 — Second Degree / 14 Second Degree

**Chapter 14 — File 3 of 5 / 第14章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **fit a second degree polynomial to the economic data**.

本脚本演示 **fit a second degree polynomial to the economic data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — fit a second degree polynomial to the economic data

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from scipy.optimize import curve_fit
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — define the true objective function

```python
def objective(x, a, b, c):
	return a * x + b * x**2 + c
```

---
## Step 3 — load the dataset

```python
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(url, header=None)
# 转换为NumPy数组 / Convert to NumPy array
data = dataframe.values
```

---
## Step 4 — choose the input and output variables

```python
x, y = data[:, 4], data[:, -1]
```

---
## Step 5 — curve fit

```python
popt, _ = curve_fit(objective, x, y)
```

---
## Step 6 — summarize the parameter values

```python
a, b, c = popt
# 打印输出 / Print output
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
```

---
## Step 7 — plot input vs output

```python
pyplot.scatter(x, y)
```

---
## Step 8 — define a sequence of inputs between the smallest and largest known inputs

```python
# 生成整数序列 / Generate integer sequence
x_line = arange(min(x), max(x), 1)
```

---
## Step 9 — calculate the output for the range

```python
y_line = objective(x_line, a, b, c)
```

---
## Step 10 — create a line plot for the mapping function

```python
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: fit a second degree polynomial to the economic data 是机器学习中的常用技术。  
  *fit a second degree polynomial to the economic data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Second Degree / 14 Second Degree
# Complete Code / 完整代码
# ===============================

# fit a second degree polynomial to the economic data
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from scipy.optimize import curve_fit
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# define the true objective function
def objective(x, a, b, c):
	return a * x + b * x**2 + c

# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(url, header=None)
# 转换为NumPy数组 / Convert to NumPy array
data = dataframe.values
# choose the input and output variables
x, y = data[:, 4], data[:, -1]
# curve fit
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b, c = popt
# 打印输出 / Print output
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
# plot input vs output
pyplot.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
# 生成整数序列 / Generate integer sequence
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b, c)
# create a line plot for the mapping function
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Fifth Degree

# 15 — Fifth Degree / 15 Fifth Degree

**Chapter 14 — File 4 of 5 / 第14章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **fit a fifth degree polynomial to the economic data**.

本脚本演示 **fit a fifth degree polynomial to the economic data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — fit a fifth degree polynomial to the economic data

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from scipy.optimize import curve_fit
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — define the true objective function

```python
def objective(x, a, b, c, d, e, f):
	return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f
```

---
## Step 3 — load the dataset

```python
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(url, header=None)
# 转换为NumPy数组 / Convert to NumPy array
data = dataframe.values
```

---
## Step 4 — choose the input and output variables

```python
x, y = data[:, 4], data[:, -1]
```

---
## Step 5 — curve fit

```python
popt, _ = curve_fit(objective, x, y)
```

---
## Step 6 — summarize the parameter values

```python
a, b, c, d, e, f = popt
```

---
## Step 7 — plot input vs output

```python
pyplot.scatter(x, y)
```

---
## Step 8 — define a sequence of inputs between the smallest and largest known inputs

```python
# 生成整数序列 / Generate integer sequence
x_line = arange(min(x), max(x), 1)
```

---
## Step 9 — calculate the output for the range

```python
y_line = objective(x_line, a, b, c, d, e, f)
```

---
## Step 10 — create a line plot for the mapping function

```python
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: fit a fifth degree polynomial to the economic data 是机器学习中的常用技术。  
  *fit a fifth degree polynomial to the economic data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Fifth Degree / 15 Fifth Degree
# Complete Code / 完整代码
# ===============================

# fit a fifth degree polynomial to the economic data
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from scipy.optimize import curve_fit
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# define the true objective function
def objective(x, a, b, c, d, e, f):
	return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(url, header=None)
# 转换为NumPy数组 / Convert to NumPy array
data = dataframe.values
# choose the input and output variables
x, y = data[:, 4], data[:, -1]
# curve fit
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b, c, d, e, f = popt
# plot input vs output
pyplot.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
# 生成整数序列 / Generate integer sequence
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b, c, d, e, f)
# create a line plot for the mapping function
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Sine

# 17 — Sine / 17 Sine

**Chapter 14 — File 5 of 5 / 第14章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **fit a line to the economic data**.

本脚本演示 **fit a line to the economic data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — fit a line to the economic data

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import sin
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from scipy.optimize import curve_fit
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — define the true objective function

```python
def objective(x, a, b, c, d):
	return a * sin(b - x) + c * x**2 + d
```

---
## Step 3 — load the dataset

```python
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(url, header=None)
# 转换为NumPy数组 / Convert to NumPy array
data = dataframe.values
```

---
## Step 4 — choose the input and output variables

```python
x, y = data[:, 4], data[:, -1]
```

---
## Step 5 — curve fit

```python
popt, _ = curve_fit(objective, x, y)
```

---
## Step 6 — summarize the parameter values

```python
a, b, c, d = popt
# 打印输出 / Print output
print(popt)
```

---
## Step 7 — plot input vs output

```python
pyplot.scatter(x, y)
```

---
## Step 8 — define a sequence of inputs between the smallest and largest known inputs

```python
# 生成整数序列 / Generate integer sequence
x_line = arange(min(x), max(x), 1)
```

---
## Step 9 — calculate the output for the range

```python
y_line = objective(x_line, a, b, c, d)
```

---
## Step 10 — create a line plot for the mapping function

```python
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: fit a line to the economic data 是机器学习中的常用技术。  
  *fit a line to the economic data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sine / 17 Sine
# Complete Code / 完整代码
# ===============================

# fit a line to the economic data
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import sin
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
from scipy.optimize import curve_fit
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# define the true objective function
def objective(x, a, b, c, d):
	return a * sin(b - x) + c * x**2 + d

# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(url, header=None)
# 转换为NumPy数组 / Convert to NumPy array
data = dataframe.values
# choose the input and output variables
x, y = data[:, 4], data[:, -1]
# curve fit
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b, c, d = popt
# 打印输出 / Print output
print(popt)
# plot input vs output
pyplot.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
# 生成整数序列 / Generate integer sequence
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b, c, d)
# create a line plot for the mapping function
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 14 Summary / 第14章总结

## Theme / 主题: Chapter 14 / Chapter 14

This chapter contains **5 code files** demonstrating chapter 14.

本章包含 **5 个代码文件**，演示Chapter 14。

---
## Evolution / 演化路线

  1. `05_plot_longley.ipynb` — Plot Longley
  2. `12_fit_longley.ipynb` — Fit Longley
  3. `14_second_degree.ipynb` — Second Degree
  4. `15_fifth_degree.ipynb` — Fifth Degree
  5. `17_sine.ipynb` — Sine

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 14) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 14）是机器学习流水线中的基础构建块。

---
