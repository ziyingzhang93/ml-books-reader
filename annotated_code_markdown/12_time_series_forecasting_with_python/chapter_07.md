# 时间序列预测
## Chapter 07

---

### Chapter Summary

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **5 code files** demonstrating chapter 07.

本章包含 **5 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `downsample_quarterly.ipynb` — Downsample Quarterly
  2. `downsample_yearly.ipynb` — Downsample Yearly
  3. `upsample.ipynb` — Upsample
  4. `upsample_interpolate.ipynb` — Upsample Interpolate
  5. `upsample_interpolate_spline.ipynb` — Upsample Interpolate Spline

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---

### Downsample Yearly

# 01 — Downsample Yearly / Downsample Yearly

**Chapter 07 — File 2 of 5 / 第07章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **downsample to yearly intervals**.

本脚本演示 **downsample to yearly intervals**。

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
## Step 1 — downsample to yearly intervals

```python
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
resample = series.resample('A')
yearly_mean_sales = resample.sum()
print(yearly_mean_sales.head())
yearly_mean_sales.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: downsample to yearly intervals 是机器学习中的常用技术。  
  *downsample to yearly intervals is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `head()` | 查看前几行数据 | View first few rows |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Downsample Yearly / Downsample Yearly
# Complete Code / 完整代码
# ===============================

# downsample to yearly intervals
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
resample = series.resample('A')
yearly_mean_sales = resample.sum()
print(yearly_mean_sales.head())
yearly_mean_sales.plot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Upsample

# 01 — Upsample / Upsample

**Chapter 07 — File 3 of 5 / 第07章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **upsample to daily intervals**.

本脚本演示 **upsample to daily intervals**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — upsample to daily intervals

```python
from pandas import read_csv
from pandas import datetime

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
upsampled = series.resample('D').mean()
print(upsampled.head(32))
```

---
## Learning Notes / 学习笔记

- **概念**: upsample to daily intervals 是机器学习中的常用技术。  
  *upsample to daily intervals is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Upsample / Upsample
# Complete Code / 完整代码
# ===============================

# upsample to daily intervals
from pandas import read_csv
from pandas import datetime

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
upsampled = series.resample('D').mean()
print(upsampled.head(32))
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Upsample Interpolate

# 01 — Upsample Interpolate / Upsample Interpolate

**Chapter 07 — File 4 of 5 / 第07章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **upsample to daily intervals with linear interpolation**.

本脚本演示 **upsample to daily intervals with linear interpolation**。

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
## Step 1 — upsample to daily intervals with linear interpolation

```python
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
upsampled = series.resample('D').mean()
interpolated = upsampled.interpolate(method='linear')
print(interpolated.head(32))
interpolated.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: upsample to daily intervals with linear interpolation 是机器学习中的常用技术。  
  *upsample to daily intervals with linear interpolation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Upsample Interpolate / Upsample Interpolate
# Complete Code / 完整代码
# ===============================

# upsample to daily intervals with linear interpolation
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
upsampled = series.resample('D').mean()
interpolated = upsampled.interpolate(method='linear')
print(interpolated.head(32))
interpolated.plot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Upsample Interpolate Spline

# 01 — Upsample Interpolate Spline / Upsample Interpolate Spline

**Chapter 07 — File 5 of 5 / 第07章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **upsample to daily intervals with spline interpolation**.

本脚本演示 **upsample to daily intervals with spline interpolation**。

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
## Step 1 — upsample to daily intervals with spline interpolation

```python
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
upsampled = series.resample('D').mean()
interpolated = upsampled.interpolate(method='spline', order=2)
print(interpolated.head(32))
interpolated.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: upsample to daily intervals with spline interpolation 是机器学习中的常用技术。  
  *upsample to daily intervals with spline interpolation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Upsample Interpolate Spline / Upsample Interpolate Spline
# Complete Code / 完整代码
# ===============================

# upsample to daily intervals with spline interpolation
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
upsampled = series.resample('D').mean()
interpolated = upsampled.interpolate(method='spline', order=2)
print(interpolated.head(32))
interpolated.plot()
pyplot.show()
```

---
