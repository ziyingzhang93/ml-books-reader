# 时间序列预测 / Time Series Forecasting with Python
## Chapter 06

---

### Chapter Summary / 章节总结

# Chapter 06 Summary / 第06章总结

## Theme / 主题: Chapter 06 / Chapter 06

This chapter contains **12 code files** demonstrating chapter 06.

本章包含 **12 个代码文件**，演示Chapter 06。

---
## Evolution / 演化路线

  1. `autocorrelation_plot.ipynb` — Autocorrelation Plot
  2. `boxplot_monthly.ipynb` — Boxplot Monthly
  3. `boxplot_yearly.ipynb` — Boxplot Yearly
  4. `density_plot.ipynb` — Density Plot
  5. `dot_line_plot.ipynb` — Dot Line Plot
  6. `heat_map_monthly.ipynb` — Heat Map Monthly
  7. `heat_map_yearly.ipynb` — Heat Map Yearly
  8. `histogram.ipynb` — Histogram
  9. `line_plot.ipynb` — Line Plot
  10. `multiple_scatterplot.ipynb` — Multiple Scatterplot
  11. `scatterplot.ipynb` — Scatterplot
  12. `stacked_line_plot.ipynb` — Stacked Line Plot

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 06) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 06）是机器学习流水线中的基础构建块。

---

### Autocorrelation Plot

# 01 — Autocorrelation Plot / Autocorrelation Plot

**Chapter 06 — File 1 of 12 / 第06章 — 第1个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create an autocorrelation plot**.

本脚本演示 **create an autocorrelation plot**。

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
## Step 1 — create an autocorrelation plot

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import autocorrelation_plot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
autocorrelation_plot(series)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create an autocorrelation plot 是机器学习中的常用技术。  
  *create an autocorrelation plot is a common technique in machine learning.*

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
# Autocorrelation Plot / Autocorrelation Plot
# Complete Code / 完整代码
# ===============================

# create an autocorrelation plot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import autocorrelation_plot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
autocorrelation_plot(series)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 12

---

### Boxplot Monthly

# 01 — Boxplot Monthly / Boxplot Monthly

**Chapter 06 — File 2 of 12 / 第06章 — 第2个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create a boxplot of monthly data**.

本脚本演示 **create a boxplot of monthly data**。

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
## Step 1 — create a boxplot of monthly data

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Grouper
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
one_year = series['1990']
groups = one_year.groupby(Grouper(freq='M'))
# 转换为NumPy数组 / Convert to NumPy array
months = concat([DataFrame(x[1].values) for x in groups], axis=1)
months = DataFrame(months)
# 获取列名 / Get column names
months.columns = range(1,13)
months.boxplot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create a boxplot of monthly data 是机器学习中的常用技术。  
  *create a boxplot of monthly data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `groupby` | 分组聚合 | Group and aggregate |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Boxplot Monthly / Boxplot Monthly
# Complete Code / 完整代码
# ===============================

# create a boxplot of monthly data
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Grouper
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
one_year = series['1990']
groups = one_year.groupby(Grouper(freq='M'))
# 转换为NumPy数组 / Convert to NumPy array
months = concat([DataFrame(x[1].values) for x in groups], axis=1)
months = DataFrame(months)
# 获取列名 / Get column names
months.columns = range(1,13)
months.boxplot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 12

---

### Boxplot Yearly

# 01 — Boxplot Yearly / Boxplot Yearly

**Chapter 06 — File 3 of 12 / 第06章 — 第3个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create a boxplot of yearly data**.

本脚本演示 **create a boxplot of yearly data**。

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
## Step 1 — create a boxplot of yearly data

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Grouper
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
 # 转换为NumPy数组 / Convert to NumPy array
	years[name.year] = group.values
years.boxplot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create a boxplot of yearly data 是机器学习中的常用技术。  
  *create a boxplot of yearly data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `groupby` | 分组聚合 | Group and aggregate |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Boxplot Yearly / Boxplot Yearly
# Complete Code / 完整代码
# ===============================

# create a boxplot of yearly data
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Grouper
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
 # 转换为NumPy数组 / Convert to NumPy array
	years[name.year] = group.values
years.boxplot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 12

---

### Density Plot

# 01 — Density Plot / Density Plot

**Chapter 06 — File 4 of 12 / 第06章 — 第4个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create a density plot**.

本脚本演示 **create a density plot**。

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
## Step 1 — create a density plot

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.plot(kind='kde')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create a density plot 是机器学习中的常用技术。  
  *create a density plot is a common technique in machine learning.*

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
# Density Plot / Density Plot
# Complete Code / 完整代码
# ===============================

# create a density plot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.plot(kind='kde')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 12

---

### Dot Line Plot

# 01 — Dot Line Plot / Dot Line Plot

**Chapter 06 — File 5 of 12 / 第06章 — 第5个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create a dot plot**.

本脚本演示 **create a dot plot**。

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
## Step 1 — create a dot plot

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.plot(style='k.')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create a dot plot 是机器学习中的常用技术。  
  *create a dot plot is a common technique in machine learning.*

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
# Dot Line Plot / Dot Line Plot
# Complete Code / 完整代码
# ===============================

# create a dot plot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.plot(style='k.')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 6 of 12

---

### Heat Map Monthly

# 01 — Heat Map Monthly / Heat Map Monthly

**Chapter 06 — File 6 of 12 / 第06章 — 第6个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create a heat map of monthly data**.

本脚本演示 **create a heat map of monthly data**。

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
## Step 1 — create a heat map of monthly data

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Grouper
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
one_year = series['1990']
groups = one_year.groupby(Grouper(freq='M'))
# 转换为NumPy数组 / Convert to NumPy array
months = concat([DataFrame(x[1].values) for x in groups], axis=1)
months = DataFrame(months)
# 获取列名 / Get column names
months.columns = range(1,13)
pyplot.matshow(months, interpolation=None, aspect='auto')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create a heat map of monthly data 是机器学习中的常用技术。  
  *create a heat map of monthly data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `groupby` | 分组聚合 | Group and aggregate |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Heat Map Monthly / Heat Map Monthly
# Complete Code / 完整代码
# ===============================

# create a heat map of monthly data
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Grouper
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
one_year = series['1990']
groups = one_year.groupby(Grouper(freq='M'))
# 转换为NumPy数组 / Convert to NumPy array
months = concat([DataFrame(x[1].values) for x in groups], axis=1)
months = DataFrame(months)
# 获取列名 / Get column names
months.columns = range(1,13)
pyplot.matshow(months, interpolation=None, aspect='auto')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 7 of 12

---

### Heat Map Yearly

# 01 — Heat Map Yearly / Heat Map Yearly

**Chapter 06 — File 7 of 12 / 第06章 — 第7个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create a heat map of yearly data**.

本脚本演示 **create a heat map of yearly data**。

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
## Step 1 — create a heat map of yearly data

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Grouper
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
 # 转换为NumPy数组 / Convert to NumPy array
	years[name.year] = group.values
years = years.T
pyplot.matshow(years, interpolation=None, aspect='auto')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create a heat map of yearly data 是机器学习中的常用技术。  
  *create a heat map of yearly data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `groupby` | 分组聚合 | Group and aggregate |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Heat Map Yearly / Heat Map Yearly
# Complete Code / 完整代码
# ===============================

# create a heat map of yearly data
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Grouper
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
 # 转换为NumPy数组 / Convert to NumPy array
	years[name.year] = group.values
years = years.T
pyplot.matshow(years, interpolation=None, aspect='auto')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 8 of 12

---

### Histogram

# 01 — Histogram / Histogram

**Chapter 06 — File 8 of 12 / 第06章 — 第8个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create a histogram plot**.

本脚本演示 **create a histogram plot**。

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
## Step 1 — create a histogram plot

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.hist()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create a histogram plot 是机器学习中的常用技术。  
  *create a histogram plot is a common technique in machine learning.*

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
# Histogram / Histogram
# Complete Code / 完整代码
# ===============================

# create a histogram plot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.hist()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 9 of 12

---

### Line Plot

# 01 — Line Plot / Line Plot

**Chapter 06 — File 9 of 12 / 第06章 — 第9个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create a line plot**.

本脚本演示 **create a line plot**。

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
## Step 1 — create a line plot

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create a line plot 是机器学习中的常用技术。  
  *create a line plot is a common technique in machine learning.*

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
# Line Plot / Line Plot
# Complete Code / 完整代码
# ===============================

# create a line plot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
series.plot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 10 of 12

---

### Multiple Scatterplot

# 01 — Multiple Scatterplot / Multiple Scatterplot

**Chapter 06 — File 10 of 12 / 第06章 — 第10个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create multiple scatter plots**.

本脚本演示 **create multiple scatter plots**。

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
## Step 1 — create multiple scatter plots

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
lags = 7
columns = [values]
# 生成整数序列 / Generate integer sequence
for i in range(1,(lags + 1)):
 # 添加元素到列表末尾 / Append element to list end
	columns.append(values.shift(i))
dataframe = concat(columns, axis=1)
columns = ['t']
# 生成整数序列 / Generate integer sequence
for i in range(1,(lags + 1)):
 # 添加元素到列表末尾 / Append element to list end
	columns.append('t-' + str(i))
# 获取列名 / Get column names
dataframe.columns = columns
pyplot.figure(1)
# 生成整数序列 / Generate integer sequence
for i in range(1,(lags + 1)):
	ax = pyplot.subplot(240 + i)
	ax.set_title('t vs t-' + str(i))
 # 转换为NumPy数组 / Convert to NumPy array
	pyplot.scatter(x=dataframe['t'].values, y=dataframe['t-'+str(i)].values)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create multiple scatter plots 是机器学习中的常用技术。  
  *create multiple scatter plots is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Multiple Scatterplot / Multiple Scatterplot
# Complete Code / 完整代码
# ===============================

# create multiple scatter plots
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import concat
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
values = DataFrame(series.values)
lags = 7
columns = [values]
# 生成整数序列 / Generate integer sequence
for i in range(1,(lags + 1)):
 # 添加元素到列表末尾 / Append element to list end
	columns.append(values.shift(i))
dataframe = concat(columns, axis=1)
columns = ['t']
# 生成整数序列 / Generate integer sequence
for i in range(1,(lags + 1)):
 # 添加元素到列表末尾 / Append element to list end
	columns.append('t-' + str(i))
# 获取列名 / Get column names
dataframe.columns = columns
pyplot.figure(1)
# 生成整数序列 / Generate integer sequence
for i in range(1,(lags + 1)):
	ax = pyplot.subplot(240 + i)
	ax.set_title('t vs t-' + str(i))
 # 转换为NumPy数组 / Convert to NumPy array
	pyplot.scatter(x=dataframe['t'].values, y=dataframe['t-'+str(i)].values)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 11 of 12

---

### Scatterplot

# 01 — Scatterplot / Scatterplot

**Chapter 06 — File 11 of 12 / 第06章 — 第11个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create a scatter plot**.

本脚本演示 **create a scatter plot**。

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
## Step 1 — create a scatter plot

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import lag_plot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
lag_plot(series)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create a scatter plot 是机器学习中的常用技术。  
  *create a scatter plot is a common technique in machine learning.*

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
# Scatterplot / Scatterplot
# Complete Code / 完整代码
# ===============================

# create a scatter plot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import lag_plot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
lag_plot(series)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 12 of 12

---

### Stacked Line Plot

# 01 — Stacked Line Plot / 堆叠方法

**Chapter 06 — File 12 of 12 / 第06章 — 第12个文件（共12个）**

---

## Summary / 总结

This script demonstrates **create stacked line plots**.

本脚本演示 **create stacked line plots**。

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
## Step 1 — create stacked line plots

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Grouper
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
 # 转换为NumPy数组 / Convert to NumPy array
	years[name.year] = group.values
years.plot(subplots=True, legend=False)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create stacked line plots 是机器学习中的常用技术。  
  *create stacked line plots is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `groupby` | 分组聚合 | Group and aggregate |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Stacked Line Plot / 堆叠方法
# Complete Code / 完整代码
# ===============================

# create stacked line plots
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Grouper
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
 # 转换为NumPy数组 / Convert to NumPy array
	years[name.year] = group.values
years.plot(subplots=True, legend=False)
pyplot.show()
```

---
