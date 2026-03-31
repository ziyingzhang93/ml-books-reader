# 时间序列预测 / Time Series Forecasting with Python
## Appendix 01

---

### Chapter Summary / 章节总结

# Chapter 01 Summary / 第01章总结

## Theme / 主题: Appendix / 附录

This chapter contains **5 code files** demonstrating appendix.

本章包含 **5 个代码文件**，演示附录。

---
## Evolution / 演化路线

  1. `load_airline.ipynb` — Load Airline
  2. `load_female_births.ipynb` — Load Female Births
  3. `load_shampoo.ipynb` — Load Shampoo
  4. `load_sunspots.ipynb` — Load Sunspots
  5. `load_temperatures.ipynb` — Load Temperatures

---
## ML Relevance / ML 关联

The techniques in this chapter (Appendix) are fundamental building blocks in machine learning pipelines.

本章技术（附录）是机器学习流水线中的基础构建块。

---

### Load Airline

# 01 — Load Airline / Load Airline

**Chapter 01 — File 1 of 5 / 第01章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **load the airline passengers dataset**.

本脚本演示 **load the airline passengers dataset**。

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
## Step 1 — load the airline passengers dataset

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 打印输出 / Print output
print(type(series))
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head())
series.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load the airline passengers dataset 是机器学习中的常用技术。  
  *load the airline passengers dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `head()` | 查看前几行数据 | View first few rows |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Airline / Load Airline
# Complete Code / 完整代码
# ===============================

# load the airline passengers dataset
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 打印输出 / Print output
print(type(series))
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head())
series.plot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Load Female Births

# 01 — Load Female Births / Load Female Births

**Chapter 01 — File 2 of 5 / 第01章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **load the female births dataset**.

本脚本演示 **load the female births dataset**。

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
## Step 1 — load the female births dataset

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head())
series.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load the female births dataset 是机器学习中的常用技术。  
  *load the female births dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `head()` | 查看前几行数据 | View first few rows |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Female Births / Load Female Births
# Complete Code / 完整代码
# ===============================

# load the female births dataset
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head())
series.plot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Load Shampoo

# 01 — Load Shampoo / Load Shampoo

**Chapter 01 — File 3 of 5 / 第01章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **load the shampoo sales dataset**.

本脚本演示 **load the shampoo sales dataset**。

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
## Step 1 — load the shampoo sales dataset

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head())
series.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load the shampoo sales dataset 是机器学习中的常用技术。  
  *load the shampoo sales dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `head()` | 查看前几行数据 | View first few rows |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Shampoo / Load Shampoo
# Complete Code / 完整代码
# ===============================

# load the shampoo sales dataset
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head())
series.plot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Load Sunspots

# 01 — Load Sunspots / Load Sunspots

**Chapter 01 — File 4 of 5 / 第01章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **load the sunspots dataset**.

本脚本演示 **load the sunspots dataset**。

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
## Step 1 — load the sunspots dataset

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('sunspots.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head())
series.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load the sunspots dataset 是机器学习中的常用技术。  
  *load the sunspots dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `head()` | 查看前几行数据 | View first few rows |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Sunspots / Load Sunspots
# Complete Code / 完整代码
# ===============================

# load the sunspots dataset
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('sunspots.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head())
series.plot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Load Temperatures

# 01 — Load Temperatures / Load Temperatures

**Chapter 01 — File 5 of 5 / 第01章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **load the minimum temperatures dataset**.

本脚本演示 **load the minimum temperatures dataset**。

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
## Step 1 — load the minimum temperatures dataset

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head())
series.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load the minimum temperatures dataset 是机器学习中的常用技术。  
  *load the minimum temperatures dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `head()` | 查看前几行数据 | View first few rows |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Temperatures / Load Temperatures
# Complete Code / 完整代码
# ===============================

# load the minimum temperatures dataset
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(series.head())
series.plot()
pyplot.show()
```

---
