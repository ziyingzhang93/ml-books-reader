# 时间序列预测 / Time Series Forecasting with Python
## Chapter 25

---

### Chapter Summary / 章节总结

# Chapter 25 Summary / 第25章总结

## Theme / 主题: Chapter 25 / Chapter 25

This chapter contains **3 code files** demonstrating chapter 25.

本章包含 **3 个代码文件**，演示Chapter 25。

---
## Evolution / 演化路线

  1. `acf.ipynb` — Acf
  2. `acf_zoom.ipynb` — Acf Zoom
  3. `pacf_zoom.ipynb` — Pacf Zoom

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 25) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 25）是机器学习流水线中的基础构建块。

---

### Acf

# 01 — Acf / Acf

**Chapter 25 — File 1 of 3 / 第25章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **ACF plot of time series**.

本脚本演示 **ACF plot of time series**。

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
## Step 1 — ACF plot of time series

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
plot_acf(series)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: ACF plot of time series 是机器学习中的常用技术。  
  *ACF plot of time series is a common technique in machine learning.*

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
# Acf / Acf
# Complete Code / 完整代码
# ===============================

# ACF plot of time series
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
plot_acf(series)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Acf Zoom

# 01 — Acf Zoom / Acf Zoom

**Chapter 25 — File 2 of 3 / 第25章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **zoomed-in ACF plot of time series**.

本脚本演示 **zoomed-in ACF plot of time series**。

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
## Step 1 — zoomed-in ACF plot of time series

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
plot_acf(series, lags=50)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: zoomed-in ACF plot of time series 是机器学习中的常用技术。  
  *zoomed-in ACF plot of time series is a common technique in machine learning.*

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
# Acf Zoom / Acf Zoom
# Complete Code / 完整代码
# ===============================

# zoomed-in ACF plot of time series
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
plot_acf(series, lags=50)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Pacf Zoom

# 01 — Pacf Zoom / Pacf Zoom

**Chapter 25 — File 3 of 3 / 第25章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **PACF plot of time series**.

本脚本演示 **PACF plot of time series**。

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
## Step 1 — PACF plot of time series

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
plot_pacf(series, lags=50)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: PACF plot of time series 是机器学习中的常用技术。  
  *PACF plot of time series is a common technique in machine learning.*

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
# Pacf Zoom / Pacf Zoom
# Complete Code / 完整代码
# ===============================

# PACF plot of time series
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
plot_pacf(series, lags=50)
pyplot.show()
```

---
