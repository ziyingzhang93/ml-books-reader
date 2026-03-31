# 时间序列预测 / Time Series Forecasting with Python
## Chapter 10

---

### Chapter Summary / 章节总结



---

### White Noise

# 01 — White Noise / White Noise

**Chapter 10 — File 1 of 1 / 第10章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **calculate and plot a white noise series**.

本脚本演示 **calculate and plot a white noise series**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — calculate and plot a white noise series

```python
from random import gauss
from random import seed
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Series
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import autocorrelation_plot
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — seed random number generator

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 3 — create white noise series

```python
# 生成整数序列 / Generate integer sequence
series = [gauss(0.0, 1.0) for i in range(1000)]
series = Series(series)
```

---
## Step 4 — summary stats

```python
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(series.describe())
```

---
## Step 5 — line plot

```python
series.plot()
pyplot.show()
```

---
## Step 6 — histogram plot

```python
series.hist()
pyplot.show()
```

---
## Step 7 — autocorrelation

```python
autocorrelation_plot(series)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: calculate and plot a white noise series 是机器学习中的常用技术。  
  *calculate and plot a white noise series is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `describe()` | 统计摘要信息 | Statistical summary |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# White Noise / White Noise
# Complete Code / 完整代码
# ===============================

# calculate and plot a white noise series
from random import gauss
from random import seed
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import Series
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import autocorrelation_plot
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# seed random number generator
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# create white noise series
# 生成整数序列 / Generate integer sequence
series = [gauss(0.0, 1.0) for i in range(1000)]
series = Series(series)
# summary stats
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(series.describe())
# line plot
series.plot()
pyplot.show()
# histogram plot
series.hist()
pyplot.show()
# autocorrelation
autocorrelation_plot(series)
pyplot.show()
```

---
