# DL时间序列
## Chapter 16

---

### Histogram Monthly Power Consumption

# 08 — Histogram Monthly Power Consumption / 08 Histogram Monthly Power Consumption

**Chapter 16 — File 8 of 8 / 第16章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **monthly histogram plots for power usage dataset**.

本脚本演示 **monthly histogram plots for power usage dataset**。

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
## Step 1 — monthly histogram plots for power usage dataset

```python
from pandas import read_csv
from matplotlib import pyplot
```

---
## Step 2 — load the new file

```python
dataset = read_csv('household_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
```

---
## Step 3 — plot active power for each year

```python
months = [x for x in range(1, 13)]
pyplot.figure()
for i in range(len(months)):
```

---
## Step 4 — prepare subplot

```python
ax = pyplot.subplot(len(months), 1, i+1)
```

---
## Step 5 — determine the month to plot

```python
month = '2007-' + str(months[i])
```

---
## Step 6 — get all observations for the month

```python
result = dataset[month]
```

---
## Step 7 — plot the active power for the month

```python
result['Global_active_power'].hist(bins=100)
```

---
## Step 8 — zoom in on the distribution

```python
ax.set_xlim(0, 5)
```

---
## Step 9 — add a title to the subplot

```python
pyplot.title(month, y=0, loc='right')
```

---
## Step 10 — turn off ticks to remove clutter

```python
pyplot.yticks([])
	pyplot.xticks([])
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: monthly histogram plots for power usage dataset 是机器学习中的常用技术。  
  *monthly histogram plots for power usage dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Histogram Monthly Power Consumption / 08 Histogram Monthly Power Consumption
# Complete Code / 完整代码
# ===============================

# monthly histogram plots for power usage dataset
from pandas import read_csv
from matplotlib import pyplot
# load the new file
dataset = read_csv('household_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# plot active power for each year
months = [x for x in range(1, 13)]
pyplot.figure()
for i in range(len(months)):
	# prepare subplot
	ax = pyplot.subplot(len(months), 1, i+1)
	# determine the month to plot
	month = '2007-' + str(months[i])
	# get all observations for the month
	result = dataset[month]
	# plot the active power for the month
	result['Global_active_power'].hist(bins=100)
	# zoom in on the distribution
	ax.set_xlim(0, 5)
	# add a title to the subplot
	pyplot.title(month, y=0, loc='right')
	# turn off ticks to remove clutter
	pyplot.yticks([])
	pyplot.xticks([])
pyplot.show()
```

---
