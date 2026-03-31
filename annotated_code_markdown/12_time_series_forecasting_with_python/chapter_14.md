# 时间序列预测 / Time Series Forecasting with Python
## Chapter 14

---

### Chapter Summary / 章节总结

# Chapter 14 Summary / 第14章总结

## Theme / 主题: Chapter 14 / Chapter 14

This chapter contains **6 code files** demonstrating chapter 14.

本章包含 **6 个代码文件**，演示Chapter 14。

---
## Evolution / 演化路线

  1. `differencing.ipynb` — Differencing
  2. `differencing_with_monthly.ipynb` — Differencing With Monthly
  3. `model_differenced.ipynb` — Model Differenced
  4. `model_seasonality.ipynb` — Model Seasonality
  5. `monthly_average.ipynb` — Monthly Average
  6. `monthly_differenced.ipynb` — Monthly Differenced

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 14) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 14）是机器学习流水线中的基础构建块。

---

### Differencing

# 01 — Differencing / Differencing

**Chapter 14 — File 1 of 6 / 第14章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **deseasonalize a time series using differencing**.

本脚本演示 **deseasonalize a time series using differencing**。

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
## Step 1 — deseasonalize a time series using differencing

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
diff = list()
days_in_year = 365
# 获取长度 / Get length
for i in range(days_in_year, len(X)):
	value = X[i] - X[i - days_in_year]
 # 添加元素到列表末尾 / Append element to list end
	diff.append(value)
pyplot.plot(diff)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: deseasonalize a time series using differencing 是机器学习中的常用技术。  
  *deseasonalize a time series using differencing is a common technique in machine learning.*

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
# Differencing / Differencing
# Complete Code / 完整代码
# ===============================

# deseasonalize a time series using differencing
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
diff = list()
days_in_year = 365
# 获取长度 / Get length
for i in range(days_in_year, len(X)):
	value = X[i] - X[i - days_in_year]
 # 添加元素到列表末尾 / Append element to list end
	diff.append(value)
pyplot.plot(diff)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Differencing With Monthly

# 01 — Differencing With Monthly / Differencing With Monthly

**Chapter 14 — File 2 of 6 / 第14章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **deseasonalize a time series using month-based differencing**.

本脚本演示 **deseasonalize a time series using month-based differencing**。

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
## Step 1 — deseasonalize a time series using month-based differencing

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
diff = list()
days_in_year = 365
# 获取长度 / Get length
for i in range(days_in_year, len(X)):
	month_str = str(series.index[i].year-1)+'-'+str(series.index[i].month)
	month_mean_last_year = series[month_str].mean()
	value = X[i] - month_mean_last_year
 # 添加元素到列表末尾 / Append element to list end
	diff.append(value)
pyplot.plot(diff)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: deseasonalize a time series using month-based differencing 是机器学习中的常用技术。  
  *deseasonalize a time series using month-based differencing is a common technique in machine learning.*

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
# Differencing With Monthly / Differencing With Monthly
# Complete Code / 完整代码
# ===============================

# deseasonalize a time series using month-based differencing
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
diff = list()
days_in_year = 365
# 获取长度 / Get length
for i in range(days_in_year, len(X)):
	month_str = str(series.index[i].year-1)+'-'+str(series.index[i].month)
	month_mean_last_year = series[month_str].mean()
	value = X[i] - month_mean_last_year
 # 添加元素到列表末尾 / Append element to list end
	diff.append(value)
pyplot.plot(diff)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Model Differenced

# 01 — Model Differenced / Model Differenced

**Chapter 14 — File 3 of 6 / 第14章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **deseasonalize by differencing with a polynomial model**.

本脚本演示 **deseasonalize by differencing with a polynomial model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
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
## Step 1 — deseasonalize by differencing with a polynomial model

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import polyfit
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 2 — fit polynomial: x^2*b1 + x*b2 + ... + bn

```python
# 获取长度 / Get length
X = [i%365 for i in range(0, len(series))]
# 转换为NumPy数组 / Convert to NumPy array
y = series.values
degree = 4
coef = polyfit(X, y, degree)
```

---
## Step 3 — create curve

```python
curve = list()
# 获取长度 / Get length
for i in range(len(X)):
	value = coef[-1]
 # 生成整数序列 / Generate integer sequence
	for d in range(degree):
		value += X[i]**(degree-d) * coef[d]
 # 添加元素到列表末尾 / Append element to list end
	curve.append(value)
```

---
## Step 4 — create seasonally adjusted

```python
# 转换为NumPy数组 / Convert to NumPy array
values = series.values
diff = list()
# 获取长度 / Get length
for i in range(len(values)):
	value = values[i] - curve[i]
 # 添加元素到列表末尾 / Append element to list end
	diff.append(value)
pyplot.plot(diff)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: deseasonalize by differencing with a polynomial model 是机器学习中的常用技术。  
  *deseasonalize by differencing with a polynomial model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model Differenced / Model Differenced
# Complete Code / 完整代码
# ===============================

# deseasonalize by differencing with a polynomial model
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import polyfit
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# fit polynomial: x^2*b1 + x*b2 + ... + bn
# 获取长度 / Get length
X = [i%365 for i in range(0, len(series))]
# 转换为NumPy数组 / Convert to NumPy array
y = series.values
degree = 4
coef = polyfit(X, y, degree)
# create curve
curve = list()
# 获取长度 / Get length
for i in range(len(X)):
	value = coef[-1]
 # 生成整数序列 / Generate integer sequence
	for d in range(degree):
		value += X[i]**(degree-d) * coef[d]
 # 添加元素到列表末尾 / Append element to list end
	curve.append(value)
# create seasonally adjusted
# 转换为NumPy数组 / Convert to NumPy array
values = series.values
diff = list()
# 获取长度 / Get length
for i in range(len(values)):
	value = values[i] - curve[i]
 # 添加元素到列表末尾 / Append element to list end
	diff.append(value)
pyplot.plot(diff)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Model Seasonality

# 01 — Model Seasonality / Model Seasonality

**Chapter 14 — File 4 of 6 / 第14章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **model seasonality with a polynomial model**.

本脚本演示 **model seasonality with a polynomial model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
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
## Step 1 — model seasonality with a polynomial model

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import polyfit
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
```

---
## Step 2 — fit polynomial: x^2*b1 + x*b2 + ... + bn

```python
# 获取长度 / Get length
X = [i%365 for i in range(0, len(series))]
# 转换为NumPy数组 / Convert to NumPy array
y = series.values
degree = 4
coef = polyfit(X, y, degree)
# 打印输出 / Print output
print('Coefficients: %s' % coef)
```

---
## Step 3 — create curve

```python
curve = list()
# 获取长度 / Get length
for i in range(len(X)):
	value = coef[-1]
 # 生成整数序列 / Generate integer sequence
	for d in range(degree):
		value += X[i]**(degree-d) * coef[d]
 # 添加元素到列表末尾 / Append element to list end
	curve.append(value)
```

---
## Step 4 — plot curve over original data

```python
# 转换为NumPy数组 / Convert to NumPy array
pyplot.plot(series.values)
pyplot.plot(curve, color='red', linewidth=3)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: model seasonality with a polynomial model 是机器学习中的常用技术。  
  *model seasonality with a polynomial model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model Seasonality / Model Seasonality
# Complete Code / 完整代码
# ===============================

# model seasonality with a polynomial model
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import polyfit
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# fit polynomial: x^2*b1 + x*b2 + ... + bn
# 获取长度 / Get length
X = [i%365 for i in range(0, len(series))]
# 转换为NumPy数组 / Convert to NumPy array
y = series.values
degree = 4
coef = polyfit(X, y, degree)
# 打印输出 / Print output
print('Coefficients: %s' % coef)
# create curve
curve = list()
# 获取长度 / Get length
for i in range(len(X)):
	value = coef[-1]
 # 生成整数序列 / Generate integer sequence
	for d in range(degree):
		value += X[i]**(degree-d) * coef[d]
 # 添加元素到列表末尾 / Append element to list end
	curve.append(value)
# plot curve over original data
# 转换为NumPy数组 / Convert to NumPy array
pyplot.plot(series.values)
pyplot.plot(curve, color='red', linewidth=3)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Monthly Average

# 01 — Monthly Average / Monthly Average

**Chapter 14 — File 5 of 6 / 第14章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **calculate and plot monthly average**.

本脚本演示 **calculate and plot monthly average**。

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
## Step 1 — calculate and plot monthly average

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
resample = series.resample('M')
monthly_mean = resample.mean()
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(monthly_mean.head(13))
monthly_mean.plot()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: calculate and plot monthly average 是机器学习中的常用技术。  
  *calculate and plot monthly average is a common technique in machine learning.*

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
# Monthly Average / Monthly Average
# Complete Code / 完整代码
# ===============================

# calculate and plot monthly average
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
resample = series.resample('M')
monthly_mean = resample.mean()
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(monthly_mean.head(13))
monthly_mean.plot()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Monthly Differenced

# 01 — Monthly Differenced / Monthly Differenced

**Chapter 14 — File 6 of 6 / 第14章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **deseasonalize monthly data by differencing**.

本脚本演示 **deseasonalize monthly data by differencing**。

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
## Step 1 — deseasonalize monthly data by differencing

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
resample = series.resample('M')
monthly_mean = resample.mean()
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
diff = list()
months_in_year = 12
# 获取长度 / Get length
for i in range(months_in_year, len(monthly_mean)):
	value = monthly_mean[i] - monthly_mean[i - months_in_year]
 # 添加元素到列表末尾 / Append element to list end
	diff.append(value)
pyplot.plot(diff)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: deseasonalize monthly data by differencing 是机器学习中的常用技术。  
  *deseasonalize monthly data by differencing is a common technique in machine learning.*

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
# Monthly Differenced / Monthly Differenced
# Complete Code / 完整代码
# ===============================

# deseasonalize monthly data by differencing
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
resample = series.resample('M')
monthly_mean = resample.mean()
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
diff = list()
months_in_year = 12
# 获取长度 / Get length
for i in range(months_in_year, len(monthly_mean)):
	value = monthly_mean[i] - monthly_mean[i - months_in_year]
 # 添加元素到列表末尾 / Append element to list end
	diff.append(value)
pyplot.plot(diff)
pyplot.show()
```

---
