# 时间序列预测 / Time Series Forecasting with Python
## Chapter 16

---

### Chapter Summary / 章节总结

# Chapter 16 Summary / 第16章总结

## Theme / 主题: Chapter 16 / Chapter 16

This chapter contains **4 code files** demonstrating chapter 16.

本章包含 **4 个代码文件**，演示Chapter 16。

---
## Evolution / 演化路线

  1. `repeated_splits.ipynb` — Repeated Splits
  2. `train_test_split.ipynb` — Train Test Split
  3. `train_test_split_plot.ipynb` — Train Test Split Plot
  4. `walk_forward_validation.ipynb` — Walk Forward Validation

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 16) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 16）是机器学习流水线中的基础构建块。

---

### Repeated Splits

# 01 — Repeated Splits / Repeated Splits

**Chapter 16 — File 1 of 4 / 第16章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **calculate repeated train-test splits of time series data**.

本脚本演示 **calculate repeated train-test splits of time series data**。

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
## Step 1 — calculate repeated train-test splits of time series data

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import TimeSeriesSplit
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('sunspots.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
splits = TimeSeriesSplit(n_splits=3)
pyplot.figure(1)
index = 1
for train_index, test_index in splits.split(X):
	train = X[train_index]
	test = X[test_index]
 # 打印输出 / Print output
	print('Observations: %d' % (len(train) + len(test)))
 # 打印输出 / Print output
	print('Training Observations: %d' % (len(train)))
 # 打印输出 / Print output
	print('Testing Observations: %d' % (len(test)))
	pyplot.subplot(310 + index)
	pyplot.plot(train)
	pyplot.plot([None for i in train] + [x for x in test])
	index += 1
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: calculate repeated train-test splits of time series data 是机器学习中的常用技术。  
  *calculate repeated train-test splits of time series data is a common technique in machine learning.*

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
# Repeated Splits / Repeated Splits
# Complete Code / 完整代码
# ===============================

# calculate repeated train-test splits of time series data
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import TimeSeriesSplit
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('sunspots.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
splits = TimeSeriesSplit(n_splits=3)
pyplot.figure(1)
index = 1
for train_index, test_index in splits.split(X):
	train = X[train_index]
	test = X[test_index]
 # 打印输出 / Print output
	print('Observations: %d' % (len(train) + len(test)))
 # 打印输出 / Print output
	print('Training Observations: %d' % (len(train)))
 # 打印输出 / Print output
	print('Testing Observations: %d' % (len(test)))
	pyplot.subplot(310 + index)
	pyplot.plot(train)
	pyplot.plot([None for i in train] + [x for x in test])
	index += 1
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Train Test Split

# 01 — Train Test Split / Train Test Split

**Chapter 16 — File 2 of 4 / 第16章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **calculate a train-test split of a time series dataset**.

本脚本演示 **calculate a train-test split of a time series dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
```

---
## Step 1 — calculate a train-test split of a time series dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('sunspots.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 获取长度 / Get length
train_size = int(len(X) * 0.66)
# 获取长度 / Get length
train, test = X[0:train_size], X[train_size:len(X)]
# 打印输出 / Print output
print('Observations: %d' % (len(X)))
# 打印输出 / Print output
print('Training Observations: %d' % (len(train)))
# 打印输出 / Print output
print('Testing Observations: %d' % (len(test)))
```

---
## Learning Notes / 学习笔记

- **概念**: calculate a train-test split of a time series dataset 是机器学习中的常用技术。  
  *calculate a train-test split of a time series dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Test Split / Train Test Split
# Complete Code / 完整代码
# ===============================

# calculate a train-test split of a time series dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('sunspots.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 获取长度 / Get length
train_size = int(len(X) * 0.66)
# 获取长度 / Get length
train, test = X[0:train_size], X[train_size:len(X)]
# 打印输出 / Print output
print('Observations: %d' % (len(X)))
# 打印输出 / Print output
print('Training Observations: %d' % (len(train)))
# 打印输出 / Print output
print('Testing Observations: %d' % (len(test)))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Train Test Split Plot

# 01 — Train Test Split Plot / Train Test Split Plot

**Chapter 16 — File 3 of 4 / 第16章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **plot train-test split of time series data**.

本脚本演示 **plot train-test split of time series data**。

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
## Step 1 — plot train-test split of time series data

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('sunspots.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 获取长度 / Get length
train_size = int(len(X) * 0.66)
# 获取长度 / Get length
train, test = X[0:train_size], X[train_size:len(X)]
# 打印输出 / Print output
print('Observations: %d' % (len(X)))
# 打印输出 / Print output
print('Training Observations: %d' % (len(train)))
# 打印输出 / Print output
print('Testing Observations: %d' % (len(test)))
pyplot.plot(train)
pyplot.plot([None for i in train] + [x for x in test])
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot train-test split of time series data 是机器学习中的常用技术。  
  *plot train-test split of time series data is a common technique in machine learning.*

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
# Train Test Split Plot / Train Test Split Plot
# Complete Code / 完整代码
# ===============================

# plot train-test split of time series data
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('sunspots.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
# 获取长度 / Get length
train_size = int(len(X) * 0.66)
# 获取长度 / Get length
train, test = X[0:train_size], X[train_size:len(X)]
# 打印输出 / Print output
print('Observations: %d' % (len(X)))
# 打印输出 / Print output
print('Training Observations: %d' % (len(train)))
# 打印输出 / Print output
print('Testing Observations: %d' % (len(test)))
pyplot.plot(train)
pyplot.plot([None for i in train] + [x for x in test])
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Walk Forward Validation

# 01 — Walk Forward Validation / 滚动前进验证

**Chapter 16 — File 4 of 4 / 第16章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **walk forward evaluation model for time series data**.

本脚本演示 **walk forward evaluation model for time series data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — walk forward evaluation model for time series data

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('sunspots.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
n_train = 500
# 获取长度 / Get length
n_records = len(X)
# 生成整数序列 / Generate integer sequence
for i in range(n_train, n_records):
	train, test = X[0:i], X[i:i+1]
 # 打印输出 / Print output
	print('train=%d, test=%d' % (len(train), len(test)))
```

---
## Learning Notes / 学习笔记

- **概念**: walk forward evaluation model for time series data 是机器学习中的常用技术。  
  *walk forward evaluation model for time series data is a common technique in machine learning.*

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
# Walk Forward Validation / 滚动前进验证
# Complete Code / 完整代码
# ===============================

# walk forward evaluation model for time series data
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('sunspots.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# 转换为NumPy数组 / Convert to NumPy array
X = series.values
n_train = 500
# 获取长度 / Get length
n_records = len(X)
# 生成整数序列 / Generate integer sequence
for i in range(n_train, n_records):
	train, test = X[0:i], X[i:i+1]
 # 打印输出 / Print output
	print('train=%d, test=%d' % (len(train), len(test)))
```

---
