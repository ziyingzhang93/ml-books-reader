# ML数据准备
## Chapter 05

---

### Num Unique

# 01 — Num Unique / 01 Num Unique

**Chapter 05 — File 1 of 10 / 第05章 — 第1个文件（共10个）**

---

## Summary / 总结

This script demonstrates **summarize the number of unique values for each column using numpy**.

本脚本演示 **summarize the number of unique values for each column using numpy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — summarize the number of unique values for each column using numpy

```python
from numpy import loadtxt
from numpy import unique
```

---
## Step 2 — load the dataset

```python
data = loadtxt('oil-spill.csv', delimiter=',')
```

---
## Step 3 — summarize the number of unique values in each column

```python
for i in range(data.shape[1]):
	print(i, len(unique(data[:, i])))
```

---
## Learning Notes / 学习笔记

- **概念**: summarize the number of unique values for each column using numpy 是机器学习中的常用技术。  
  *summarize the number of unique values for each column using numpy is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Num Unique / 01 Num Unique
# Complete Code / 完整代码
# ===============================

# summarize the number of unique values for each column using numpy
from numpy import loadtxt
from numpy import unique
# load the dataset
data = loadtxt('oil-spill.csv', delimiter=',')
# summarize the number of unique values in each column
for i in range(data.shape[1]):
	print(i, len(unique(data[:, i])))
```

---

➡️ **Next / 下一步**: File 2 of 10

---

### Num Unique Simpler

# 02 — Num Unique Simpler / 02 Num Unique Simpler

**Chapter 05 — File 2 of 10 / 第05章 — 第2个文件（共10个）**

---

## Summary / 总结

This script demonstrates **summarize the number of unique values for each column using pandas**.

本脚本演示 **summarize the number of unique values for each column using pandas**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — summarize the number of unique values for each column using pandas

```python
from pandas import read_csv
```

---
## Step 2 — load the dataset

```python
df = read_csv('oil-spill.csv', header=None)
```

---
## Step 3 — summarize the number of unique values in each column

```python
print(df.nunique())
```

---
## Learning Notes / 学习笔记

- **概念**: summarize the number of unique values for each column using pandas 是机器学习中的常用技术。  
  *summarize the number of unique values for each column using pandas is a common technique in machine learning.*

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
# Num Unique Simpler / 02 Num Unique Simpler
# Complete Code / 完整代码
# ===============================

# summarize the number of unique values for each column using pandas
from pandas import read_csv
# load the dataset
df = read_csv('oil-spill.csv', header=None)
# summarize the number of unique values in each column
print(df.nunique())
```

---

➡️ **Next / 下一步**: File 3 of 10

---

### Drop Column

# 03 — Drop Column / 03 Drop Column

**Chapter 05 — File 3 of 10 / 第05章 — 第3个文件（共10个）**

---

## Summary / 总结

This script demonstrates **delete columns with a single unique value**.

本脚本演示 **delete columns with a single unique value**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — delete columns with a single unique value

```python
from pandas import read_csv
```

---
## Step 2 — load the dataset

```python
df = read_csv('oil-spill.csv', header=None)
print(df.shape)
```

---
## Step 3 — get number of unique values for each column

```python
counts = df.nunique()
```

---
## Step 4 — record columns to delete

```python
to_del = [i for i,v in enumerate(counts) if v == 1]
print(to_del)
```

---
## Step 5 — drop useless columns

```python
df.drop(to_del, axis=1, inplace=True)
print(df.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: delete columns with a single unique value 是机器学习中的常用技术。  
  *delete columns with a single unique value is a common technique in machine learning.*

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
# Drop Column / 03 Drop Column
# Complete Code / 完整代码
# ===============================

# delete columns with a single unique value
from pandas import read_csv
# load the dataset
df = read_csv('oil-spill.csv', header=None)
print(df.shape)
# get number of unique values for each column
counts = df.nunique()
# record columns to delete
to_del = [i for i,v in enumerate(counts) if v == 1]
print(to_del)
# drop useless columns
df.drop(to_del, axis=1, inplace=True)
print(df.shape)
```

---

➡️ **Next / 下一步**: File 4 of 10

---

### Column Variance

# 04 — Column Variance / 04 Column Variance

**Chapter 05 — File 4 of 10 / 第05章 — 第4个文件（共10个）**

---

## Summary / 总结

This script demonstrates **summarize the percentage of unique values for each column using numpy**.

本脚本演示 **summarize the percentage of unique values for each column using numpy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — summarize the percentage of unique values for each column using numpy

```python
from numpy import loadtxt
from numpy import unique
```

---
## Step 2 — load the dataset

```python
data = loadtxt('oil-spill.csv', delimiter=',')
```

---
## Step 3 — summarize the number of unique values in each column

```python
for i in range(data.shape[1]):
	num = len(unique(data[:, i]))
	percentage = float(num) / data.shape[0] * 100
	print('%d, %d, %.1f%%' % (i, num, percentage))
```

---
## Learning Notes / 学习笔记

- **概念**: summarize the percentage of unique values for each column using numpy 是机器学习中的常用技术。  
  *summarize the percentage of unique values for each column using numpy is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Column Variance / 04 Column Variance
# Complete Code / 完整代码
# ===============================

# summarize the percentage of unique values for each column using numpy
from numpy import loadtxt
from numpy import unique
# load the dataset
data = loadtxt('oil-spill.csv', delimiter=',')
# summarize the number of unique values in each column
for i in range(data.shape[1]):
	num = len(unique(data[:, i]))
	percentage = float(num) / data.shape[0] * 100
	print('%d, %d, %.1f%%' % (i, num, percentage))
```

---

➡️ **Next / 下一步**: File 5 of 10

---

### Low Variance

# 05 — Low Variance / 05 Low Variance

**Chapter 05 — File 5 of 10 / 第05章 — 第5个文件（共10个）**

---

## Summary / 总结

This script demonstrates **summarize the percentage of unique values for each column using numpy**.

本脚本演示 **summarize the percentage of unique values for each column using numpy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — summarize the percentage of unique values for each column using numpy

```python
from numpy import loadtxt
from numpy import unique
```

---
## Step 2 — load the dataset

```python
data = loadtxt('oil-spill.csv', delimiter=',')
```

---
## Step 3 — summarize the number of unique values in each column

```python
for i in range(data.shape[1]):
	num = len(unique(data[:, i]))
	percentage = float(num) / data.shape[0] * 100
	if percentage < 1:
		print('%d, %d, %.1f%%' % (i, num, percentage))
```

---
## Learning Notes / 学习笔记

- **概念**: summarize the percentage of unique values for each column using numpy 是机器学习中的常用技术。  
  *summarize the percentage of unique values for each column using numpy is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Low Variance / 05 Low Variance
# Complete Code / 完整代码
# ===============================

# summarize the percentage of unique values for each column using numpy
from numpy import loadtxt
from numpy import unique
# load the dataset
data = loadtxt('oil-spill.csv', delimiter=',')
# summarize the number of unique values in each column
for i in range(data.shape[1]):
	num = len(unique(data[:, i]))
	percentage = float(num) / data.shape[0] * 100
	if percentage < 1:
		print('%d, %d, %.1f%%' % (i, num, percentage))
```

---

➡️ **Next / 下一步**: File 6 of 10

---

### Remove Low Variance

# 06 — Remove Low Variance / 06 Remove Low Variance

**Chapter 05 — File 6 of 10 / 第05章 — 第6个文件（共10个）**

---

## Summary / 总结

This script demonstrates **delete columns where number of unique values is less than 1% of the rows**.

本脚本演示 **delete columns where number of unique values is less than 1% of the rows**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — delete columns where number of unique values is less than 1% of the rows

```python
from pandas import read_csv
```

---
## Step 2 — load the dataset

```python
df = read_csv('oil-spill.csv', header=None)
print(df.shape)
```

---
## Step 3 — get number of unique values for each column

```python
counts = df.nunique()
```

---
## Step 4 — record columns to delete

```python
to_del = [i for i,v in enumerate(counts) if (float(v)/df.shape[0]*100) < 1]
print(to_del)
```

---
## Step 5 — drop useless columns

```python
df.drop(to_del, axis=1, inplace=True)
print(df.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: delete columns where number of unique values is less than 1% of the rows 是机器学习中的常用技术。  
  *delete columns where number of unique values is less than 1% of the rows is a common technique in machine learning.*

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
# Remove Low Variance / 06 Remove Low Variance
# Complete Code / 完整代码
# ===============================

# delete columns where number of unique values is less than 1% of the rows
from pandas import read_csv
# load the dataset
df = read_csv('oil-spill.csv', header=None)
print(df.shape)
# get number of unique values for each column
counts = df.nunique()
# record columns to delete
to_del = [i for i,v in enumerate(counts) if (float(v)/df.shape[0]*100) < 1]
print(to_del)
# drop useless columns
df.drop(to_del, axis=1, inplace=True)
print(df.shape)
```

---

➡️ **Next / 下一步**: File 7 of 10

---

### Auto Remove Low Variance

# 07 — Auto Remove Low Variance / 07 Auto Remove Low Variance

**Chapter 05 — File 7 of 10 / 第05章 — 第7个文件（共10个）**

---

## Summary / 总结

This script demonstrates **example of applying the variance threshold for feature selection**.

本脚本演示 **example of applying the variance threshold for feature selection**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Step 1 — example of applying the variance threshold for feature selection

```python
from pandas import read_csv
from sklearn.feature_selection import VarianceThreshold
```

---
## Step 2 — load the dataset

```python
df = read_csv('oil-spill.csv', header=None)
```

---
## Step 3 — split data into inputs and outputs

```python
data = df.values
X = data[:, :-1]
y = data[:, -1]
print(X.shape, y.shape)
```

---
## Step 4 — define the transform

```python
transform = VarianceThreshold()
```

---
## Step 5 — transform the input data

```python
X_sel = transform.fit_transform(X)
print(X_sel.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: example of applying the variance threshold for feature selection 是机器学习中的常用技术。  
  *example of applying the variance threshold for feature selection is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Auto Remove Low Variance / 07 Auto Remove Low Variance
# Complete Code / 完整代码
# ===============================

# example of applying the variance threshold for feature selection
from pandas import read_csv
from sklearn.feature_selection import VarianceThreshold
# load the dataset
df = read_csv('oil-spill.csv', header=None)
# split data into inputs and outputs
data = df.values
X = data[:, :-1]
y = data[:, -1]
print(X.shape, y.shape)
# define the transform
transform = VarianceThreshold()
# transform the input data
X_sel = transform.fit_transform(X)
print(X_sel.shape)
```

---

➡️ **Next / 下一步**: File 8 of 10

---

### Compare Variance Thresholds

# 08 — Compare Variance Thresholds / 阈值调优

**Chapter 05 — File 8 of 10 / 第05章 — 第8个文件（共10个）**

---

## Summary / 总结

This script demonstrates **explore the effect of the variance thresholds on the number of selected features**.

本脚本演示 **explore the effect of the variance thresholds on the number of selected features**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — explore the effect of the variance thresholds on the number of selected features

```python
from numpy import arange
from pandas import read_csv
from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot
```

---
## Step 2 — load the dataset

```python
df = read_csv('oil-spill.csv', header=None)
```

---
## Step 3 — split data into inputs and outputs

```python
data = df.values
X = data[:, :-1]
y = data[:, -1]
print(X.shape, y.shape)
```

---
## Step 4 — define thresholds to check

```python
thresholds = arange(0.0, 0.55, 0.05)
```

---
## Step 5 — apply transform with each threshold

```python
results = list()
for t in thresholds:
```

---
## Step 6 — define the transform

```python
transform = VarianceThreshold(threshold=t)
```

---
## Step 7 — transform the input data

```python
X_sel = transform.fit_transform(X)
```

---
## Step 8 — determine the number of input features

```python
n_features = X_sel.shape[1]
	print('>Threshold=%.2f, Features=%d' % (t, n_features))
```

---
## Step 9 — store the result

```python
results.append(n_features)
```

---
## Step 10 — plot the threshold vs the number of selected features

```python
pyplot.plot(thresholds, results)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: explore the effect of the variance thresholds on the number of selected features 是机器学习中的常用技术。  
  *explore the effect of the variance thresholds on the number of selected features is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Compare Variance Thresholds / 阈值调优
# Complete Code / 完整代码
# ===============================

# explore the effect of the variance thresholds on the number of selected features
from numpy import arange
from pandas import read_csv
from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot
# load the dataset
df = read_csv('oil-spill.csv', header=None)
# split data into inputs and outputs
data = df.values
X = data[:, :-1]
y = data[:, -1]
print(X.shape, y.shape)
# define thresholds to check
thresholds = arange(0.0, 0.55, 0.05)
# apply transform with each threshold
results = list()
for t in thresholds:
	# define the transform
	transform = VarianceThreshold(threshold=t)
	# transform the input data
	X_sel = transform.fit_transform(X)
	# determine the number of input features
	n_features = X_sel.shape[1]
	print('>Threshold=%.2f, Features=%d' % (t, n_features))
	# store the result
	results.append(n_features)
# plot the threshold vs the number of selected features
pyplot.plot(thresholds, results)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 9 of 10

---

### Report Duplicates

# 09 — Report Duplicates / 09 Report Duplicates

**Chapter 05 — File 9 of 10 / 第05章 — 第9个文件（共10个）**

---

## Summary / 总结

This script demonstrates **locate rows of duplicate data**.

本脚本演示 **locate rows of duplicate data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — locate rows of duplicate data

```python
from pandas import read_csv
```

---
## Step 2 — load the dataset

```python
df = read_csv('iris.csv', header=None)
```

---
## Step 3 — calculate duplicates

```python
dups = df.duplicated()
```

---
## Step 4 — report if there are any duplicates

```python
print(dups.any())
```

---
## Step 5 — list all duplicate rows

```python
print(df[dups])
```

---
## Learning Notes / 学习笔记

- **概念**: locate rows of duplicate data 是机器学习中的常用技术。  
  *locate rows of duplicate data is a common technique in machine learning.*

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
# Report Duplicates / 09 Report Duplicates
# Complete Code / 完整代码
# ===============================

# locate rows of duplicate data
from pandas import read_csv
# load the dataset
df = read_csv('iris.csv', header=None)
# calculate duplicates
dups = df.duplicated()
# report if there are any duplicates
print(dups.any())
# list all duplicate rows
print(df[dups])
```

---

➡️ **Next / 下一步**: File 10 of 10

---

### Remove Duplicates

# 10 — Remove Duplicates / 10 Remove Duplicates

**Chapter 05 — File 10 of 10 / 第05章 — 第10个文件（共10个）**

---

## Summary / 总结

This script demonstrates **delete rows of duplicate data from the dataset**.

本脚本演示 **delete rows of duplicate data from the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — delete rows of duplicate data from the dataset

```python
from pandas import read_csv
```

---
## Step 2 — load the dataset

```python
df = read_csv('iris.csv', header=None)
print(df.shape)
```

---
## Step 3 — delete duplicate rows

```python
df.drop_duplicates(inplace=True)
print(df.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: delete rows of duplicate data from the dataset 是机器学习中的常用技术。  
  *delete rows of duplicate data from the dataset is a common technique in machine learning.*

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
# Remove Duplicates / 10 Remove Duplicates
# Complete Code / 完整代码
# ===============================

# delete rows of duplicate data from the dataset
from pandas import read_csv
# load the dataset
df = read_csv('iris.csv', header=None)
print(df.shape)
# delete duplicate rows
df.drop_duplicates(inplace=True)
print(df.shape)
```

---

### Chapter Summary

# Chapter 05 Summary / 第05章总结

## Theme / 主题: Chapter 05 / Chapter 05

This chapter contains **10 code files** demonstrating chapter 05.

本章包含 **10 个代码文件**，演示Chapter 05。

---
## Evolution / 演化路线

  1. `01_num_unique.ipynb` — Num Unique
  2. `02_num_unique_simpler.ipynb` — Num Unique Simpler
  3. `03_drop_column.ipynb` — Drop Column
  4. `04_column_variance.ipynb` — Column Variance
  5. `05_low_variance.ipynb` — Low Variance
  6. `06_remove_low_variance.ipynb` — Remove Low Variance
  7. `07_auto_remove_low_variance.ipynb` — Auto Remove Low Variance
  8. `08_compare_variance_thresholds.ipynb` — Compare Variance Thresholds
  9. `09_report_duplicates.ipynb` — Report Duplicates
  10. `10_remove_duplicates.ipynb` — Remove Duplicates

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 05) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 05）是机器学习流水线中的基础构建块。

---
