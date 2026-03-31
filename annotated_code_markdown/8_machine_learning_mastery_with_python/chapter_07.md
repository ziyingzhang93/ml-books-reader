# Python 机器学习实战 / ML Mastery with Python
## Chapter 07

---

### Chapter Summary / 章节总结

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **4 code files** demonstrating chapter 07.

本章包含 **4 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `binarization.ipynb` — Binarization
  2. `normalize_data.ipynb` — Normalize Data
  3. `rescale_data.ipynb` — Rescale Data
  4. `standardize_data.ipynb` — Standardize Data

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---

### Binarization

# 01 — Binarization / Binarization

**Chapter 07 — File 1 of 4 / 第07章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **binarization**.

本脚本演示 **binarization**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 训练模型 / Train the model

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — binarization

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import Binarizer
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import set_printoptions
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
```

---
## Step 2 — separate array into input and output components

```python
X = array[:,0:8]
Y = array[:,8]
binarizer = Binarizer(threshold=0.0).fit(X)
# 用已拟合的模型转换数据 / Transform data with fitted model
binaryX = binarizer.transform(X)
```

---
## Step 3 — summarize transformed data

```python
set_printoptions(precision=3)
# 打印输出 / Print output
print(binaryX[0:5,:])
```

---
## Learning Notes / 学习笔记

- **概念**: binarization 是机器学习中的常用技术。  
  *binarization is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Binarization / Binarization
# Complete Code / 完整代码
# ===============================

# binarization
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import Binarizer
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import set_printoptions
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
binarizer = Binarizer(threshold=0.0).fit(X)
# 用已拟合的模型转换数据 / Transform data with fitted model
binaryX = binarizer.transform(X)
# summarize transformed data
set_printoptions(precision=3)
# 打印输出 / Print output
print(binaryX[0:5,:])
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Normalize Data

# 01 — Normalize Data / Normalize Data

**Chapter 07 — File 2 of 4 / 第07章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Normalize data (length of 1)**.

本脚本演示 **Normalize data (length of 1)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 训练模型 / Train the model

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — Normalize data (length of 1)

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import Normalizer
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import set_printoptions
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
```

---
## Step 2 — separate array into input and output components

```python
X = array[:,0:8]
Y = array[:,8]
scaler = Normalizer().fit(X)
# 用已拟合的模型转换数据 / Transform data with fitted model
normalizedX = scaler.transform(X)
```

---
## Step 3 — summarize transformed data

```python
set_printoptions(precision=3)
# 打印输出 / Print output
print(normalizedX[0:5,:])
```

---
## Learning Notes / 学习笔记

- **概念**: Normalize data (length of 1) 是机器学习中的常用技术。  
  *Normalize data (length of 1) is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Normalize Data / Normalize Data
# Complete Code / 完整代码
# ===============================

# Normalize data (length of 1)
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import Normalizer
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import set_printoptions
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = Normalizer().fit(X)
# 用已拟合的模型转换数据 / Transform data with fitted model
normalizedX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
# 打印输出 / Print output
print(normalizedX[0:5,:])
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Rescale Data

# 01 — Rescale Data / 数据缩放

**Chapter 07 — File 3 of 4 / 第07章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Rescale data (between 0 and 1)**.

本脚本演示 **Rescale data (between 0 and 1)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
```

---
## Step 1 — Rescale data (between 0 and 1)

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import set_printoptions
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
```

---
## Step 2 — separate array into input and output components

```python
X = array[:,0:8]
Y = array[:,8]
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
rescaledX = scaler.fit_transform(X)
```

---
## Step 3 — summarize transformed data

```python
set_printoptions(precision=3)
# 打印输出 / Print output
print(rescaledX[0:5,:])
```

---
## Learning Notes / 学习笔记

- **概念**: Rescale data (between 0 and 1) 是机器学习中的常用技术。  
  *Rescale data (between 0 and 1) is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Rescale Data / 数据缩放
# Complete Code / 完整代码
# ===============================

# Rescale data (between 0 and 1)
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import set_printoptions
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
rescaledX = scaler.fit_transform(X)
# summarize transformed data
set_printoptions(precision=3)
# 打印输出 / Print output
print(rescaledX[0:5,:])
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Standardize Data

# 01 — Standardize Data / Standardize Data

**Chapter 07 — File 4 of 4 / 第07章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Standardize data (0 mean, 1 stdev)**.

本脚本演示 **Standardize data (0 mean, 1 stdev)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 训练模型 / Train the model

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — Standardize data (0 mean, 1 stdev)

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import set_printoptions
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
```

---
## Step 2 — separate array into input and output components

```python
X = array[:,0:8]
Y = array[:,8]
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler().fit(X)
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledX = scaler.transform(X)
```

---
## Step 3 — summarize transformed data

```python
set_printoptions(precision=3)
# 打印输出 / Print output
print(rescaledX[0:5,:])
```

---
## Learning Notes / 学习笔记

- **概念**: Standardize data (0 mean, 1 stdev) 是机器学习中的常用技术。  
  *Standardize data (0 mean, 1 stdev) is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Standardize Data / Standardize Data
# Complete Code / 完整代码
# ===============================

# Standardize data (0 mean, 1 stdev)
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import set_printoptions
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler().fit(X)
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
# 打印输出 / Print output
print(rescaledX[0:5,:])
```

---
