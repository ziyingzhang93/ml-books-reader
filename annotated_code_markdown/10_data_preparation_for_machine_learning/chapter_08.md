# ML数据准备
## Chapter 08

---

### Mark Missing

# 01 — Mark Missing / 缺失值处理

**Chapter 08 — File 1 of 5 / 第08章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **summarize the horse colic dataset**.

本脚本演示 **summarize the horse colic dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — summarize the horse colic dataset

```python
from pandas import read_csv
```

---
## Step 2 — load dataset

```python
dataframe = read_csv('horse-colic.csv', header=None, na_values='?')
```

---
## Step 3 — summarize the first few rows

```python
print(dataframe.head())
```

---
## Step 4 — summarize the number of rows with missing values for each column

```python
for i in range(dataframe.shape[1]):
```

---
## Step 5 — count number of rows with missing values

```python
n_miss = dataframe[[i]].isnull().sum()
	perc = n_miss / dataframe.shape[0] * 100
	print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))
```

---
## Learning Notes / 学习笔记

- **概念**: summarize the horse colic dataset 是机器学习中的常用技术。  
  *summarize the horse colic dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `head()` | 查看前几行数据 | View first few rows |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mark Missing / 缺失值处理
# Complete Code / 完整代码
# ===============================

# summarize the horse colic dataset
from pandas import read_csv
# load dataset
dataframe = read_csv('horse-colic.csv', header=None, na_values='?')
# summarize the first few rows
print(dataframe.head())
# summarize the number of rows with missing values for each column
for i in range(dataframe.shape[1]):
	# count number of rows with missing values
	n_miss = dataframe[[i]].isnull().sum()
	perc = n_miss / dataframe.shape[0] * 100
	print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Impute Missing

# 02 — Impute Missing / 缺失值处理

**Chapter 08 — File 2 of 5 / 第08章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **statistical imputation transform for the horse colic dataset**.

本脚本演示 **statistical imputation transform for the horse colic dataset**。

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
## Step 1 — statistical imputation transform for the horse colic dataset

```python
from numpy import isnan
from pandas import read_csv
from sklearn.impute import SimpleImputer
```

---
## Step 2 — load dataset

```python
dataframe = read_csv('horse-colic.csv', header=None, na_values='?')
```

---
## Step 3 — split into input and output elements

```python
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
```

---
## Step 4 — summarize total missing

```python
print('Missing: %d' % sum(isnan(X).flatten()))
```

---
## Step 5 — define imputer

```python
imputer = SimpleImputer(strategy='mean')
```

---
## Step 6 — fit on the dataset

```python
imputer.fit(X)
```

---
## Step 7 — transform the dataset

```python
Xtrans = imputer.transform(X)
```

---
## Step 8 — summarize total missing

```python
print('Missing: %d' % sum(isnan(Xtrans).flatten()))
```

---
## Learning Notes / 学习笔记

- **概念**: statistical imputation transform for the horse colic dataset 是机器学习中的常用技术。  
  *statistical imputation transform for the horse colic dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Impute Missing / 缺失值处理
# Complete Code / 完整代码
# ===============================

# statistical imputation transform for the horse colic dataset
from numpy import isnan
from pandas import read_csv
from sklearn.impute import SimpleImputer
# load dataset
dataframe = read_csv('horse-colic.csv', header=None, na_values='?')
# split into input and output elements
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
# summarize total missing
print('Missing: %d' % sum(isnan(X).flatten()))
# define imputer
imputer = SimpleImputer(strategy='mean')
# fit on the dataset
imputer.fit(X)
# transform the dataset
Xtrans = imputer.transform(X)
# summarize total missing
print('Missing: %d' % sum(isnan(Xtrans).flatten()))
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Impute Pipeline

# 03 — Impute Pipeline / 管道

**Chapter 08 — File 3 of 5 / 第08章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **evaluate mean imputation and random forest for the horse colic dataset**.

本脚本演示 **evaluate mean imputation and random forest for the horse colic dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
```

---
## Step 1 — evaluate mean imputation and random forest for the horse colic dataset

```python
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
```

---
## Step 2 — load dataset

```python
dataframe = read_csv('horse-colic.csv', header=None, na_values='?')
```

---
## Step 3 — split into input and output elements

```python
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
```

---
## Step 4 — define modeling pipeline

```python
model = RandomForestClassifier()
imputer = SimpleImputer(strategy='mean')
pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
```

---
## Step 5 — define model evaluation

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 6 — evaluate model

```python
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate mean imputation and random forest for the horse colic dataset 是机器学习中的常用技术。  
  *evaluate mean imputation and random forest for the horse colic dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `RandomForestClassifier` | 随机森林分类器 | Random Forest classifier |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Impute Pipeline / 管道
# Complete Code / 完整代码
# ===============================

# evaluate mean imputation and random forest for the horse colic dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
# load dataset
dataframe = read_csv('horse-colic.csv', header=None, na_values='?')
# split into input and output elements
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
# define modeling pipeline
model = RandomForestClassifier()
imputer = SimpleImputer(strategy='mean')
pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Compare Strategies

# 04 — Compare Strategies / 04 Compare Strategies

**Chapter 08 — File 4 of 5 / 第08章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **compare statistical imputation strategies for the horse colic dataset**.

本脚本演示 **compare statistical imputation strategies for the horse colic dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — compare statistical imputation strategies for the horse colic dataset

```python
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
```

---
## Step 2 — load dataset

```python
dataframe = read_csv('horse-colic.csv', header=None, na_values='?')
```

---
## Step 3 — split into input and output elements

```python
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
```

---
## Step 4 — evaluate each strategy on the dataset

```python
results = list()
strategies = ['mean', 'median', 'most_frequent', 'constant']
for s in strategies:
```

---
## Step 5 — create the modeling pipeline

```python
pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)), ('m', RandomForestClassifier())])
```

---
## Step 6 — evaluate the model

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 7 — store results

```python
results.append(scores)
	print('>%s %.3f (%.3f)' % (s, mean(scores), std(scores)))
```

---
## Step 8 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=strategies, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: compare statistical imputation strategies for the horse colic dataset 是机器学习中的常用技术。  
  *compare statistical imputation strategies for the horse colic dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `RandomForestClassifier` | 随机森林分类器 | Random Forest classifier |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Compare Strategies / 04 Compare Strategies
# Complete Code / 完整代码
# ===============================

# compare statistical imputation strategies for the horse colic dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
# load dataset
dataframe = read_csv('horse-colic.csv', header=None, na_values='?')
# split into input and output elements
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
# evaluate each strategy on the dataset
results = list()
strategies = ['mean', 'median', 'most_frequent', 'constant']
for s in strategies:
	# create the modeling pipeline
	pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)), ('m', RandomForestClassifier())])
	# evaluate the model
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# store results
	results.append(scores)
	print('>%s %.3f (%.3f)' % (s, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=strategies, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Predict

# 05 — Predict / 05 Predict

**Chapter 08 — File 5 of 5 / 第08章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **constant imputation strategy and prediction for the horse colic dataset**.

本脚本演示 **constant imputation strategy and prediction for the horse colic dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

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
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — constant imputation strategy and prediction for the horse colic dataset

```python
from numpy import nan
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
```

---
## Step 2 — load dataset

```python
dataframe = read_csv('horse-colic.csv', header=None, na_values='?')
```

---
## Step 3 — split into input and output elements

```python
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
```

---
## Step 4 — create the modeling pipeline

```python
pipeline = Pipeline(steps=[('i', SimpleImputer(strategy='constant')), ('m', RandomForestClassifier())])
```

---
## Step 5 — fit the model

```python
pipeline.fit(X, y)
```

---
## Step 6 — define new data

```python
row = [2, 1, 530101, 38.50, 66, 28, 3, 3, nan, 2, 5, 4, 4, nan, nan, nan, 3, 5, 45.00, 8.40, nan, nan, 2, 11300, 00000, 00000, 2]
```

---
## Step 7 — make a prediction

```python
yhat = pipeline.predict([row])
```

---
## Step 8 — summarize prediction

```python
print('Predicted Class: %d' % yhat[0])
```

---
## Learning Notes / 学习笔记

- **概念**: constant imputation strategy and prediction for the horse colic dataset 是机器学习中的常用技术。  
  *constant imputation strategy and prediction for the horse colic dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `RandomForestClassifier` | 随机森林分类器 | Random Forest classifier |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Predict / 05 Predict
# Complete Code / 完整代码
# ===============================

# constant imputation strategy and prediction for the horse colic dataset
from numpy import nan
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
# load dataset
dataframe = read_csv('horse-colic.csv', header=None, na_values='?')
# split into input and output elements
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
# create the modeling pipeline
pipeline = Pipeline(steps=[('i', SimpleImputer(strategy='constant')), ('m', RandomForestClassifier())])
# fit the model
pipeline.fit(X, y)
# define new data
row = [2, 1, 530101, 38.50, 66, 28, 3, 3, nan, 2, 5, 4, 4, nan, nan, nan, 3, 5, 45.00, 8.40, nan, nan, 2, 11300, 00000, 00000, 2]
# make a prediction
yhat = pipeline.predict([row])
# summarize prediction
print('Predicted Class: %d' % yhat[0])
```

---

### Chapter Summary

# Chapter 08 Summary / 第08章总结

## Theme / 主题: Chapter 08 / Chapter 08

This chapter contains **5 code files** demonstrating chapter 08.

本章包含 **5 个代码文件**，演示Chapter 08。

---
## Evolution / 演化路线

  1. `01_mark_missing.ipynb` — Mark Missing
  2. `02_impute_missing.ipynb` — Impute Missing
  3. `03_impute_pipeline.ipynb` — Impute Pipeline
  4. `04_compare_strategies.ipynb` — Compare Strategies
  5. `05_predict.ipynb` — Predict

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 08) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 08）是机器学习流水线中的基础构建块。

---
