# 机器学习数据准备 / Data Preparation for ML
## Chapter 24

---

### Load Dataset

# 01 — Load Dataset / 01 Load Dataset

**Chapter 24 — File 1 of 2 / 第24章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **load the dataset**.

本脚本演示 **load the dataset**。

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
## Step 1 — load the dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('abalone.csv', header=None)
```

---
## Step 3 — split into inputs and outputs

```python
# 获取列名 / Get column names
last_ix = len(dataframe.columns) - 1
# 删除指定列或行 / Drop specified columns or rows
X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: load the dataset 是机器学习中的常用技术。  
  *load the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Dataset / 01 Load Dataset
# Complete Code / 完整代码
# ===============================

# load the dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('abalone.csv', header=None)
# split into inputs and outputs
# 获取列名 / Get column names
last_ix = len(dataframe.columns) - 1
# 删除指定列或行 / Drop specified columns or rows
X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Evaluate Model With Transforms

# 02 — Evaluate Model With Transforms / 数据变换

**Chapter 24 — File 2 of 2 / 第24章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **example of using the ColumnTransformer for the Abalone dataset**.

本脚本演示 **example of using the ColumnTransformer for the Abalone dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
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
## Step 1 — example of using the ColumnTransformer for the Abalone dataset

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import absolute
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('abalone.csv', header=None)
```

---
## Step 3 — split into inputs and outputs

```python
# 获取列名 / Get column names
last_ix = len(dataframe.columns) - 1
# 删除指定列或行 / Drop specified columns or rows
X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---
## Step 4 — determine categorical and numerical features

```python
# 获取列名 / Get column names
numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
# 获取列名 / Get column names
categorical_ix = X.select_dtypes(include=['object', 'bool']).columns
```

---
## Step 5 — define the data preparation for the columns

```python
# 归一化到[0,1]范围 / Normalize to [0,1] range
t = [('cat', OneHotEncoder(), categorical_ix), ('num', MinMaxScaler(), numerical_ix)]
col_transform = ColumnTransformer(transformers=t)
```

---
## Step 6 — define the model

```python
# 支持向量机 / Support Vector Machine
model = SVR(kernel='rbf',gamma='scale',C=100)
```

---
## Step 7 — define the data preparation and modeling pipeline

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])
```

---
## Step 8 — define the model cross-validation configuration

```python
cv = KFold(n_splits=10, shuffle=True, random_state=1)
```

---
## Step 9 — evaluate the pipeline using cross validation and calculate MAE

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
```

---
## Step 10 — convert MAE scores to positive values

```python
scores = absolute(scores)
```

---
## Step 11 — summarize the model performance

```python
# 打印输出 / Print output
print('MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: example of using the ColumnTransformer for the Abalone dataset 是机器学习中的常用技术。  
  *example of using the ColumnTransformer for the Abalone dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `SVM` | 支持向量机 | Support Vector Machine |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Model With Transforms / 数据变换
# Complete Code / 完整代码
# ===============================

# example of using the ColumnTransformer for the Abalone dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import absolute
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('abalone.csv', header=None)
# split into inputs and outputs
# 获取列名 / Get column names
last_ix = len(dataframe.columns) - 1
# 删除指定列或行 / Drop specified columns or rows
X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
# determine categorical and numerical features
# 获取列名 / Get column names
numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
# 获取列名 / Get column names
categorical_ix = X.select_dtypes(include=['object', 'bool']).columns
# define the data preparation for the columns
# 归一化到[0,1]范围 / Normalize to [0,1] range
t = [('cat', OneHotEncoder(), categorical_ix), ('num', MinMaxScaler(), numerical_ix)]
col_transform = ColumnTransformer(transformers=t)
# define the model
# 支持向量机 / Support Vector Machine
model = SVR(kernel='rbf',gamma='scale',C=100)
# define the data preparation and modeling pipeline
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])
# define the model cross-validation configuration
cv = KFold(n_splits=10, shuffle=True, random_state=1)
# evaluate the pipeline using cross validation and calculate MAE
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# convert MAE scores to positive values
scores = absolute(scores)
# summarize the model performance
# 打印输出 / Print output
print('MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---

### Chapter Summary / 章节总结

# Chapter 24 Summary / 第24章总结

## Theme / 主题: Chapter 24 / Chapter 24

This chapter contains **2 code files** demonstrating chapter 24.

本章包含 **2 个代码文件**，演示Chapter 24。

---
## Evolution / 演化路线

  1. `01_load_dataset.ipynb` — Load Dataset
  2. `02_evaluate_model_with_transforms.ipynb` — Evaluate Model With Transforms

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 24) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 24）是机器学习流水线中的基础构建块。

---
