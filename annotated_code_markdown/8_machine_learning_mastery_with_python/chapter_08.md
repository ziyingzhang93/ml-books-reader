# Python 机器学习实战 / ML Mastery with Python
## Chapter 08

---

### Chapter Summary / 章节总结

# Chapter 08 Summary / 第08章总结

## Theme / 主题: Chapter 08 / Chapter 08

This chapter contains **4 code files** demonstrating chapter 08.

本章包含 **4 个代码文件**，演示Chapter 08。

---
## Evolution / 演化路线

  1. `feature_importance.ipynb` — Feature Importance
  2. `pca.ipynb` — Pca
  3. `recursive_feature_elimination.ipynb` — Recursive Feature Elimination
  4. `univariate_selection.ipynb` — Univariate Selection

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 08) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 08）是机器学习流水线中的基础构建块。

---

### Feature Importance

# 01 — Feature Importance / 特征工程

**Chapter 08 — File 1 of 4 / 第08章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Feature Importance with Extra Trees Classifier**.

本脚本演示 **Feature Importance with Extra Trees Classifier**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — Feature Importance with Extra Trees Classifier

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import ExtraTreesClassifier
```

---
## Step 2 — load data

```python
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
```

---
## Step 3 — feature extraction

```python
model = ExtraTreesClassifier(n_estimators=100)
# 训练模型 / Train the model
model.fit(X, Y)
# 打印输出 / Print output
print(model.feature_importances_)
```

---
## Learning Notes / 学习笔记

- **概念**: Feature Importance with Extra Trees Classifier 是机器学习中的常用技术。  
  *Feature Importance with Extra Trees Classifier is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `model.fit` | 训练模型 | Train the model |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Feature Importance / 特征工程
# Complete Code / 完整代码
# ===============================

# Feature Importance with Extra Trees Classifier
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import ExtraTreesClassifier
# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = ExtraTreesClassifier(n_estimators=100)
# 训练模型 / Train the model
model.fit(X, Y)
# 打印输出 / Print output
print(model.feature_importances_)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Pca

# 01 — Pca / Pca

**Chapter 08 — File 2 of 4 / 第08章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Feature Extraction with PCA**.

本脚本演示 **Feature Extraction with PCA**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — Feature Extraction with PCA

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
```

---
## Step 2 — load data

```python
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
```

---
## Step 3 — feature extraction

```python
# 主成分分析：降维，保留最重要的特征 / PCA: reduce dimensions, keep key features
pca = PCA(n_components=3)
fit = pca.fit(X)
```

---
## Step 4 — summarize components

```python
# 打印输出 / Print output
print("Explained Variance: %s" % fit.explained_variance_ratio_)
# 打印输出 / Print output
print(fit.components_)
```

---
## Learning Notes / 学习笔记

- **概念**: Feature Extraction with PCA 是机器学习中的常用技术。  
  *Feature Extraction with PCA is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pca / Pca
# Complete Code / 完整代码
# ===============================

# Feature Extraction with PCA
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
# 主成分分析：降维，保留最重要的特征 / PCA: reduce dimensions, keep key features
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
# 打印输出 / Print output
print("Explained Variance: %s" % fit.explained_variance_ratio_)
# 打印输出 / Print output
print(fit.components_)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Recursive Feature Elimination

# 01 — Recursive Feature Elimination / 特征工程

**Chapter 08 — File 3 of 4 / 第08章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Feature Selection with RFE**.

本脚本演示 **Feature Selection with RFE**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — Feature Selection with RFE

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import RFE
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
```

---
## Step 2 — load data

```python
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
```

---
## Step 3 — feature extraction

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='liblinear')
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
# 打印输出 / Print output
print("Num Features: %d" % fit.n_features_)
# 打印输出 / Print output
print("Selected Features: %s" % fit.support_)
# 打印输出 / Print output
print("Feature Ranking: %s" % fit.ranking_)
```

---
## Learning Notes / 学习笔记

- **概念**: Feature Selection with RFE 是机器学习中的常用技术。  
  *Feature Selection with RFE is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Recursive Feature Elimination / 特征工程
# Complete Code / 完整代码
# ===============================

# Feature Selection with RFE
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import RFE
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='liblinear')
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
# 打印输出 / Print output
print("Num Features: %d" % fit.n_features_)
# 打印输出 / Print output
print("Selected Features: %s" % fit.support_)
# 打印输出 / Print output
print("Feature Ranking: %s" % fit.ranking_)
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Univariate Selection

# 01 — Univariate Selection / 特征选择

**Chapter 08 — File 4 of 4 / 第08章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Feature Selection with Univariate Statistical Tests**.

本脚本演示 **Feature Selection with Univariate Statistical Tests**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
```

---
## Step 1 — Feature Selection with Univariate Statistical Tests

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import set_printoptions
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import f_classif
```

---
## Step 2 — load data

```python
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
```

---
## Step 3 — feature extraction

```python
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)
```

---
## Step 4 — summarize scores

```python
set_printoptions(precision=3)
# 打印输出 / Print output
print(fit.scores_)
# 用已拟合的模型转换数据 / Transform data with fitted model
features = fit.transform(X)
```

---
## Step 5 — summarize selected features

```python
# 打印输出 / Print output
print(features[0:5,:])
```

---
## Learning Notes / 学习笔记

- **概念**: Feature Selection with Univariate Statistical Tests 是机器学习中的常用技术。  
  *Feature Selection with Univariate Statistical Tests is a common technique in machine learning.*

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
# Univariate Selection / 特征选择
# Complete Code / 完整代码
# ===============================

# Feature Selection with Univariate Statistical Tests
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import set_printoptions
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import f_classif
# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
# 打印输出 / Print output
print(fit.scores_)
# 用已拟合的模型转换数据 / Transform data with fitted model
features = fit.transform(X)
# summarize selected features
# 打印输出 / Print output
print(features[0:5,:])
```

---
