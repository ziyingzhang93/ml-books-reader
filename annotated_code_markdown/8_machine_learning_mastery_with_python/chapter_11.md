# Python 机器学习实战 / ML Mastery with Python
## Chapter 11

---

### Chapter Summary / 章节总结

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **6 code files** demonstrating chapter 11.

本章包含 **6 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `classification_and_regression_trees_classification.ipynb` — Classification And Regression Trees Classification
  2. `gaussian_naive_bayes.ipynb` — Gaussian Naive Bayes
  3. `k_nearest_neighbors_classification.ipynb` — K Nearest Neighbors Classification
  4. `linear_discriminant_analysis.ipynb` — Linear Discriminant Analysis
  5. `logistic_regression.ipynb` — Logistic Regression
  6. `support_vector_machines_classification.ipynb` — Support Vector Machines Classification

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---

### Classification And Regression Trees Classification

# 01 — Classification And Regression Trees Classification / 回归

**Chapter 11 — File 1 of 6 / 第11章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **CART Classification**.

本脚本演示 **CART Classification**。

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
## Step 1 — CART Classification

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
model = DecisionTreeClassifier()
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: CART Classification 是机器学习中的常用技术。  
  *CART Classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Classification And Regression Trees Classification / 回归
# Complete Code / 完整代码
# ===============================

# CART Classification
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
model = DecisionTreeClassifier()
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Gaussian Naive Bayes

# 01 — Gaussian Naive Bayes / Gaussian Naive Bayes

**Chapter 11 — File 2 of 6 / 第11章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Gaussian Naive Bayes Classification**.

本脚本演示 **Gaussian Naive Bayes Classification**。

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
## Step 1 — Gaussian Naive Bayes Classification

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = GaussianNB()
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: Gaussian Naive Bayes Classification 是机器学习中的常用技术。  
  *Gaussian Naive Bayes Classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gaussian Naive Bayes / Gaussian Naive Bayes
# Complete Code / 完整代码
# ===============================

# Gaussian Naive Bayes Classification
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = GaussianNB()
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### K Nearest Neighbors Classification

# 01 — K Nearest Neighbors Classification / 分类

**Chapter 11 — File 3 of 6 / 第11章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **KNN Classification**.

本脚本演示 **KNN Classification**。

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
## Step 1 — KNN Classification

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
model = KNeighborsClassifier()
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: KNN Classification 是机器学习中的常用技术。  
  *KNN Classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# K Nearest Neighbors Classification / 分类
# Complete Code / 完整代码
# ===============================

# KNN Classification
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
model = KNeighborsClassifier()
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Linear Discriminant Analysis

# 01 — Linear Discriminant Analysis / 线性模型

**Chapter 11 — File 4 of 6 / 第11章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **LDA Classification**.

本脚本演示 **LDA Classification**。

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
## Step 1 — LDA Classification

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearDiscriminantAnalysis()
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: LDA Classification 是机器学习中的常用技术。  
  *LDA Classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Linear Discriminant Analysis / 线性模型
# Complete Code / 完整代码
# ===============================

# LDA Classification
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearDiscriminantAnalysis()
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Logistic Regression

# 01 — Logistic Regression / 回归

**Chapter 11 — File 5 of 6 / 第11章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Logistic Regression Classification**.

本脚本演示 **Logistic Regression Classification**。

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
## Step 1 — Logistic Regression Classification

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='liblinear')
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: Logistic Regression Classification 是机器学习中的常用技术。  
  *Logistic Regression Classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Logistic Regression / 回归
# Complete Code / 完整代码
# ===============================

# Logistic Regression Classification
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='liblinear')
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Support Vector Machines Classification

# 01 — Support Vector Machines Classification / 分类

**Chapter 11 — File 6 of 6 / 第11章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **SVM Classification**.

本脚本演示 **SVM Classification**。

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
## Step 1 — SVM Classification

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# 支持向量机 / Support Vector Machine
model = SVC()
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: SVM Classification 是机器学习中的常用技术。  
  *SVM Classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `SVM` | 支持向量机 | Support Vector Machine |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Support Vector Machines Classification / 分类
# Complete Code / 完整代码
# ===============================

# SVM Classification
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# 支持向量机 / Support Vector Machine
model = SVC()
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---
