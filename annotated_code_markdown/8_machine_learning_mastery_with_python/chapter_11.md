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
## Step 1 — CART Classification

```python
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = DecisionTreeClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: CART Classification 是机器学习中的常用技术。  
  *CART Classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Classification And Regression Trees Classification / 回归
# Complete Code / 完整代码
# ===============================

# CART Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = DecisionTreeClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
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
## Step 1 — Gaussian Naive Bayes Classification

```python
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = GaussianNB()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: Gaussian Naive Bayes Classification 是机器学习中的常用技术。  
  *Gaussian Naive Bayes Classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gaussian Naive Bayes / Gaussian Naive Bayes
# Complete Code / 完整代码
# ===============================

# Gaussian Naive Bayes Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = GaussianNB()
results = cross_val_score(model, X, Y, cv=kfold)
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
## Step 1 — KNN Classification

```python
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: KNN Classification 是机器学习中的常用技术。  
  *KNN Classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# K Nearest Neighbors Classification / 分类
# Complete Code / 完整代码
# ===============================

# KNN Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
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
## Step 1 — LDA Classification

```python
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: LDA Classification 是机器学习中的常用技术。  
  *LDA Classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Linear Discriminant Analysis / 线性模型
# Complete Code / 完整代码
# ===============================

# LDA Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, Y, cv=kfold)
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
## Step 1 — Logistic Regression Classification

```python
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: Logistic Regression Classification 是机器学习中的常用技术。  
  *Logistic Regression Classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Logistic Regression / 回归
# Complete Code / 完整代码
# ===============================

# Logistic Regression Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, Y, cv=kfold)
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
## Step 1 — SVM Classification

```python
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = SVC()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: SVM Classification 是机器学习中的常用技术。  
  *SVM Classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Support Vector Machines Classification / 分类
# Complete Code / 完整代码
# ===============================

# SVM Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = SVC()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

---
