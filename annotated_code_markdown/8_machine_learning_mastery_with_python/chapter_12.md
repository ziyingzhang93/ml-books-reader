# Python 机器学习实战 / ML Mastery with Python
## Chapter 12

---

### Chapter Summary / 章节总结



---

### Classification And Regression Trees Regression



---

### Elastic Net

# 01 — Elastic Net / Elastic Net

**Chapter 12 — File 2 of 7 / 第12章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **ElasticNet Regression**.

本脚本演示 **ElasticNet Regression**。

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
## Step 1 — ElasticNet Regression

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import ElasticNet
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, delim_whitespace=True, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = ElasticNet()
scoring = 'neg_mean_squared_error'
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: ElasticNet Regression 是机器学习中的常用技术。  
  *ElasticNet Regression is a common technique in machine learning.*

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
# Elastic Net / Elastic Net
# Complete Code / 完整代码
# ===============================

# ElasticNet Regression
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import ElasticNet
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, delim_whitespace=True, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = ElasticNet()
scoring = 'neg_mean_squared_error'
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# 打印输出 / Print output
print(results.mean())
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### K Nearest Neighbors Regression

# 01 — K Nearest Neighbors Regression / 回归

**Chapter 12 — File 3 of 7 / 第12章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **KNN Regression**.

本脚本演示 **KNN Regression**。

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
## Step 1 — KNN Regression

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsRegressor
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, delim_whitespace=True, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: KNN Regression 是机器学习中的常用技术。  
  *KNN Regression is a common technique in machine learning.*

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
# K Nearest Neighbors Regression / 回归
# Complete Code / 完整代码
# ===============================

# KNN Regression
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsRegressor
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, delim_whitespace=True, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# 打印输出 / Print output
print(results.mean())
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Lasso Regression

# 01 — Lasso Regression / 回归

**Chapter 12 — File 4 of 7 / 第12章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Lasso Regression**.

本脚本演示 **Lasso Regression**。

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
## Step 1 — Lasso Regression

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, delim_whitespace=True, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = Lasso()
scoring = 'neg_mean_squared_error'
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: Lasso Regression 是机器学习中的常用技术。  
  *Lasso Regression is a common technique in machine learning.*

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
# Lasso Regression / 回归
# Complete Code / 完整代码
# ===============================

# Lasso Regression
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, delim_whitespace=True, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = Lasso()
scoring = 'neg_mean_squared_error'
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# 打印输出 / Print output
print(results.mean())
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Linear Regression

# 01 — Linear Regression / 线性模型

**Chapter 12 — File 5 of 7 / 第12章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Linear Regression**.

本脚本演示 **Linear Regression**。

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
## Step 1 — Linear Regression

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, delim_whitespace=True, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: Linear Regression 是机器学习中的常用技术。  
  *Linear Regression is a common technique in machine learning.*

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
# Linear Regression / 线性模型
# Complete Code / 完整代码
# ===============================

# Linear Regression
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, delim_whitespace=True, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# 打印输出 / Print output
print(results.mean())
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Ridge Regression

# 01 — Ridge Regression / 回归

**Chapter 12 — File 6 of 7 / 第12章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Ridge Regression**.

本脚本演示 **Ridge Regression**。

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
## Step 1 — Ridge Regression

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Ridge
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, delim_whitespace=True, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = Ridge()
scoring = 'neg_mean_squared_error'
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: Ridge Regression 是机器学习中的常用技术。  
  *Ridge Regression is a common technique in machine learning.*

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
# Ridge Regression / 回归
# Complete Code / 完整代码
# ===============================

# Ridge Regression
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Ridge
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, delim_whitespace=True, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = Ridge()
scoring = 'neg_mean_squared_error'
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# 打印输出 / Print output
print(results.mean())
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Support Vector Machines Regression

# 01 — Support Vector Machines Regression / 回归

**Chapter 12 — File 7 of 7 / 第12章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **SVM Regression**.

本脚本演示 **SVM Regression**。

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
## Step 1 — SVM Regression

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, delim_whitespace=True, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# 支持向量机 / Support Vector Machine
model = SVR(gamma='auto')
scoring = 'neg_mean_squared_error'
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: SVM Regression 是机器学习中的常用技术。  
  *SVM Regression is a common technique in machine learning.*

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
# Support Vector Machines Regression / 回归
# Complete Code / 完整代码
# ===============================

# SVM Regression
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, delim_whitespace=True, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# 支持向量机 / Support Vector Machine
model = SVR(gamma='auto')
scoring = 'neg_mean_squared_error'
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# 打印输出 / Print output
print(results.mean())
```

---
