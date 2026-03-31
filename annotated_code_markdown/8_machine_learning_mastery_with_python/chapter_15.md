# Python 机器学习实战 / ML Mastery with Python
## Chapter 15

---

### Chapter Summary / 章节总结



---

### Adaboost Classification

# 01 — Adaboost Classification / 分类

**Chapter 15 — File 1 of 6 / 第15章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **AdaBoost Classification**.

本脚本演示 **AdaBoost Classification**。

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
## Step 1 — AdaBoost Classification

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_trees = 30
seed=7
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: AdaBoost Classification 是机器学习中的常用技术。  
  *AdaBoost Classification is a common technique in machine learning.*

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
# Adaboost Classification / 分类
# Complete Code / 完整代码
# ===============================

# AdaBoost Classification
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_trees = 30
seed=7
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Bagged Cart Classification

# 01 — Bagged Cart Classification / 分类

**Chapter 15 — File 2 of 6 / 第15章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Bagged Decision Trees for Classification**.

本脚本演示 **Bagged Decision Trees for Classification**。

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
## Step 1 — Bagged Decision Trees for Classification

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import BaggingClassifier
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
seed = 7
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: Bagged Decision Trees for Classification 是机器学习中的常用技术。  
  *Bagged Decision Trees for Classification is a common technique in machine learning.*

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
# Bagged Cart Classification / 分类
# Complete Code / 完整代码
# ===============================

# Bagged Decision Trees for Classification
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import BaggingClassifier
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
seed = 7
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Extra Trees Classification



---

### Gradient Boosting Classification



---

### Random Forest Classification



---

### Voting Ensemble Classification

# 01 — Voting Ensemble Classification / 分类

**Chapter 15 — File 6 of 6 / 第15章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Voting Ensemble for Classification**.

本脚本演示 **Voting Ensemble for Classification**。

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
## Step 1 — Voting Ensemble for Classification

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import VotingClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
```

---
## Step 2 — create the sub models

```python
estimators = []
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model1 = LogisticRegression(solver='liblinear')
# 添加元素到列表末尾 / Append element to list end
estimators.append(('logistic', model1))
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
model2 = DecisionTreeClassifier()
# 添加元素到列表末尾 / Append element to list end
estimators.append(('cart', model2))
# 支持向量机 / Support Vector Machine
model3 = SVC(gamma='auto')
# 添加元素到列表末尾 / Append element to list end
estimators.append(('svm', model3))
```

---
## Step 3 — create the ensemble model

```python
ensemble = VotingClassifier(estimators)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(ensemble, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: Voting Ensemble for Classification 是机器学习中的常用技术。  
  *Voting Ensemble for Classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `SVM` | 支持向量机 | Support Vector Machine |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Voting Ensemble Classification / 分类
# Complete Code / 完整代码
# ===============================

# Voting Ensemble for Classification
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import VotingClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# create the sub models
estimators = []
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model1 = LogisticRegression(solver='liblinear')
# 添加元素到列表末尾 / Append element to list end
estimators.append(('logistic', model1))
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
model2 = DecisionTreeClassifier()
# 添加元素到列表末尾 / Append element to list end
estimators.append(('cart', model2))
# 支持向量机 / Support Vector Machine
model3 = SVC(gamma='auto')
# 添加元素到列表末尾 / Append element to list end
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(ensemble, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---
