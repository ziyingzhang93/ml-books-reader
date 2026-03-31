# Python 机器学习实战 / ML Mastery with Python
## Chapter 13

---

### Chapter Summary / 章节总结



---

### Race Algorithms

# 01 — Race Algorithms / Race Algorithms

**Chapter 13 — File 1 of 1 / 第13章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Compare Algorithms**.

本脚本演示 **Compare Algorithms**。

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
## Step 1 — Compare Algorithms

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
```

---
## Step 2 — load dataset

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
## Step 3 — prepare models

```python
models = []
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
models.append(('LR', LogisticRegression(solver='liblinear')))
# 添加元素到列表末尾 / Append element to list end
models.append(('LDA', LinearDiscriminantAnalysis()))
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
models.append(('KNN', KNeighborsClassifier()))
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
models.append(('CART', DecisionTreeClassifier()))
# 添加元素到列表末尾 / Append element to list end
models.append(('NB', GaussianNB()))
# 支持向量机 / Support Vector Machine
models.append(('SVM', SVC()))
```

---
## Step 4 — evaluate each model in turn

```python
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
 # 添加元素到列表末尾 / Append element to list end
	results.append(cv_results)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 # 打印输出 / Print output
	print(msg)
```

---
## Step 5 — boxplot algorithm comparison

```python
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Compare Algorithms 是机器学习中的常用技术。  
  *Compare Algorithms is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `SVM` | 支持向量机 | Support Vector Machine |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Race Algorithms / Race Algorithms
# Complete Code / 完整代码
# ===============================

# Compare Algorithms
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# load dataset
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# prepare models
models = []
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
models.append(('LR', LogisticRegression(solver='liblinear')))
# 添加元素到列表末尾 / Append element to list end
models.append(('LDA', LinearDiscriminantAnalysis()))
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
models.append(('KNN', KNeighborsClassifier()))
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
models.append(('CART', DecisionTreeClassifier()))
# 添加元素到列表末尾 / Append element to list end
models.append(('NB', GaussianNB()))
# 支持向量机 / Support Vector Machine
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
 # 添加元素到列表末尾 / Append element to list end
	results.append(cv_results)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 # 打印输出 / Print output
	print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
```

---
