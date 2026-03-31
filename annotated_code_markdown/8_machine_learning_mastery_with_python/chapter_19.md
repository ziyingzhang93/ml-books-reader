# Python 机器学习实战 / ML Mastery with Python
## Chapter 19

---

### Chapter Summary / 章节总结



---

### Project Classification Iris

# 01 — Project Classification Iris / 分类

**Chapter 19 — File 1 of 1 / 第19章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Hello World Classification: Iris flowers prediction**.

本脚本演示 **Hello World Classification: Iris flowers prediction**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
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
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Hello World Classification: Iris flowers prediction
Prepare Problem
Load libraries

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import scatter_matrix
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import classification_report
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import confusion_matrix
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
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
## Step 2 — Load dataset

```python
filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv(filename, names=names)
```

---
## Step 3 — Summarize Data
Descriptive statistics
shape

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
```

---
## Step 4 — head

```python
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataset.head(20))
```

---
## Step 5 — descriptions

```python
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(dataset.describe())
```

---
## Step 6 — class distribution

```python
# 打印输出 / Print output
print(dataset.groupby('class').size())
```

---
## Step 7 — Data visualizations
box and whisker plots

```python
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
```

---
## Step 8 — histograms

```python
dataset.hist()
pyplot.show()
```

---
## Step 9 — scatter plot matrix

```python
scatter_matrix(dataset)
pyplot.show()
```

---
## Step 10 — Prepare Data
Split-out validation dataset

```python
# 转换为NumPy数组 / Convert to NumPy array
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
# 划分训练集和测试集 / Split into train and test sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
```

---
## Step 11 — Spot-Check Algorithms

```python
models = []
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# 添加元素到列表末尾 / Append element to list end
models.append(('LDA', LinearDiscriminantAnalysis()))
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
models.append(('KNN', KNeighborsClassifier()))
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
models.append(('CART', DecisionTreeClassifier()))
# 添加元素到列表末尾 / Append element to list end
models.append(('NB', GaussianNB()))
# 支持向量机 / Support Vector Machine
models.append(('SVM', SVC(gamma='auto')))
```

---
## Step 12 — evaluate each model in turn

```python
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
 # 添加元素到列表末尾 / Append element to list end
	results.append(cv_results)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 # 打印输出 / Print output
	print(msg)
```

---
## Step 13 — Compare Algorithms

```python
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
```

---
## Step 14 — Make predictions on validation dataset

```python
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
print(accuracy_score(Y_validation, predictions))
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
print(confusion_matrix(Y_validation, predictions))
# 生成分类报告：精确率/召回率/F1 / Classification report: precision/recall/F1
print(classification_report(Y_validation, predictions))
```

---
## Learning Notes / 学习笔记

- **概念**: Hello World Classification: Iris flowers prediction 是机器学习中的常用技术。  
  *Hello World Classification: Iris flowers prediction is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `SVM` | 支持向量机 | Support Vector Machine |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `classification_report` | 分类报告：精确率/召回率/F1 | Classification report: precision/recall/F1 |
| `confusion_matrix` | 混淆矩阵：展示预测对错分布 | Confusion matrix: prediction error distribution |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `describe()` | 统计摘要信息 | Statistical summary |
| `groupby` | 分组聚合 | Group and aggregate |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Project Classification Iris / 分类
# Complete Code / 完整代码
# ===============================

# Hello World Classification: Iris flowers prediction

# Prepare Problem

# Load libraries
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import scatter_matrix
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import classification_report
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import confusion_matrix
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
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

# Load dataset
filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv(filename, names=names)

# Summarize Data

# Descriptive statistics
# shape
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
# head
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataset.head(20))
# descriptions
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(dataset.describe())
# class distribution
# 打印输出 / Print output
print(dataset.groupby('class').size())

# Data visualizations

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
# histograms
dataset.hist()
pyplot.show()
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# Prepare Data

# Split-out validation dataset
# 转换为NumPy数组 / Convert to NumPy array
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
# 划分训练集和测试集 / Split into train and test sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Spot-Check Algorithms
models = []
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# 添加元素到列表末尾 / Append element to list end
models.append(('LDA', LinearDiscriminantAnalysis()))
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
models.append(('KNN', KNeighborsClassifier()))
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
models.append(('CART', DecisionTreeClassifier()))
# 添加元素到列表末尾 / Append element to list end
models.append(('NB', GaussianNB()))
# 支持向量机 / Support Vector Machine
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
 # 添加元素到列表末尾 / Append element to list end
	results.append(cv_results)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 # 打印输出 / Print output
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Make predictions on validation dataset
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
print(accuracy_score(Y_validation, predictions))
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
print(confusion_matrix(Y_validation, predictions))
# 生成分类报告：精确率/召回率/F1 / Classification report: precision/recall/F1
print(classification_report(Y_validation, predictions))
```

---
