# Python 机器学习实战 / ML Mastery with Python
## Chapter 21

---

### Chapter Summary / 章节总结

# Chapter 21 Summary / 第21章总结

## Theme / 主题: Chapter 21 / Chapter 21

This chapter contains **1 code files** demonstrating chapter 21.

本章包含 **1 个代码文件**，演示Chapter 21。

---
## Evolution / 演化路线

  1. `project_classification_sonar.ipynb` — Project Classification Sonar

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 21) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 21）是机器学习流水线中的基础构建块。

---

### Project Classification Sonar

# 01 — Project Classification Sonar / 分类

**Chapter 21 — File 1 of 1 / 第21章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Classification Project: Sonar rocks or mines**.

本脚本演示 **Classification Project: Sonar rocks or mines**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — Classification Project: Sonar rocks or mines
Load libraries

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import set_option
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import scatter_matrix
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import classification_report
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import confusion_matrix
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import GradientBoostingClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import RandomForestClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import ExtraTreesClassifier
```

---
## Step 2 — Load dataset

```python
url = 'sonar.all-data.csv'
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv(url, header=None)
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
## Step 4 — types

```python
set_option('display.max_rows', 500)
# 打印输出 / Print output
print(dataset.dtypes)
```

---
## Step 5 — head

```python
set_option('display.width', 100)
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataset.head(20))
```

---
## Step 6 — descriptions, change precision to 3 places

```python
set_option('precision', 3)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(dataset.describe())
```

---
## Step 7 — class distribution

```python
# 打印输出 / Print output
print(dataset.groupby(60).size())
```

---
## Step 8 — Data visualizations
histograms

```python
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()
```

---
## Step 9 — density

```python
dataset.plot(kind='density', subplots=True, layout=(8,8), sharex=False, legend=False, fontsize=1)
pyplot.show()
```

---
## Step 10 — correlation matrix

```python
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
pyplot.show()
```

---
## Step 11 — Prepare Data
Split-out validation dataset

```python
# 转换为NumPy数组 / Convert to NumPy array
array = dataset.values
# 转换数据类型 / Convert data type
X = array[:,0:60].astype(float)
Y = array[:,60]
validation_size = 0.20
seed = 7
# 划分训练集和测试集 / Split into train and test sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
```

---
## Step 12 — Evaluate Algorithms
Test options and evaluation metric

```python
num_folds = 10
seed = 7
scoring = 'accuracy'
```

---
## Step 13 — Spot Check Algorithms

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
models.append(('SVM', SVC(gamma='auto')))
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
 # 添加元素到列表末尾 / Append element to list end
	results.append(cv_results)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 # 打印输出 / Print output
	print(msg)
```

---
## Step 14 — Compare Algorithms

```python
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
```

---
## Step 15 — Standardize the dataset

```python
pipelines = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression(solver='liblinear'))])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC(gamma='auto'))])))
results = []
names = []
for name, model in pipelines:
	kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
 # 添加元素到列表末尾 / Append element to list end
	results.append(cv_results)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 # 打印输出 / Print output
	print(msg)
```

---
## Step 16 — Compare Algorithms

```python
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
```

---
## Step 17 — Tune scaled KNN

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler().fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledX = scaler.transform(X_train)
neighbors = [1,3,5,7,9,11,13,15,17,19,21]
param_grid = dict(n_neighbors=neighbors)
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
model = KNeighborsClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---
## Step 18 — Tune scaled SVM

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler().fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
# 支持向量机 / Support Vector Machine
model = SVC(gamma='auto')
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---
## Step 19 — ensembles

```python
ensembles = []
# 添加元素到列表末尾 / Append element to list end
ensembles.append(('AB', AdaBoostClassifier()))
# 梯度提升：逐步修正前一棵树的错误 / Gradient Boosting: iteratively correct errors
ensembles.append(('GBM', GradientBoostingClassifier()))
# 随机森林：多棵决策树投票 / Random Forest: multiple trees vote
ensembles.append(('RF', RandomForestClassifier(n_estimators=10)))
# 添加元素到列表末尾 / Append element to list end
ensembles.append(('ET', ExtraTreesClassifier(n_estimators=10)))
results = []
names = []
for name, model in ensembles:
	kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
 # 添加元素到列表末尾 / Append element to list end
	results.append(cv_results)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 # 打印输出 / Print output
	print(msg)
```

---
## Step 20 — Compare Algorithms

```python
fig = pyplot.figure()
fig.suptitle('Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
```

---
## Step 21 — Finalize Model
prepare the model

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler().fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledX = scaler.transform(X_train)
# 支持向量机 / Support Vector Machine
model = SVC(gamma='auto', C=1.5)
# 训练模型 / Train the model
model.fit(rescaledX, Y_train)
```

---
## Step 22 — estimate accuracy on validation dataset

```python
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledValidationX = scaler.transform(X_validation)
# 用模型做预测 / Make predictions with model
predictions = model.predict(rescaledValidationX)
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
print(accuracy_score(Y_validation, predictions))
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
print(confusion_matrix(Y_validation, predictions))
# 生成分类报告：精确率/召回率/F1 / Classification report: precision/recall/F1
print(classification_report(Y_validation, predictions))
```

---
## Learning Notes / 学习笔记

- **概念**: Classification Project: Sonar rocks or mines 是机器学习中的常用技术。  
  *Classification Project: Sonar rocks or mines is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `GradientBoosting` | 梯度提升算法 | Gradient Boosting algorithm |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `RandomForestClassifier` | 随机森林分类器 | Random Forest classifier |
| `SVM` | 支持向量机 | Support Vector Machine |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `classification_report` | 分类报告：精确率/召回率/F1 | Classification report: precision/recall/F1 |
| `confusion_matrix` | 混淆矩阵：展示预测对错分布 | Confusion matrix: prediction error distribution |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `describe()` | 统计摘要信息 | Statistical summary |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `groupby` | 分组聚合 | Group and aggregate |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Project Classification Sonar / 分类
# Complete Code / 完整代码
# ===============================

# Classification Project: Sonar rocks or mines

# Load libraries
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import set_option
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import scatter_matrix
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import classification_report
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import confusion_matrix
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import GradientBoostingClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import RandomForestClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import ExtraTreesClassifier

# Load dataset
url = 'sonar.all-data.csv'
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv(url, header=None)


# Summarize Data

# Descriptive statistics
# shape
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
# types
set_option('display.max_rows', 500)
# 打印输出 / Print output
print(dataset.dtypes)
# head
set_option('display.width', 100)
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataset.head(20))
# descriptions, change precision to 3 places
set_option('precision', 3)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(dataset.describe())
# class distribution
# 打印输出 / Print output
print(dataset.groupby(60).size())


# Data visualizations

# histograms
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()
# density
dataset.plot(kind='density', subplots=True, layout=(8,8), sharex=False, legend=False, fontsize=1)
pyplot.show()

# correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
pyplot.show()

# Prepare Data

# Split-out validation dataset
# 转换为NumPy数组 / Convert to NumPy array
array = dataset.values
# 转换数据类型 / Convert data type
X = array[:,0:60].astype(float)
Y = array[:,60]
validation_size = 0.20
seed = 7
# 划分训练集和测试集 / Split into train and test sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Evaluate Algorithms

# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
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
models.append(('SVM', SVC(gamma='auto')))
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
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


# Standardize the dataset
pipelines = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression(solver='liblinear'))])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC(gamma='auto'))])))
results = []
names = []
for name, model in pipelines:
	kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
 # 添加元素到列表末尾 / Append element to list end
	results.append(cv_results)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 # 打印输出 / Print output
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# Tune scaled KNN
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler().fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledX = scaler.transform(X_train)
neighbors = [1,3,5,7,9,11,13,15,17,19,21]
param_grid = dict(n_neighbors=neighbors)
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
model = KNeighborsClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))


# Tune scaled SVM
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler().fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
# 支持向量机 / Support Vector Machine
model = SVC(gamma='auto')
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))




# ensembles
ensembles = []
# 添加元素到列表末尾 / Append element to list end
ensembles.append(('AB', AdaBoostClassifier()))
# 梯度提升：逐步修正前一棵树的错误 / Gradient Boosting: iteratively correct errors
ensembles.append(('GBM', GradientBoostingClassifier()))
# 随机森林：多棵决策树投票 / Random Forest: multiple trees vote
ensembles.append(('RF', RandomForestClassifier(n_estimators=10)))
# 添加元素到列表末尾 / Append element to list end
ensembles.append(('ET', ExtraTreesClassifier(n_estimators=10)))
results = []
names = []
for name, model in ensembles:
	kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
 # 添加元素到列表末尾 / Append element to list end
	results.append(cv_results)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 # 打印输出 / Print output
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()



# Finalize Model

# prepare the model
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler().fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledX = scaler.transform(X_train)
# 支持向量机 / Support Vector Machine
model = SVC(gamma='auto', C=1.5)
# 训练模型 / Train the model
model.fit(rescaledX, Y_train)
# estimate accuracy on validation dataset
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledValidationX = scaler.transform(X_validation)
# 用模型做预测 / Make predictions with model
predictions = model.predict(rescaledValidationX)
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
print(accuracy_score(Y_validation, predictions))
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
print(confusion_matrix(Y_validation, predictions))
# 生成分类报告：精确率/召回率/F1 / Classification report: precision/recall/F1
print(classification_report(Y_validation, predictions))
```

---
