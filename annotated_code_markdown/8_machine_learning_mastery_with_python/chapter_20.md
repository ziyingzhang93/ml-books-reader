# Python 机器学习实战 / ML Mastery with Python
## Chapter 20

---

### Chapter Summary / 章节总结

# Chapter 20 Summary / 第20章总结

## Theme / 主题: Chapter 20 / Chapter 20

This chapter contains **1 code files** demonstrating chapter 20.

本章包含 **1 个代码文件**，演示Chapter 20。

---
## Evolution / 演化路线

  1. `project_regression_boston.ipynb` — Project Regression Boston

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 20) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 20）是机器学习流水线中的基础构建块。

---

### Project Regression Boston

# 01 — Project Regression Boston / 回归

**Chapter 20 — File 1 of 1 / 第20章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Regression Project: Boston House Prices**.

本脚本演示 **Regression Project: Boston House Prices**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — Regression Project: Boston House Prices
Load libraries

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
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
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import RandomForestRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import GradientBoostingRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import ExtraTreesRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
```

---
## Step 2 — Load dataset

```python
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv(filename, delim_whitespace=True, names=names)
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
# 打印输出 / Print output
print(dataset.dtypes)
```

---
## Step 5 — head

```python
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataset.head(20))
```

---
## Step 6 — descriptions, change precision to 2 places

```python
set_option('precision', 1)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(dataset.describe())
```

---
## Step 7 — correlation

```python
set_option('precision', 2)
# 打印输出 / Print output
print(dataset.corr(method='pearson'))
```

---
## Step 8 — Data visualizations
histograms

```python
dataset.hist()
pyplot.show()
```

---
## Step 9 — density

```python
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False)
pyplot.show()
```

---
## Step 10 — box and whisker plots

```python
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
pyplot.show()
```

---
## Step 11 — scatter plot matrix

```python
scatter_matrix(dataset)
pyplot.show()
```

---
## Step 12 — correlation matrix

```python
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
# 生成整数序列 / Generate integer sequence
ticks = arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()
```

---
## Step 13 — Prepare Data
Split-out validation dataset

```python
# 转换为NumPy数组 / Convert to NumPy array
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
validation_size = 0.20
seed = 7
# 划分训练集和测试集 / Split into train and test sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
```

---
## Step 14 — Evaluate Algorithms
Test options and evaluation metric

```python
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'
```

---
## Step 15 — Spot Check Algorithms

```python
models = []
# 添加元素到列表末尾 / Append element to list end
models.append(('LR', LinearRegression()))
# 添加元素到列表末尾 / Append element to list end
models.append(('LASSO', Lasso()))
# 添加元素到列表末尾 / Append element to list end
models.append(('EN', ElasticNet()))
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
models.append(('KNN', KNeighborsRegressor()))
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
models.append(('CART', DecisionTreeRegressor()))
# 支持向量机 / Support Vector Machine
models.append(('SVR', SVR(gamma='auto')))
```

---
## Step 16 — evaluate each model in turn

```python
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
## Step 17 — Compare Algorithms

```python
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
```

---
## Step 18 — Standardize the dataset

```python
pipelines = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR(gamma='auto'))])))
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
## Step 19 — Compare Algorithms

```python
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
```

---
## Step 20 — KNN Algorithm tuning

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler().fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
model = KNeighborsRegressor()
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
## Step 21 — ensembles

```python
ensembles = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor(n_estimators=10))])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor(n_estimators=10))])))
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
## Step 22 — Compare Algorithms

```python
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
```

---
## Step 23 — Tune scaled GBM

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler().fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
# 梯度提升：逐步修正前一棵树的错误 / Gradient Boosting: iteratively correct errors
model = GradientBoostingRegressor(random_state=seed)
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
## Step 24 — Make predictions on validation dataset
prepare the model

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler().fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledX = scaler.transform(X_train)
# 梯度提升：逐步修正前一棵树的错误 / Gradient Boosting: iteratively correct errors
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
# 训练模型 / Train the model
model.fit(rescaledX, Y_train)
```

---
## Step 25 — transform the validation dataset

```python
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledValidationX = scaler.transform(X_validation)
# 用模型做预测 / Make predictions with model
predictions = model.predict(rescaledValidationX)
# 计算均方误差 / Calculate Mean Squared Error
print(mean_squared_error(Y_validation, predictions))
```

---
## Learning Notes / 学习笔记

- **概念**: Regression Project: Boston House Prices 是机器学习中的常用技术。  
  *Regression Project: Boston House Prices is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `GradientBoosting` | 梯度提升算法 | Gradient Boosting algorithm |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `SVM` | 支持向量机 | Support Vector Machine |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `describe()` | 统计摘要信息 | Statistical summary |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Project Regression Boston / 回归
# Complete Code / 完整代码
# ===============================

# Regression Project: Boston House Prices

# Load libraries
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
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
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import RandomForestRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import GradientBoostingRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import ExtraTreesRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error

# Load dataset
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv(filename, delim_whitespace=True, names=names)

# Summarize Data

# Descriptive statistics
# shape
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
# types
# 打印输出 / Print output
print(dataset.dtypes)
# head
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataset.head(20))
# descriptions, change precision to 2 places
set_option('precision', 1)
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(dataset.describe())
# correlation
set_option('precision', 2)
# 打印输出 / Print output
print(dataset.corr(method='pearson'))


# Data visualizations

# histograms
dataset.hist()
pyplot.show()
# density
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False)
pyplot.show()
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()
# correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
# 生成整数序列 / Generate integer sequence
ticks = arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# Prepare Data

# Split-out validation dataset
# 转换为NumPy数组 / Convert to NumPy array
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
validation_size = 0.20
seed = 7
# 划分训练集和测试集 / Split into train and test sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Evaluate Algorithms
# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

# Spot Check Algorithms
models = []
# 添加元素到列表末尾 / Append element to list end
models.append(('LR', LinearRegression()))
# 添加元素到列表末尾 / Append element to list end
models.append(('LASSO', Lasso()))
# 添加元素到列表末尾 / Append element to list end
models.append(('EN', ElasticNet()))
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
models.append(('KNN', KNeighborsRegressor()))
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
models.append(('CART', DecisionTreeRegressor()))
# 支持向量机 / Support Vector Machine
models.append(('SVR', SVR(gamma='auto')))

# evaluate each model in turn
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
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR(gamma='auto'))])))
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


# KNN Algorithm tuning
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler().fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
model = KNeighborsRegressor()
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
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor(n_estimators=10))])))
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor(n_estimators=10))])))
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
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# Tune scaled GBM
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler().fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
# 梯度提升：逐步修正前一棵树的错误 / Gradient Boosting: iteratively correct errors
model = GradientBoostingRegressor(random_state=seed)
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


# Make predictions on validation dataset

# prepare the model
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler().fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledX = scaler.transform(X_train)
# 梯度提升：逐步修正前一棵树的错误 / Gradient Boosting: iteratively correct errors
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
# 训练模型 / Train the model
model.fit(rescaledX, Y_train)
# transform the validation dataset
# 用已拟合的模型转换数据 / Transform data with fitted model
rescaledValidationX = scaler.transform(X_validation)
# 用模型做预测 / Make predictions with model
predictions = model.predict(rescaledValidationX)
# 计算均方误差 / Calculate Mean Squared Error
print(mean_squared_error(Y_validation, predictions))
```

---
