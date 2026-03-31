# Python XGBoost 实战 / XGBoost with Python
## Chapter 14

---

### Chapter Summary / 章节总结



---

### Tune Depth

# 01 — Tune Depth / 超参数调优

**Chapter 14 — File 1 of 3 / 第14章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **XGBoost on Otto dataset, Tune max_depth**.

本脚本演示 **XGBoost on Otto dataset, Tune max_depth**。

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
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — XGBoost on Otto dataset, Tune max_depth

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib
matplotlib.use('Agg')
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv('train.csv')
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
```

---
## Step 3 — split data into X and y

```python
X = dataset[:,0:94]
y = dataset[:,94]
```

---
## Step 4 — encode string class values as integers

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
label_encoded_y = LabelEncoder().fit_transform(y)
```

---
## Step 5 — grid search

```python
model = XGBClassifier()
# 生成整数序列 / Generate integer sequence
max_depth = range(1, 11, 2)
param_grid = dict(max_depth=max_depth)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X, label_encoded_y)
```

---
## Step 6 — summarize results

```python
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
## Step 7 — plot

```python
pyplot.errorbar(max_depth, means, yerr=stds)
pyplot.title("XGBoost max_depth vs Log Loss")
pyplot.xlabel('max_depth')
pyplot.ylabel('Log Loss')
pyplot.savefig('max_depth.png')
```

---
## Learning Notes / 学习笔记

- **概念**: XGBoost on Otto dataset, Tune max_depth 是机器学习中的常用技术。  
  *XGBoost on Otto dataset, Tune max_depth is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `XGBClassifier` | XGBoost分类器 | XGBoost classifier |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Tune Depth / 超参数调优
# Complete Code / 完整代码
# ===============================

# XGBoost on Otto dataset, Tune max_depth
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib
matplotlib.use('Agg')
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv('train.csv')
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
# split data into X and y
X = dataset[:,0:94]
y = dataset[:,94]
# encode string class values as integers
# 将类别标签编码为数字 / Encode categorical labels to numbers
label_encoded_y = LabelEncoder().fit_transform(y)
# grid search
model = XGBClassifier()
# 生成整数序列 / Generate integer sequence
max_depth = range(1, 11, 2)
param_grid = dict(max_depth=max_depth)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X, label_encoded_y)
# summarize results
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
 # 打印输出 / Print output
	print("%f (%f) with: %r" % (mean, stdev, param))
# plot
pyplot.errorbar(max_depth, means, yerr=stds)
pyplot.title("XGBoost max_depth vs Log Loss")
pyplot.xlabel('max_depth')
pyplot.ylabel('Log Loss')
pyplot.savefig('max_depth.png')
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Tune Num Trees And Depth



---

### Tune Trees



---
