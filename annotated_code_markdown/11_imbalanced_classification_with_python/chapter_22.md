# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 22

---

### Svm Uncalibrated



---

### Svm Calibrated



---

### Svm Balanced Calibrated



---

### Decision Tree Uncalibrated



---

### Decision Tree Calibrated



---

### Knn Uncalibrated



---

### Knn Grid Search

# 07 — Knn Grid Search / 07 Knn Grid Search

**Chapter 22 — File 7 of 7 / 第22章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **grid search probability calibration with knn for imbalanced classification**.

本脚本演示 **grid search probability calibration with knn for imbalanced classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏋️ 训练模型 / Train Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — grid search probability calibration with knn for imbalanced classification

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.calibration import CalibratedClassifierCV
```

---
## Step 2 — generate dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
```

---
## Step 3 — define model

```python
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
model = KNeighborsClassifier()
```

---
## Step 4 — wrap the model

```python
calibrated = CalibratedClassifierCV(model)
```

---
## Step 5 — define grid

```python
param_grid = dict(cv=[2,3,4], method=['sigmoid','isotonic'])
```

---
## Step 6 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 7 — define grid search

```python
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=calibrated, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
```

---
## Step 8 — execute the grid search

```python
grid_result = grid.fit(X, y)
```

---
## Step 9 — report the best configuration

```python
# 打印输出 / Print output
print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
```

---
## Step 10 — report all configurations

```python
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print('%f (%f) with: %r' % (mean, stdev, param))
```

---
## Learning Notes / 学习笔记

- **概念**: grid search probability calibration with knn for imbalanced classification 是机器学习中的常用技术。  
  *grid search probability calibration with knn for imbalanced classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Knn Grid Search / 07 Knn Grid Search
# Complete Code / 完整代码
# ===============================

# grid search probability calibration with knn for imbalanced classification
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.calibration import CalibratedClassifierCV
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
model = KNeighborsClassifier()
# wrap the model
calibrated = CalibratedClassifierCV(model)
# define grid
param_grid = dict(cv=[2,3,4], method=['sigmoid','isotonic'])
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=calibrated, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
# execute the grid search
grid_result = grid.fit(X, y)
# report the best configuration
# 打印输出 / Print output
print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
# report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print('%f (%f) with: %r' % (mean, stdev, param))
```

---

### Chapter Summary / 章节总结



---
