# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 20

---

### Dataset

# 01 — Dataset / 01 Dataset

**Chapter 20 — File 1 of 5 / 第20章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Generate and plot a synthetic imbalanced classification dataset**.

本脚本演示 **Generate and plot a synthetic imbalanced classification dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — Generate and plot a synthetic imbalanced classification dataset

```python
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
```

---
## Step 3 — summarize class distribution

```python
counter = Counter(y)
# 打印输出 / Print output
print(counter)
```

---
## Step 4 — scatter plot of examples by class label

```python
# 获取字典的键值对 / Get dict key-value pairs
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Generate and plot a synthetic imbalanced classification dataset 是机器学习中的常用技术。  
  *Generate and plot a synthetic imbalanced classification dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dataset / 01 Dataset
# Complete Code / 完整代码
# ===============================

# Generate and plot a synthetic imbalanced classification dataset
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
# summarize class distribution
counter = Counter(y)
# 打印输出 / Print output
print(counter)
# scatter plot of examples by class label
# 获取字典的键值对 / Get dict key-value pairs
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Xgboost

# 02 — Xgboost / 提升方法

**Chapter 20 — File 2 of 5 / 第20章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **fit xgboost on an imbalanced classification dataset**.

本脚本演示 **fit xgboost on an imbalanced classification dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — fit xgboost on an imbalanced classification dataset

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
```

---
## Step 2 — generate dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
```

---
## Step 3 — define model

```python
model = XGBClassifier()
```

---
## Step 4 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 5 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
```

---
## Step 6 — summarize performance

```python
# 打印输出 / Print output
print('Mean ROC AUC: %.5f' % mean(scores))
```

---
## Learning Notes / 学习笔记

- **概念**: fit xgboost on an imbalanced classification dataset 是机器学习中的常用技术。  
  *fit xgboost on an imbalanced classification dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBClassifier` | XGBoost分类器 | XGBoost classifier |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Xgboost / 提升方法
# Complete Code / 完整代码
# ===============================

# fit xgboost on an imbalanced classification dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
# define model
model = XGBClassifier()
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
# 打印输出 / Print output
print('Mean ROC AUC: %.5f' % mean(scores))
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Estimate Weight

# 03 — Estimate Weight / 03 Estimate Weight

**Chapter 20 — File 3 of 5 / 第20章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **estimate a value for the scale_pos_weight xgboost hyperparameter**.

本脚本演示 **estimate a value for the scale_pos_weight xgboost hyperparameter**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — estimate a value for the scale_pos_weight xgboost hyperparameter

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
```

---
## Step 2 — generate dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
```

---
## Step 3 — count examples in each class

```python
counter = Counter(y)
```

---
## Step 4 — estimate scale_pos_weight value

```python
estimate = counter[0] / counter[1]
# 打印输出 / Print output
print('Estimate: %.3f' % estimate)
```

---
## Learning Notes / 学习笔记

- **概念**: estimate a value for the scale_pos_weight xgboost hyperparameter 是机器学习中的常用技术。  
  *estimate a value for the scale_pos_weight xgboost hyperparameter is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Estimate Weight / 03 Estimate Weight
# Complete Code / 完整代码
# ===============================

# estimate a value for the scale_pos_weight xgboost hyperparameter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
# count examples in each class
counter = Counter(y)
# estimate scale_pos_weight value
estimate = counter[0] / counter[1]
# 打印输出 / Print output
print('Estimate: %.3f' % estimate)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Balanced Xgboost

# 04 — Balanced Xgboost / 提升方法

**Chapter 20 — File 4 of 5 / 第20章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **fit balanced xgboost on an imbalanced classification dataset**.

本脚本演示 **fit balanced xgboost on an imbalanced classification dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — fit balanced xgboost on an imbalanced classification dataset

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
```

---
## Step 2 — generate dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
```

---
## Step 3 — define model

```python
model = XGBClassifier(scale_pos_weight=99)
```

---
## Step 4 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 5 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
```

---
## Step 6 — summarize performance

```python
# 打印输出 / Print output
print('Mean ROC AUC: %.5f' % mean(scores))
```

---
## Learning Notes / 学习笔记

- **概念**: fit balanced xgboost on an imbalanced classification dataset 是机器学习中的常用技术。  
  *fit balanced xgboost on an imbalanced classification dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBClassifier` | XGBoost分类器 | XGBoost classifier |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Balanced Xgboost / 提升方法
# Complete Code / 完整代码
# ===============================

# fit balanced xgboost on an imbalanced classification dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
# define model
model = XGBClassifier(scale_pos_weight=99)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
# 打印输出 / Print output
print('Mean ROC AUC: %.5f' % mean(scores))
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Grid Xgboost

# 05 — Grid Xgboost / 提升方法

**Chapter 20 — File 5 of 5 / 第20章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **grid search positive class weights with xgboost for imbalance classification**.

本脚本演示 **grid search positive class weights with xgboost for imbalance classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  🏋️ 训练模型 / Train Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — grid search positive class weights with xgboost for imbalance classification

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
```

---
## Step 2 — generate dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
```

---
## Step 3 — define model

```python
model = XGBClassifier()
```

---
## Step 4 — define grid

```python
weights = [1, 10, 25, 50, 75, 99, 100, 1000]
param_grid = dict(scale_pos_weight=weights)
```

---
## Step 5 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 6 — define grid search

```python
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
```

---
## Step 7 — execute the grid search

```python
grid_result = grid.fit(X, y)
```

---
## Step 8 — report the best configuration

```python
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
```

---
## Step 9 — report all configurations

```python
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---
## Learning Notes / 学习笔记

- **概念**: grid search positive class weights with xgboost for imbalance classification 是机器学习中的常用技术。  
  *grid search positive class weights with xgboost for imbalance classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `XGBClassifier` | XGBoost分类器 | XGBoost classifier |
| `numpy` | 数值计算库 | Numerical computing library |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grid Xgboost / 提升方法
# Complete Code / 完整代码
# ===============================

# grid search positive class weights with xgboost for imbalance classification
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
# define model
model = XGBClassifier()
# define grid
weights = [1, 10, 25, 50, 75, 99, 100, 1000]
param_grid = dict(scale_pos_weight=weights)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
# execute the grid search
grid_result = grid.fit(X, y)
# report the best configuration
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---

### Chapter Summary / 章节总结



---
