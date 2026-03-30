# 不平衡分类
## Chapter 12

---

### Smote Grid Search

# 05 — Smote Grid Search / SMOTE 过采样

**Chapter 12 — File 5 of 8 / 第12章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **grid search k value for SMOTE oversampling for imbalanced classification**.

本脚本演示 **grid search k value for SMOTE oversampling for imbalanced classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — grid search k value for SMOTE oversampling for imbalanced classification

```python
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
```

---
## Step 3 — values to evaluate

```python
k_values = [1, 2, 3, 4, 5, 6, 7]
for k in k_values:
```

---
## Step 4 — define pipeline

```python
model = DecisionTreeClassifier()
	over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
	pipeline = Pipeline(steps=[('over', over), ('model', model)])
```

---
## Step 5 — evaluate pipeline

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
	score = mean(scores)
	print('> k=%d, Mean ROC AUC: %.3f' % (k, score))
```

---
## Learning Notes / 学习笔记

- **概念**: grid search k value for SMOTE oversampling for imbalanced classification 是机器学习中的常用技术。  
  *grid search k value for SMOTE oversampling for imbalanced classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Smote Grid Search / SMOTE 过采样
# Complete Code / 完整代码
# ===============================

# grid search k value for SMOTE oversampling for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# values to evaluate
k_values = [1, 2, 3, 4, 5, 6, 7]
for k in k_values:
	# define pipeline
	model = DecisionTreeClassifier()
	over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
	pipeline = Pipeline(steps=[('over', over), ('model', model)])
	# evaluate pipeline
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
	score = mean(scores)
	print('> k=%d, Mean ROC AUC: %.3f' % (k, score))
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Borderline Smote

# 06 — Borderline Smote / SMOTE 过采样

**Chapter 12 — File 6 of 8 / 第12章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **borderline-SMOTE for imbalanced dataset**.

本脚本演示 **borderline-SMOTE for imbalanced dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Step 1 — borderline-SMOTE for imbalanced dataset

```python
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import BorderlineSMOTE
from matplotlib import pyplot
from numpy import where
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
```

---
## Step 3 — summarize class distribution

```python
counter = Counter(y)
print(counter)
```

---
## Step 4 — transform the dataset

```python
oversample = BorderlineSMOTE()
X, y = oversample.fit_resample(X, y)
```

---
## Step 5 — summarize the new class distribution

```python
counter = Counter(y)
print(counter)
```

---
## Step 6 — scatter plot of examples by class label

```python
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: borderline-SMOTE for imbalanced dataset 是机器学习中的常用技术。  
  *borderline-SMOTE for imbalanced dataset is a common technique in machine learning.*

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
# Borderline Smote / SMOTE 过采样
# Complete Code / 完整代码
# ===============================

# borderline-SMOTE for imbalanced dataset
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import BorderlineSMOTE
from matplotlib import pyplot
from numpy import where
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# summarize class distribution
counter = Counter(y)
print(counter)
# transform the dataset
oversample = BorderlineSMOTE()
X, y = oversample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Chapter Summary

# Chapter 12 Summary / 第12章总结

## Theme / 主题: Chapter 12 / Chapter 12

This chapter contains **8 code files** demonstrating chapter 12.

本章包含 **8 个代码文件**，演示Chapter 12。

---
## Evolution / 演化路线

  1. `01_dataset.ipynb` — Dataset
  2. `02_smote.ipynb` — Smote
  3. `03_evaluate_model.ipynb` — Evaluate Model
  4. `04_smote_evaluate_model.ipynb` — Smote Evaluate Model
  5. `05_smote_grid_search.ipynb` — Smote Grid Search
  6. `06_borderline_smote.ipynb` — Borderline Smote
  7. `07_svm_smote.ipynb` — Svm Smote
  8. `08_adasyn.ipynb` — Adasyn

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 12) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 12）是机器学习流水线中的基础构建块。

---
