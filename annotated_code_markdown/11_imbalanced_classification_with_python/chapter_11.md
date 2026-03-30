# 不平衡分类
## Chapter 11

---

### Random Oversampling

# 01 — Random Oversampling / 过采样

**Chapter 11 — File 1 of 4 / 第11章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of random oversampling to balance the class distribution**.

本脚本演示 **example of random oversampling to balance the class distribution**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Step 1 — example of random oversampling to balance the class distribution

```python
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
```

---
## Step 3 — summarize class distribution

```python
print(Counter(y))
```

---
## Step 4 — define oversampling strategy

```python
oversample = RandomOverSampler(sampling_strategy='minority')
```

---
## Step 5 — fit and apply the transform

```python
X_over, y_over = oversample.fit_resample(X, y)
```

---
## Step 6 — summarize class distribution

```python
print(Counter(y_over))
```

---
## Learning Notes / 学习笔记

- **概念**: example of random oversampling to balance the class distribution 是机器学习中的常用技术。  
  *example of random oversampling to balance the class distribution is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Random Oversampling / 过采样
# Complete Code / 完整代码
# ===============================

# example of random oversampling to balance the class distribution
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
# summarize class distribution
print(Counter(y))
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)
# summarize class distribution
print(Counter(y_over))
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Oversample Evaluation

# 02 — Oversample Evaluation / 模型评估

**Chapter 11 — File 2 of 4 / 第11章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of evaluating a decision tree with random oversampling**.

本脚本演示 **example of evaluating a decision tree with random oversampling**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of evaluating a decision tree with random oversampling

```python
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
```

---
## Step 3 — define pipeline

```python
steps = [('over', RandomOverSampler()), ('model', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)
```

---
## Step 4 — evaluate pipeline

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1)
score = mean(scores)
print('F-measure: %.3f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: example of evaluating a decision tree with random oversampling 是机器学习中的常用技术。  
  *example of evaluating a decision tree with random oversampling is a common technique in machine learning.*

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
# Oversample Evaluation / 模型评估
# Complete Code / 完整代码
# ===============================

# example of evaluating a decision tree with random oversampling
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
# define pipeline
steps = [('over', RandomOverSampler()), ('model', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1)
score = mean(scores)
print('F-measure: %.3f' % score)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Chapter Summary

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **4 code files** demonstrating chapter 11.

本章包含 **4 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `01_random_oversampling.ipynb` — Random Oversampling
  2. `02_oversample_evaluation.ipynb` — Oversample Evaluation
  3. `03_random_undersampling.ipynb` — Random Undersampling
  4. `04_undersample_evaluation.ipynb` — Undersample Evaluation

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---
