# 集成学习
## Chapter 10

---

### Dataset

# 01 — Dataset / 01 Dataset

**Chapter 10 — File 1 of 10 / 第10章 — 第1个文件（共10个）**

---

## Summary / 总结

This script demonstrates **example of multioutput regression dataset**.

本脚本演示 **example of multioutput regression dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of multioutput regression dataset

```python
from sklearn.datasets import make_regression
```

---
## Step 2 — create datasets

```python
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
```

---
## Step 3 — summarize dataset

```python
print(X.shape, y.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: example of multioutput regression dataset 是机器学习中的常用技术。  
  *example of multioutput regression dataset is a common technique in machine learning.*

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
# Dataset / 01 Dataset
# Complete Code / 完整代码
# ===============================

# example of multioutput regression dataset
from sklearn.datasets import make_regression
# create datasets
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# summarize dataset
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 2 of 10

---

### Failure

# 06 — Failure / 06 Failure

**Chapter 10 — File 6 of 10 / 第10章 — 第6个文件（共10个）**

---

## Summary / 总结

This script demonstrates **failure of support vector regression for multioutput regression (causes an error)**.

本脚本演示 **failure of support vector regression for multioutput regression (causes an error)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model


---
## Step 1 — failure of support vector regression for multioutput regression (causes an error)

```python
from sklearn.datasets import make_regression
from sklearn.svm import LinearSVR
```

---
## Step 2 — create datasets

```python
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)
```

---
## Step 3 — define model

```python
model = LinearSVR()
```

---
## Step 4 — fit model
(THIS WILL CAUSE AN ERROR!)

```python
model.fit(X, y)
```

---
## Learning Notes / 学习笔记

- **概念**: failure of support vector regression for multioutput regression (causes an error) 是机器学习中的常用技术。  
  *failure of support vector regression for multioutput regression (causes an error) is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SVM` | 支持向量机 | Support Vector Machine |
| `model.fit` | 训练模型 | Train the model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Failure / 06 Failure
# Complete Code / 完整代码
# ===============================

# failure of support vector regression for multioutput regression (causes an error)
from sklearn.datasets import make_regression
from sklearn.svm import LinearSVR
# create datasets
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)
# define model
model = LinearSVR()
# fit model
# (THIS WILL CAUSE AN ERROR!)
model.fit(X, y)
```

---

➡️ **Next / 下一步**: File 7 of 10

---

### Chapter Summary

# Chapter 10 Summary / 第10章总结

## Theme / 主题: Chapter 10 / Chapter 10

This chapter contains **10 code files** demonstrating chapter 10.

本章包含 **10 个代码文件**，演示Chapter 10。

---
## Evolution / 演化路线

  1. `01_dataset.ipynb` — Dataset
  2. `02_linear_regression.ipynb` — Linear Regression
  3. `03_knn.ipynb` — Knn
  4. `04_decision_tree.ipynb` — Decision Tree
  5. `05_evaluate_tree.ipynb` — Evaluate Tree
  6. `06_failure.ipynb` — Failure
  7. `07_direct_evaluate.ipynb` — Direct Evaluate
  8. `08_direct_predict.ipynb` — Direct Predict
  9. `09_chained_evaluate.ipynb` — Chained Evaluate
  10. `10_chained_predict.ipynb` — Chained Predict

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 10) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 10）是机器学习流水线中的基础构建块。

---
