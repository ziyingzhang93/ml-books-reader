# 进阶数据科学
## Chapter 06

---

### Coefficient

# 01 — Coefficient / 01 Coefficient

**Chapter 06 — File 1 of 4 / 第06章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Set up to obtain CV model performance and coefficient using k-fold**.

本脚本演示 **Set up to obtain CV model performance and coefficient using k-fold**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

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
```

---
## Step 1 — Set up to obtain CV model performance and coefficient using k-fold

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

Ames = pd.read_csv("Ames.csv")
X = Ames[["GrLivArea"]].values  # get 2D matrix
y = Ames["SalePrice"].values    # get 1D vector

model = LinearRegression()
kf = KFold(n_splits=5)
coefs = []
scores = []
```

---
## Step 2 — Manually perform k-fold cross-validation

```python
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
```

---
## Step 3 — Split the data into training and testing sets

```python
X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

---
## Step 4 — Fit the model, obtain fold performance and coefficient

```python
model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
    coefs.append(model.coef_)

mean_score = np.mean(scores)
print(f"Mean CV R^2 = {mean_score:.4f}")

mean_coefs = np.mean(coefs)
print(f"Mean Coefficient = {mean_coefs:.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Set up to obtain CV model performance and coefficient using k-fold 是机器学习中的常用技术。  
  *Set up to obtain CV model performance and coefficient using k-fold is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `model.fit` | 训练模型 | Train the model |
| `np.mean` | 计算均值 | Calculate mean |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Coefficient / 01 Coefficient
# Complete Code / 完整代码
# ===============================

# Set up to obtain CV model performance and coefficient using k-fold
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

Ames = pd.read_csv("Ames.csv")
X = Ames[["GrLivArea"]].values  # get 2D matrix
y = Ames["SalePrice"].values    # get 1D vector

model = LinearRegression()
kf = KFold(n_splits=5)
coefs = []
scores = []

# Manually perform k-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model, obtain fold performance and coefficient
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
    coefs.append(model.coef_)

mean_score = np.mean(scores)
print(f"Mean CV R^2 = {mean_score:.4f}")

mean_coefs = np.mean(coefs)
print(f"Mean Coefficient = {mean_coefs:.4f}")
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Rank

# 02 — Rank / 02 Rank

**Chapter 06 — File 2 of 4 / 第06章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Rank**.

本脚本演示 **02 Rank**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
import pandas as pd

Ames = pd.read_csv("Ames.csv")
neighbor_stats = Ames.groupby("Neighborhood")["SalePrice"] \
                     .agg(["count", "mean"]) \
                     .sort_values(by="mean")
print(neighbor_stats.round(0).astype(int))
```

---
## Learning Notes / 学习笔记

- **概念**: Rank 是机器学习中的常用技术。  
  *Rank is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `groupby` | 分组聚合 | Group and aggregate |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Rank / 02 Rank
# Complete Code / 完整代码
# ===============================

import pandas as pd

Ames = pd.read_csv("Ames.csv")
neighbor_stats = Ames.groupby("Neighborhood")["SalePrice"] \
                     .agg(["count", "mean"]) \
                     .sort_values(by="mean")
print(neighbor_stats.round(0).astype(int))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Chapter Summary

# Chapter 06 Summary / 第06章总结

## Theme / 主题: Chapter 06 / Chapter 06

This chapter contains **4 code files** demonstrating chapter 06.

本章包含 **4 个代码文件**，演示Chapter 06。

---
## Evolution / 演化路线

  1. `01_coefficient.ipynb` — Coefficient
  2. `02_rank.ipynb` — Rank
  3. `03_regression.ipynb` — Regression
  4. `04_columntransformer.ipynb` — Columntransformer

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 06) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 06）是机器学习流水线中的基础构建块。

---
