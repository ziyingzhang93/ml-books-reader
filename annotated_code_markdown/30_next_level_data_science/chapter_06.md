# 进阶数据科学 / Next Level Data Science
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
# 转换为NumPy数组 / Convert to NumPy array
X = Ames[["GrLivArea"]].values  # get 2D matrix
# 转换为NumPy数组 / Convert to NumPy array
y = Ames["SalePrice"].values    # get 1D vector

model = LinearRegression()
kf = KFold(n_splits=5)
coefs = []
scores = []
```

---
## Step 2 — Manually perform k-fold cross-validation

```python
# 同时获取索引和值 / Get both index and value
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
# 训练模型 / Train the model
model.fit(X_train, y_train)
    # 添加元素到列表末尾 / Append element to list end
    scores.append(model.score(X_test, y_test))
    # 添加元素到列表末尾 / Append element to list end
    coefs.append(model.coef_)

# 计算均值 / Calculate mean
mean_score = np.mean(scores)
# 打印输出 / Print output
print(f"Mean CV R^2 = {mean_score:.4f}")

# 计算均值 / Calculate mean
mean_coefs = np.mean(coefs)
# 打印输出 / Print output
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
# 转换为NumPy数组 / Convert to NumPy array
X = Ames[["GrLivArea"]].values  # get 2D matrix
# 转换为NumPy数组 / Convert to NumPy array
y = Ames["SalePrice"].values    # get 1D vector

model = LinearRegression()
kf = KFold(n_splits=5)
coefs = []
scores = []

# Manually perform k-fold cross-validation
# 同时获取索引和值 / Get both index and value
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model, obtain fold performance and coefficient
    # 训练模型 / Train the model
    model.fit(X_train, y_train)
    # 添加元素到列表末尾 / Append element to list end
    scores.append(model.score(X_test, y_test))
    # 添加元素到列表末尾 / Append element to list end
    coefs.append(model.coef_)

# 计算均值 / Calculate mean
mean_score = np.mean(scores)
# 打印输出 / Print output
print(f"Mean CV R^2 = {mean_score:.4f}")

# 计算均值 / Calculate mean
mean_coefs = np.mean(coefs)
# 打印输出 / Print output
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
neighbor_stats = Ames.groupby("Neighborhood")["SalePrice"] \
                     .agg(["count", "mean"]) \
                     .sort_values(by="mean")
# 转换数据类型 / Convert data type
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

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
neighbor_stats = Ames.groupby("Neighborhood")["SalePrice"] \
                     .agg(["count", "mean"]) \
                     .sort_values(by="mean")
# 转换数据类型 / Convert data type
print(neighbor_stats.round(0).astype(int))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Regression



---

### Columntransformer



---

### Chapter Summary / 章节总结

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
