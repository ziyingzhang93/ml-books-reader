# 进阶数据科学 / Next Level Data Science
## Chapter 03

---

### Sfs

# 01 — Sfs / 01 Sfs

**Chapter 03 — File 1 of 3 / 第03章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Load only the numeric columns from the Ames dataset**.

本脚本演示 **Load only the numeric columns from the Ames dataset**。

---
## Step 1 — Load only the numeric columns from the Ames dataset

```python
import pandas as pd
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"])
```

---
## Step 2 — Drop any columns with missing values

```python
Ames = Ames.dropna(axis=1)
```

---
## Step 3 — Import Linear Regression and Sequential Feature Selector from scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
```

---
## Step 4 — Initializing the Linear Regression model

```python
model = LinearRegression()
```

---
## Step 5 — Perform Sequential Feature Selector

```python
sfs = SequentialFeatureSelector(model, n_features_to_select=1)
X = Ames.drop("SalePrice", axis=1)  # Features
y = Ames["SalePrice"]  # Target variable
sfs.fit(X,y)           # Uses a default of cv=5
selected_feature = X.columns[sfs.get_support()]
print("Feature selected for highest predictability:", selected_feature[0])
```

---
## Learning Notes / 学习笔记

- **概念**: Load only the numeric columns from the Ames dataset 是机器学习中的常用技术。  
  *Load only the numeric columns from the Ames dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sfs / 01 Sfs
# Complete Code / 完整代码
# ===============================

# Load only the numeric columns from the Ames dataset
import pandas as pd
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"])

# Drop any columns with missing values
Ames = Ames.dropna(axis=1)

# Import Linear Regression and Sequential Feature Selector from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

# Initializing the Linear Regression model
model = LinearRegression()

# Perform Sequential Feature Selector
sfs = SequentialFeatureSelector(model, n_features_to_select=1)
X = Ames.drop("SalePrice", axis=1)  # Features
y = Ames["SalePrice"]  # Target variable
sfs.fit(X,y)           # Uses a default of cv=5
selected_feature = X.columns[sfs.get_support()]
print("Feature selected for highest predictability:", selected_feature[0])
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Cv

# 02 — Cv / 02 Cv

**Chapter 03 — File 2 of 3 / 第03章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Dictionary to hold feature names and their corresponding mean CV R^2 scores**.

本脚本演示 **Dictionary to hold feature names and their corresponding mean CV R^2 scores**。

---
## Step 1 — Step 1

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
X = Ames.drop("SalePrice", axis=1)  # Features
y = Ames["SalePrice"]  # Target variable
model = LinearRegression()
```

---
## Step 2 — Dictionary to hold feature names and their corresponding mean CV R^2 scores

```python
feature_scores = {}
```

---
## Step 3 — Iterate over each feature, perform CV, and store the mean R^2 score

```python
for feature in X.columns:
    X_single = X[[feature]]
    cv_scores = cross_val_score(model, X_single, y, cv=5)
    feature_scores[feature] = cv_scores.mean()
```

---
## Step 4 — Sort features based on their mean CV R^2 scores in descending order

```python
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
```

---
## Step 5 — Print the top 3 features and their scores

```python
top_3 = sorted_features[0:3]
for feature, score in top_3:
    print(f"Feature: {feature}, Mean CV R^2: {score:.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Dictionary to hold feature names and their corresponding mean CV R^2 scores 是机器学习中的常用技术。  
  *Dictionary to hold feature names and their corresponding mean CV R^2 scores is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cv / 02 Cv
# Complete Code / 完整代码
# ===============================

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
X = Ames.drop("SalePrice", axis=1)  # Features
y = Ames["SalePrice"]  # Target variable
model = LinearRegression()

# Dictionary to hold feature names and their corresponding mean CV R^2 scores
feature_scores = {}

# Iterate over each feature, perform CV, and store the mean R^2 score
for feature in X.columns:
    X_single = X[[feature]]
    cv_scores = cross_val_score(model, X_single, y, cv=5)
    feature_scores[feature] = cv_scores.mean()

# Sort features based on their mean CV R^2 scores in descending order
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)

# Print the top 3 features and their scores
top_3 = sorted_features[0:3]
for feature, score in top_3:
    print(f"Feature: {feature}, Mean CV R^2: {score:.4f}")
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Synthetic

# 03 — Synthetic / 03 Synthetic

**Chapter 03 — File 3 of 3 / 第03章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Create a new feature**.

本脚本演示 **Create a new feature**。

---
## Step 1 — Step 1

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
X = Ames.drop("SalePrice", axis=1)  # Features
y = Ames["SalePrice"]  # Target variable
```

---
## Step 2 — Create a new feature

```python
Ames['QualityArea'] = Ames['OverallQual'] * Ames['GrLivArea']
```

---
## Step 3 — Setting up the feature and target variable for the new 'QualityArea' feature

```python
X = Ames[['QualityArea']]  # New feature
y = Ames['SalePrice']
```

---
## Step 4 — 5-Fold CV on Linear Regression

```python
model = LinearRegression()
cv_scores = cross_val_score(model, X, y, cv=5)
```

---
## Step 5 — Calculating the mean of the CV scores

```python
mean_cv_score = cv_scores.mean()
print(f"Mean CV R^2 score using 'Quality Weighted Area': {mean_cv_score:.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Create a new feature 是机器学习中的常用技术。  
  *Create a new feature is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Synthetic / 03 Synthetic
# Complete Code / 完整代码
# ===============================

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
X = Ames.drop("SalePrice", axis=1)  # Features
y = Ames["SalePrice"]  # Target variable

# Create a new feature
Ames['QualityArea'] = Ames['OverallQual'] * Ames['GrLivArea']

# Setting up the feature and target variable for the new 'QualityArea' feature
X = Ames[['QualityArea']]  # New feature
y = Ames['SalePrice']

# 5-Fold CV on Linear Regression
model = LinearRegression()
cv_scores = cross_val_score(model, X, y, cv=5)

# Calculating the mean of the CV scores
mean_cv_score = cv_scores.mean()
print(f"Mean CV R^2 score using 'Quality Weighted Area': {mean_cv_score:.4f}")
```

---

### Chapter Summary / 章节总结

# Chapter 03 Summary / 第03章总结

## Theme / 主题: Chapter 03 / Chapter 03

This chapter contains **3 code files** demonstrating chapter 03.

本章包含 **3 个代码文件**，演示Chapter 03。

---
## Evolution / 演化路线

  1. `01_sfs.ipynb` — Sfs
  2. `02_cv.ipynb` — Cv
  3. `03_synthetic.ipynb` — Synthetic

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 03) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 03）是机器学习流水线中的基础构建块。

---
