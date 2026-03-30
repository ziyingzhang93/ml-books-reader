# 进阶数据科学 / Next Level Data Science
## Chapter 18

---

### Catboost

# 02 — Catboost / 提升方法

**Chapter 18 — File 1 of 2 / 第18章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Import libraries to run CatBoost Regressor**.

本脚本演示 **Import libraries to run CatBoost Regressor**。

---
## Step 1 — Import libraries to run CatBoost Regressor

```python
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
```

---
## Step 2 — Load dataset

```python
data = pd.read_csv("Ames.csv")
X = data.drop(["SalePrice"], axis=1)
y = data["SalePrice"]
```

---
## Step 3 — Identify and fill NaNs in categorical columns

```python
cat_features = [col for col in X.columns if X[col].dtype == "object"]
X["Electrical"] = X["Electrical"].fillna(X["Electrical"].mode()[0])
X[cat_features] = X[cat_features].fillna("Missing")
```

---
## Step 4 — Identify categorical columns

```python
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
```

---
## Step 5 — Define and train the default CatBoost model

```python
default_model = CatBoostRegressor(cat_features=cat_features, random_state=42, verbose=0)
default_scores = cross_val_score(default_model, X, y, cv=5, scoring="r2")
print(f"Average R^2 score for default CatBoost: {default_scores.mean():.4f}")
```

---
## Step 6 — Define and train the CatBoost model with ordered boosting

```python
ordered_model = CatBoostRegressor(cat_features=cat_features, random_state=42,
                                  boosting_type="Ordered", verbose=0)
ordered_scores = cross_val_score(ordered_model, X, y, cv=5, scoring="r2")
print("Average R^2 score for CatBoost with ordered boosting: "
      f"{ordered_scores.mean():.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Import libraries to run CatBoost Regressor 是机器学习中的常用技术。  
  *Import libraries to run CatBoost Regressor is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Catboost / 提升方法
# Complete Code / 完整代码
# ===============================

# Import libraries to run CatBoost Regressor
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score

# Load dataset
data = pd.read_csv("Ames.csv")
X = data.drop(["SalePrice"], axis=1)
y = data["SalePrice"]

# Identify and fill NaNs in categorical columns
cat_features = [col for col in X.columns if X[col].dtype == "object"]
X["Electrical"] = X["Electrical"].fillna(X["Electrical"].mode()[0])
X[cat_features] = X[cat_features].fillna("Missing")

# Identify categorical columns
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Define and train the default CatBoost model
default_model = CatBoostRegressor(cat_features=cat_features, random_state=42, verbose=0)
default_scores = cross_val_score(default_model, X, y, cv=5, scoring="r2")
print(f"Average R^2 score for default CatBoost: {default_scores.mean():.4f}")

# Define and train the CatBoost model with ordered boosting
ordered_model = CatBoostRegressor(cat_features=cat_features, random_state=42,
                                  boosting_type="Ordered", verbose=0)
ordered_scores = cross_val_score(ordered_model, X, y, cv=5, scoring="r2")
print("Average R^2 score for CatBoost with ordered boosting: "
      f"{ordered_scores.mean():.4f}")
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Plot

# 03 — Plot / 03 Plot

**Chapter 18 — File 2 of 2 / 第18章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Load dataset**.

本脚本演示 **Load dataset**。

---
## Step 1 — Step 1

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
```

---
## Step 2 — Load dataset

```python
data = pd.read_csv("Ames.csv")
X = data.drop(["SalePrice"], axis=1)
y = data["SalePrice"]
cat_features = [col for col in X.columns if X[col].dtype == "object"]
X["Electrical"] = X["Electrical"].fillna(X["Electrical"].mode()[0])
X[cat_features] = X[cat_features].fillna("Missing")
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
```

---
## Step 3 — Set up k-fold cross-validation

```python
kf = KFold(n_splits=5)
feature_importances = []
```

---
## Step 4 — Iterate over each split

```python
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
```

---
## Step 5 — Train default CatBoost model

```python
model = CatBoostRegressor(cat_features=cat_features, random_state=42, verbose=0)
    model.fit(X_train, y_train)
    feature_importances.append(model.get_feature_importance())
```

---
## Step 6 — Average feature importance across all folds

```python
avg_importance = np.mean(feature_importances, axis=0)
```

---
## Step 7 — Convert to DataFrame

```python
feat_imp_df = pd.DataFrame({"Feature": X.columns, "Importance": avg_importance})
```

---
## Step 8 — Sort and take the top 20 features

```python
top_features = feat_imp_df.sort_values(by="Importance", ascending=False).head(20)
```

---
## Step 9 — Set the style and color palette

```python
sns.set_style("whitegrid")
palette = sns.color_palette("rocket", len(top_features))
```

---
## Step 10 — Create the plot

```python
plt.figure(figsize=(12, 10))
ax = sns.barplot(x="Importance", y="Feature", hue="Feature",
                 data=top_features, palette=palette, legend=False)
```

---
## Step 11 — Customize the plot

```python
plt.title("Top 20 Most Important Features - CatBoost Model",
          fontsize=20, fontweight="bold")
plt.xlabel("Importance Score", fontsize=15)
plt.ylabel("Features", fontsize=15)
```

---
## Step 12 — Add value labels to the end of each bar

```python
for i, v in enumerate(top_features["Importance"]):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=13)
```

---
## Step 13 — Extend x-axis by 10% and feature names font size

```python
plt.xlim(0, max(top_features["Importance"]) * 1.1)
plt.yticks(fontsize=13)
```

---
## Step 14 — Adjust layout and display

```python
plt.tight_layout()
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load dataset 是机器学习中的常用技术。  
  *Load dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot / 03 Plot
# Complete Code / 完整代码
# ===============================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Load dataset
data = pd.read_csv("Ames.csv")
X = data.drop(["SalePrice"], axis=1)
y = data["SalePrice"]
cat_features = [col for col in X.columns if X[col].dtype == "object"]
X["Electrical"] = X["Electrical"].fillna(X["Electrical"].mode()[0])
X[cat_features] = X[cat_features].fillna("Missing")
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Set up k-fold cross-validation
kf = KFold(n_splits=5)
feature_importances = []

# Iterate over each split
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train default CatBoost model
    model = CatBoostRegressor(cat_features=cat_features, random_state=42, verbose=0)
    model.fit(X_train, y_train)
    feature_importances.append(model.get_feature_importance())

# Average feature importance across all folds
avg_importance = np.mean(feature_importances, axis=0)

# Convert to DataFrame
feat_imp_df = pd.DataFrame({"Feature": X.columns, "Importance": avg_importance})

# Sort and take the top 20 features
top_features = feat_imp_df.sort_values(by="Importance", ascending=False).head(20)

# Set the style and color palette
sns.set_style("whitegrid")
palette = sns.color_palette("rocket", len(top_features))

# Create the plot
plt.figure(figsize=(12, 10))
ax = sns.barplot(x="Importance", y="Feature", hue="Feature",
                 data=top_features, palette=palette, legend=False)

# Customize the plot
plt.title("Top 20 Most Important Features - CatBoost Model",
          fontsize=20, fontweight="bold")
plt.xlabel("Importance Score", fontsize=15)
plt.ylabel("Features", fontsize=15)

# Add value labels to the end of each bar
for i, v in enumerate(top_features["Importance"]):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=13)

# Extend x-axis by 10% and feature names font size
plt.xlim(0, max(top_features["Importance"]) * 1.1)
plt.yticks(fontsize=13)

# Adjust layout and display
plt.tight_layout()
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 18 Summary / 第18章总结

## Theme / 主题: Chapter 18 / Chapter 18

This chapter contains **2 code files** demonstrating chapter 18.

本章包含 **2 个代码文件**，演示Chapter 18。

---
## Evolution / 演化路线

  1. `02_catboost.ipynb` — Catboost
  2. `03_plot.ipynb` — Plot

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 18) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 18）是机器学习流水线中的基础构建块。

---
