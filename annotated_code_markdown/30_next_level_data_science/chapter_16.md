# 进阶数据科学
## Chapter 16

---

### Xgboost

# 02 — Xgboost / 提升方法

**Chapter 16 — File 1 of 4 / 第16章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Import XGBoost to demonstrate native handling of missing values**.

本脚本演示 **Import XGBoost to demonstrate native handling of missing values**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
```

---
## Step 1 — Import XGBoost to demonstrate native handling of missing values

```python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Select numeric features with missing values

```python
cols_with_missing = Ames.isnull().any()
X = Ames.loc[:, cols_with_missing].select_dtypes(include=["int", "float"])
y = Ames["SalePrice"]
```

---
## Step 4 — Check and print the total number of missing values

```python
total_missing_values = X.isna().sum().sum()
print(f"Total number of missing values: {total_missing_values}")
```

---
## Step 5 — Initialize XGBoost regressor with default settings, fixed seed for reproducibility

```python
xgb_model = xgb.XGBRegressor(seed=42)
```

---
## Step 6 — Perform 5-fold cross-validation

```python
scores = cross_val_score(xgb_model, X, y, cv=5, scoring="r2")
```

---
## Step 7 — Calculate and display the average R-squared score

```python
mean_r2 = scores.mean()
print(f"XGB with native imputing, average R^2 score: {mean_r2:.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Import XGBoost to demonstrate native handling of missing values 是机器学习中的常用技术。  
  *Import XGBoost to demonstrate native handling of missing values is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBRegressor` | XGBoost回归器 | XGBoost regressor |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Xgboost / 提升方法
# Complete Code / 完整代码
# ===============================

# Import XGBoost to demonstrate native handling of missing values
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Select numeric features with missing values
cols_with_missing = Ames.isnull().any()
X = Ames.loc[:, cols_with_missing].select_dtypes(include=["int", "float"])
y = Ames["SalePrice"]

# Check and print the total number of missing values
total_missing_values = X.isna().sum().sum()
print(f"Total number of missing values: {total_missing_values}")

# Initialize XGBoost regressor with default settings, fixed seed for reproducibility
xgb_model = xgb.XGBRegressor(seed=42)

# Perform 5-fold cross-validation
scores = cross_val_score(xgb_model, X, y, cv=5, scoring="r2")

# Calculate and display the average R-squared score
mean_r2 = scores.mean()
print(f"XGB with native imputing, average R^2 score: {mean_r2:.4f}")
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Categorical

# 03 — Categorical / 03 Categorical

**Chapter 16 — File 2 of 4 / 第16章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Demonstrate native handling of categorical features**.

本脚本演示 **Demonstrate native handling of categorical features**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
```

---
## Step 1 — Demonstrate native handling of categorical features

```python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Convert specified categorical features to "category" type

```python
for col in ["Neighborhood", "BldgType", "HouseStyle"]:
    Ames[col] = Ames[col].astype("category")
```

---
## Step 4 — Include some numeric features for a balanced model

```python
selected_features = ["OverallQual", "GrLivArea", "YearBuilt", "TotalBsmtSF", "1stFlrSF",
                     "Neighborhood", "BldgType", "HouseStyle"]
X = Ames[selected_features]
y = Ames["SalePrice"]
```

---
## Step 5 — Initialize XGBoost regressor with native handling for categorical data

```python
xgb_model = xgb.XGBRegressor(
    seed=42,
    enable_categorical=True
)
```

---
## Step 6 — Perform 5-fold cross-validation

```python
scores = cross_val_score(xgb_model, X, y, cv=5, scoring="r2")
```

---
## Step 7 — Calculate the average R-squared score

```python
mean_r2 = scores.mean()

print(f"Average model R^2 score with selected categorical features: {mean_r2:.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Demonstrate native handling of categorical features 是机器学习中的常用技术。  
  *Demonstrate native handling of categorical features is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBRegressor` | XGBoost回归器 | XGBoost regressor |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Categorical / 03 Categorical
# Complete Code / 完整代码
# ===============================

# Demonstrate native handling of categorical features
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Convert specified categorical features to "category" type
for col in ["Neighborhood", "BldgType", "HouseStyle"]:
    Ames[col] = Ames[col].astype("category")

# Include some numeric features for a balanced model
selected_features = ["OverallQual", "GrLivArea", "YearBuilt", "TotalBsmtSF", "1stFlrSF",
                     "Neighborhood", "BldgType", "HouseStyle"]
X = Ames[selected_features]
y = Ames["SalePrice"]

# Initialize XGBoost regressor with native handling for categorical data
xgb_model = xgb.XGBRegressor(
    seed=42,
    enable_categorical=True
)

# Perform 5-fold cross-validation
scores = cross_val_score(xgb_model, X, y, cv=5, scoring="r2")

# Calculate the average R-squared score
mean_r2 = scores.mean()

print(f"Average model R^2 score with selected categorical features: {mean_r2:.4f}")
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Rfecv

# 04 — Rfecv / 04 Rfecv

**Chapter 16 — File 3 of 4 / 第16章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Perform Cross-Validated Recursive Feature Elimination for XGB**.

本脚本演示 **Perform Cross-Validated Recursive Feature Elimination for XGB**。

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
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — Perform Cross-Validated Recursive Feature Elimination for XGB

```python
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Convert selected features to "object" type to treat them as categorical

```python
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")
```

---
## Step 4 — Convert all object-type features to categorical and then to codes

```python
categorical_features = Ames.select_dtypes(include=["object"]).columns
for col in categorical_features:
    Ames[col] = Ames[col].astype("category").cat.codes
```

---
## Step 5 — Optional: Show that the categorical data encoded into integers
print(Ames)
Select features and target

```python
X = Ames.drop(columns=["SalePrice", "PID"])
y = Ames["SalePrice"]
```

---
## Step 6 — Initialize XGBoost regressor

```python
xgb_model = xgb.XGBRegressor(seed=42, enable_categorical=True)
```

---
## Step 7 — Initialize RFECV

```python
rfecv = RFECV(estimator=xgb_model, step=1, cv=5,
              scoring="r2", min_features_to_select=1)
```

---
## Step 8 — Fit RFECV

```python
rfecv.fit(X, y)
```

---
## Step 9 — Print the optimal number of features and their names

```python
print("Optimal number of features: ", rfecv.n_features_)
print("Best features: ", X.columns[rfecv.support_])
```

---
## Learning Notes / 学习笔记

- **概念**: Perform Cross-Validated Recursive Feature Elimination for XGB 是机器学习中的常用技术。  
  *Perform Cross-Validated Recursive Feature Elimination for XGB is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBRegressor` | XGBoost回归器 | XGBoost regressor |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Rfecv / 04 Rfecv
# Complete Code / 完整代码
# ===============================

# Perform Cross-Validated Recursive Feature Elimination for XGB
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Convert selected features to "object" type to treat them as categorical
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")

# Convert all object-type features to categorical and then to codes
categorical_features = Ames.select_dtypes(include=["object"]).columns
for col in categorical_features:
    Ames[col] = Ames[col].astype("category").cat.codes

# Optional: Show that the categorical data encoded into integers
# print(Ames)

# Select features and target
X = Ames.drop(columns=["SalePrice", "PID"])
y = Ames["SalePrice"]

# Initialize XGBoost regressor
xgb_model = xgb.XGBRegressor(seed=42, enable_categorical=True)

# Initialize RFECV
rfecv = RFECV(estimator=xgb_model, step=1, cv=5,
              scoring="r2", min_features_to_select=1)

# Fit RFECV
rfecv.fit(X, y)

# Print the optimal number of features and their names
print("Optimal number of features: ", rfecv.n_features_)
print("Best features: ", X.columns[rfecv.support_])
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Cv

# 05 — Cv / 05 Cv

**Chapter 16 — File 4 of 4 / 第16章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load the dataset and convert features**.

本脚本演示 **Load the dataset and convert features**。

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
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — Step 1

```python
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
```

---
## Step 2 — Load the dataset and convert features

```python
Ames = pd.read_csv("Ames.csv")
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")
categorical_features = Ames.select_dtypes(include=["object"]).columns
for col in categorical_features:
    Ames[col] = Ames[col].astype("category").cat.codes
```

---
## Step 3 — RFECV on an XGBoost regressor

```python
X = Ames.drop(columns=["SalePrice", "PID"])
y = Ames["SalePrice"]
xgb_model = xgb.XGBRegressor(seed=42, enable_categorical=True)
rfecv = RFECV(estimator=xgb_model, step=1, cv=5,
              scoring="r2", min_features_to_select=1)
rfecv.fit(X, y)
```

---
## Step 4 — Cross-validate the final model using only the selected features

```python
final_model = xgb.XGBRegressor(seed=42, enable_categorical=True)
cv_scores = cross_val_score(final_model, X.iloc[:, rfecv.support_], y, cv=5, scoring="r2")
```

---
## Step 5 — Calculate the average R-squared score

```python
mean_r2 = cv_scores.mean()

print(f"Average Cross-validated R^2 score with remaining features: {mean_r2:.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Load the dataset and convert features 是机器学习中的常用技术。  
  *Load the dataset and convert features is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBRegressor` | XGBoost回归器 | XGBoost regressor |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cv / 05 Cv
# Complete Code / 完整代码
# ===============================

import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score

# Load the dataset and convert features
Ames = pd.read_csv("Ames.csv")
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")
categorical_features = Ames.select_dtypes(include=["object"]).columns
for col in categorical_features:
    Ames[col] = Ames[col].astype("category").cat.codes

# RFECV on an XGBoost regressor
X = Ames.drop(columns=["SalePrice", "PID"])
y = Ames["SalePrice"]
xgb_model = xgb.XGBRegressor(seed=42, enable_categorical=True)
rfecv = RFECV(estimator=xgb_model, step=1, cv=5,
              scoring="r2", min_features_to_select=1)
rfecv.fit(X, y)

# Cross-validate the final model using only the selected features
final_model = xgb.XGBRegressor(seed=42, enable_categorical=True)
cv_scores = cross_val_score(final_model, X.iloc[:, rfecv.support_], y, cv=5, scoring="r2")

# Calculate the average R-squared score
mean_r2 = cv_scores.mean()

print(f"Average Cross-validated R^2 score with remaining features: {mean_r2:.4f}")
```

---
