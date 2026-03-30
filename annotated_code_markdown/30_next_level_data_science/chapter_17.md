# 进阶数据科学
## Chapter 17

---

### Lightgbm

# 02 — Lightgbm / LightGBM

**Chapter 17 — File 1 of 3 / 第17章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Import libraries to run LightGBM**.

本脚本演示 **Import libraries to run LightGBM**。

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
## Step 1 — Import libraries to run LightGBM

```python
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
```

---
## Step 2 — Load the Ames Housing Dataset

```python
data = pd.read_csv("Ames.csv")
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
```

---
## Step 3 — Convert categorical columns to "category" dtype

```python
categorical_cols = X.select_dtypes(include=["object"]).columns
X[categorical_cols] = X[categorical_cols].apply(lambda x: x.astype("category"))
```

---
## Step 4 — Define the default GBDT model

```python
gbdt_model = lgb.LGBMRegressor(verbose=-1)
gbdt_scores = cross_val_score(gbdt_model, X, y, cv=5)
print(f"Average R^2 score for default Light GBM (with GBDT): {gbdt_scores.mean():.4f}")
```

---
## Step 5 — Define the GOSS model

```python
goss_model = lgb.LGBMRegressor(boosting_type="goss", verbose=-1)
goss_scores = cross_val_score(goss_model, X, y, cv=5)
print(f"Average R^2 score for Light GBM with GOSS: {goss_scores.mean():.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Import libraries to run LightGBM 是机器学习中的常用技术。  
  *Import libraries to run LightGBM is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lightgbm / LightGBM
# Complete Code / 完整代码
# ===============================

# Import libraries to run LightGBM
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

# Load the Ames Housing Dataset
data = pd.read_csv("Ames.csv")
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# Convert categorical columns to "category" dtype
categorical_cols = X.select_dtypes(include=["object"]).columns
X[categorical_cols] = X[categorical_cols].apply(lambda x: x.astype("category"))

# Define the default GBDT model
gbdt_model = lgb.LGBMRegressor(verbose=-1)
gbdt_scores = cross_val_score(gbdt_model, X, y, cv=5)
print(f"Average R^2 score for default Light GBM (with GBDT): {gbdt_scores.mean():.4f}")

# Define the GOSS model
goss_model = lgb.LGBMRegressor(boosting_type="goss", verbose=-1)
goss_scores = cross_val_score(goss_model, X, y, cv=5)
print(f"Average R^2 score for Light GBM with GOSS: {goss_scores.mean():.4f}")
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Training

# 03 — Training / 03 Training

**Chapter 17 — File 2 of 3 / 第17章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Experiment with Leaf-wise Tree Growth**.

本脚本演示 **Experiment with Leaf-wise Tree Growth**。

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
## Step 1 — Experiment with Leaf-wise Tree Growth

```python
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
```

---
## Step 2 — Load the Ames Housing Dataset

```python
data = pd.read_csv("Ames.csv")
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
```

---
## Step 3 — Convert categorical columns to "category" dtype

```python
categorical_cols = X.select_dtypes(include=["object"]).columns
X[categorical_cols] = X[categorical_cols].apply(lambda x: x.astype("category"))
```

---
## Step 4 — Define a range of leaf sizes to test

```python
leaf_sizes = [5, 10, 15, 31, 50, 100]
```

---
## Step 5 — Results storage

```python
results = {}
```

---
## Step 6 — Experiment with different leaf sizes for GBDT

```python
results["GBDT"] = {}
print('Testing different "num_leaves" for GBDT:')
for leaf_size in leaf_sizes:
    model = lgb.LGBMRegressor(boosting_type="gbdt", num_leaves=leaf_size, verbose=-1)
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    results["GBDT"][leaf_size] = scores.mean()
    print(f"num_leaves = {leaf_size}: Average R^2 score = {scores.mean():.4f}")
```

---
## Step 7 — Experiment with different leaf sizes for GOSS

```python
results["GOSS"] = {}
print('\nTesting different "num_leaves" for GOSS:')
for leaf_size in leaf_sizes:
    model = lgb.LGBMRegressor(boosting_type="goss", num_leaves=leaf_size, verbose=-1)
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    results["GOSS"][leaf_size] = scores.mean()
    print(f"num_leaves = {leaf_size}: Average R^2 score = {scores.mean():.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Experiment with Leaf-wise Tree Growth 是机器学习中的常用技术。  
  *Experiment with Leaf-wise Tree Growth is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Training / 03 Training
# Complete Code / 完整代码
# ===============================

# Experiment with Leaf-wise Tree Growth
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

# Load the Ames Housing Dataset
data = pd.read_csv("Ames.csv")
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# Convert categorical columns to "category" dtype
categorical_cols = X.select_dtypes(include=["object"]).columns
X[categorical_cols] = X[categorical_cols].apply(lambda x: x.astype("category"))

# Define a range of leaf sizes to test
leaf_sizes = [5, 10, 15, 31, 50, 100]

# Results storage
results = {}

# Experiment with different leaf sizes for GBDT
results["GBDT"] = {}
print('Testing different "num_leaves" for GBDT:')
for leaf_size in leaf_sizes:
    model = lgb.LGBMRegressor(boosting_type="gbdt", num_leaves=leaf_size, verbose=-1)
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    results["GBDT"][leaf_size] = scores.mean()
    print(f"num_leaves = {leaf_size}: Average R^2 score = {scores.mean():.4f}")

# Experiment with different leaf sizes for GOSS
results["GOSS"] = {}
print('\nTesting different "num_leaves" for GOSS:')
for leaf_size in leaf_sizes:
    model = lgb.LGBMRegressor(boosting_type="goss", num_leaves=leaf_size, verbose=-1)
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    results["GOSS"][leaf_size] = scores.mean()
    print(f"num_leaves = {leaf_size}: Average R^2 score = {scores.mean():.4f}")
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Compare

# 04 — Compare / 04 Compare

**Chapter 17 — File 3 of 3 / 第17章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Importing libraries to compare feature importance between GBDT and GOSS:**.

本脚本演示 **Importing libraries to compare feature importance between GBDT and GOSS:**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 训练模型 / Train the model
- 可视化结果 / Visualize results

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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Importing libraries to compare feature importance between GBDT and GOSS:

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
```

---
## Step 2 — Prepare data

```python
data = pd.read_csv("Ames.csv")
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
categorical_cols = X.select_dtypes(include=["object"]).columns
X[categorical_cols] = X[categorical_cols].apply(lambda x: x.astype("category"))
```

---
## Step 3 — Set up K-fold cross-validation

```python
kf = KFold(n_splits=5)
gbdt_feature_importances = []
goss_feature_importances = []
```

---
## Step 4 — Iterate over each split

```python
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
```

---
## Step 5 — Train GBDT model with optimal num_leaves

```python
gbdt_model = lgb.LGBMRegressor(boosting_type="gbdt", num_leaves=10, verbose=-1)
    gbdt_model.fit(X_train, y_train)
    gbdt_feature_importances.append(gbdt_model.feature_importances_)
```

---
## Step 6 — Train GOSS model with optimal num_leaves

```python
goss_model = lgb.LGBMRegressor(boosting_type="goss", num_leaves=10, verbose=-1)
    goss_model.fit(X_train, y_train)
    goss_feature_importances.append(goss_model.feature_importances_)
```

---
## Step 7 — Average feature importance across all folds for each model

```python
avg_gbdt_feature_importance = np.mean(gbdt_feature_importances, axis=0)
avg_goss_feature_importance = np.mean(goss_feature_importances, axis=0)
```

---
## Step 8 — Convert to DataFrame

```python
feat_importances_gbdt = pd.DataFrame({"Feature": X.columns,
                                      "Importance": avg_gbdt_feature_importance})
feat_importances_goss = pd.DataFrame({"Feature": X.columns,
                                      "Importance": avg_goss_feature_importance})
```

---
## Step 9 — Sort and take the top 10 features

```python
top_gbdt_features = feat_importances_gbdt \
                    .sort_values(by="Importance", ascending=False) \
                    .head(10)
top_goss_features = feat_importances_goss \
                    .sort_values(by="Importance", ascending=False) \
                    .head(10)
```

---
## Step 10 — Plotting

```python
plt.figure(figsize=(16, 12))
plt.subplot(1, 2, 1)
sns.barplot(data=top_gbdt_features, y="Feature", x="Importance",
            hue="Feature", orient="h", legend=False, palette="viridis")
plt.title("Top 10 LightGBM GBDT Features", fontsize=18)
plt.xlabel("Importance", fontsize=16)
plt.ylabel("Feature", fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=14)

plt.subplot(1, 2, 2)
sns.barplot(data=top_goss_features, y="Feature", x="Importance",
            hue="Feature", orient="h", legend=False, palette="viridis")
plt.title("Top 10 LightGBM GOSS Features", fontsize=18)
plt.xlabel("Importance", fontsize=16)
plt.ylabel("Feature", fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Importing libraries to compare feature importance between GBDT and GOSS: 是机器学习中的常用技术。  
  *Importing libraries to compare feature importance between GBDT and GOSS: is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `np.mean` | 计算均值 | Calculate mean |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Compare / 04 Compare
# Complete Code / 完整代码
# ===============================

# Importing libraries to compare feature importance between GBDT and GOSS:
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data
data = pd.read_csv("Ames.csv")
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
categorical_cols = X.select_dtypes(include=["object"]).columns
X[categorical_cols] = X[categorical_cols].apply(lambda x: x.astype("category"))

# Set up K-fold cross-validation
kf = KFold(n_splits=5)
gbdt_feature_importances = []
goss_feature_importances = []

# Iterate over each split
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train GBDT model with optimal num_leaves
    gbdt_model = lgb.LGBMRegressor(boosting_type="gbdt", num_leaves=10, verbose=-1)
    gbdt_model.fit(X_train, y_train)
    gbdt_feature_importances.append(gbdt_model.feature_importances_)

    # Train GOSS model with optimal num_leaves
    goss_model = lgb.LGBMRegressor(boosting_type="goss", num_leaves=10, verbose=-1)
    goss_model.fit(X_train, y_train)
    goss_feature_importances.append(goss_model.feature_importances_)

# Average feature importance across all folds for each model
avg_gbdt_feature_importance = np.mean(gbdt_feature_importances, axis=0)
avg_goss_feature_importance = np.mean(goss_feature_importances, axis=0)

# Convert to DataFrame
feat_importances_gbdt = pd.DataFrame({"Feature": X.columns,
                                      "Importance": avg_gbdt_feature_importance})
feat_importances_goss = pd.DataFrame({"Feature": X.columns,
                                      "Importance": avg_goss_feature_importance})

# Sort and take the top 10 features
top_gbdt_features = feat_importances_gbdt \
                    .sort_values(by="Importance", ascending=False) \
                    .head(10)
top_goss_features = feat_importances_goss \
                    .sort_values(by="Importance", ascending=False) \
                    .head(10)

# Plotting
plt.figure(figsize=(16, 12))
plt.subplot(1, 2, 1)
sns.barplot(data=top_gbdt_features, y="Feature", x="Importance",
            hue="Feature", orient="h", legend=False, palette="viridis")
plt.title("Top 10 LightGBM GBDT Features", fontsize=18)
plt.xlabel("Importance", fontsize=16)
plt.ylabel("Feature", fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=14)

plt.subplot(1, 2, 2)
sns.barplot(data=top_goss_features, y="Feature", x="Importance",
            hue="Feature", orient="h", legend=False, palette="viridis")
plt.title("Top 10 LightGBM GOSS Features", fontsize=18)
plt.xlabel("Importance", fontsize=16)
plt.ylabel("Feature", fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()
```

---
