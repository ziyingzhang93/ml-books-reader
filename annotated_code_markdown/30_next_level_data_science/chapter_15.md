# 进阶数据科学
## Chapter 15

---

### Baseline

# 01 — Baseline / 01 Baseline

**Chapter 15 — File 1 of 6 / 第15章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Import necessary libraries for preprocessing and modeling**.

本脚本演示 **Import necessary libraries for preprocessing and modeling**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — Import necessary libraries for preprocessing and modeling

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import \
    GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Adjust data types for categorical variables

```python
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")
```

---
## Step 4 — Exclude "PID" and "SalePrice" from features and handle the "Electrical" column

```python
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]
```

---
## Step 5 — Manually specify the categories for ordinal encoding according to the data dictionary

```python
ordinal_order = {
```

---
## Step 6 — Electrical system

```python
"Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
```

---
## Step 7 — General shape of property

```python
"LotShape": ["IR3", "IR2", "IR1", "Reg"],
```

---
## Step 8 — Type of utilities available

```python
"Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
```

---
## Step 9 — Slope of property

```python
"LandSlope": ["Sev", "Mod", "Gtl"],
```

---
## Step 10 — Evaluates the quality of the material on the exterior

```python
"ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 11 — Evaluates the present condition of the material on the exterior

```python
"ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 12 — Height of the basement

```python
"BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 13 — General condition of the basement

```python
"BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 14 — Walkout or garden level basement walls

```python
"BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
```

---
## Step 15 — Quality of basement finished area

```python
"BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 16 — Quality of second basement finished area

```python
"BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 17 — Heating quality and condition

```python
"HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 18 — Kitchen quality

```python
"KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 19 — Home functionality

```python
"Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
```

---
## Step 20 — Fireplace quality

```python
"FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 21 — Interior finish of the garage

```python
"GarageFinish": ["None", "Unf", "RFn", "Fin"],
```

---
## Step 22 — Garage quality

```python
"GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 23 — Garage condition

```python
"GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 24 — Paved driveway

```python
"PavedDrive": ["N", "P", "Y"],
```

---
## Step 25 — Pool quality

```python
"PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 26 — Fence quality

```python
"Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"]
}
```

---
## Step 27 — Extract list of ALL ordinal features from dictionary

```python
ordinal_features = list(ordinal_order.keys())
```

---
## Step 28 — List of ordinal features except Electrical

```python
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]
```

---
## Step 29 — Define transformations for various feature types

```python
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])
```

---
## Step 30 — Updated categorical imputer using SimpleImputer

```python
categorical_imputer = SimpleImputer(strategy="constant", fill_value="None")

ordinal_transformer = Pipeline([
    ("impute_ordinal", categorical_imputer),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline([
    ("impute_nominal", categorical_imputer),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 31 — Combined preprocessor for numeric, ordinal, nominal, and specific electrical data

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("electrical", electrical_transformer, ["Electrical"]),
        ("num", numeric_transformer, numeric_features),
        ("ordinal", ordinal_transformer, ordinal_except_electrical),
        ("nominal", categorical_transformer, nominal_features)
])
```

---
## Step 32 — Define model pipelines including Gradient Boosting Regressor

```python
models = {
    "Decision Tree (1 Tree)": DecisionTreeRegressor(random_state=42),
    "Bagging Regressor (100 Decision Trees)":
        BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42),
                         n_estimators=100, random_state=42),
    "Bagging Regressor (200 Decision Trees)":
        BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42),
                         n_estimators=200, random_state=42),
    "Random Forest (Default of 100 Trees)":
        RandomForestRegressor(random_state=42),
    "Random Forest (200 Trees)":
        RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting Regressor (Default of 100 Trees)":
        GradientBoostingRegressor(random_state=42),
    "Gradient Boosting Regressor (200 Trees)":
        GradientBoostingRegressor(n_estimators=200, random_state=42)
}
```

---
## Step 33 — Evaluate models using cross-validation and print results

```python
results = {}
for name, model in models.items():
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    scores = cross_val_score(model_pipeline,
                             Ames.drop(columns="SalePrice"), Ames["SalePrice"], cv=5)
    results[name] = round(scores.mean(), 4)
    print(f"{name}: Mean CV R^2 = {results[name]}")
```

---
## Learning Notes / 学习笔记

- **概念**: Import necessary libraries for preprocessing and modeling 是机器学习中的常用技术。  
  *Import necessary libraries for preprocessing and modeling is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `GradientBoosting` | 梯度提升算法 | Gradient Boosting algorithm |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Baseline / 01 Baseline
# Complete Code / 完整代码
# ===============================

# Import necessary libraries for preprocessing and modeling
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import \
    GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Adjust data types for categorical variables
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")

# Exclude "PID" and "SalePrice" from features and handle the "Electrical" column
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]

# Manually specify the categories for ordinal encoding according to the data dictionary
ordinal_order = {
    # Electrical system
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    # General shape of property
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    # Type of utilities available
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    # Slope of property
    "LandSlope": ["Sev", "Mod", "Gtl"],
    # Evaluates the quality of the material on the exterior
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Evaluates the present condition of the material on the exterior
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Height of the basement
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # General condition of the basement
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Walkout or garden level basement walls
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    # Quality of basement finished area
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    # Quality of second basement finished area
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    # Heating quality and condition
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Kitchen quality
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Home functionality
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    # Fireplace quality
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Interior finish of the garage
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    # Garage quality
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Garage condition
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Paved driveway
    "PavedDrive": ["N", "P", "Y"],
    # Pool quality
    "PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
    # Fence quality
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"]
}

# Extract list of ALL ordinal features from dictionary
ordinal_features = list(ordinal_order.keys())

# List of ordinal features except Electrical
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]

# Define transformations for various feature types
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])

# Updated categorical imputer using SimpleImputer
categorical_imputer = SimpleImputer(strategy="constant", fill_value="None")

ordinal_transformer = Pipeline([
    ("impute_ordinal", categorical_imputer),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline([
    ("impute_nominal", categorical_imputer),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessor for numeric, ordinal, nominal, and specific electrical data
preprocessor = ColumnTransformer(
    transformers=[
        ("electrical", electrical_transformer, ["Electrical"]),
        ("num", numeric_transformer, numeric_features),
        ("ordinal", ordinal_transformer, ordinal_except_electrical),
        ("nominal", categorical_transformer, nominal_features)
])

# Define model pipelines including Gradient Boosting Regressor
models = {
    "Decision Tree (1 Tree)": DecisionTreeRegressor(random_state=42),
    "Bagging Regressor (100 Decision Trees)":
        BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42),
                         n_estimators=100, random_state=42),
    "Bagging Regressor (200 Decision Trees)":
        BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42),
                         n_estimators=200, random_state=42),
    "Random Forest (Default of 100 Trees)":
        RandomForestRegressor(random_state=42),
    "Random Forest (200 Trees)":
        RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting Regressor (Default of 100 Trees)":
        GradientBoostingRegressor(random_state=42),
    "Gradient Boosting Regressor (200 Trees)":
        GradientBoostingRegressor(n_estimators=200, random_state=42)
}

# Evaluate models using cross-validation and print results
results = {}
for name, model in models.items():
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    scores = cross_val_score(model_pipeline,
                             Ames.drop(columns="SalePrice"), Ames["SalePrice"], cv=5)
    results[name] = round(scores.mean(), 4)
    print(f"{name}: Mean CV R^2 = {results[name]}")
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Learning Rate

# 02 — Learning Rate / 02 Learning Rate

**Chapter 15 — File 2 of 6 / 第15章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Experiment with GridSearchCV**.

本脚本演示 **Experiment with GridSearchCV**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — Experiment with GridSearchCV

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

Ames = pd.read_csv("Ames.csv")
```

---
## Step 2 — Adjust data types for categorical variables

```python
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")
```

---
## Step 3 — Exclude "PID" and "SalePrice" from features and handle the "Electrical" column

```python
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]
```

---
## Step 4 — Manually specify the categories for ordinal encoding according to the data dictionary

```python
ordinal_order = {
```

---
## Step 5 — Electrical system

```python
"Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
```

---
## Step 6 — General shape of property

```python
"LotShape": ["IR3", "IR2", "IR1", "Reg"],
```

---
## Step 7 — Type of utilities available

```python
"Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
```

---
## Step 8 — Slope of property

```python
"LandSlope": ["Sev", "Mod", "Gtl"],
```

---
## Step 9 — Evaluates the quality of the material on the exterior

```python
"ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 10 — Evaluates the present condition of the material on the exterior

```python
"ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 11 — Height of the basement

```python
"BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 12 — General condition of the basement

```python
"BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 13 — Walkout or garden level basement walls

```python
"BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
```

---
## Step 14 — Quality of basement finished area

```python
"BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 15 — Quality of second basement finished area

```python
"BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 16 — Heating quality and condition

```python
"HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 17 — Kitchen quality

```python
"KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 18 — Home functionality

```python
"Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
```

---
## Step 19 — Fireplace quality

```python
"FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 20 — Interior finish of the garage

```python
"GarageFinish": ["None", "Unf", "RFn", "Fin"],
```

---
## Step 21 — Garage quality

```python
"GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 22 — Garage condition

```python
"GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 23 — Paved driveway

```python
"PavedDrive": ["N", "P", "Y"],
```

---
## Step 24 — Pool quality

```python
"PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 25 — Fence quality

```python
"Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"]
}
```

---
## Step 26 — Extract list of ALL ordinal features from dictionary

```python
ordinal_features = list(ordinal_order.keys())
```

---
## Step 27 — List of ordinal features except Electrical

```python
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]
```

---
## Step 28 — Define transformations for various feature types

```python
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])
```

---
## Step 29 — Updated categorical imputer using SimpleImputer

```python
categorical_imputer = SimpleImputer(strategy="constant", fill_value="None")

ordinal_transformer = Pipeline([
    ("impute_ordinal", categorical_imputer),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline([
    ("impute_nominal", categorical_imputer),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 30 — Combined preprocessor for numeric, ordinal, nominal, and specific electrical data

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("electrical", electrical_transformer, ["Electrical"]),
        ("num", numeric_transformer, numeric_features),
        ("ordinal", ordinal_transformer, ordinal_except_electrical),
        ("nominal", categorical_transformer, nominal_features)
])

model = GradientBoostingRegressor(n_estimators=200, random_state=42)

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", model)
])
```

---
## Step 31 — Parameter grid for GridSearchCV

```python
param_grid = {
    "regressor__learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3]
}
```

---
## Step 32 — Setup the GridSearchCV

```python
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="r2", verbose=1)
```

---
## Step 33 — Fit the GridSearchCV to the data

```python
grid_search.fit(Ames.drop(columns="SalePrice"), Ames["SalePrice"])
```

---
## Step 34 — Best parameters and best score from Grid Search

```python
print("Best parameters (Grid Search):", grid_search.best_params_)
print("Best score (Grid Search):", round(grid_search.best_score_, 4))
```

---
## Learning Notes / 学习笔记

- **概念**: Experiment with GridSearchCV 是机器学习中的常用技术。  
  *Experiment with GridSearchCV is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `GradientBoosting` | 梯度提升算法 | Gradient Boosting algorithm |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Learning Rate / 02 Learning Rate
# Complete Code / 完整代码
# ===============================

# Experiment with GridSearchCV
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

Ames = pd.read_csv("Ames.csv")

# Adjust data types for categorical variables
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")

# Exclude "PID" and "SalePrice" from features and handle the "Electrical" column
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]

# Manually specify the categories for ordinal encoding according to the data dictionary
ordinal_order = {
    # Electrical system
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    # General shape of property
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    # Type of utilities available
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    # Slope of property
    "LandSlope": ["Sev", "Mod", "Gtl"],
    # Evaluates the quality of the material on the exterior
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Evaluates the present condition of the material on the exterior
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Height of the basement
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # General condition of the basement
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Walkout or garden level basement walls
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    # Quality of basement finished area
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    # Quality of second basement finished area
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    # Heating quality and condition
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Kitchen quality
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Home functionality
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    # Fireplace quality
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Interior finish of the garage
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    # Garage quality
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Garage condition
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Paved driveway
    "PavedDrive": ["N", "P", "Y"],
    # Pool quality
    "PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
    # Fence quality
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"]
}

# Extract list of ALL ordinal features from dictionary
ordinal_features = list(ordinal_order.keys())

# List of ordinal features except Electrical
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]

# Define transformations for various feature types
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])

# Updated categorical imputer using SimpleImputer
categorical_imputer = SimpleImputer(strategy="constant", fill_value="None")

ordinal_transformer = Pipeline([
    ("impute_ordinal", categorical_imputer),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline([
    ("impute_nominal", categorical_imputer),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessor for numeric, ordinal, nominal, and specific electrical data
preprocessor = ColumnTransformer(
    transformers=[
        ("electrical", electrical_transformer, ["Electrical"]),
        ("num", numeric_transformer, numeric_features),
        ("ordinal", ordinal_transformer, ordinal_except_electrical),
        ("nominal", categorical_transformer, nominal_features)
])

model = GradientBoostingRegressor(n_estimators=200, random_state=42)

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", model)
])

# Parameter grid for GridSearchCV
param_grid = {
    "regressor__learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3]
}

# Setup the GridSearchCV
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="r2", verbose=1)

# Fit the GridSearchCV to the data
grid_search.fit(Ames.drop(columns="SalePrice"), Ames["SalePrice"])

# Best parameters and best score from Grid Search
print("Best parameters (Grid Search):", grid_search.best_params_)
print("Best score (Grid Search):", round(grid_search.best_score_, 4))
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Randomized Search

# 03 — Randomized Search / 03 Randomized Search

**Chapter 15 — File 3 of 6 / 第15章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Experiment with RandomizedSearchCV**.

本脚本演示 **Experiment with RandomizedSearchCV**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — Experiment with RandomizedSearchCV

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

Ames = pd.read_csv("Ames.csv")
```

---
## Step 2 — Adjust data types for categorical variables

```python
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")
```

---
## Step 3 — Exclude "PID" and "SalePrice" from features and handle the "Electrical" column

```python
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]
```

---
## Step 4 — Manually specify the categories for ordinal encoding according to the data dictionary

```python
ordinal_order = {
```

---
## Step 5 — Electrical system

```python
"Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
```

---
## Step 6 — General shape of property

```python
"LotShape": ["IR3", "IR2", "IR1", "Reg"],
```

---
## Step 7 — Type of utilities available

```python
"Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
```

---
## Step 8 — Slope of property

```python
"LandSlope": ["Sev", "Mod", "Gtl"],
```

---
## Step 9 — Evaluates the quality of the material on the exterior

```python
"ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 10 — Evaluates the present condition of the material on the exterior

```python
"ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 11 — Height of the basement

```python
"BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 12 — General condition of the basement

```python
"BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 13 — Walkout or garden level basement walls

```python
"BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
```

---
## Step 14 — Quality of basement finished area

```python
"BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 15 — Quality of second basement finished area

```python
"BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 16 — Heating quality and condition

```python
"HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 17 — Kitchen quality

```python
"KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 18 — Home functionality

```python
"Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
```

---
## Step 19 — Fireplace quality

```python
"FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 20 — Interior finish of the garage

```python
"GarageFinish": ["None", "Unf", "RFn", "Fin"],
```

---
## Step 21 — Garage quality

```python
"GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 22 — Garage condition

```python
"GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 23 — Paved driveway

```python
"PavedDrive": ["N", "P", "Y"],
```

---
## Step 24 — Pool quality

```python
"PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 25 — Fence quality

```python
"Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"]
}
```

---
## Step 26 — Extract list of ALL ordinal features from dictionary

```python
ordinal_features = list(ordinal_order.keys())
```

---
## Step 27 — List of ordinal features except Electrical

```python
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]
```

---
## Step 28 — Define transformations for various feature types

```python
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])
```

---
## Step 29 — Updated categorical imputer using SimpleImputer

```python
categorical_imputer = SimpleImputer(strategy="constant", fill_value="None")

ordinal_transformer = Pipeline([
    ("impute_ordinal", categorical_imputer),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline([
    ("impute_nominal", categorical_imputer),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 30 — Combined preprocessor for numeric, ordinal, nominal, and specific electrical data

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("electrical", electrical_transformer, ["Electrical"]),
        ("num", numeric_transformer, numeric_features),
        ("ordinal", ordinal_transformer, ordinal_except_electrical),
        ("nominal", categorical_transformer, nominal_features)
])

model = GradientBoostingRegressor(n_estimators=200, random_state=42)

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", model)
])
```

---
## Step 31 — Parameter distribution for RandomizedSearchCV

```python
param_dist = {
```

---
## Step 32 — Uniform distribution between 0.001 and 0.3

```python
"regressor__learning_rate": uniform(0.001, 0.299)
}
```

---
## Step 33 — Setup the RandomizedSearchCV

```python
random_search = RandomizedSearchCV(model_pipeline, param_distributions=param_dist,
                                   n_iter=50, cv=5, scoring="r2", verbose=1,
                                   random_state=42)
```

---
## Step 34 — Fit the RandomizedSearchCV to the data

```python
random_search.fit(Ames.drop(columns="SalePrice"), Ames["SalePrice"])
```

---
## Step 35 — Best parameters and best score from Random Search

```python
print("Best parameters (Random Search):", random_search.best_params_)
print("Best score (Random Search):", round(random_search.best_score_, 4))
```

---
## Learning Notes / 学习笔记

- **概念**: Experiment with RandomizedSearchCV 是机器学习中的常用技术。  
  *Experiment with RandomizedSearchCV is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `GradientBoosting` | 梯度提升算法 | Gradient Boosting algorithm |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Randomized Search / 03 Randomized Search
# Complete Code / 完整代码
# ===============================

# Experiment with RandomizedSearchCV
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

Ames = pd.read_csv("Ames.csv")

# Adjust data types for categorical variables
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")

# Exclude "PID" and "SalePrice" from features and handle the "Electrical" column
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]

# Manually specify the categories for ordinal encoding according to the data dictionary
ordinal_order = {
    # Electrical system
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    # General shape of property
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    # Type of utilities available
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    # Slope of property
    "LandSlope": ["Sev", "Mod", "Gtl"],
    # Evaluates the quality of the material on the exterior
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Evaluates the present condition of the material on the exterior
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Height of the basement
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # General condition of the basement
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Walkout or garden level basement walls
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    # Quality of basement finished area
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    # Quality of second basement finished area
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    # Heating quality and condition
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Kitchen quality
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Home functionality
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    # Fireplace quality
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Interior finish of the garage
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    # Garage quality
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Garage condition
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Paved driveway
    "PavedDrive": ["N", "P", "Y"],
    # Pool quality
    "PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
    # Fence quality
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"]
}

# Extract list of ALL ordinal features from dictionary
ordinal_features = list(ordinal_order.keys())

# List of ordinal features except Electrical
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]

# Define transformations for various feature types
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])

# Updated categorical imputer using SimpleImputer
categorical_imputer = SimpleImputer(strategy="constant", fill_value="None")

ordinal_transformer = Pipeline([
    ("impute_ordinal", categorical_imputer),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline([
    ("impute_nominal", categorical_imputer),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessor for numeric, ordinal, nominal, and specific electrical data
preprocessor = ColumnTransformer(
    transformers=[
        ("electrical", electrical_transformer, ["Electrical"]),
        ("num", numeric_transformer, numeric_features),
        ("ordinal", ordinal_transformer, ordinal_except_electrical),
        ("nominal", categorical_transformer, nominal_features)
])

model = GradientBoostingRegressor(n_estimators=200, random_state=42)

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", model)
])

# Parameter distribution for RandomizedSearchCV
param_dist = {
    # Uniform distribution between 0.001 and 0.3
    "regressor__learning_rate": uniform(0.001, 0.299)
}

# Setup the RandomizedSearchCV
random_search = RandomizedSearchCV(model_pipeline, param_distributions=param_dist,
                                   n_iter=50, cv=5, scoring="r2", verbose=1,
                                   random_state=42)

# Fit the RandomizedSearchCV to the data
random_search.fit(Ames.drop(columns="SalePrice"), Ames["SalePrice"])

# Best parameters and best score from Random Search
print("Best parameters (Random Search):", random_search.best_params_)
print("Best score (Random Search):", round(random_search.best_score_, 4))
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Lr And Tree

# 05 — Lr And Tree / 决策树

**Chapter 15 — File 5 of 6 / 第15章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Adjust data types for categorical variables**.

本脚本演示 **Adjust data types for categorical variables**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — Step 1

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

Ames = pd.read_csv("Ames.csv")
```

---
## Step 2 — Adjust data types for categorical variables

```python
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")
```

---
## Step 3 — Exclude "PID" and "SalePrice" from features and handle the "Electrical" column

```python
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]
```

---
## Step 4 — Manually specify the categories for ordinal encoding according to the data dictionary

```python
ordinal_order = {
```

---
## Step 5 — Electrical system

```python
"Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
```

---
## Step 6 — General shape of property

```python
"LotShape": ["IR3", "IR2", "IR1", "Reg"],
```

---
## Step 7 — Type of utilities available

```python
"Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
```

---
## Step 8 — Slope of property

```python
"LandSlope": ["Sev", "Mod", "Gtl"],
```

---
## Step 9 — Evaluates the quality of the material on the exterior

```python
"ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 10 — Evaluates the present condition of the material on the exterior

```python
"ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 11 — Height of the basement

```python
"BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 12 — General condition of the basement

```python
"BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 13 — Walkout or garden level basement walls

```python
"BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
```

---
## Step 14 — Quality of basement finished area

```python
"BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 15 — Quality of second basement finished area

```python
"BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 16 — Heating quality and condition

```python
"HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 17 — Kitchen quality

```python
"KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 18 — Home functionality

```python
"Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
```

---
## Step 19 — Fireplace quality

```python
"FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 20 — Interior finish of the garage

```python
"GarageFinish": ["None", "Unf", "RFn", "Fin"],
```

---
## Step 21 — Garage quality

```python
"GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 22 — Garage condition

```python
"GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 23 — Paved driveway

```python
"PavedDrive": ["N", "P", "Y"],
```

---
## Step 24 — Pool quality

```python
"PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 25 — Fence quality

```python
"Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"]
}
```

---
## Step 26 — Extract list of ALL ordinal features from dictionary

```python
ordinal_features = list(ordinal_order.keys())
```

---
## Step 27 — List of ordinal features except Electrical

```python
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]
```

---
## Step 28 — Define transformations for various feature types

```python
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])
```

---
## Step 29 — Updated categorical imputer using SimpleImputer

```python
categorical_imputer = SimpleImputer(strategy="constant", fill_value="None")

ordinal_transformer = Pipeline([
    ("impute_ordinal", categorical_imputer),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline([
    ("impute_nominal", categorical_imputer),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 30 — Combined preprocessor for numeric, ordinal, nominal, and specific electrical data

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("electrical", electrical_transformer, ["Electrical"]),
        ("num", numeric_transformer, numeric_features),
        ("ordinal", ordinal_transformer, ordinal_except_electrical),
        ("nominal", categorical_transformer, nominal_features)
])
```

---
## Step 31 — "preprocessor" is already set up as your preprocessing pipeline

```python
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42))
])
```

---
## Step 32 — Parameter distribution for RandomizedSearchCV

```python
param_dist = {
```

---
## Step 33 — Uniform distribution between 0.001 and 0.3

```python
"regressor__learning_rate": uniform(0.001, 0.299),
```

---
## Step 34 — Uniform distribution of integers from 100 to 500

```python
"regressor__n_estimators": randint(100, 501)
}
```

---
## Step 35 — Setup the RandomizedSearchCV

```python
random_search = RandomizedSearchCV(model_pipeline, param_distributions=param_dist,
                                   n_iter=50, cv=5, scoring="r2", verbose=1,
                                   random_state=42)
```

---
## Step 36 — Fit the RandomizedSearchCV to the data

```python
random_search.fit(Ames.drop(columns="SalePrice"), Ames["SalePrice"])
```

---
## Step 37 — Best parameters and best score from Random Search

```python
print("Best parameters (Random Search):", random_search.best_params_)
print("Best score (Random Search):", round((random_search.best_score_), 4))
```

---
## Learning Notes / 学习笔记

- **概念**: Adjust data types for categorical variables 是机器学习中的常用技术。  
  *Adjust data types for categorical variables is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `GradientBoosting` | 梯度提升算法 | Gradient Boosting algorithm |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lr And Tree / 决策树
# Complete Code / 完整代码
# ===============================

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

Ames = pd.read_csv("Ames.csv")

# Adjust data types for categorical variables
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")

# Exclude "PID" and "SalePrice" from features and handle the "Electrical" column
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]

# Manually specify the categories for ordinal encoding according to the data dictionary
ordinal_order = {
    # Electrical system
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    # General shape of property
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    # Type of utilities available
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    # Slope of property
    "LandSlope": ["Sev", "Mod", "Gtl"],
    # Evaluates the quality of the material on the exterior
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Evaluates the present condition of the material on the exterior
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Height of the basement
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # General condition of the basement
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Walkout or garden level basement walls
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    # Quality of basement finished area
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    # Quality of second basement finished area
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    # Heating quality and condition
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Kitchen quality
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Home functionality
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    # Fireplace quality
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Interior finish of the garage
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    # Garage quality
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Garage condition
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Paved driveway
    "PavedDrive": ["N", "P", "Y"],
    # Pool quality
    "PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
    # Fence quality
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"]
}

# Extract list of ALL ordinal features from dictionary
ordinal_features = list(ordinal_order.keys())

# List of ordinal features except Electrical
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]

# Define transformations for various feature types
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])

# Updated categorical imputer using SimpleImputer
categorical_imputer = SimpleImputer(strategy="constant", fill_value="None")

ordinal_transformer = Pipeline([
    ("impute_ordinal", categorical_imputer),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline([
    ("impute_nominal", categorical_imputer),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessor for numeric, ordinal, nominal, and specific electrical data
preprocessor = ColumnTransformer(
    transformers=[
        ("electrical", electrical_transformer, ["Electrical"]),
        ("num", numeric_transformer, numeric_features),
        ("ordinal", ordinal_transformer, ordinal_except_electrical),
        ("nominal", categorical_transformer, nominal_features)
])

# "preprocessor" is already set up as your preprocessing pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42))
])


# Parameter distribution for RandomizedSearchCV
param_dist = {
    # Uniform distribution between 0.001 and 0.3
    "regressor__learning_rate": uniform(0.001, 0.299),
    # Uniform distribution of integers from 100 to 500
    "regressor__n_estimators": randint(100, 501)
}

# Setup the RandomizedSearchCV
random_search = RandomizedSearchCV(model_pipeline, param_distributions=param_dist,
                                   n_iter=50, cv=5, scoring="r2", verbose=1,
                                   random_state=42)

# Fit the RandomizedSearchCV to the data
random_search.fit(Ames.drop(columns="SalePrice"), Ames["SalePrice"])

# Best parameters and best score from Random Search
print("Best parameters (Random Search):", random_search.best_params_)
print("Best score (Random Search):", round((random_search.best_score_), 4))
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Optimal Regressor

# 06 — Optimal Regressor / 优化

**Chapter 15 — File 6 of 6 / 第15章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Adjust data types for categorical variables**.

本脚本演示 **Adjust data types for categorical variables**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — Step 1

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

Ames = pd.read_csv("Ames.csv")
```

---
## Step 2 — Adjust data types for categorical variables

```python
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")
```

---
## Step 3 — Exclude "PID" and "SalePrice" from features and handle the "Electrical" column

```python
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]
```

---
## Step 4 — Manually specify the categories for ordinal encoding according to the data dictionary

```python
ordinal_order = {
```

---
## Step 5 — Electrical system

```python
"Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
```

---
## Step 6 — General shape of property

```python
"LotShape": ["IR3", "IR2", "IR1", "Reg"],
```

---
## Step 7 — Type of utilities available

```python
"Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
```

---
## Step 8 — Slope of property

```python
"LandSlope": ["Sev", "Mod", "Gtl"],
```

---
## Step 9 — Evaluates the quality of the material on the exterior

```python
"ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 10 — Evaluates the present condition of the material on the exterior

```python
"ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 11 — Height of the basement

```python
"BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 12 — General condition of the basement

```python
"BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 13 — Walkout or garden level basement walls

```python
"BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
```

---
## Step 14 — Quality of basement finished area

```python
"BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 15 — Quality of second basement finished area

```python
"BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 16 — Heating quality and condition

```python
"HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 17 — Kitchen quality

```python
"KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 18 — Home functionality

```python
"Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
```

---
## Step 19 — Fireplace quality

```python
"FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 20 — Interior finish of the garage

```python
"GarageFinish": ["None", "Unf", "RFn", "Fin"],
```

---
## Step 21 — Garage quality

```python
"GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 22 — Garage condition

```python
"GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 23 — Paved driveway

```python
"PavedDrive": ["N", "P", "Y"],
```

---
## Step 24 — Pool quality

```python
"PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 25 — Fence quality

```python
"Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"]
}
```

---
## Step 26 — Extract list of ALL ordinal features from dictionary

```python
ordinal_features = list(ordinal_order.keys())
```

---
## Step 27 — List of ordinal features except Electrical

```python
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]
```

---
## Step 28 — Define transformations for various feature types

```python
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])
```

---
## Step 29 — Updated categorical imputer using SimpleImputer

```python
categorical_imputer = SimpleImputer(strategy="constant", fill_value="None")

ordinal_transformer = Pipeline([
    ("impute_ordinal", categorical_imputer),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline([
    ("impute_nominal", categorical_imputer),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 30 — Combined preprocessor for numeric, ordinal, nominal, and specific electrical data

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("electrical", electrical_transformer, ["Electrical"]),
        ("num", numeric_transformer, numeric_features),
        ("ordinal", ordinal_transformer, ordinal_except_electrical),
        ("nominal", categorical_transformer, nominal_features)
])
```

---
## Step 31 — Cross check model performance of gradient boosting regressor with tuned parameters
"preprocessor" is already set up as your preprocessing pipeline

```python
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(n_estimators=287,
                                            learning_rate=0.12055843054286139,
                                            random_state=42))
])
```

---
## Step 32 — Using the full dataset X, y

```python
X = Ames.drop(columns="SalePrice")
y = Ames["SalePrice"]
```

---
## Step 33 — Perform 5-fold cross-validation

```python
cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring="r2")
```

---
## Step 34 — Output the mean cross-validated score of tuned model

```python
print("Performance of gradient boosting regressor with tuned parameters:",
      round(cv_scores.mean(), 4))
```

---
## Learning Notes / 学习笔记

- **概念**: Adjust data types for categorical variables 是机器学习中的常用技术。  
  *Adjust data types for categorical variables is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GradientBoosting` | 梯度提升算法 | Gradient Boosting algorithm |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Optimal Regressor / 优化
# Complete Code / 完整代码
# ===============================

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

Ames = pd.read_csv("Ames.csv")

# Adjust data types for categorical variables
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")

# Exclude "PID" and "SalePrice" from features and handle the "Electrical" column
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]

# Manually specify the categories for ordinal encoding according to the data dictionary
ordinal_order = {
    # Electrical system
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    # General shape of property
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    # Type of utilities available
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    # Slope of property
    "LandSlope": ["Sev", "Mod", "Gtl"],
    # Evaluates the quality of the material on the exterior
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Evaluates the present condition of the material on the exterior
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Height of the basement
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # General condition of the basement
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Walkout or garden level basement walls
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    # Quality of basement finished area
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    # Quality of second basement finished area
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    # Heating quality and condition
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Kitchen quality
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Home functionality
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    # Fireplace quality
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Interior finish of the garage
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    # Garage quality
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Garage condition
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Paved driveway
    "PavedDrive": ["N", "P", "Y"],
    # Pool quality
    "PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
    # Fence quality
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"]
}

# Extract list of ALL ordinal features from dictionary
ordinal_features = list(ordinal_order.keys())

# List of ordinal features except Electrical
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]

# Define transformations for various feature types
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])

# Updated categorical imputer using SimpleImputer
categorical_imputer = SimpleImputer(strategy="constant", fill_value="None")

ordinal_transformer = Pipeline([
    ("impute_ordinal", categorical_imputer),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline([
    ("impute_nominal", categorical_imputer),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessor for numeric, ordinal, nominal, and specific electrical data
preprocessor = ColumnTransformer(
    transformers=[
        ("electrical", electrical_transformer, ["Electrical"]),
        ("num", numeric_transformer, numeric_features),
        ("ordinal", ordinal_transformer, ordinal_except_electrical),
        ("nominal", categorical_transformer, nominal_features)
])


# Cross check model performance of gradient boosting regressor with tuned parameters

# "preprocessor" is already set up as your preprocessing pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(n_estimators=287,
                                            learning_rate=0.12055843054286139,
                                            random_state=42))
])

# Using the full dataset X, y
X = Ames.drop(columns="SalePrice")
y = Ames["SalePrice"]

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring="r2")

# Output the mean cross-validated score of tuned model
print("Performance of gradient boosting regressor with tuned parameters:",
      round(cv_scores.mean(), 4))
```

---
