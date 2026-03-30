# 进阶数据科学
## Chapter 14

---

### Preprocessing

# 01 — Preprocessing / 01 Preprocessing

**Chapter 14 — File 1 of 7 / 第14章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Import necessary libraries for preprocessing**.

本脚本演示 **Import necessary libraries for preprocessing**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Step 1 — Import necessary libraries for preprocessing

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Convert the below numeric features to categorical features

```python
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")
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
## Step 29 — Helper function to fill "None" for missing categorical data

```python
def fill_none(X):
    return X.infer_objects(copy=False).fillna("None")
```

---
## Step 30 — Pipeline for "Electrical": Fill missing value with mode then apply ordinal encoding

```python
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])
```

---
## Step 31 — Pipeline for numeric features: Impute missing values using mean

```python
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])
```

---
## Step 32 — Pipeline for ordinal features: Fill missing values with "None" then apply
ordinal encoding

```python
ordinal_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])
```

---
## Step 33 — Pipeline for nominal categorical features: Fill missing values with "None" then apply
one-hot encoding

```python
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 34 — Combined preprocessor for numeric, ordinal, nominal, and specific electrical data

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
## Step 35 — Apply the preprocessing pipeline to Ames

```python
transformed_data = preprocessor.fit_transform(Ames).toarray()
```

---
## Step 36 — Generate column names for the one-hot encoded features

```python
onehot_features = preprocessor.named_transformers_["nominal"] \
                              .named_steps["onehot"].get_feature_names_out()
```

---
## Step 37 — Combine all feature names

```python
all_feature_names = ["Electrical"] + list(numeric_features) + \
                    list(ordinal_except_electrical) + list(onehot_features)
```

---
## Step 38 — Convert the transformed array to a DataFrame

```python
transformed_df = pd.DataFrame(transformed_data, columns=all_feature_names)
```

---
## Learning Notes / 学习笔记

- **概念**: Import necessary libraries for preprocessing 是机器学习中的常用技术。  
  *Import necessary libraries for preprocessing is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `fillna` | 填充缺失值 | Fill missing values |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Preprocessing / 01 Preprocessing
# Complete Code / 完整代码
# ===============================

# Import necessary libraries for preprocessing
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Convert the below numeric features to categorical features
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")

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

# Helper function to fill "None" for missing categorical data
def fill_none(X):
    return X.infer_objects(copy=False).fillna("None")

# Pipeline for "Electrical": Fill missing value with mode then apply ordinal encoding
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

# Pipeline for numeric features: Impute missing values using mean
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])

# Pipeline for ordinal features: Fill missing values with "None" then apply
# ordinal encoding
ordinal_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

# Pipeline for nominal categorical features: Fill missing values with "None" then apply
# one-hot encoding
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
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

# Apply the preprocessing pipeline to Ames
transformed_data = preprocessor.fit_transform(Ames).toarray()

# Generate column names for the one-hot encoded features
onehot_features = preprocessor.named_transformers_["nominal"] \
                              .named_steps["onehot"].get_feature_names_out()

# Combine all feature names
all_feature_names = ["Electrical"] + list(numeric_features) + \
                    list(ordinal_except_electrical) + list(onehot_features)

# Convert the transformed array to a DataFrame
transformed_df = pd.DataFrame(transformed_data, columns=all_feature_names)
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Show

# 02 — Show / 02 Show

**Chapter 14 — File 2 of 7 / 第14章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Load the dataset**.

本脚本演示 **Load the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Convert the below numeric features to categorical features

```python
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")
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
## Step 29 — Helper function to fill "None" for missing categorical data

```python
def fill_none(X):
    return X.infer_objects(copy=False).fillna("None")
```

---
## Step 30 — Pipeline for "Electrical": Fill missing value with mode then apply ordinal encoding

```python
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])
```

---
## Step 31 — Pipeline for numeric features: Impute missing values using mean

```python
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])
```

---
## Step 32 — Pipeline for ordinal features: Fill missing values with "None" then apply
ordinal encoding

```python
ordinal_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])
```

---
## Step 33 — Pipeline for nominal categorical features: Fill missing values with "None" then apply
one-hot encoding

```python
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 34 — Combined preprocessor for numeric, ordinal, nominal, and specific electrical data

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
## Step 35 — Apply the preprocessing pipeline to Ames

```python
transformed_data = preprocessor.fit_transform(Ames).toarray()
```

---
## Step 36 — Generate column names for the one-hot encoded features

```python
onehot_features = preprocessor.named_transformers_["nominal"] \
                              .named_steps["onehot"].get_feature_names_out()
```

---
## Step 37 — Combine all feature names

```python
all_feature_names = ["Electrical"] + list(numeric_features) + \
                    list(ordinal_except_electrical) + list(onehot_features)
```

---
## Step 38 — Convert the transformed array to a DataFrame

```python
transformed_df = pd.DataFrame(transformed_data, columns=all_feature_names)
```

---
## Step 39 — # Optional command for expanded view
pd.set_option('display.max_columns', None)
View the transformation

```python
print(transformed_df)
```

---
## Learning Notes / 学习笔记

- **概念**: Load the dataset 是机器学习中的常用技术。  
  *Load the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `fillna` | 填充缺失值 | Fill missing values |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Show / 02 Show
# Complete Code / 完整代码
# ===============================

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Convert the below numeric features to categorical features
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")

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

# Helper function to fill "None" for missing categorical data
def fill_none(X):
    return X.infer_objects(copy=False).fillna("None")

# Pipeline for "Electrical": Fill missing value with mode then apply ordinal encoding
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

# Pipeline for numeric features: Impute missing values using mean
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])

# Pipeline for ordinal features: Fill missing values with "None" then apply
# ordinal encoding
ordinal_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

# Pipeline for nominal categorical features: Fill missing values with "None" then apply
# one-hot encoding
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
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

# Apply the preprocessing pipeline to Ames
transformed_data = preprocessor.fit_transform(Ames).toarray()

# Generate column names for the one-hot encoded features
onehot_features = preprocessor.named_transformers_["nominal"] \
                              .named_steps["onehot"].get_feature_names_out()

# Combine all feature names
all_feature_names = ["Electrical"] + list(numeric_features) + \
                    list(ordinal_except_electrical) + list(onehot_features)

# Convert the transformed array to a DataFrame
transformed_df = pd.DataFrame(transformed_data, columns=all_feature_names)

# # Optional command for expanded view
# pd.set_option('display.max_columns', None)

# View the transformation
print(transformed_df)
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Cross Check

# 03 — Cross Check / 03 Cross Check

**Chapter 14 — File 3 of 7 / 第14章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Load the dataset**.

本脚本演示 **Load the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Convert the below numeric features to categorical features

```python
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")
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

ordinal_features = list(ordinal_order.keys())
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]

print(len(numeric_features) +
      len(ordinal_features) +
      Ames[nominal_features].infer_objects(copy=False).fillna("None").nunique().sum())
```

---
## Learning Notes / 学习笔记

- **概念**: Load the dataset 是机器学习中的常用技术。  
  *Load the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `fillna` | 填充缺失值 | Fill missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cross Check / 03 Cross Check
# Complete Code / 完整代码
# ===============================

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Convert the below numeric features to categorical features
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")

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

ordinal_features = list(ordinal_order.keys())
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]

print(len(numeric_features) +
      len(ordinal_features) +
      Ames[nominal_features].infer_objects(copy=False).fillna("None").nunique().sum())
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Tree

# 04 — Tree / 决策树

**Chapter 14 — File 4 of 7 / 第14章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Load the dataset**.

本脚本演示 **Load the dataset**。

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
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Convert the below numeric features to categorical features

```python
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")
```

---
## Step 4 — Identify numeric and categorical features from the dataset based on the data type

```python
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
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
## Step 29 — Helper function to fill "None" for missing categorical data

```python
def fill_none(X):
    return X.infer_objects(copy=False).fillna("None")
```

---
## Step 30 — Pipeline for "Electrical": Fill missing value with mode then apply ordinal encoding

```python
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])
```

---
## Step 31 — Pipeline for numeric features: Impute missing values using mean

```python
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])
```

---
## Step 32 — Pipeline for ordinal features: Fill missing values with "None" then apply
ordinal encoding

```python
ordinal_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])
```

---
## Step 33 — Pipeline for nominal categorical features: Fill missing values with "None" then apply
one-hot encoding

```python
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 34 — Combined preprocessor for numeric, ordinal, nominal, and specific electrical data

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
## Step 35 — Define the full model pipeline

```python
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(random_state=42))
])
```

---
## Step 36 — Evaluate the model using cross-validation

```python
scores = cross_val_score(model_pipeline, Ames.drop(columns="SalePrice"), Ames["SalePrice"])
```

---
## Step 37 — Output the result

```python
print("Decision Tree Regressor Mean CV R^2:", round(scores.mean(),4))
```

---
## Learning Notes / 学习笔记

- **概念**: Load the dataset 是机器学习中的常用技术。  
  *Load the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `fillna` | 填充缺失值 | Fill missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Tree / 决策树
# Complete Code / 完整代码
# ===============================

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Convert the below numeric features to categorical features
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")

# Identify numeric and categorical features from the dataset based on the data type
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])

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

# Helper function to fill "None" for missing categorical data
def fill_none(X):
    return X.infer_objects(copy=False).fillna("None")

# Pipeline for "Electrical": Fill missing value with mode then apply ordinal encoding
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

# Pipeline for numeric features: Impute missing values using mean
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])

# Pipeline for ordinal features: Fill missing values with "None" then apply
# ordinal encoding
ordinal_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

# Pipeline for nominal categorical features: Fill missing values with "None" then apply
# one-hot encoding
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
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

# Define the full model pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

# Evaluate the model using cross-validation
scores = cross_val_score(model_pipeline, Ames.drop(columns="SalePrice"), Ames["SalePrice"])

# Output the result
print("Decision Tree Regressor Mean CV R^2:", round(scores.mean(),4))
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Bagging

# 05 — Bagging / 装袋方法

**Chapter 14 — File 5 of 7 / 第14章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Import Bagging Regressor and compare how performance is affected by Bagging**.

本脚本演示 **Import Bagging Regressor and compare how performance is affected by Bagging**。

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
## Step 1 — Import Bagging Regressor and compare how performance is affected by Bagging
(i.e. increasing number of trees)

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Convert the below numeric features to categorical features

```python
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")
```

---
## Step 4 — Identify numeric and categorical features from the dataset based on the data type

```python
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
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
## Step 29 — Helper function to fill "None" for missing categorical data

```python
def fill_none(X):
    return X.infer_objects(copy=False).fillna("None")
```

---
## Step 30 — Pipeline for "Electrical": Fill missing value with mode then apply ordinal encoding

```python
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])
```

---
## Step 31 — Pipeline for numeric features: Impute missing values using mean

```python
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])
```

---
## Step 32 — Pipeline for ordinal features: Fill missing values with "None" then apply
ordinal encoding

```python
ordinal_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])
```

---
## Step 33 — Pipeline for nominal categorical features: Fill missing values with "None" then apply
one-hot encoding

```python
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 34 — Combined preprocessor for numeric, ordinal, nominal, and specific electrical data

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("electrical", electrical_transformer, ["Electrical"]),
        ("num", numeric_transformer, numeric_features),
        ("ordinal", ordinal_transformer, ordinal_except_electrical),
        ("nominal", categorical_transformer, nominal_features)
])

models = {
    "Decision Tree (1 Tree)":
        DecisionTreeRegressor(random_state=42),
    "Bagging Regressor (10 Trees)":
        BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42),
                         n_estimators=10, random_state=42)
}

results = {}
for name, model in models.items():
```

---
## Step 35 — Define the full model pipeline for each model

```python
model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
```

---
## Step 36 — Perform cross-validation

```python
scores = cross_val_score(model_pipeline,
                             Ames.drop(columns="SalePrice"), Ames["SalePrice"])
```

---
## Step 37 — Store and print the mean of the scores

```python
results[name] = round(scores.mean(), 4)
```

---
## Step 38 — Output the cross-validation scores

```python
print("Cross-validation scores:", results)
```

---
## Learning Notes / 学习笔记

- **概念**: Import Bagging Regressor and compare how performance is affected by Bagging 是机器学习中的常用技术。  
  *Import Bagging Regressor and compare how performance is affected by Bagging is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `fillna` | 填充缺失值 | Fill missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Bagging / 装袋方法
# Complete Code / 完整代码
# ===============================

# Import Bagging Regressor and compare how performance is affected by Bagging
# (i.e. increasing number of trees)

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Convert the below numeric features to categorical features
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")

# Identify numeric and categorical features from the dataset based on the data type
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])

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

# Helper function to fill "None" for missing categorical data
def fill_none(X):
    return X.infer_objects(copy=False).fillna("None")

# Pipeline for "Electrical": Fill missing value with mode then apply ordinal encoding
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

# Pipeline for numeric features: Impute missing values using mean
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])

# Pipeline for ordinal features: Fill missing values with "None" then apply
# ordinal encoding
ordinal_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

# Pipeline for nominal categorical features: Fill missing values with "None" then apply
# one-hot encoding
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
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

models = {
    "Decision Tree (1 Tree)":
        DecisionTreeRegressor(random_state=42),
    "Bagging Regressor (10 Trees)":
        BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42),
                         n_estimators=10, random_state=42)
}

results = {}
for name, model in models.items():
    # Define the full model pipeline for each model
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    # Perform cross-validation
    scores = cross_val_score(model_pipeline,
                             Ames.drop(columns="SalePrice"), Ames["SalePrice"])

    # Store and print the mean of the scores
    results[name] = round(scores.mean(), 4)

# Output the cross-validation scores
print("Cross-validation scores:", results)
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Number Of Trees

# 06 — Number Of Trees / 决策树

**Chapter 14 — File 6 of 7 / 第14章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Compare how performance is affected by Bagging in increments of 10 trees**.

本脚本演示 **Compare how performance is affected by Bagging in increments of 10 trees**。

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
## Step 1 — Compare how performance is affected by Bagging in increments of 10 trees

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Convert the below numeric features to categorical features

```python
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")
```

---
## Step 4 — Identify numeric and categorical features from the dataset based on the data type

```python
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
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
## Step 29 — Helper function to fill "None" for missing categorical data

```python
def fill_none(X):
    return X.infer_objects(copy=False).fillna("None")
```

---
## Step 30 — Pipeline for "Electrical": Fill missing value with mode then apply ordinal encoding

```python
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])
```

---
## Step 31 — Pipeline for numeric features: Impute missing values using mean

```python
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])
```

---
## Step 32 — Pipeline for ordinal features: Fill missing values with "None" then apply
ordinal encoding

```python
ordinal_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])
```

---
## Step 33 — Pipeline for nominal categorical features: Fill missing values with "None" then apply
one-hot encoding

```python
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 34 — Combined preprocessor for numeric, ordinal, nominal, and specific electrical data

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
## Step 35 — Number of trees to test

```python
n_trees = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
```

---
## Step 36 — Define the model pipelines with various regressors

```python
models = {
    'Decision Tree (1 Tree)': DecisionTreeRegressor(random_state=42)
}
```

---
## Step 37 — Adding Bagging models for each tree count

```python
for n in n_trees:
    models[f'Bagging Regressor {n} Trees'] = BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=42),
        n_estimators=n,
        random_state=42
    )

results = {}
for name, model in models.items():
```

---
## Step 38 — Define the full model pipeline for each model

```python
model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
```

---
## Step 39 — Perform cross-validation

```python
scores = cross_val_score(model_pipeline,
                             Ames.drop(columns='SalePrice'), Ames['SalePrice'])
```

---
## Step 40 — Store and print the mean of the scores

```python
results[name] = round(scores.mean(), 4)
```

---
## Step 41 — Output the cross-validation scores

```python
print("Cross-validation scores:")
for name, score in results.items():
    print(f"{name}: {score}")
```

---
## Learning Notes / 学习笔记

- **概念**: Compare how performance is affected by Bagging in increments of 10 trees 是机器学习中的常用技术。  
  *Compare how performance is affected by Bagging in increments of 10 trees is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `fillna` | 填充缺失值 | Fill missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Number Of Trees / 决策树
# Complete Code / 完整代码
# ===============================

# Compare how performance is affected by Bagging in increments of 10 trees

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Convert the below numeric features to categorical features
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")

# Identify numeric and categorical features from the dataset based on the data type
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])

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

# Helper function to fill "None" for missing categorical data
def fill_none(X):
    return X.infer_objects(copy=False).fillna("None")

# Pipeline for "Electrical": Fill missing value with mode then apply ordinal encoding
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

# Pipeline for numeric features: Impute missing values using mean
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])

# Pipeline for ordinal features: Fill missing values with "None" then apply
# ordinal encoding
ordinal_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

# Pipeline for nominal categorical features: Fill missing values with "None" then apply
# one-hot encoding
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
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

# Number of trees to test
n_trees = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Define the model pipelines with various regressors
models = {
    'Decision Tree (1 Tree)': DecisionTreeRegressor(random_state=42)
}

# Adding Bagging models for each tree count
for n in n_trees:
    models[f'Bagging Regressor {n} Trees'] = BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=42),
        n_estimators=n,
        random_state=42
    )

results = {}
for name, model in models.items():
    # Define the full model pipeline for each model
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # Perform cross-validation
    scores = cross_val_score(model_pipeline,
                             Ames.drop(columns='SalePrice'), Ames['SalePrice'])

    # Store and print the mean of the scores
    results[name] = round(scores.mean(), 4)

# Output the cross-validation scores
print("Cross-validation scores:")
for name, score in results.items():
    print(f"{name}: {score}")
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Compare

# 07 — Compare / 07 Compare

**Chapter 14 — File 7 of 7 / 第14章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Evaluate performance of Random Forest against Bagging Regressor**.

本脚本演示 **Evaluate performance of Random Forest against Bagging Regressor**。

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
## Step 1 — Evaluate performance of Random Forest against Bagging Regressor

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Convert the below numeric features to categorical features

```python
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")
```

---
## Step 4 — Identify numeric and categorical features from the dataset based on the data type

```python
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
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
## Step 29 — Helper function to fill "None" for missing categorical data

```python
def fill_none(X):
    return X.infer_objects(copy=False).fillna("None")
```

---
## Step 30 — Pipeline for "Electrical": Fill missing value with mode then apply ordinal encoding

```python
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])
```

---
## Step 31 — Pipeline for numeric features: Impute missing values using mean

```python
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])
```

---
## Step 32 — Pipeline for ordinal features: Fill missing values with "None" then apply
ordinal encoding

```python
ordinal_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])
```

---
## Step 33 — Pipeline for nominal categorical features: Fill missing values with "None" then apply
one-hot encoding

```python
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 34 — Combined preprocessor for numeric, ordinal, nominal, and specific electrical data

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
## Step 35 — Number of trees to test

```python
n_trees = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
```

---
## Step 36 — Define the model pipelines with various regressors

```python
models = {
    "Decision Tree (1 Tree)": DecisionTreeRegressor(random_state=42),
}
```

---
## Step 37 — Adding Bagging and Random Forest models for each tree count

```python
for n in n_trees:
    models[f"Bagging Regressor {n} Trees"] = BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=42),
        n_estimators=n,
        random_state=42
    )
    models[f"Random Forest {n} Trees"] = RandomForestRegressor(
        n_estimators=n,
        random_state=42
    )

results = {}
for name, model in models.items():
```

---
## Step 38 — Define the full model pipeline for each model

```python
model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
```

---
## Step 39 — Perform cross-validation

```python
scores = cross_val_score(model_pipeline,
                             Ames.drop(columns="SalePrice"), Ames["SalePrice"])
```

---
## Step 40 — Store and print the mean of the scores

```python
results[name] = round(scores.mean(), 4)
```

---
## Step 41 — Output the cross-validation scores

```python
print("Cross-validation scores:")
for name, score in results.items():
    print(f"{name}: {score}")
```

---
## Learning Notes / 学习笔记

- **概念**: Evaluate performance of Random Forest against Bagging Regressor 是机器学习中的常用技术。  
  *Evaluate performance of Random Forest against Bagging Regressor is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `fillna` | 填充缺失值 | Fill missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Compare / 07 Compare
# Complete Code / 完整代码
# ===============================

# Evaluate performance of Random Forest against Bagging Regressor

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Convert the below numeric features to categorical features
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")

# Identify numeric and categorical features from the dataset based on the data type
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])

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

# Helper function to fill "None" for missing categorical data
def fill_none(X):
    return X.infer_objects(copy=False).fillna("None")

# Pipeline for "Electrical": Fill missing value with mode then apply ordinal encoding
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

# Pipeline for numeric features: Impute missing values using mean
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])

# Pipeline for ordinal features: Fill missing values with "None" then apply
# ordinal encoding
ordinal_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

# Pipeline for nominal categorical features: Fill missing values with "None" then apply
# one-hot encoding
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
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

# Number of trees to test
n_trees = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Define the model pipelines with various regressors
models = {
    "Decision Tree (1 Tree)": DecisionTreeRegressor(random_state=42),
}

# Adding Bagging and Random Forest models for each tree count
for n in n_trees:
    models[f"Bagging Regressor {n} Trees"] = BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=42),
        n_estimators=n,
        random_state=42
    )
    models[f"Random Forest {n} Trees"] = RandomForestRegressor(
        n_estimators=n,
        random_state=42
    )

results = {}
for name, model in models.items():
    # Define the full model pipeline for each model
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    # Perform cross-validation
    scores = cross_val_score(model_pipeline,
                             Ames.drop(columns="SalePrice"), Ames["SalePrice"])

    # Store and print the mean of the scores
    results[name] = round(scores.mean(), 4)

# Output the cross-validation scores
print("Cross-validation scores:")
for name, score in results.items():
    print(f"{name}: {score}")
```

---
