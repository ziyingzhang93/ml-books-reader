# 进阶数据科学
## Chapter 13

---

### Ordinal

# 01 — Ordinal / 01 Ordinal

**Chapter 13 — File 1 of 5 / 第13章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Import necessary libraries**.

本脚本演示 **Import necessary libraries**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Step 1 — Import necessary libraries

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Manually specify the categories for ordinal encoding according to the data dictionary

```python
ordinal_order = {
```

---
## Step 4 — Electrical system

```python
"Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
```

---
## Step 5 — General shape of property

```python
"LotShape": ["IR3", "IR2", "IR1", "Reg"],
```

---
## Step 6 — Type of utilities available

```python
"Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
```

---
## Step 7 — Slope of property

```python
"LandSlope": ["Sev", "Mod", "Gtl"],
```

---
## Step 8 — Evaluates the quality of the material on the exterior

```python
"ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 9 — Evaluates the present condition of the material on the exterior

```python
"ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 10 — Height of the basement

```python
"BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 11 — General condition of the basement

```python
"BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 12 — Walkout or garden level basement walls

```python
"BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
```

---
## Step 13 — Quality of basement finished area

```python
"BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 14 — Quality of second basement finished area

```python
"BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 15 — Heating quality and condition

```python
"HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 16 — Kitchen quality

```python
"KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 17 — Home functionality

```python
"Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
```

---
## Step 18 — Fireplace quality

```python
"FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 19 — Interior finish of the garage

```python
"GarageFinish": ["None", "Unf", "RFn", "Fin"],
```

---
## Step 20 — Garage quality

```python
"GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 21 — Garage condition

```python
"GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 22 — Paved driveway

```python
"PavedDrive": ["N", "P", "Y"],
```

---
## Step 23 — Pool quality

```python
"PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 24 — Fence quality

```python
"Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
}
```

---
## Step 25 — Extract list of ALL ordinal features from dictionary

```python
ordinal_features = list(ordinal_order.keys())
```

---
## Step 26 — List of ordinal features except Electrical

```python
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]
```

---
## Step 27 — Specific transformer for "Electrical" using the mode for imputation

```python
electrical_imputer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent"))
])
```

---
## Step 28 — Helper function to fill "None" for other ordinal features

```python
def fill_none(X):
    return X.fillna("None")
```

---
## Step 29 — Pipeline for ordinal features: Fill missing values with "None"

```python
ordinal_imputer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False))
])
```

---
## Step 30 — Preprocessor for filling missing values

```python
preprocessor_fill = ColumnTransformer(transformers=[
    ("electrical", electrical_imputer, ["Electrical"]),
    ("cat", ordinal_imputer, ordinal_except_electrical)
])
```

---
## Step 31 — Apply preprocessor for filling missing values

```python
Ames_ordinal = preprocessor_fill.fit_transform(Ames[ordinal_features])
```

---
## Step 32 — Convert back to DataFrame to apply OrdinalEncoder

```python
Ames_ordinal = pd.DataFrame(Ames_ordinal,
                            columns=["Electrical"] + ordinal_except_electrical)
```

---
## Step 33 — Apply Ordinal Encoding

```python
categories = [ordinal_order[feature] for feature in ordinal_features]
ordinal_encoder = OrdinalEncoder(categories=categories)
Ames_ordinal_encoded = ordinal_encoder.fit_transform(Ames_ordinal)
Ames_ordinal_encoded = pd.DataFrame(Ames_ordinal_encoded,
                                    columns=["Electrical"] + ordinal_except_electrical)
```

---
## Learning Notes / 学习笔记

- **概念**: Import necessary libraries 是机器学习中的常用技术。  
  *Import necessary libraries is a common technique in machine learning.*

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
# Ordinal / 01 Ordinal
# Complete Code / 完整代码
# ===============================

# Import necessary libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

# Load the dataset
Ames = pd.read_csv("Ames.csv")

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
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
}

# Extract list of ALL ordinal features from dictionary
ordinal_features = list(ordinal_order.keys())

# List of ordinal features except Electrical
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]

# Specific transformer for "Electrical" using the mode for imputation
electrical_imputer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent"))
])

# Helper function to fill "None" for other ordinal features
def fill_none(X):
    return X.fillna("None")

# Pipeline for ordinal features: Fill missing values with "None"
ordinal_imputer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False))
])

# Preprocessor for filling missing values
preprocessor_fill = ColumnTransformer(transformers=[
    ("electrical", electrical_imputer, ["Electrical"]),
    ("cat", ordinal_imputer, ordinal_except_electrical)
])

# Apply preprocessor for filling missing values
Ames_ordinal = preprocessor_fill.fit_transform(Ames[ordinal_features])

# Convert back to DataFrame to apply OrdinalEncoder
Ames_ordinal = pd.DataFrame(Ames_ordinal,
                            columns=["Electrical"] + ordinal_except_electrical)

# Apply Ordinal Encoding
categories = [ordinal_order[feature] for feature in ordinal_features]
ordinal_encoder = OrdinalEncoder(categories=categories)
Ames_ordinal_encoded = ordinal_encoder.fit_transform(Ames_ordinal)
Ames_ordinal_encoded = pd.DataFrame(Ames_ordinal_encoded,
                                    columns=["Electrical"] + ordinal_except_electrical)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Show Features

# 02 — Show Features / 特征工程

**Chapter 13 — File 2 of 5 / 第13章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Import necessary libraries**.

本脚本演示 **Import necessary libraries**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Step 1 — Import necessary libraries

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Manually specify the categories for ordinal encoding according to the data dictionary

```python
ordinal_order = {
```

---
## Step 4 — Electrical system

```python
"Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
```

---
## Step 5 — General shape of property

```python
"LotShape": ["IR3", "IR2", "IR1", "Reg"],
```

---
## Step 6 — Type of utilities available

```python
"Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
```

---
## Step 7 — Slope of property

```python
"LandSlope": ["Sev", "Mod", "Gtl"],
```

---
## Step 8 — Evaluates the quality of the material on the exterior

```python
"ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 9 — Evaluates the present condition of the material on the exterior

```python
"ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 10 — Height of the basement

```python
"BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 11 — General condition of the basement

```python
"BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 12 — Walkout or garden level basement walls

```python
"BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
```

---
## Step 13 — Quality of basement finished area

```python
"BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 14 — Quality of second basement finished area

```python
"BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 15 — Heating quality and condition

```python
"HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 16 — Kitchen quality

```python
"KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 17 — Home functionality

```python
"Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
```

---
## Step 18 — Fireplace quality

```python
"FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 19 — Interior finish of the garage

```python
"GarageFinish": ["None", "Unf", "RFn", "Fin"],
```

---
## Step 20 — Garage quality

```python
"GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 21 — Garage condition

```python
"GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 22 — Paved driveway

```python
"PavedDrive": ["N", "P", "Y"],
```

---
## Step 23 — Pool quality

```python
"PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 24 — Fence quality

```python
"Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
}
```

---
## Step 25 — Extract list of ALL ordinal features from dictionary

```python
ordinal_features = list(ordinal_order.keys())
```

---
## Step 26 — List of ordinal features except Electrical

```python
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]
```

---
## Step 27 — Specific transformer for "Electrical" using the mode for imputation

```python
electrical_imputer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent"))
])
```

---
## Step 28 — Helper function to fill "None" for other ordinal features

```python
def fill_none(X):
    return X.fillna("None")
```

---
## Step 29 — Pipeline for ordinal features: Fill missing values with "None"

```python
ordinal_imputer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False))
])
```

---
## Step 30 — Preprocessor for filling missing values

```python
preprocessor_fill = ColumnTransformer(transformers=[
    ("electrical", electrical_imputer, ["Electrical"]),
    ("cat", ordinal_imputer, ordinal_except_electrical)
])
```

---
## Step 31 — Apply preprocessor for filling missing values

```python
Ames_ordinal = preprocessor_fill.fit_transform(Ames[ordinal_features])
```

---
## Step 32 — Convert back to DataFrame to apply OrdinalEncoder

```python
Ames_ordinal = pd.DataFrame(Ames_ordinal,
                            columns=["Electrical"] + ordinal_except_electrical)
```

---
## Step 33 — Ames dataset of ordinal features prior to ordinal encoding

```python
print(Ames_ordinal)
```

---
## Learning Notes / 学习笔记

- **概念**: Import necessary libraries 是机器学习中的常用技术。  
  *Import necessary libraries is a common technique in machine learning.*

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
# Show Features / 特征工程
# Complete Code / 完整代码
# ===============================

# Import necessary libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# Load the dataset
Ames = pd.read_csv("Ames.csv")

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
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
}

# Extract list of ALL ordinal features from dictionary
ordinal_features = list(ordinal_order.keys())

# List of ordinal features except Electrical
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]

# Specific transformer for "Electrical" using the mode for imputation
electrical_imputer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent"))
])

# Helper function to fill "None" for other ordinal features
def fill_none(X):
    return X.fillna("None")

# Pipeline for ordinal features: Fill missing values with "None"
ordinal_imputer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False))
])

# Preprocessor for filling missing values
preprocessor_fill = ColumnTransformer(transformers=[
    ("electrical", electrical_imputer, ["Electrical"]),
    ("cat", ordinal_imputer, ordinal_except_electrical)
])

# Apply preprocessor for filling missing values
Ames_ordinal = preprocessor_fill.fit_transform(Ames[ordinal_features])

# Convert back to DataFrame to apply OrdinalEncoder
Ames_ordinal = pd.DataFrame(Ames_ordinal,
                            columns=["Electrical"] + ordinal_except_electrical)

# Ames dataset of ordinal features prior to ordinal encoding
print(Ames_ordinal)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Show Values

# 03 — Show Values / 03 Show Values

**Chapter 13 — File 3 of 5 / 第13章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Import necessary libraries**.

本脚本演示 **Import necessary libraries**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Import necessary libraries

```python
import pandas as pd
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Manually specify the categories for ordinal encoding according to the data dictionary

```python
ordinal_order = {
```

---
## Step 4 — Electrical system

```python
"Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
```

---
## Step 5 — General shape of property

```python
"LotShape": ["IR3", "IR2", "IR1", "Reg"],
```

---
## Step 6 — Type of utilities available

```python
"Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
```

---
## Step 7 — Slope of property

```python
"LandSlope": ["Sev", "Mod", "Gtl"],
```

---
## Step 8 — Evaluates the quality of the material on the exterior

```python
"ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 9 — Evaluates the present condition of the material on the exterior

```python
"ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 10 — Height of the basement

```python
"BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 11 — General condition of the basement

```python
"BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 12 — Walkout or garden level basement walls

```python
"BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
```

---
## Step 13 — Quality of basement finished area

```python
"BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 14 — Quality of second basement finished area

```python
"BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 15 — Heating quality and condition

```python
"HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 16 — Kitchen quality

```python
"KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 17 — Home functionality

```python
"Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
```

---
## Step 18 — Fireplace quality

```python
"FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 19 — Interior finish of the garage

```python
"GarageFinish": ["None", "Unf", "RFn", "Fin"],
```

---
## Step 20 — Garage quality

```python
"GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 21 — Garage condition

```python
"GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 22 — Paved driveway

```python
"PavedDrive": ["N", "P", "Y"],
```

---
## Step 23 — Pool quality

```python
"PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 24 — Fence quality

```python
"Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
}
```

---
## Step 25 — Extract list of ALL ordinal features from dictionary

```python
ordinal_features = list(ordinal_order.keys())
```

---
## Step 26 — Apply Ordinal Encoding

```python
categories = [ordinal_order[feature] for feature in ordinal_features]
```

---
## Step 27 — The information we input into ordinal encoder, it will automatically
assign 0, 1, 2, 3, etc.

```python
print(categories)
```

---
## Learning Notes / 学习笔记

- **概念**: Import necessary libraries 是机器学习中的常用技术。  
  *Import necessary libraries is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Show Values / 03 Show Values
# Complete Code / 完整代码
# ===============================

# Import necessary libraries
import pandas as pd

# Load the dataset
Ames = pd.read_csv("Ames.csv")

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
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
}

# Extract list of ALL ordinal features from dictionary
ordinal_features = list(ordinal_order.keys())

# Apply Ordinal Encoding
categories = [ordinal_order[feature] for feature in ordinal_features]

# The information we input into ordinal encoder, it will automatically
# assign 0, 1, 2, 3, etc.
print(categories)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Regressor

# 05 — Regressor / 05 Regressor

**Chapter 13 — File 5 of 5 / 第13章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Import the necessary libraries**.

本脚本演示 **Import the necessary libraries**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 训练模型 / Train the model

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
## Step 1 — Import the necessary libraries

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import dtreeviz
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Manually specify the categories for ordinal encoding according to the data dictionary

```python
ordinal_order = {
```

---
## Step 4 — Electrical system

```python
"Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
```

---
## Step 5 — General shape of property

```python
"LotShape": ["IR3", "IR2", "IR1", "Reg"],
```

---
## Step 6 — Type of utilities available

```python
"Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
```

---
## Step 7 — Slope of property

```python
"LandSlope": ["Sev", "Mod", "Gtl"],
```

---
## Step 8 — Evaluates the quality of the material on the exterior

```python
"ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 9 — Evaluates the present condition of the material on the exterior

```python
"ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 10 — Height of the basement

```python
"BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 11 — General condition of the basement

```python
"BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 12 — Walkout or garden level basement walls

```python
"BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
```

---
## Step 13 — Quality of basement finished area

```python
"BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 14 — Quality of second basement finished area

```python
"BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
```

---
## Step 15 — Heating quality and condition

```python
"HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 16 — Kitchen quality

```python
"KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 17 — Home functionality

```python
"Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
```

---
## Step 18 — Fireplace quality

```python
"FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 19 — Interior finish of the garage

```python
"GarageFinish": ["None", "Unf", "RFn", "Fin"],
```

---
## Step 20 — Garage quality

```python
"GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 21 — Garage condition

```python
"GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 22 — Paved driveway

```python
"PavedDrive": ["N", "P", "Y"],
```

---
## Step 23 — Pool quality

```python
"PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
```

---
## Step 24 — Fence quality

```python
"Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
}
```

---
## Step 25 — Extract list of ALL ordinal features from dictionary

```python
ordinal_features = list(ordinal_order.keys())
```

---
## Step 26 — List of ordinal features except Electrical

```python
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]
```

---
## Step 27 — Specific transformer for "Electrical" using the mode for imputation

```python
electrical_imputer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent"))
])
```

---
## Step 28 — Helper function to fill "None" for other ordinal features

```python
def fill_none(X):
    return X.fillna("None")
```

---
## Step 29 — Pipeline for ordinal features: Fill missing values with "None"

```python
ordinal_imputer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False))
])
```

---
## Step 30 — Preprocessor for filling missing values

```python
preprocessor_fill = ColumnTransformer(transformers=[
    ("electrical", electrical_imputer, ["Electrical"]),
    ("cat", ordinal_imputer, ordinal_except_electrical)
])
```

---
## Step 31 — Apply preprocessor for filling missing values

```python
Ames_ordinal = preprocessor_fill.fit_transform(Ames[ordinal_features])
```

---
## Step 32 — Convert back to DataFrame to apply OrdinalEncoder

```python
Ames_ordinal = pd.DataFrame(Ames_ordinal,
                            columns=["Electrical"] + ordinal_except_electrical)
```

---
## Step 33 — Apply Ordinal Encoding

```python
categories = [ordinal_order[feature] for feature in ordinal_features]
ordinal_encoder = OrdinalEncoder(categories=categories)
Ames_ordinal_encoded = ordinal_encoder.fit_transform(Ames_ordinal)
Ames_ordinal_encoded = pd.DataFrame(Ames_ordinal_encoded,
                                    columns=["Electrical"] + ordinal_except_electrical)
```

---
## Step 34 — Load and split the data

```python
X_ordinal = Ames_ordinal_encoded  # Use only the ordinal features for fitting the model
y = Ames["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X_ordinal, y,
                                                    test_size=0.2, random_state=42)
```

---
## Step 35 — Initialize and fit the decision tree

```python
tree_model = DecisionTreeRegressor(max_depth=3)
tree_model.fit(X_train.values, y_train)
```

---
## Step 36 — Visualize the decision tree using dtreeviz

```python
viz = dtreeviz.model(tree_model, X_train, y_train,
               target_name="SalePrice", feature_names=X_train.columns.tolist())
```

---
## Step 37 — In Jupyter Notebook, you can directly view the visual using the below:
viz.view()  # Renders and displays the SVG visualization
In PyCharm, you can render and display the SVG image:

```python
v = viz.view()     # render as SVG into internal object
v.show()           # pop up window
```

---
## Learning Notes / 学习笔记

- **概念**: Import the necessary libraries 是机器学习中的常用技术。  
  *Import the necessary libraries is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `fillna` | 填充缺失值 | Fill missing values |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `model.fit` | 训练模型 | Train the model |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Regressor / 05 Regressor
# Complete Code / 完整代码
# ===============================

# Import the necessary libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import dtreeviz

# Load the dataset
Ames = pd.read_csv("Ames.csv")

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
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
}

# Extract list of ALL ordinal features from dictionary
ordinal_features = list(ordinal_order.keys())

# List of ordinal features except Electrical
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]

# Specific transformer for "Electrical" using the mode for imputation
electrical_imputer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent"))
])

# Helper function to fill "None" for other ordinal features
def fill_none(X):
    return X.fillna("None")

# Pipeline for ordinal features: Fill missing values with "None"
ordinal_imputer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False))
])

# Preprocessor for filling missing values
preprocessor_fill = ColumnTransformer(transformers=[
    ("electrical", electrical_imputer, ["Electrical"]),
    ("cat", ordinal_imputer, ordinal_except_electrical)
])

# Apply preprocessor for filling missing values
Ames_ordinal = preprocessor_fill.fit_transform(Ames[ordinal_features])

# Convert back to DataFrame to apply OrdinalEncoder
Ames_ordinal = pd.DataFrame(Ames_ordinal,
                            columns=["Electrical"] + ordinal_except_electrical)

# Apply Ordinal Encoding
categories = [ordinal_order[feature] for feature in ordinal_features]
ordinal_encoder = OrdinalEncoder(categories=categories)
Ames_ordinal_encoded = ordinal_encoder.fit_transform(Ames_ordinal)
Ames_ordinal_encoded = pd.DataFrame(Ames_ordinal_encoded,
                                    columns=["Electrical"] + ordinal_except_electrical)

# Load and split the data
X_ordinal = Ames_ordinal_encoded  # Use only the ordinal features for fitting the model
y = Ames["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X_ordinal, y,
                                                    test_size=0.2, random_state=42)

# Initialize and fit the decision tree
tree_model = DecisionTreeRegressor(max_depth=3)
tree_model.fit(X_train.values, y_train)

# Visualize the decision tree using dtreeviz
viz = dtreeviz.model(tree_model, X_train, y_train,
               target_name="SalePrice", feature_names=X_train.columns.tolist())

# In Jupyter Notebook, you can directly view the visual using the below:
# viz.view()  # Renders and displays the SVG visualization

# In PyCharm, you can render and display the SVG image:
v = viz.view()     # render as SVG into internal object
v.show()           # pop up window
```

---

### Chapter Summary

# Chapter 13 Summary / 第13章总结

## Theme / 主题: Chapter 13 / Chapter 13

This chapter contains **5 code files** demonstrating chapter 13.

本章包含 **5 个代码文件**，演示Chapter 13。

---
## Evolution / 演化路线

  1. `01_ordinal.ipynb` — Ordinal
  2. `02_show_features.ipynb` — Show Features
  3. `03_show_values.ipynb` — Show Values
  4. `04_encoded.ipynb` — Encoded
  5. `05_regressor.ipynb` — Regressor

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 13) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 13）是机器学习流水线中的基础构建块。

---
