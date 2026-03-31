# 进阶数据科学 / Next Level Data Science
## Chapter 11

---

### Simpleimputer

# 01 — Simpleimputer / 01 Simpleimputer

**Chapter 11 — File 1 of 3 / 第11章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Import the necessary libraries**.

本脚本演示 **Import the necessary libraries**。

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
## Step 1 — Import the necessary libraries

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.impute import SimpleImputer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso, Ridge, ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
```

---
## Step 2 — Load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Exclude "PID" and "SalePrice" from features and handle the "Electrical" column

```python
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       # 获取列名 / Get column names
                       .drop(columns=["PID", "SalePrice"]).columns
# 获取列名 / Get column names
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]  # Specifically handle the "Electrical" column
```

---
## Step 4 — Helper function to fill "Missing" for missing categorical data

```python
def fill_none(X):
    # 填充缺失值 / Fill missing values
    return X.fillna("Missing")
```

---
## Step 5 — Pipeline for numeric features: Impute missing values then scale

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean")),
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    ("scaler", StandardScaler())
])
```

---
## Step 6 — Pipeline for general categorical features: Fill missing values with "Missing" then
apply one-hot encoding

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 7 — Specific transformer for "Electrical" using the mode for imputation

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot_electrical", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 8 — Combined preprocessor for numeric, general categorical, and electrical data

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("electrical", electrical_transformer, electrical_feature)
    ])
```

---
## Step 9 — Target variable

```python
y = Ames["SalePrice"]
```

---
## Step 10 — All features

```python
X = Ames[numeric_features.tolist() + categorical_features.tolist() + electrical_feature]
```

---
## Step 11 — Define the model pipelines with preprocessor and regressor

```python
models = {
    "Lasso": Lasso(max_iter=20000),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet()
}

results = {}
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
```

---
## Step 12 — Perform cross-validation

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(pipeline, X, y)
    results[name] = round(scores.mean(), 4)
```

---
## Step 13 — Output the cross-validation scores

```python
# 打印输出 / Print output
print("Cross-validation scores with Simple Imputer:", results)
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
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
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
# Simpleimputer / 01 Simpleimputer
# Complete Code / 完整代码
# ===============================

# Import the necessary libraries
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.impute import SimpleImputer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso, Ridge, ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score

# Load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")

# Exclude "PID" and "SalePrice" from features and handle the "Electrical" column
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       # 获取列名 / Get column names
                       .drop(columns=["PID", "SalePrice"]).columns
# 获取列名 / Get column names
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]  # Specifically handle the "Electrical" column

# Helper function to fill "Missing" for missing categorical data
def fill_none(X):
    # 填充缺失值 / Fill missing values
    return X.fillna("Missing")

# Pipeline for numeric features: Impute missing values then scale
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean")),
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    ("scaler", StandardScaler())
])

# Pipeline for general categorical features: Fill missing values with "Missing" then
# apply one-hot encoding
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Specific transformer for "Electrical" using the mode for imputation
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot_electrical", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessor for numeric, general categorical, and electrical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("electrical", electrical_transformer, electrical_feature)
    ])

# Target variable
y = Ames["SalePrice"]

# All features
X = Ames[numeric_features.tolist() + categorical_features.tolist() + electrical_feature]

# Define the model pipelines with preprocessor and regressor
models = {
    "Lasso": Lasso(max_iter=20000),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet()
}

results = {}
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    # Perform cross-validation
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    scores = cross_val_score(pipeline, X, y)
    results[name] = round(scores.mean(), 4)

# Output the cross-validation scores
# 打印输出 / Print output
print("Cross-validation scores with Simple Imputer:", results)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Iterativeimputer

# 02 — Iterativeimputer / 02 Iterativeimputer

**Chapter 11 — File 2 of 3 / 第11章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Import the necessary libraries**.

本脚本演示 **Import the necessary libraries**。

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
## Step 1 — Import the necessary libraries

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.experimental import enable_iterative_imputer  # needed for IterativeImputer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.impute import SimpleImputer, IterativeImputer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso, Ridge, ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
```

---
## Step 2 — Load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Exclude "PID" and "SalePrice" from features and handle the "Electrical" column

```python
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       # 获取列名 / Get column names
                       .drop(columns=["PID", "SalePrice"]).columns
# 获取列名 / Get column names
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]  # Specifically handle the "Electrical" column
```

---
## Step 4 — Helper function to fill "Missing" for missing categorical data

```python
def fill_none(X):
    # 填充缺失值 / Fill missing values
    return X.fillna("Missing")
```

---
## Step 5 — Pipeline for numeric features: Iterative imputation then scale

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
numeric_transformer_advanced = Pipeline(steps=[
    ("impute_iterative", IterativeImputer(random_state=42)),
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    ("scaler", StandardScaler())
])
```

---
## Step 6 — Pipeline for general categorical features: Fill missing values with "Missing" then
apply one-hot encoding

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 7 — Specific transformer for "Electrical" using the mode for imputation

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot_electrical", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 8 — Combined preprocessor for numeric, general categorical, and electrical data

```python
preprocessor_advanced = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_advanced, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("electrical", electrical_transformer, electrical_feature)
    ])
```

---
## Step 9 — Target variable

```python
y = Ames["SalePrice"]
```

---
## Step 10 — All features

```python
X = Ames[numeric_features.tolist() + categorical_features.tolist() + electrical_feature]
```

---
## Step 11 — Define the model pipelines with preprocessor and regressor

```python
models = {
    "Lasso": Lasso(max_iter=20000),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet()
}

results_advanced = {}
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor_advanced),
        ("regressor", model)
    ])
```

---
## Step 12 — Perform cross-validation

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(pipeline, X, y)
    results_advanced[name] = round(scores.mean(), 4)
```

---
## Step 13 — Output the cross-validation scores for advanced imputation

```python
# 打印输出 / Print output
print("Cross-validation scores with Iterative Imputer:", results_advanced)
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
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
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
# Iterativeimputer / 02 Iterativeimputer
# Complete Code / 完整代码
# ===============================

# Import the necessary libraries
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.experimental import enable_iterative_imputer  # needed for IterativeImputer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.impute import SimpleImputer, IterativeImputer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso, Ridge, ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score

# Load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")

# Exclude "PID" and "SalePrice" from features and handle the "Electrical" column
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       # 获取列名 / Get column names
                       .drop(columns=["PID", "SalePrice"]).columns
# 获取列名 / Get column names
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]  # Specifically handle the "Electrical" column

# Helper function to fill "Missing" for missing categorical data
def fill_none(X):
    # 填充缺失值 / Fill missing values
    return X.fillna("Missing")

# Pipeline for numeric features: Iterative imputation then scale
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
numeric_transformer_advanced = Pipeline(steps=[
    ("impute_iterative", IterativeImputer(random_state=42)),
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    ("scaler", StandardScaler())
])

# Pipeline for general categorical features: Fill missing values with "Missing" then
# apply one-hot encoding
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Specific transformer for "Electrical" using the mode for imputation
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot_electrical", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessor for numeric, general categorical, and electrical data
preprocessor_advanced = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_advanced, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("electrical", electrical_transformer, electrical_feature)
    ])

# Target variable
y = Ames["SalePrice"]

# All features
X = Ames[numeric_features.tolist() + categorical_features.tolist() + electrical_feature]

# Define the model pipelines with preprocessor and regressor
models = {
    "Lasso": Lasso(max_iter=20000),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet()
}

results_advanced = {}
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor_advanced),
        ("regressor", model)
    ])
    # Perform cross-validation
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    scores = cross_val_score(pipeline, X, y)
    results_advanced[name] = round(scores.mean(), 4)

# Output the cross-validation scores for advanced imputation
# 打印输出 / Print output
print("Cross-validation scores with Iterative Imputer:", results_advanced)
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Knnimputer

# 03 — Knnimputer / 03 Knnimputer

**Chapter 11 — File 3 of 3 / 第11章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Import the necessary libraries**.

本脚本演示 **Import the necessary libraries**。

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
## Step 1 — Import the necessary libraries

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.impute import SimpleImputer, KNNImputer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso, Ridge, ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
```

---
## Step 2 — Load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Exclude "PID" and "SalePrice" from features and handle the "Electrical" column

```python
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       # 获取列名 / Get column names
                       .drop(columns=["PID", "SalePrice"]).columns
# 获取列名 / Get column names
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]  # Specifically handle the "Electrical" column
```

---
## Step 4 — Helper function to fill "Missing" for missing categorical data

```python
def fill_none(X):
    # 填充缺失值 / Fill missing values
    return X.fillna("Missing")
```

---
## Step 5 — Pipeline for numeric features: K-Nearest Neighbors Imputation then scale

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
numeric_transformer_knn = Pipeline(steps=[
    ("impute_knn", KNNImputer(n_neighbors=5)),
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    ("scaler", StandardScaler())
])
```

---
## Step 6 — Pipeline for general categorical features: Fill missing values with "Missing" then
apply one-hot encoding

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 7 — Specific transformer for "Electrical" using the mode for imputation

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot_electrical", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 8 — Combined preprocessor for numeric, general categorical, and electrical data

```python
preprocessor_knn = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_knn, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("electrical", electrical_transformer, electrical_feature)
    ])
```

---
## Step 9 — Target variable

```python
y = Ames["SalePrice"]
```

---
## Step 10 — All features

```python
X = Ames[numeric_features.tolist() + categorical_features.tolist() + electrical_feature]
```

---
## Step 11 — Define the model pipelines with preprocessor and regressor

```python
models = {
    "Lasso": Lasso(max_iter=20000),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet()
}

results_knn = {}
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor_knn),
        ("regressor", model)
    ])
```

---
## Step 12 — Perform cross-validation

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(pipeline, X, y)
    results_knn[name] = round(scores.mean(), 4)
```

---
## Step 13 — Output the cross-validation scores for KNN imputation

```python
# 打印输出 / Print output
print("Cross-validation scores with KNN Imputer:", results_knn)
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
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
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
# Knnimputer / 03 Knnimputer
# Complete Code / 完整代码
# ===============================

# Import the necessary libraries
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.impute import SimpleImputer, KNNImputer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso, Ridge, ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score

# Load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")

# Exclude "PID" and "SalePrice" from features and handle the "Electrical" column
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       # 获取列名 / Get column names
                       .drop(columns=["PID", "SalePrice"]).columns
# 获取列名 / Get column names
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]  # Specifically handle the "Electrical" column

# Helper function to fill "Missing" for missing categorical data
def fill_none(X):
    # 填充缺失值 / Fill missing values
    return X.fillna("Missing")

# Pipeline for numeric features: K-Nearest Neighbors Imputation then scale
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
numeric_transformer_knn = Pipeline(steps=[
    ("impute_knn", KNNImputer(n_neighbors=5)),
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    ("scaler", StandardScaler())
])

# Pipeline for general categorical features: Fill missing values with "Missing" then
# apply one-hot encoding
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Specific transformer for "Electrical" using the mode for imputation
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot_electrical", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessor for numeric, general categorical, and electrical data
preprocessor_knn = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_knn, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("electrical", electrical_transformer, electrical_feature)
    ])

# Target variable
y = Ames["SalePrice"]

# All features
X = Ames[numeric_features.tolist() + categorical_features.tolist() + electrical_feature]

# Define the model pipelines with preprocessor and regressor
models = {
    "Lasso": Lasso(max_iter=20000),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet()
}

results_knn = {}
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor_knn),
        ("regressor", model)
    ])
    # Perform cross-validation
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    scores = cross_val_score(pipeline, X, y)
    results_knn[name] = round(scores.mean(), 4)

# Output the cross-validation scores for KNN imputation
# 打印输出 / Print output
print("Cross-validation scores with KNN Imputer:", results_knn)
```

---

### Chapter Summary / 章节总结

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **3 code files** demonstrating chapter 11.

本章包含 **3 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `01_simpleimputer.ipynb` — Simpleimputer
  2. `02_iterativeimputer.ipynb` — Iterativeimputer
  3. `03_knnimputer.ipynb` — Knnimputer

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---
