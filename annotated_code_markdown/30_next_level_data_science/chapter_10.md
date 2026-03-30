# 进阶数据科学
## Chapter 10

---

### Scale

# 01 — Scale / 数据缩放

**Chapter 10 — File 1 of 4 / 第10章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Import necessary libraries**.

本脚本演示 **Import necessary libraries**。

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
## Step 1 — Import necessary libraries

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
```

---
## Step 2 — Load the dataset and remove columns with missing values

```python
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
```

---
## Step 3 — Identify numeric and categorical features, excluding "PID" and "SalePrice"

```python
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns
X = Ames[numeric_features.tolist() + categorical_features.tolist()]
```

---
## Step 4 — Target variable

```python
y = Ames["SalePrice"]
```

---
## Step 5 — Pipeline for numeric features

```python
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])
```

---
## Step 6 — Pipeline for categorical features

```python
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

---
## Step 7 — Combined preprocessor for both numeric and categorical data

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
```

---
## Step 8 — Define the model pipelines with preprocessor and regressor

```python
pipelines = {
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}
```

---
## Step 9 — Perform cross-validation and store results in a dictionary

```python
cv_results = {}
for name, pipeline in pipelines.items():
    scores = cross_val_score(pipeline, X, y)
    cv_results[name] = round(scores.mean(), 4)
```

---
## Step 10 — Output the mean cross-validation scores

```python
print(cv_results)
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
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `dropna` | 删除缺失值 | Drop missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Scale / 数据缩放
# Complete Code / 完整代码
# ===============================

# Import necessary libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# Load the dataset and remove columns with missing values
Ames = pd.read_csv("Ames.csv").dropna(axis=1)

# Identify numeric and categorical features, excluding "PID" and "SalePrice"
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns
X = Ames[numeric_features.tolist() + categorical_features.tolist()]

# Target variable
y = Ames["SalePrice"]

# Pipeline for numeric features
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# Pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessor for both numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

# Define the model pipelines with preprocessor and regressor
pipelines = {
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}

# Perform cross-validation and store results in a dictionary
cv_results = {}
for name, pipeline in pipelines.items():
    scores = cross_val_score(pipeline, X, y)
    cv_results[name] = round(scores.mean(), 4)

# Output the mean cross-validation scores
print(cv_results)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Gridsearch

# 02 — Gridsearch / 02 Gridsearch

**Chapter 10 — File 2 of 4 / 第10章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Implement GridSearchCV on Lasso to obtain optimal alpha**.

本脚本演示 **Implement GridSearchCV on Lasso to obtain optimal alpha**。

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
## Step 1 — Implement GridSearchCV on Lasso to obtain optimal alpha

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV
```

---
## Step 2 — Load dataset and identify numeric and categorical features

```python
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns
```

---
## Step 3 — Features and target variables

```python
X = Ames[numeric_features.tolist() + categorical_features.tolist()]
y = Ames["SalePrice"]
```

---
## Step 4 — Set up transformers and pipelines

```python
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
pipelines = {
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}
```

---
## Step 5 — Define range of alpha values for Lasso

```python
alpha = list(range(1, 21, 1))  # Ranges from 1 to 20 in increments of 1
```

---
## Step 6 — Setup Grid Search for Lasso

```python
lasso_grid = GridSearchCV(estimator=pipelines["Lasso"],
                          param_grid={"regressor__alpha": alpha},
                          verbose=1)  # Prints out progress
lasso_grid.fit(X, y)
```

---
## Step 7 — Extract the best alpha and best score Lasso

```python
lasso_best_alpha = lasso_grid.best_params_["regressor__alpha"]
lasso_best_score = lasso_grid.best_score_

print(f"Best alpha for Lasso: {lasso_best_alpha}")
print(f"Best cross-validation score: {round(lasso_best_score, 4)}")
```

---
## Learning Notes / 学习笔记

- **概念**: Implement GridSearchCV on Lasso to obtain optimal alpha 是机器学习中的常用技术。  
  *Implement GridSearchCV on Lasso to obtain optimal alpha is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `dropna` | 删除缺失值 | Drop missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gridsearch / 02 Gridsearch
# Complete Code / 完整代码
# ===============================

# Implement GridSearchCV on Lasso to obtain optimal alpha

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV

# Load dataset and identify numeric and categorical features
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns

# Features and target variables
X = Ames[numeric_features.tolist() + categorical_features.tolist()]
y = Ames["SalePrice"]

# Set up transformers and pipelines
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
pipelines = {
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}

# Define range of alpha values for Lasso
alpha = list(range(1, 21, 1))  # Ranges from 1 to 20 in increments of 1

# Setup Grid Search for Lasso
lasso_grid = GridSearchCV(estimator=pipelines["Lasso"],
                          param_grid={"regressor__alpha": alpha},
                          verbose=1)  # Prints out progress
lasso_grid.fit(X, y)

# Extract the best alpha and best score Lasso
lasso_best_alpha = lasso_grid.best_params_["regressor__alpha"]
lasso_best_score = lasso_grid.best_score_

print(f"Best alpha for Lasso: {lasso_best_alpha}")
print(f"Best cross-validation score: {round(lasso_best_score, 4)}")
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Alpha

# 03 — Alpha / 03 Alpha

**Chapter 10 — File 3 of 4 / 第10章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Implement GridSearchCV on Ridge to obtain optimal alpha**.

本脚本演示 **Implement GridSearchCV on Ridge to obtain optimal alpha**。

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
## Step 1 — Implement GridSearchCV on Ridge to obtain optimal alpha

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV
```

---
## Step 2 — Load dataset and identify numeric and categorical features

```python
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns
```

---
## Step 3 — Features and target variables

```python
X = Ames[numeric_features.tolist() + categorical_features.tolist()]
y = Ames["SalePrice"]
```

---
## Step 4 — Set up transformers and pipelines

```python
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
pipelines = {
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}
```

---
## Step 5 — Define range of alpha for Ridge

```python
alpha = list(range(1, 21, 1))  # Ranges from 1 to 20 in increments of 1
```

---
## Step 6 — Setup Grid Search for Ridge

```python
ridge_grid = GridSearchCV(estimator=pipelines["Ridge"],
                          param_grid={"regressor__alpha": alpha},
                          verbose=1)  # Prints out progress
ridge_grid.fit(X, y)
```

---
## Step 7 — Extract the best alpha and best score for Ridge

```python
ridge_best_alpha = ridge_grid.best_params_["regressor__alpha"]
ridge_best_score = ridge_grid.best_score_

print(f"Best alpha for Ridge: {ridge_best_alpha}")
print(f"Best cross-validation score: {round(ridge_best_score, 4)}")
```

---
## Learning Notes / 学习笔记

- **概念**: Implement GridSearchCV on Ridge to obtain optimal alpha 是机器学习中的常用技术。  
  *Implement GridSearchCV on Ridge to obtain optimal alpha is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `dropna` | 删除缺失值 | Drop missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Alpha / 03 Alpha
# Complete Code / 完整代码
# ===============================

# Implement GridSearchCV on Ridge to obtain optimal alpha

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV

# Load dataset and identify numeric and categorical features
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns

# Features and target variables
X = Ames[numeric_features.tolist() + categorical_features.tolist()]
y = Ames["SalePrice"]

# Set up transformers and pipelines
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
pipelines = {
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}

# Define range of alpha for Ridge
alpha = list(range(1, 21, 1))  # Ranges from 1 to 20 in increments of 1

# Setup Grid Search for Ridge
ridge_grid = GridSearchCV(estimator=pipelines["Ridge"],
                          param_grid={"regressor__alpha": alpha},
                          verbose=1)  # Prints out progress
ridge_grid.fit(X, y)

# Extract the best alpha and best score for Ridge
ridge_best_alpha = ridge_grid.best_params_["regressor__alpha"]
ridge_best_score = ridge_grid.best_score_

print(f"Best alpha for Ridge: {ridge_best_alpha}")
print(f"Best cross-validation score: {round(ridge_best_score, 4)}")
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Alpha L2Ratio

# 04 — Alpha L2Ratio / 04 Alpha L2Ratio

**Chapter 10 — File 4 of 4 / 第10章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Implement GridSearchCV on ElasticNet to obtain optimal parameters**.

本脚本演示 **Implement GridSearchCV on ElasticNet to obtain optimal parameters**。

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
## Step 1 — Implement GridSearchCV on ElasticNet to obtain optimal parameters

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV
```

---
## Step 2 — Load dataset and identify numeric and categorical features

```python
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns
```

---
## Step 3 — Features and target variables

```python
X = Ames[numeric_features.tolist() + categorical_features.tolist()]
y = Ames["SalePrice"]
```

---
## Step 4 — Set up transformers and pipelines

```python
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
pipelines = {
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}
```

---
## Step 5 — Define range of alpha for ElasticNet

```python
alpha = list(range(1, 21, 1))  # Ranges from 1 to 20 in increments of 1
```

---
## Step 6 — Define range of L1 ratio for ElasticNet

```python
l1_ratio = [0.05, 0.5, 0.95]
```

---
## Step 7 — Setup Grid Search for ElasticNet

```python
elasticnet_grid = GridSearchCV(estimator=pipelines['ElasticNet'],
                               param_grid={'regressor__alpha': alpha,
                                           'regressor__l1_ratio': l1_ratio},
                               verbose=1)  # Prints out progress
elasticnet_grid.fit(X, y)
```

---
## Step 8 — Extract the best parameters and best score for ElasticNet

```python
elasticnet_best_params = elasticnet_grid.best_params_
elasticnet_best_score = elasticnet_grid.best_score_

print(f"Best parameters for ElasticNet: {elasticnet_best_params}")
print(f"Best cross-validation score: {round(elasticnet_best_score, 4)}")
```

---
## Learning Notes / 学习笔记

- **概念**: Implement GridSearchCV on ElasticNet to obtain optimal parameters 是机器学习中的常用技术。  
  *Implement GridSearchCV on ElasticNet to obtain optimal parameters is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `dropna` | 删除缺失值 | Drop missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Alpha L2Ratio / 04 Alpha L2Ratio
# Complete Code / 完整代码
# ===============================

# Implement GridSearchCV on ElasticNet to obtain optimal parameters

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV

# Load dataset and identify numeric and categorical features
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns

# Features and target variables
X = Ames[numeric_features.tolist() + categorical_features.tolist()]
y = Ames["SalePrice"]

# Set up transformers and pipelines
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
pipelines = {
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}

# Define range of alpha for ElasticNet
alpha = list(range(1, 21, 1))  # Ranges from 1 to 20 in increments of 1

# Define range of L1 ratio for ElasticNet
l1_ratio = [0.05, 0.5, 0.95]

# Setup Grid Search for ElasticNet
elasticnet_grid = GridSearchCV(estimator=pipelines['ElasticNet'],
                               param_grid={'regressor__alpha': alpha,
                                           'regressor__l1_ratio': l1_ratio},
                               verbose=1)  # Prints out progress
elasticnet_grid.fit(X, y)

# Extract the best parameters and best score for ElasticNet
elasticnet_best_params = elasticnet_grid.best_params_
elasticnet_best_score = elasticnet_grid.best_score_

print(f"Best parameters for ElasticNet: {elasticnet_best_params}")
print(f"Best cross-validation score: {round(elasticnet_best_score, 4)}")
```

---

### Chapter Summary

# Chapter 10 Summary / 第10章总结

## Theme / 主题: Chapter 10 / Chapter 10

This chapter contains **4 code files** demonstrating chapter 10.

本章包含 **4 个代码文件**，演示Chapter 10。

---
## Evolution / 演化路线

  1. `01_scale.ipynb` — Scale
  2. `02_gridsearch.ipynb` — Gridsearch
  3. `03_alpha.ipynb` — Alpha
  4. `04_alpha_l2ratio.ipynb` — Alpha L2Ratio

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 10) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 10）是机器学习流水线中的基础构建块。

---
