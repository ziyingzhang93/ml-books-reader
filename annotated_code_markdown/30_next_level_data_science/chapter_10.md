# 进阶数据科学 / Next Level Data Science
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso, Ridge, ElasticNet
```

---
## Step 2 — Load the dataset and remove columns with missing values

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
```

---
## Step 3 — Identify numeric and categorical features, excluding "PID" and "SalePrice"

```python
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       # 获取列名 / Get column names
                       .drop(columns=["PID", "SalePrice"]).columns
# 获取列名 / Get column names
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
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
numeric_transformer = Pipeline(steps=[
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    ("scaler", StandardScaler())
])
```

---
## Step 6 — Pipeline for categorical features

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
categorical_transformer = Pipeline(steps=[
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
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
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}
```

---
## Step 9 — Perform cross-validation and store results in a dictionary

```python
cv_results = {}
# 获取字典的键值对 / Get dict key-value pairs
for name, pipeline in pipelines.items():
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    scores = cross_val_score(pipeline, X, y)
    cv_results[name] = round(scores.mean(), 4)
```

---
## Step 10 — Output the mean cross-validation scores

```python
# 打印输出 / Print output
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# Load the dataset and remove columns with missing values
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").dropna(axis=1)

# Identify numeric and categorical features, excluding "PID" and "SalePrice"
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       # 获取列名 / Get column names
                       .drop(columns=["PID", "SalePrice"]).columns
# 获取列名 / Get column names
categorical_features = Ames.select_dtypes(include=["object"]).columns
X = Ames[numeric_features.tolist() + categorical_features.tolist()]

# Target variable
y = Ames["SalePrice"]

# Pipeline for numeric features
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
numeric_transformer = Pipeline(steps=[
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    ("scaler", StandardScaler())
])

# Pipeline for categorical features
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
categorical_transformer = Pipeline(steps=[
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
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
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}

# Perform cross-validation and store results in a dictionary
cv_results = {}
# 获取字典的键值对 / Get dict key-value pairs
for name, pipeline in pipelines.items():
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    scores = cross_val_score(pipeline, X, y)
    cv_results[name] = round(scores.mean(), 4)

# Output the mean cross-validation scores
# 打印输出 / Print output
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso, Ridge, ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
```

---
## Step 2 — Load dataset and identify numeric and categorical features

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       # 获取列名 / Get column names
                       .drop(columns=["PID", "SalePrice"]).columns
# 获取列名 / Get column names
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
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
numeric_transformer = Pipeline(steps=[
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    ("scaler", StandardScaler())
])
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
categorical_transformer = Pipeline(steps=[
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
pipelines = {
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}
```

---
## Step 5 — Define range of alpha values for Lasso

```python
# 生成整数序列 / Generate integer sequence
alpha = list(range(1, 21, 1))  # Ranges from 1 to 20 in increments of 1
```

---
## Step 6 — Setup Grid Search for Lasso

```python
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
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

# 打印输出 / Print output
print(f"Best alpha for Lasso: {lasso_best_alpha}")
# 打印输出 / Print output
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

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso, Ridge, ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV

# Load dataset and identify numeric and categorical features
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       # 获取列名 / Get column names
                       .drop(columns=["PID", "SalePrice"]).columns
# 获取列名 / Get column names
categorical_features = Ames.select_dtypes(include=["object"]).columns

# Features and target variables
X = Ames[numeric_features.tolist() + categorical_features.tolist()]
y = Ames["SalePrice"]

# Set up transformers and pipelines
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
numeric_transformer = Pipeline(steps=[
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    ("scaler", StandardScaler())
])
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
categorical_transformer = Pipeline(steps=[
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
pipelines = {
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}

# Define range of alpha values for Lasso
# 生成整数序列 / Generate integer sequence
alpha = list(range(1, 21, 1))  # Ranges from 1 to 20 in increments of 1

# Setup Grid Search for Lasso
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
lasso_grid = GridSearchCV(estimator=pipelines["Lasso"],
                          param_grid={"regressor__alpha": alpha},
                          verbose=1)  # Prints out progress
lasso_grid.fit(X, y)

# Extract the best alpha and best score Lasso
lasso_best_alpha = lasso_grid.best_params_["regressor__alpha"]
lasso_best_score = lasso_grid.best_score_

# 打印输出 / Print output
print(f"Best alpha for Lasso: {lasso_best_alpha}")
# 打印输出 / Print output
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso, Ridge, ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
```

---
## Step 2 — Load dataset and identify numeric and categorical features

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       # 获取列名 / Get column names
                       .drop(columns=["PID", "SalePrice"]).columns
# 获取列名 / Get column names
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
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
numeric_transformer = Pipeline(steps=[
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    ("scaler", StandardScaler())
])
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
categorical_transformer = Pipeline(steps=[
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
pipelines = {
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}
```

---
## Step 5 — Define range of alpha for Ridge

```python
# 生成整数序列 / Generate integer sequence
alpha = list(range(1, 21, 1))  # Ranges from 1 to 20 in increments of 1
```

---
## Step 6 — Setup Grid Search for Ridge

```python
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
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

# 打印输出 / Print output
print(f"Best alpha for Ridge: {ridge_best_alpha}")
# 打印输出 / Print output
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

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso, Ridge, ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV

# Load dataset and identify numeric and categorical features
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       # 获取列名 / Get column names
                       .drop(columns=["PID", "SalePrice"]).columns
# 获取列名 / Get column names
categorical_features = Ames.select_dtypes(include=["object"]).columns

# Features and target variables
X = Ames[numeric_features.tolist() + categorical_features.tolist()]
y = Ames["SalePrice"]

# Set up transformers and pipelines
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
numeric_transformer = Pipeline(steps=[
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    ("scaler", StandardScaler())
])
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
categorical_transformer = Pipeline(steps=[
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
pipelines = {
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}

# Define range of alpha for Ridge
# 生成整数序列 / Generate integer sequence
alpha = list(range(1, 21, 1))  # Ranges from 1 to 20 in increments of 1

# Setup Grid Search for Ridge
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
ridge_grid = GridSearchCV(estimator=pipelines["Ridge"],
                          param_grid={"regressor__alpha": alpha},
                          verbose=1)  # Prints out progress
ridge_grid.fit(X, y)

# Extract the best alpha and best score for Ridge
ridge_best_alpha = ridge_grid.best_params_["regressor__alpha"]
ridge_best_score = ridge_grid.best_score_

# 打印输出 / Print output
print(f"Best alpha for Ridge: {ridge_best_alpha}")
# 打印输出 / Print output
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso, Ridge, ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
```

---
## Step 2 — Load dataset and identify numeric and categorical features

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       # 获取列名 / Get column names
                       .drop(columns=["PID", "SalePrice"]).columns
# 获取列名 / Get column names
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
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
numeric_transformer = Pipeline(steps=[
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    ("scaler", StandardScaler())
])
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
categorical_transformer = Pipeline(steps=[
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
pipelines = {
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}
```

---
## Step 5 — Define range of alpha for ElasticNet

```python
# 生成整数序列 / Generate integer sequence
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
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
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

# 打印输出 / Print output
print(f"Best parameters for ElasticNet: {elasticnet_best_params}")
# 打印输出 / Print output
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

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso, Ridge, ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV

# Load dataset and identify numeric and categorical features
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").dropna(axis=1)
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       # 获取列名 / Get column names
                       .drop(columns=["PID", "SalePrice"]).columns
# 获取列名 / Get column names
categorical_features = Ames.select_dtypes(include=["object"]).columns

# Features and target variables
X = Ames[numeric_features.tolist() + categorical_features.tolist()]
y = Ames["SalePrice"]

# Set up transformers and pipelines
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
numeric_transformer = Pipeline(steps=[
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    ("scaler", StandardScaler())
])
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
categorical_transformer = Pipeline(steps=[
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
pipelines = {
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Lasso": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Lasso(max_iter=20000))]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "Ridge": Pipeline(steps=[("preprocessor", preprocessor),
                             ("regressor", Ridge())]),
    # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
    "ElasticNet": Pipeline(steps=[("preprocessor", preprocessor),
                                  ("regressor", ElasticNet())])
}

# Define range of alpha for ElasticNet
# 生成整数序列 / Generate integer sequence
alpha = list(range(1, 21, 1))  # Ranges from 1 to 20 in increments of 1

# Define range of L1 ratio for ElasticNet
l1_ratio = [0.05, 0.5, 0.95]

# Setup Grid Search for ElasticNet
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
elasticnet_grid = GridSearchCV(estimator=pipelines['ElasticNet'],
                               param_grid={'regressor__alpha': alpha,
                                           'regressor__l1_ratio': l1_ratio},
                               verbose=1)  # Prints out progress
elasticnet_grid.fit(X, y)

# Extract the best parameters and best score for ElasticNet
elasticnet_best_params = elasticnet_grid.best_params_
elasticnet_best_score = elasticnet_grid.best_score_

# 打印输出 / Print output
print(f"Best parameters for ElasticNet: {elasticnet_best_params}")
# 打印输出 / Print output
print(f"Best cross-validation score: {round(elasticnet_best_score, 4)}")
```

---

### Chapter Summary / 章节总结

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
