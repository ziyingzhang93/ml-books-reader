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
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model

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
## Step 1 — Load only the numeric columns from the Ames dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"])
```

---
## Step 2 — Drop any columns with missing values

```python
# 删除含缺失值的行 / Drop rows with missing values
Ames = Ames.dropna(axis=1)
```

---
## Step 3 — Import Linear Regression and Sequential Feature Selector from scikit-learn

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)  # Features
y = Ames["SalePrice"]  # Target variable
sfs.fit(X,y)           # Uses a default of cv=5
# 获取列名 / Get column names
selected_feature = X.columns[sfs.get_support()]
# 打印输出 / Print output
print("Feature selected for highest predictability:", selected_feature[0])
```

---
## Learning Notes / 学习笔记

- **概念**: Load only the numeric columns from the Ames dataset 是机器学习中的常用技术。  
  *Load only the numeric columns from the Ames dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `dropna` | 删除缺失值 | Drop missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sfs / 01 Sfs
# Complete Code / 完整代码
# ===============================

# Load only the numeric columns from the Ames dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"])

# Drop any columns with missing values
# 删除含缺失值的行 / Drop rows with missing values
Ames = Ames.dropna(axis=1)

# Import Linear Regression and Sequential Feature Selector from scikit-learn
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SequentialFeatureSelector

# Initializing the Linear Regression model
model = LinearRegression()

# Perform Sequential Feature Selector
sfs = SequentialFeatureSelector(model, n_features_to_select=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)  # Features
y = Ames["SalePrice"]  # Target variable
sfs.fit(X,y)           # Uses a default of cv=5
# 获取列名 / Get column names
selected_feature = X.columns[sfs.get_support()]
# 打印输出 / Print output
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
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
# 获取列名 / Get column names
for feature in X.columns:
    X_single = X[[feature]]
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    cv_scores = cross_val_score(model, X_single, y, cv=5)
    feature_scores[feature] = cv_scores.mean()
```

---
## Step 4 — Sort features based on their mean CV R^2 scores in descending order

```python
# 获取字典的键值对 / Get dict key-value pairs
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
```

---
## Step 5 — Print the top 3 features and their scores

```python
top_3 = sorted_features[0:3]
for feature, score in top_3:
    # 打印输出 / Print output
    print(f"Feature: {feature}, Mean CV R^2: {score:.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Dictionary to hold feature names and their corresponding mean CV R^2 scores 是机器学习中的常用技术。  
  *Dictionary to hold feature names and their corresponding mean CV R^2 scores is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `dropna` | 删除缺失值 | Drop missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cv / 02 Cv
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)  # Features
y = Ames["SalePrice"]  # Target variable
model = LinearRegression()

# Dictionary to hold feature names and their corresponding mean CV R^2 scores
feature_scores = {}

# Iterate over each feature, perform CV, and store the mean R^2 score
# 获取列名 / Get column names
for feature in X.columns:
    X_single = X[[feature]]
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    cv_scores = cross_val_score(model, X_single, y, cv=5)
    feature_scores[feature] = cv_scores.mean()

# Sort features based on their mean CV R^2 scores in descending order
# 获取字典的键值对 / Get dict key-value pairs
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)

# Print the top 3 features and their scores
top_3 = sorted_features[0:3]
for feature, score in top_3:
    # 打印输出 / Print output
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
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
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
cv_scores = cross_val_score(model, X, y, cv=5)
```

---
## Step 5 — Calculating the mean of the CV scores

```python
mean_cv_score = cv_scores.mean()
# 打印输出 / Print output
print(f"Mean CV R^2 score using 'Quality Weighted Area': {mean_cv_score:.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Create a new feature 是机器学习中的常用技术。  
  *Create a new feature is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `dropna` | 删除缺失值 | Drop missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Synthetic / 03 Synthetic
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)  # Features
y = Ames["SalePrice"]  # Target variable

# Create a new feature
Ames['QualityArea'] = Ames['OverallQual'] * Ames['GrLivArea']

# Setting up the feature and target variable for the new 'QualityArea' feature
X = Ames[['QualityArea']]  # New feature
y = Ames['SalePrice']

# 5-Fold CV on Linear Regression
model = LinearRegression()
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
cv_scores = cross_val_score(model, X, y, cv=5)

# Calculating the mean of the CV scores
mean_cv_score = cv_scores.mean()
# 打印输出 / Print output
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
