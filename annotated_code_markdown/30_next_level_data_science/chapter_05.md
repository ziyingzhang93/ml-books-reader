# 进阶数据科学 / Next Level Data Science
## Chapter 05

---

### Onehot

# 01 — Onehot / 01 Onehot

**Chapter 05 — File 1 of 3 / 第05章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Load only categorical columns without missing values from the Ames dataset**.

本脚本演示 **Load only categorical columns without missing values from the Ames dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
```

---
## Step 1 — Load only categorical columns without missing values from the Ames dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["object"]).dropna(axis=1)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"The shape of the DataFrame before one-hot encoding is: {Ames.shape}")
```

---
## Step 2 — Import OneHotEncoder and apply it to Ames:

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder

# 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
encoder = OneHotEncoder(sparse=False)
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
Ames_One_Hot = encoder.fit_transform(Ames)
```

---
## Step 3 — Convert the encoded result back to a DataFrame

```python
Ames_encoded_df = pd.DataFrame(Ames_One_Hot,
                               # 获取列名 / Get column names
                               columns=encoder.get_feature_names_out(Ames.columns))
```

---
## Step 4 — Display the new DataFrame and it's expanded shape

```python
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(Ames_encoded_df.head())
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"The shape of the DataFrame after one-hot encoding is: {Ames_encoded_df.shape}")
```

---
## Learning Notes / 学习笔记

- **概念**: Load only categorical columns without missing values from the Ames dataset 是机器学习中的常用技术。  
  *Load only categorical columns without missing values from the Ames dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `dropna` | 删除缺失值 | Drop missing values |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `head()` | 查看前几行数据 | View first few rows |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Onehot / 01 Onehot
# Complete Code / 完整代码
# ===============================

# Load only categorical columns without missing values from the Ames dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["object"]).dropna(axis=1)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"The shape of the DataFrame before one-hot encoding is: {Ames.shape}")

# Import OneHotEncoder and apply it to Ames:
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder

# 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
encoder = OneHotEncoder(sparse=False)
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
Ames_One_Hot = encoder.fit_transform(Ames)

# Convert the encoded result back to a DataFrame
Ames_encoded_df = pd.DataFrame(Ames_One_Hot,
                               # 获取列名 / Get column names
                               columns=encoder.get_feature_names_out(Ames.columns))

# Display the new DataFrame and it's expanded shape
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(Ames_encoded_df.head())
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"The shape of the DataFrame after one-hot encoding is: {Ames_encoded_df.shape}")
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Preprocess

# 02 — Preprocess / 02 Preprocess

**Chapter 05 — File 2 of 3 / 第05章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Set "SalePrice" as the target variable**.

本脚本演示 **Set "SalePrice" as the target variable**。

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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["object"]).dropna(axis=1)
```

---
## Step 2 — Set "SalePrice" as the target variable

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
y = pd.read_csv("Ames.csv")["SalePrice"]
```

---
## Step 3 — Dictionary to store feature names and their corresponding mean CV R^2 scores

```python
feature_scores = {}

# 获取列名 / Get column names
for feature in Ames.columns:
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    encoder = OneHotEncoder(drop="first")
    # 拟合并转换数据（一步完成） / Fit and transform data (one step)
    X_encoded = encoder.fit_transform(Ames[[feature]])
```

---
## Step 4 — Initialize the linear regression model

```python
model = LinearRegression()
```

---
## Step 5 — Perform 5-fold cross-validation and calculate R^2 scores

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X_encoded, y)
    mean_score = scores.mean()
```

---
## Step 6 — Store the mean R^2 score

```python
feature_scores[feature] = mean_score
```

---
## Step 7 — Sort features based on their mean CV R^2 scores in descending order

```python
# 获取字典的键值对 / Get dict key-value pairs
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
# 打印输出 / Print output
print("Feature selected for highest predictability:", sorted_features[0][0])
```

---
## Learning Notes / 学习笔记

- **概念**: Set "SalePrice" as the target variable 是机器学习中的常用技术。  
  *Set "SalePrice" as the target variable is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `dropna` | 删除缺失值 | Drop missing values |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Preprocess / 02 Preprocess
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["object"]).dropna(axis=1)

# Set "SalePrice" as the target variable
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
y = pd.read_csv("Ames.csv")["SalePrice"]

# Dictionary to store feature names and their corresponding mean CV R^2 scores
feature_scores = {}

# 获取列名 / Get column names
for feature in Ames.columns:
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    encoder = OneHotEncoder(drop="first")
    # 拟合并转换数据（一步完成） / Fit and transform data (one step)
    X_encoded = encoder.fit_transform(Ames[[feature]])

    # Initialize the linear regression model
    model = LinearRegression()

    # Perform 5-fold cross-validation and calculate R^2 scores
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    scores = cross_val_score(model, X_encoded, y)
    mean_score = scores.mean()

    # Store the mean R^2 score
    feature_scores[feature] = mean_score

# Sort features based on their mean CV R^2 scores in descending order
# 获取字典的键值对 / Get dict key-value pairs
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
# 打印输出 / Print output
print("Feature selected for highest predictability:", sorted_features[0][0])
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Top5

# 03 — Top5 / 03 Top5

**Chapter 05 — File 3 of 3 / 第05章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Top5**.

本脚本演示 **03 Top5**。

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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["object"]).dropna(axis=1)
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
y = pd.read_csv("Ames.csv")["SalePrice"]

feature_scores = {}
# 获取列名 / Get column names
for feature in Ames.columns:
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    encoder = OneHotEncoder(drop="first")
    # 拟合并转换数据（一步完成） / Fit and transform data (one step)
    X_encoded = encoder.fit_transform(Ames[[feature]])
    model = LinearRegression()
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    scores = cross_val_score(model, X_encoded, y)
    mean_score = scores.mean()
    feature_scores[feature] = mean_score

# 获取字典的键值对 / Get dict key-value pairs
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)

# 打印输出 / Print output
print("Top 5 Categorical Features:")
for feature, score in sorted_features[0:5]:
    # 打印输出 / Print output
    print(f"{feature}: Mean CV R^2 = {score:.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Top5 是机器学习中的常用技术。  
  *Top5 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `dropna` | 删除缺失值 | Drop missing values |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Top5 / 03 Top5
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["object"]).dropna(axis=1)
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
y = pd.read_csv("Ames.csv")["SalePrice"]

feature_scores = {}
# 获取列名 / Get column names
for feature in Ames.columns:
    # 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
    encoder = OneHotEncoder(drop="first")
    # 拟合并转换数据（一步完成） / Fit and transform data (one step)
    X_encoded = encoder.fit_transform(Ames[[feature]])
    model = LinearRegression()
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    scores = cross_val_score(model, X_encoded, y)
    mean_score = scores.mean()
    feature_scores[feature] = mean_score

# 获取字典的键值对 / Get dict key-value pairs
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)

# 打印输出 / Print output
print("Top 5 Categorical Features:")
for feature, score in sorted_features[0:5]:
    # 打印输出 / Print output
    print(f"{feature}: Mean CV R^2 = {score:.4f}")
```

---

### Chapter Summary / 章节总结

# Chapter 05 Summary / 第05章总结

## Theme / 主题: Chapter 05 / Chapter 05

This chapter contains **3 code files** demonstrating chapter 05.

本章包含 **3 个代码文件**，演示Chapter 05。

---
## Evolution / 演化路线

  1. `01_onehot.ipynb` — Onehot
  2. `02_preprocess.ipynb` — Preprocess
  3. `03_top5.ipynb` — Top5

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 05) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 05）是机器学习流水线中的基础构建块。

---
