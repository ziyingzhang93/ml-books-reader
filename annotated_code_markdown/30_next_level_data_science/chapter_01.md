# 进阶数据科学
## Chapter 01

---

### Loaddata

# 01 — Loaddata / 01 Loaddata

**Chapter 01 — File 1 of 3 / 第01章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Load the Ames dataset**.

本脚本演示 **Load the Ames dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Load the Ames dataset

```python
import pandas as pd
Ames = pd.read_csv("Ames.csv")
```

---
## Step 2 — Display the first few rows of the dataset and the data type of "SalePrice"

```python
print(Ames.head())

sale_price_dtype = Ames["SalePrice"].dtype
print(f"The data type of 'SalePrice' is {sale_price_dtype}.")
```

---
## Learning Notes / 学习笔记

- **概念**: Load the Ames dataset 是机器学习中的常用技术。  
  *Load the Ames dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `head()` | 查看前几行数据 | View first few rows |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Loaddata / 01 Loaddata
# Complete Code / 完整代码
# ===============================

# Load the Ames dataset
import pandas as pd
Ames = pd.read_csv("Ames.csv")

# Display the first few rows of the dataset and the data type of "SalePrice"
print(Ames.head())

sale_price_dtype = Ames["SalePrice"].dtype
print(f"The data type of 'SalePrice' is {sale_price_dtype}.")
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Sklearn

# 02 — Sklearn / 02 Sklearn

**Chapter 01 — File 2 of 3 / 第01章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Import Linear Regression from scikit-learn**.

本脚本演示 **Import Linear Regression from scikit-learn**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
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
```

---
## Step 2 — Import Linear Regression from scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```

---
## Step 3 — Load the Ames dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 4 — Select features and target

```python
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, 1D vector
```

---
## Step 5 — Split data into training (80%) and testing sets (20%)
Setting the random state to a fixed number to make the output reproducible

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---
## Step 6 — Initialize and fit the Linear Regression model

```python
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
```

---
## Step 7 — Scoring the model

```python
score = round(model.score(X_test, y_test), 4)
print(f"Model R^2 Score: {score}")
```

---
## Learning Notes / 学习笔记

- **概念**: Import Linear Regression from scikit-learn 是机器学习中的常用技术。  
  *Import Linear Regression from scikit-learn is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sklearn / 02 Sklearn
# Complete Code / 完整代码
# ===============================

import pandas as pd
# Import Linear Regression from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the Ames dataset
Ames = pd.read_csv("Ames.csv")

# Select features and target
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, 1D vector

# Split data into training (80%) and testing sets (20%)
# Setting the random state to a fixed number to make the output reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

# Scoring the model
score = round(model.score(X_test, y_test), 4)
print(f"Model R^2 Score: {score}")
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Statsmodels

# 03 — Statsmodels / 03 Statsmodels

**Chapter 01 — File 3 of 3 / 第01章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Load the Ames dataset**.

本脚本演示 **Load the Ames dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — Step 1

```python
import pandas as pd
import statsmodels.api as sm
```

---
## Step 2 — Load the Ames dataset

```python
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Select features and target

```python
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, 1D vector
```

---
## Step 4 — statsmodels requires to add a constant explicitly to model the intercept

```python
X_with_constant = sm.add_constant(X)
```

---
## Step 5 — Fit the OLS model

```python
model_stats = sm.OLS(y, X_with_constant).fit()
```

---
## Step 6 — Print the summary of the model

```python
print(model_stats.summary())
```

---
## Learning Notes / 学习笔记

- **概念**: Load the Ames dataset 是机器学习中的常用技术。  
  *Load the Ames dataset is a common technique in machine learning.*

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
# Statsmodels / 03 Statsmodels
# Complete Code / 完整代码
# ===============================

import pandas as pd
import statsmodels.api as sm

# Load the Ames dataset
Ames = pd.read_csv("Ames.csv")

# Select features and target
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, 1D vector

# statsmodels requires to add a constant explicitly to model the intercept
X_with_constant = sm.add_constant(X)

# Fit the OLS model
model_stats = sm.OLS(y, X_with_constant).fit()

# Print the summary of the model
print(model_stats.summary())
```

---

### Chapter Summary

# Chapter 01 Summary / 第01章总结

## Theme / 主题: Chapter 01 / Chapter 01

This chapter contains **3 code files** demonstrating chapter 01.

本章包含 **3 个代码文件**，演示Chapter 01。

---
## Evolution / 演化路线

  1. `01_loaddata.ipynb` — Loaddata
  2. `02_sklearn.ipynb` — Sklearn
  3. `03_statsmodels.ipynb` — Statsmodels

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 01) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 01）是机器学习流水线中的基础构建块。

---
