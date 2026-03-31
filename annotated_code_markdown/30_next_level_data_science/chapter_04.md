# 进阶数据科学 / Next Level Data Science
## Chapter 04

---

### Topfive

# 01 — Topfive / 01 Topfive

**Chapter 04 — File 1 of 6 / 第04章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Load the essential libraries and Ames dataset**.

本脚本演示 **Load the essential libraries and Ames dataset**。

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
## Step 1 — Load the essential libraries and Ames dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]
```

---
## Step 2 — Initialize the Linear Regression model

```python
model = LinearRegression()
```

---
## Step 3 — Prepare to collect feature scores

```python
feature_scores = {}
```

---
## Step 4 — Evaluate each feature with cross-validation

```python
# 获取列名 / Get column names
for feature in X.columns:
    X_single = X[[feature]]
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    cv_scores = cross_val_score(model, X_single, y)
    feature_scores[feature] = cv_scores.mean()
```

---
## Step 5 — Identify the top 5 features based on mean CV R^2 scores

```python
# 获取字典的键值对 / Get dict key-value pairs
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
top_5 = sorted_features[0:5]
```

---
## Step 6 — Display the top 5 features and their individual performance

```python
for feature, score in top_5:
    # 打印输出 / Print output
    print(f"Feature: {feature}, Mean CV R^2: {score:.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Load the essential libraries and Ames dataset 是机器学习中的常用技术。  
  *Load the essential libraries and Ames dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `dropna` | 删除缺失值 | Drop missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Topfive / 01 Topfive
# Complete Code / 完整代码
# ===============================

# Load the essential libraries and Ames dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]

# Initialize the Linear Regression model
model = LinearRegression()

# Prepare to collect feature scores
feature_scores = {}

# Evaluate each feature with cross-validation
# 获取列名 / Get column names
for feature in X.columns:
    X_single = X[[feature]]
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    cv_scores = cross_val_score(model, X_single, y)
    feature_scores[feature] = cv_scores.mean()

# Identify the top 5 features based on mean CV R^2 scores
# 获取字典的键值对 / Get dict key-value pairs
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
top_5 = sorted_features[0:5]

# Display the top 5 features and their individual performance
for feature, score in top_5:
    # 打印输出 / Print output
    print(f"Feature: {feature}, Mean CV R^2: {score:.4f}")
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Regression

# 02 — Regression / 回归

**Chapter 04 — File 2 of 6 / 第04章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Evaluate each feature to find the top 5**.

本脚本演示 **Evaluate each feature to find the top 5**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
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
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]
```

---
## Step 2 — Evaluate each feature to find the top 5

```python
model = LinearRegression()
feature_scores = {}
# 获取列名 / Get column names
for feature in X.columns:
    X_single = X[[feature]]
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    cv_scores = cross_val_score(model, X_single, y)
    feature_scores[feature] = cv_scores.mean()

# 获取字典的键值对 / Get dict key-value pairs
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
top_5 = sorted_features[0:5]
```

---
## Step 3 — Extracting the top 5 features for our multiple linear regression

```python
top_features = [feature for feature, score in top_5]
```

---
## Step 4 — Building the model with the top 5 features

```python
X_top = Ames[top_features]
```

---
## Step 5 — Evaluating the model with cross-validation

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
cv_scores_mlr = cross_val_score(model, X_top, y, cv=5, scoring="r2")
mean_mlr_score = cv_scores_mlr.mean()

# 打印输出 / Print output
print(f"Mean CV R^2 Score for Multiple Linear Regression Model: {mean_mlr_score:.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Evaluate each feature to find the top 5 是机器学习中的常用技术。  
  *Evaluate each feature to find the top 5 is a common technique in machine learning.*

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
# Regression / 回归
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]

# Evaluate each feature to find the top 5
model = LinearRegression()
feature_scores = {}
# 获取列名 / Get column names
for feature in X.columns:
    X_single = X[[feature]]
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    cv_scores = cross_val_score(model, X_single, y)
    feature_scores[feature] = cv_scores.mean()

# 获取字典的键值对 / Get dict key-value pairs
sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
top_5 = sorted_features[0:5]

# Extracting the top 5 features for our multiple linear regression
top_features = [feature for feature, score in top_5]

# Building the model with the top 5 features
X_top = Ames[top_features]

# Evaluating the model with cross-validation
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
cv_scores_mlr = cross_val_score(model, X_top, y, cv=5, scoring="r2")
mean_mlr_score = cv_scores_mlr.mean()

# 打印输出 / Print output
print(f"Mean CV R^2 Score for Multiple Linear Regression Model: {mean_mlr_score:.4f}")
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Sfs

# 03 — Sfs / 03 Sfs

**Chapter 04 — File 3 of 6 / 第04章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Perform Sequential Feature Selector with n=5**.

本脚本演示 **Perform Sequential Feature Selector with n=5**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SequentialFeatureSelector

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]
model = LinearRegression()
```

---
## Step 2 — Perform Sequential Feature Selector with n=5

```python
sfs = SequentialFeatureSelector(model, n_features_to_select=5)
sfs.fit(X, y)

# 获取列名 / Get column names
selected_features = X.columns[sfs.get_support()].to_list()
# 打印输出 / Print output
print(f"Features selected by SFS: {selected_features}")

# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, Ames[selected_features], y)
# 打印输出 / Print output
print(f"Mean CV R^2 Score using SFS with n=5: {scores.mean():.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Perform Sequential Feature Selector with n=5 是机器学习中的常用技术。  
  *Perform Sequential Feature Selector with n=5 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `dropna` | 删除缺失值 | Drop missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sfs / 03 Sfs
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SequentialFeatureSelector

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]
model = LinearRegression()

# Perform Sequential Feature Selector with n=5
sfs = SequentialFeatureSelector(model, n_features_to_select=5)
sfs.fit(X, y)

# 获取列名 / Get column names
selected_features = X.columns[sfs.get_support()].to_list()
# 打印输出 / Print output
print(f"Features selected by SFS: {selected_features}")

# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, Ames[selected_features], y)
# 打印输出 / Print output
print(f"Mean CV R^2 Score using SFS with n=5: {scores.mean():.4f}")
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Plot

# 04 — Plot / 04 Plot

**Chapter 04 — File 4 of 6 / 第04章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Prepare to store the mean CV R^2 scores for each number of features**.

本脚本演示 **Prepare to store the mean CV R^2 scores for each number of features**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SequentialFeatureSelector
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]
model = LinearRegression()
```

---
## Step 2 — Prepare to store the mean CV R^2 scores for each number of features

```python
mean_scores = []
```

---
## Step 3 — Performance of SFS from 1 feature to the maximum number of features available

```python
# 获取列名 / Get column names
for n_features_to_select in range(1, len(X.columns)):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select)
    sfs.fit(X, y)
    # 获取列名 / Get column names
    selected_features = X.columns[sfs.get_support()]
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    score = cross_val_score(model, X[selected_features], y, cv=5, scoring="r2").mean()
    # 添加元素到列表末尾 / Append element to list end
    mean_scores.append(score)
```

---
## Step 4 — Plot the mean CV R^2 scores against the number of features selected

```python
# 创建画布 / Create figure canvas
plt.figure(figsize=(10, 6))
# 获取列名 / Get column names
plt.plot(range(1, len(X.columns)), mean_scores, marker="o")
# 设置图表标题 / Set chart title
plt.title("Performance vs. Number of Features Selected")
# 设置X轴标签 / Set X-axis label
plt.xlabel("Number of Features")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("Mean CV R^2 Score")
plt.grid(True)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Prepare to store the mean CV R^2 scores for each number of features 是机器学习中的常用技术。  
  *Prepare to store the mean CV R^2 scores for each number of features is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `dropna` | 删除缺失值 | Drop missing values |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot / 04 Plot
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SequentialFeatureSelector
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]
model = LinearRegression()

# Prepare to store the mean CV R^2 scores for each number of features
mean_scores = []

# Performance of SFS from 1 feature to the maximum number of features available
# 获取列名 / Get column names
for n_features_to_select in range(1, len(X.columns)):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select)
    sfs.fit(X, y)
    # 获取列名 / Get column names
    selected_features = X.columns[sfs.get_support()]
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    score = cross_val_score(model, X[selected_features], y, cv=5, scoring="r2").mean()
    # 添加元素到列表末尾 / Append element to list end
    mean_scores.append(score)

# Plot the mean CV R^2 scores against the number of features selected
# 创建画布 / Create figure canvas
plt.figure(figsize=(10, 6))
# 获取列名 / Get column names
plt.plot(range(1, len(X.columns)), mean_scores, marker="o")
# 设置图表标题 / Set chart title
plt.title("Performance vs. Number of Features Selected")
# 设置X轴标签 / Set X-axis label
plt.xlabel("Number of Features")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("Mean CV R^2 Score")
plt.grid(True)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Tolerance

# 05 — Tolerance / 05 Tolerance

**Chapter 04 — File 5 of 6 / 第04章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Apply Sequential Feature Selector with tolerance = 0.005**.

本脚本演示 **Apply Sequential Feature Selector with tolerance = 0.005**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SequentialFeatureSelector
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]
model = LinearRegression()
```

---
## Step 2 — Apply Sequential Feature Selector with tolerance = 0.005

```python
sfs_tol = SequentialFeatureSelector(model, n_features_to_select="auto", tol=0.005)
sfs_tol.fit(X, y)
```

---
## Step 3 — Get the number of features selected with tolerance

```python
n_features_selected = sum(sfs_tol.get_support())
```

---
## Step 4 — Prepare to store the mean CV R^2 scores for each number of features

```python
mean_scores_tol = []
```

---
## Step 5 — Iterate over a range from 1 feature to the Sweet Spot

```python
# 生成整数序列 / Generate integer sequence
for n_features_to_select in range(1, n_features_selected + 1):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select)
    sfs.fit(X, y)
    # 获取列名 / Get column names
    selected_features = X.columns[sfs.get_support()]
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    score = cross_val_score(model, X[selected_features], y, cv=5, scoring="r2").mean()
    # 添加元素到列表末尾 / Append element to list end
    mean_scores_tol.append(score)
```

---
## Step 6 — Plot the mean CV R^2 scores against the number of features selected

```python
# 创建画布 / Create figure canvas
plt.figure(figsize=(10, 6))
# 绘制折线图 / Draw line plot
plt.plot(range(1, n_features_selected + 1), mean_scores_tol, marker="o")
# 设置图表标题 / Set chart title
plt.title("The Sweet Spot: Performance vs. Number of Features Selected")
# 设置X轴标签 / Set X-axis label
plt.xlabel("Number of Features")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("Mean CV R^2 Score")
plt.grid(True)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Apply Sequential Feature Selector with tolerance = 0.005 是机器学习中的常用技术。  
  *Apply Sequential Feature Selector with tolerance = 0.005 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `dropna` | 删除缺失值 | Drop missing values |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Tolerance / 05 Tolerance
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SequentialFeatureSelector
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]
model = LinearRegression()

# Apply Sequential Feature Selector with tolerance = 0.005
sfs_tol = SequentialFeatureSelector(model, n_features_to_select="auto", tol=0.005)
sfs_tol.fit(X, y)

# Get the number of features selected with tolerance
n_features_selected = sum(sfs_tol.get_support())

# Prepare to store the mean CV R^2 scores for each number of features
mean_scores_tol = []

# Iterate over a range from 1 feature to the Sweet Spot
# 生成整数序列 / Generate integer sequence
for n_features_to_select in range(1, n_features_selected + 1):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select)
    sfs.fit(X, y)
    # 获取列名 / Get column names
    selected_features = X.columns[sfs.get_support()]
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    score = cross_val_score(model, X[selected_features], y, cv=5, scoring="r2").mean()
    # 添加元素到列表末尾 / Append element to list end
    mean_scores_tol.append(score)

# Plot the mean CV R^2 scores against the number of features selected
# 创建画布 / Create figure canvas
plt.figure(figsize=(10, 6))
# 绘制折线图 / Draw line plot
plt.plot(range(1, n_features_selected + 1), mean_scores_tol, marker="o")
# 设置图表标题 / Set chart title
plt.title("The Sweet Spot: Performance vs. Number of Features Selected")
# 设置X轴标签 / Set X-axis label
plt.xlabel("Number of Features")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("Mean CV R^2 Score")
plt.grid(True)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Features

# 06 — Features / 特征工程

**Chapter 04 — File 6 of 6 / 第04章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Apply Sequential Feature Selector with tolerance = 0.005**.

本脚本演示 **Apply Sequential Feature Selector with tolerance = 0.005**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SequentialFeatureSelector

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]
model = LinearRegression()
```

---
## Step 2 — Apply Sequential Feature Selector with tolerance = 0.005

```python
sfs_tol = SequentialFeatureSelector(model, n_features_to_select="auto", tol=0.005)
sfs_tol.fit(X, y)
```

---
## Step 3 — Get the number of features selected with tolerance

```python
n_features_selected = sum(sfs_tol.get_support())
```

---
## Step 4 — Prepare to store the mean CV R^2 scores for each number of features

```python
mean_scores_tol = []
```

---
## Step 5 — Iterate over a range from 1 feature to the Sweet Spot

```python
# 生成整数序列 / Generate integer sequence
for n_features_to_select in range(1, n_features_selected + 1):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select)
    sfs.fit(X, y)
    # 获取列名 / Get column names
    selected_features = X.columns[sfs.get_support()]
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    score = cross_val_score(model, X[selected_features], y, cv=5, scoring="r2").mean()
    # 添加元素到列表末尾 / Append element to list end
    mean_scores_tol.append(score)
```

---
## Step 6 — Print the selected features and their performance

```python
# 获取列名 / Get column names
selected_features = X.columns[sfs_tol.get_support()]
# 打印输出 / Print output
print(f"Number of features selected: {n_features_selected}")
# 打印输出 / Print output
print(f"Selected features: {selected_features.tolist()}")
# 打印输出 / Print output
print(f"Mean CV R^2 Score using SFS with tol=0.005: {mean_scores_tol[-1]:.4f}")
```

---
## Learning Notes / 学习笔记

- **概念**: Apply Sequential Feature Selector with tolerance = 0.005 是机器学习中的常用技术。  
  *Apply Sequential Feature Selector with tolerance = 0.005 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `dropna` | 删除缺失值 | Drop missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Features / 特征工程
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SequentialFeatureSelector

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv").select_dtypes(include=["int64", "float64"]).dropna(axis=1)
# 删除指定列或行 / Drop specified columns or rows
X = Ames.drop("SalePrice", axis=1)
y = Ames["SalePrice"]
model = LinearRegression()

# Apply Sequential Feature Selector with tolerance = 0.005
sfs_tol = SequentialFeatureSelector(model, n_features_to_select="auto", tol=0.005)
sfs_tol.fit(X, y)

# Get the number of features selected with tolerance
n_features_selected = sum(sfs_tol.get_support())

# Prepare to store the mean CV R^2 scores for each number of features
mean_scores_tol = []

# Iterate over a range from 1 feature to the Sweet Spot
# 生成整数序列 / Generate integer sequence
for n_features_to_select in range(1, n_features_selected + 1):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select)
    sfs.fit(X, y)
    # 获取列名 / Get column names
    selected_features = X.columns[sfs.get_support()]
    # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
    score = cross_val_score(model, X[selected_features], y, cv=5, scoring="r2").mean()
    # 添加元素到列表末尾 / Append element to list end
    mean_scores_tol.append(score)

# Print the selected features and their performance
# 获取列名 / Get column names
selected_features = X.columns[sfs_tol.get_support()]
# 打印输出 / Print output
print(f"Number of features selected: {n_features_selected}")
# 打印输出 / Print output
print(f"Selected features: {selected_features.tolist()}")
# 打印输出 / Print output
print(f"Mean CV R^2 Score using SFS with tol=0.005: {mean_scores_tol[-1]:.4f}")
```

---

### Chapter Summary / 章节总结

# Chapter 04 Summary / 第04章总结

## Theme / 主题: Chapter 04 / Chapter 04

This chapter contains **6 code files** demonstrating chapter 04.

本章包含 **6 个代码文件**，演示Chapter 04。

---
## Evolution / 演化路线

  1. `01_topfive.ipynb` — Topfive
  2. `02_regression.ipynb` — Regression
  3. `03_sfs.ipynb` — Sfs
  4. `04_plot.ipynb` — Plot
  5. `05_tolerance.ipynb` — Tolerance
  6. `06_features.ipynb` — Features

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 04) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 04）是机器学习流水线中的基础构建块。

---
