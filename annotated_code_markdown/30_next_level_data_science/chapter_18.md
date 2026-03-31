# 进阶数据科学 / Next Level Data Science
## Chapter 18

---

### Catboost



---

### Plot

# 03 — Plot / 03 Plot

**Chapter 18 — File 2 of 2 / 第18章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Load dataset**.

本脚本演示 **Load dataset**。

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
import seaborn as sns
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
```

---
## Step 2 — Load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("Ames.csv")
# 删除指定列或行 / Drop specified columns or rows
X = data.drop(["SalePrice"], axis=1)
y = data["SalePrice"]
# 获取列名 / Get column names
cat_features = [col for col in X.columns if X[col].dtype == "object"]
# 填充缺失值 / Fill missing values
X["Electrical"] = X["Electrical"].fillna(X["Electrical"].mode()[0])
# 填充缺失值 / Fill missing values
X[cat_features] = X[cat_features].fillna("Missing")
# 获取列名 / Get column names
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
```

---
## Step 3 — Set up k-fold cross-validation

```python
kf = KFold(n_splits=5)
feature_importances = []
```

---
## Step 4 — Iterate over each split

```python
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
```

---
## Step 5 — Train default CatBoost model

```python
model = CatBoostRegressor(cat_features=cat_features, random_state=42, verbose=0)
    # 训练模型 / Train the model
    model.fit(X_train, y_train)
    # 添加元素到列表末尾 / Append element to list end
    feature_importances.append(model.get_feature_importance())
```

---
## Step 6 — Average feature importance across all folds

```python
# 计算均值 / Calculate mean
avg_importance = np.mean(feature_importances, axis=0)
```

---
## Step 7 — Convert to DataFrame

```python
# 获取列名 / Get column names
feat_imp_df = pd.DataFrame({"Feature": X.columns, "Importance": avg_importance})
```

---
## Step 8 — Sort and take the top 20 features

```python
# 查看前几行数据（快速预览） / View first rows (quick preview)
top_features = feat_imp_df.sort_values(by="Importance", ascending=False).head(20)
```

---
## Step 9 — Set the style and color palette

```python
sns.set_style("whitegrid")
# 获取长度 / Get length
palette = sns.color_palette("rocket", len(top_features))
```

---
## Step 10 — Create the plot

```python
# 创建画布 / Create figure canvas
plt.figure(figsize=(12, 10))
ax = sns.barplot(x="Importance", y="Feature", hue="Feature",
                 data=top_features, palette=palette, legend=False)
```

---
## Step 11 — Customize the plot

```python
# 设置图表标题 / Set chart title
plt.title("Top 20 Most Important Features - CatBoost Model",
          fontsize=20, fontweight="bold")
# 设置X轴标签 / Set X-axis label
plt.xlabel("Importance Score", fontsize=15)
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("Features", fontsize=15)
```

---
## Step 12 — Add value labels to the end of each bar

```python
# 同时获取索引和值 / Get both index and value
for i, v in enumerate(top_features["Importance"]):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=13)
```

---
## Step 13 — Extend x-axis by 10% and feature names font size

```python
plt.xlim(0, max(top_features["Importance"]) * 1.1)
plt.yticks(fontsize=13)
```

---
## Step 14 — Adjust layout and display

```python
plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load dataset 是机器学习中的常用技术。  
  *Load dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `fillna` | 填充缺失值 | Fill missing values |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `np.mean` | 计算均值 | Calculate mean |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.show` | 显示图表 | Display plot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot / 03 Plot
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
import seaborn as sns
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold

# Load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = pd.read_csv("Ames.csv")
# 删除指定列或行 / Drop specified columns or rows
X = data.drop(["SalePrice"], axis=1)
y = data["SalePrice"]
# 获取列名 / Get column names
cat_features = [col for col in X.columns if X[col].dtype == "object"]
# 填充缺失值 / Fill missing values
X["Electrical"] = X["Electrical"].fillna(X["Electrical"].mode()[0])
# 填充缺失值 / Fill missing values
X[cat_features] = X[cat_features].fillna("Missing")
# 获取列名 / Get column names
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Set up k-fold cross-validation
kf = KFold(n_splits=5)
feature_importances = []

# Iterate over each split
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train default CatBoost model
    model = CatBoostRegressor(cat_features=cat_features, random_state=42, verbose=0)
    # 训练模型 / Train the model
    model.fit(X_train, y_train)
    # 添加元素到列表末尾 / Append element to list end
    feature_importances.append(model.get_feature_importance())

# Average feature importance across all folds
# 计算均值 / Calculate mean
avg_importance = np.mean(feature_importances, axis=0)

# Convert to DataFrame
# 获取列名 / Get column names
feat_imp_df = pd.DataFrame({"Feature": X.columns, "Importance": avg_importance})

# Sort and take the top 20 features
# 查看前几行数据（快速预览） / View first rows (quick preview)
top_features = feat_imp_df.sort_values(by="Importance", ascending=False).head(20)

# Set the style and color palette
sns.set_style("whitegrid")
# 获取长度 / Get length
palette = sns.color_palette("rocket", len(top_features))

# Create the plot
# 创建画布 / Create figure canvas
plt.figure(figsize=(12, 10))
ax = sns.barplot(x="Importance", y="Feature", hue="Feature",
                 data=top_features, palette=palette, legend=False)

# Customize the plot
# 设置图表标题 / Set chart title
plt.title("Top 20 Most Important Features - CatBoost Model",
          fontsize=20, fontweight="bold")
# 设置X轴标签 / Set X-axis label
plt.xlabel("Importance Score", fontsize=15)
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("Features", fontsize=15)

# Add value labels to the end of each bar
# 同时获取索引和值 / Get both index and value
for i, v in enumerate(top_features["Importance"]):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=13)

# Extend x-axis by 10% and feature names font size
plt.xlim(0, max(top_features["Importance"]) * 1.1)
plt.yticks(fontsize=13)

# Adjust layout and display
plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结



---
