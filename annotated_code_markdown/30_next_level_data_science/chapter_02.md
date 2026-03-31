# 进阶数据科学 / Next Level Data Science
## Chapter 02

---

### Crossvalidate

# 01 — Crossvalidate / 01 Crossvalidate

**Chapter 02 — File 1 of 3 / 第02章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Load the Ames dataset**.

本脚本演示 **Load the Ames dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — Load the Ames dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
```

---
## Step 2 — Import Linear Regression, Train-Test, Cross-Validation from scikit-learn

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split, cross_val_score
```

---
## Step 3 — Select features and target

```python
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, a 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, a 1D vector
```

---
## Step 4 — Split data into training and testing sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---
## Step 5 — Linear Regression model using Train-Test

```python
model = LinearRegression()
# 训练模型 / Train the model
model.fit(X_train, y_train)
train_test_score = round(model.score(X_test, y_test), 4)
# 打印输出 / Print output
print(f"Train-Test R^2 Score: {train_test_score}")
```

---
## Step 6 — Perform 5-Fold Cross-Validation

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
cv_scores = cross_val_score(model, X, y, cv=5)
cv_scores_rounded = [round(score, 4) for score in cv_scores]
# 打印输出 / Print output
print(f"Cross-Validation R^2 Scores: {cv_scores_rounded}")
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
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `model.fit` | 训练模型 | Train the model |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Crossvalidate / 01 Crossvalidate
# Complete Code / 完整代码
# ===============================

# Load the Ames dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")

# Import Linear Regression, Train-Test, Cross-Validation from scikit-learn
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split, cross_val_score

# Select features and target
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, a 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, a 1D vector

# Split data into training and testing sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model using Train-Test
model = LinearRegression()
# 训练模型 / Train the model
model.fit(X_train, y_train)
train_test_score = round(model.score(X_test, y_test), 4)
# 打印输出 / Print output
print(f"Train-Test R^2 Score: {train_test_score}")

# Perform 5-Fold Cross-Validation
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
cv_scores = cross_val_score(model, X, y, cv=5)
cv_scores_rounded = [round(score, 4) for score in cv_scores]
# 打印输出 / Print output
print(f"Cross-Validation R^2 Scores: {cv_scores_rounded}")
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Plot Cv

# 02 — Plot Cv / 02 Plot Cv

**Chapter 02 — File 2 of 3 / 第02章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Import Seaborn and Matplotlib**.

本脚本演示 **Import Seaborn and Matplotlib**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split, cross_val_score
```

---
## Step 2 — Import Seaborn and Matplotlib

```python
import seaborn as sns
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
```

---
## Step 3 — Perform 5-fold cross-validation. Let cv_scores_rounded contains your
cross-validation scores, and train_test_score is your single train-test R^2 score

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, a 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, a 1D vector
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
# 训练模型 / Train the model
model.fit(X_train, y_train)
train_test_score = round(model.score(X_test, y_test), 4)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
cv_scores = cross_val_score(model, X, y, cv=5)
cv_scores_rounded = [round(score, 4) for score in cv_scores]
```

---
## Step 4 — Plot the box plot for cross-validation scores

```python
cv_scores_df = pd.DataFrame(cv_scores_rounded, columns=["Cross-Validation Scores"])
sns.boxplot(data=cv_scores_df, y="Cross-Validation Scores",
            width=0.3, color="lightblue", fliersize=0)
```

---
## Step 5 — Overlay individual scores as points

```python
# 绘制散点图 / Draw scatter plot
plt.scatter([0] * len(cv_scores_rounded), cv_scores_rounded,
            color="blue", label="Cross-Validation Scores")
# 绘制散点图 / Draw scatter plot
plt.scatter(0, train_test_score, color="red", zorder=5, label="Train-Test Score")
```

---
## Step 6 — Plot the visual

```python
# 设置图表标题 / Set chart title
plt.title("Model Evaluation: Cross-Validation vs. Train-Test")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("R^2 Score")
plt.xticks([0], ["Evaluation Scores"])
# 显示图例 / Show legend
plt.legend(loc="lower left", bbox_to_anchor=(0, +0.1))
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Import Seaborn and Matplotlib 是机器学习中的常用技术。  
  *Import Seaborn and Matplotlib is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Cv / 02 Plot Cv
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split, cross_val_score
# Import Seaborn and Matplotlib
import seaborn as sns
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# Perform 5-fold cross-validation. Let cv_scores_rounded contains your
# cross-validation scores, and train_test_score is your single train-test R^2 score
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
X = Ames[["GrLivArea"]]  # Feature: GrLivArea, a 2D matrix
y = Ames["SalePrice"]    # Target: SalePrice, a 1D vector
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
# 训练模型 / Train the model
model.fit(X_train, y_train)
train_test_score = round(model.score(X_test, y_test), 4)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
cv_scores = cross_val_score(model, X, y, cv=5)
cv_scores_rounded = [round(score, 4) for score in cv_scores]

# Plot the box plot for cross-validation scores
cv_scores_df = pd.DataFrame(cv_scores_rounded, columns=["Cross-Validation Scores"])
sns.boxplot(data=cv_scores_df, y="Cross-Validation Scores",
            width=0.3, color="lightblue", fliersize=0)

# Overlay individual scores as points
# 绘制散点图 / Draw scatter plot
plt.scatter([0] * len(cv_scores_rounded), cv_scores_rounded,
            color="blue", label="Cross-Validation Scores")
# 绘制散点图 / Draw scatter plot
plt.scatter(0, train_test_score, color="red", zorder=5, label="Train-Test Score")

# Plot the visual
# 设置图表标题 / Set chart title
plt.title("Model Evaluation: Cross-Validation vs. Train-Test")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("R^2 Score")
plt.xticks([0], ["Evaluation Scores"])
# 显示图例 / Show legend
plt.legend(loc="lower left", bbox_to_anchor=(0, +0.1))
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Kfold

# 03 — Kfold / 03 Kfold

**Chapter 02 — File 3 of 3 / 第02章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Import k-fold and necessary libraries**.

本脚本演示 **Import k-fold and necessary libraries**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
```

---
## Step 2 — Import k-fold and necessary libraries

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import r2_score

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Select features and target

```python
# 转换为NumPy数组 / Convert to NumPy array
X = Ames[['GrLivArea']].values  # Convert to numpy array for KFold
# 转换为NumPy数组 / Convert to NumPy array
y = Ames['SalePrice'].values    # Convert to numpy array for KFold
```

---
## Step 4 — Initialize linear regression and k-fold

```python
model = LinearRegression()
kf = KFold(n_splits=5)
```

---
## Step 5 — k-fold cross-validation in detailed steps

```python
# 同时获取索引和值 / Get both index and value
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
```

---
## Step 6 — Split the data into training and testing sets

```python
X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

---
## Step 7 — Fit the model and predict

```python
# 训练模型 / Train the model
model.fit(X_train, y_train)
    # 用模型做预测 / Make predictions with model
    y_pred = model.predict(X_test)
```

---
## Step 8 — Calculate and print the R^2 score for the current fold

```python
# 打印输出 / Print output
print(f"Fold {fold}:")
    # 打印输出 / Print output
    print(f"TRAIN set size: {len(train_index)}")
    # 打印输出 / Print output
    print(f"TEST set size: {len(test_index)}")
    # 计算R²决定系数（越接近1越好） / R² score (closer to 1 is better)
    print(f"R^2 score: {round(r2_score(y_test, y_pred), 4)}\n")
```

---
## Learning Notes / 学习笔记

- **概念**: Import k-fold and necessary libraries 是机器学习中的常用技术。  
  *Import k-fold and necessary libraries is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Kfold / 03 Kfold
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# Import k-fold and necessary libraries
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import r2_score

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")

# Select features and target
# 转换为NumPy数组 / Convert to NumPy array
X = Ames[['GrLivArea']].values  # Convert to numpy array for KFold
# 转换为NumPy数组 / Convert to NumPy array
y = Ames['SalePrice'].values    # Convert to numpy array for KFold

# Initialize linear regression and k-fold
model = LinearRegression()
kf = KFold(n_splits=5)

# k-fold cross-validation in detailed steps
# 同时获取索引和值 / Get both index and value
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model and predict
    # 训练模型 / Train the model
    model.fit(X_train, y_train)
    # 用模型做预测 / Make predictions with model
    y_pred = model.predict(X_test)

    # Calculate and print the R^2 score for the current fold
    # 打印输出 / Print output
    print(f"Fold {fold}:")
    # 打印输出 / Print output
    print(f"TRAIN set size: {len(train_index)}")
    # 打印输出 / Print output
    print(f"TEST set size: {len(test_index)}")
    # 计算R²决定系数（越接近1越好） / R² score (closer to 1 is better)
    print(f"R^2 score: {round(r2_score(y_test, y_pred), 4)}\n")
```

---

### Chapter Summary / 章节总结

# Chapter 02 Summary / 第02章总结

## Theme / 主题: Chapter 02 / Chapter 02

This chapter contains **3 code files** demonstrating chapter 02.

本章包含 **3 个代码文件**，演示Chapter 02。

---
## Evolution / 演化路线

  1. `01_crossvalidate.ipynb` — Crossvalidate
  2. `02_plot_cv.ipynb` — Plot Cv
  3. `03_kfold.ipynb` — Kfold

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 02) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 02）是机器学习流水线中的基础构建块。

---
