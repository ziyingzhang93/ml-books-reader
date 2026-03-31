# 进阶数据科学 / Next Level Data Science
## Chapter 09

---

### Matrix

# 01 — Matrix / 01 Matrix

**Chapter 09 — File 1 of 6 / 第09章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Import necessary libraries to check and compare number of columns vs rank of dataset**.

本脚本演示 **Import necessary libraries to check and compare number of columns vs rank of dataset**。

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
## Step 1 — Import necessary libraries to check and compare number of columns vs rank of dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
```

---
## Step 2 — Load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
```

---
## Step 3 — Select numerical columns without missing values

```python
# 删除含缺失值的行 / Drop rows with missing values
numerical_data = Ames.select_dtypes(include=[np.number]).dropna(axis=1)
```

---
## Step 4 — Calculate the matrix rank

```python
# 转换为NumPy数组 / Convert to NumPy array
rank = np.linalg.matrix_rank(numerical_data.values)
```

---
## Step 5 — Number of features

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
num_features = numerical_data.shape[1]
```

---
## Step 6 — Print the rank and the number of features

```python
# 打印输出 / Print output
print(f"Numerical features without missing values: {num_features}")
# 打印输出 / Print output
print(f"Rank: {rank}")
```

---
## Learning Notes / 学习笔记

- **概念**: Import necessary libraries to check and compare number of columns vs rank of dataset 是机器学习中的常用技术。  
  *Import necessary libraries to check and compare number of columns vs rank of dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `dropna` | 删除缺失值 | Drop missing values |
| `np.linalg` | 线性代数运算 | Linear algebra operations |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Matrix / 01 Matrix
# Complete Code / 完整代码
# ===============================

# Import necessary libraries to check and compare number of columns vs rank of dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# Load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")

# Select numerical columns without missing values
# 删除含缺失值的行 / Drop rows with missing values
numerical_data = Ames.select_dtypes(include=[np.number]).dropna(axis=1)

# Calculate the matrix rank
# 转换为NumPy数组 / Convert to NumPy array
rank = np.linalg.matrix_rank(numerical_data.values)

# Number of features
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
num_features = numerical_data.shape[1]

# Print the rank and the number of features
# 打印输出 / Print output
print(f"Numerical features without missing values: {num_features}")
# 打印输出 / Print output
print(f"Rank: {rank}")
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Redundant

# 02 — Redundant / 02 Redundant

**Chapter 09 — File 2 of 6 / 第09章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Creating and using a function to identify redundant features in a dataset**.

本脚本演示 **Creating and using a function to identify redundant features in a dataset**。

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
## Step 1 — Creating and using a function to identify redundant features in a dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

def find_redundant_features(data):
    """
    Identifies and returns redundant features in a dataset based on matrix rank.
    A feature is considered redundant if removing it does not decrease the rank of the
    dataset, indicating that it can be expressed as a linear combination of other
    features.

    Parameters:
        data (DataFrame): The numerical dataset to analyze.

    Returns:
        list: A list of redundant feature names.
    """
```

---
## Step 2 — Calculate the matrix rank of the original dataset

```python
original_rank = np.linalg.matrix_rank(data)
    redundant_features = []

    # 获取列名 / Get column names
    for column in data.columns:
```

---
## Step 3 — Create a new dataset without this column

```python
# 删除指定列或行 / Drop specified columns or rows
temp_data = data.drop(column, axis=1)
```

---
## Step 4 — Calculate the rank of the new dataset

```python
temp_rank = np.linalg.matrix_rank(temp_data)
```

---
## Step 5 — If the rank does not decrease, the removed column is redundant

```python
if temp_rank == original_rank:
            # 添加元素到列表末尾 / Append element to list end
            redundant_features.append(column)

    return redundant_features
```

---
## Step 6 — Usage of the function with the numerical data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
# 删除含缺失值的行 / Drop rows with missing values
numerical_data = Ames.select_dtypes(include=[np.number]).dropna(axis=1)
redundant_features = find_redundant_features(numerical_data)
# 打印输出 / Print output
print("Redundant features:", redundant_features)
```

---
## Learning Notes / 学习笔记

- **概念**: Creating and using a function to identify redundant features in a dataset 是机器学习中的常用技术。  
  *Creating and using a function to identify redundant features in a dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `dropna` | 删除缺失值 | Drop missing values |
| `np.linalg` | 线性代数运算 | Linear algebra operations |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Redundant / 02 Redundant
# Complete Code / 完整代码
# ===============================

# Creating and using a function to identify redundant features in a dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

def find_redundant_features(data):
    """
    Identifies and returns redundant features in a dataset based on matrix rank.
    A feature is considered redundant if removing it does not decrease the rank of the
    dataset, indicating that it can be expressed as a linear combination of other
    features.

    Parameters:
        data (DataFrame): The numerical dataset to analyze.

    Returns:
        list: A list of redundant feature names.
    """

    # Calculate the matrix rank of the original dataset
    original_rank = np.linalg.matrix_rank(data)
    redundant_features = []

    # 获取列名 / Get column names
    for column in data.columns:
        # Create a new dataset without this column
        # 删除指定列或行 / Drop specified columns or rows
        temp_data = data.drop(column, axis=1)
        # Calculate the rank of the new dataset
        temp_rank = np.linalg.matrix_rank(temp_data)

        # If the rank does not decrease, the removed column is redundant
        if temp_rank == original_rank:
            # 添加元素到列表末尾 / Append element to list end
            redundant_features.append(column)

    return redundant_features

# Usage of the function with the numerical data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
# 删除含缺失值的行 / Drop rows with missing values
numerical_data = Ames.select_dtypes(include=[np.number]).dropna(axis=1)
redundant_features = find_redundant_features(numerical_data)
# 打印输出 / Print output
print("Redundant features:", redundant_features)
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Verify



---

### Plot

# 04 — Plot / 04 Plot

**Chapter 09 — File 4 of 6 / 第09章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Import necessary libraries**.

本脚本演示 **Import necessary libraries**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — Import necessary libraries

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
```

---
## Step 2 — Load the data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
features = ["GrLivArea", "1stFlrSF", "2ndFlrSF", "LowQualFinSF"]
X = Ames[features]
y = Ames["SalePrice"]
```

---
## Step 3 — Initialize a K-Fold cross-validation

```python
kf = KFold(n_splits=5, shuffle=True, random_state=1)
```

---
## Step 4 — Collect coefficients and CV scores

```python
coefficients = []
cv_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
```

---
## Step 5 — Initialize and fit the linear regression model

```python
model = LinearRegression()
    # 训练模型 / Train the model
    model.fit(X_train, y_train)
    # 添加元素到列表末尾 / Append element to list end
    coefficients.append(model.coef_)
```

---
## Step 6 — Calculate R^2 score using the model's score method

```python
score = model.score(X_test, y_test)
```

---
## Step 7 — print(score)

```python
# 添加元素到列表末尾 / Append element to list end
cv_scores.append(score)
```

---
## Step 8 — Plotting the coefficients

```python
# 创建画布 / Create figure canvas
plt.figure(figsize=(12, 6))
# 创建子图 / Create subplot
plt.subplot(1, 2, 1)
# 创建NumPy数组 / Create NumPy array
plt.boxplot(np.array(coefficients), labels=features)
# 设置图表标题 / Set chart title
plt.title("Box Plot of Coefficients Across Folds (MLR)")
# 设置X轴标签 / Set X-axis label
plt.xlabel("Features")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("Coefficient Value")
plt.grid(True)
```

---
## Step 9 — Plotting the CV scores

```python
# 创建子图 / Create subplot
plt.subplot(1, 2, 2)
# 绘制折线图 / Draw line plot
plt.plot(range(1, 6), cv_scores, marker="o", linestyle="-")  # Make x-axis to start from 1
# 设置图表标题 / Set chart title
plt.title("Cross-Validation R^2 Scores (MLR)")
# 设置X轴标签 / Set X-axis label
plt.xlabel("Fold")
# 生成整数序列 / Generate integer sequence
plt.xticks(range(1, 6))  # Set x-ticks to match fold numbers
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("R^2 Score")
plt.ylim(min(cv_scores) - 0.05, max(cv_scores) + 0.05)  # Dynamically adjust y-axis limits
plt.grid(True)
```

---
## Step 10 — Annotate mean R^2 score

```python
# 计算均值 / Calculate mean
mean_r2 = np.mean(cv_scores)
plt.annotate(f"Mean CV R^2: {mean_r2:.3f}", xy=(1.25, 0.65), color="red", fontsize=14),

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
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
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.mean` | 计算均值 | Calculate mean |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot / 04 Plot
# Complete Code / 完整代码
# ===============================

# Import necessary libraries
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# Load the data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
features = ["GrLivArea", "1stFlrSF", "2ndFlrSF", "LowQualFinSF"]
X = Ames[features]
y = Ames["SalePrice"]

# Initialize a K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Collect coefficients and CV scores
coefficients = []
cv_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Initialize and fit the linear regression model
    model = LinearRegression()
    # 训练模型 / Train the model
    model.fit(X_train, y_train)
    # 添加元素到列表末尾 / Append element to list end
    coefficients.append(model.coef_)

    # Calculate R^2 score using the model's score method
    score = model.score(X_test, y_test)
    # print(score)
    # 添加元素到列表末尾 / Append element to list end
    cv_scores.append(score)

# Plotting the coefficients
# 创建画布 / Create figure canvas
plt.figure(figsize=(12, 6))
# 创建子图 / Create subplot
plt.subplot(1, 2, 1)
# 创建NumPy数组 / Create NumPy array
plt.boxplot(np.array(coefficients), labels=features)
# 设置图表标题 / Set chart title
plt.title("Box Plot of Coefficients Across Folds (MLR)")
# 设置X轴标签 / Set X-axis label
plt.xlabel("Features")
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("Coefficient Value")
plt.grid(True)

# Plotting the CV scores
# 创建子图 / Create subplot
plt.subplot(1, 2, 2)
# 绘制折线图 / Draw line plot
plt.plot(range(1, 6), cv_scores, marker="o", linestyle="-")  # Make x-axis to start from 1
# 设置图表标题 / Set chart title
plt.title("Cross-Validation R^2 Scores (MLR)")
# 设置X轴标签 / Set X-axis label
plt.xlabel("Fold")
# 生成整数序列 / Generate integer sequence
plt.xticks(range(1, 6))  # Set x-ticks to match fold numbers
# 设置Y轴标签 / Set Y-axis label
plt.ylabel("R^2 Score")
plt.ylim(min(cv_scores) - 0.05, max(cv_scores) + 0.05)  # Dynamically adjust y-axis limits
plt.grid(True)

# Annotate mean R^2 score
# 计算均值 / Calculate mean
mean_r2 = np.mean(cv_scores)
plt.annotate(f"Mean CV R^2: {mean_r2:.3f}", xy=(1.25, 0.65), color="red", fontsize=14),

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Lasso

# 05 — Lasso / 05 Lasso

**Chapter 09 — File 5 of 6 / 第09章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Import necessary libraries**.

本脚本演示 **Import necessary libraries**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — Import necessary libraries

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
```

---
## Step 2 — Load the data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
features = ["GrLivArea", "1stFlrSF", "2ndFlrSF", "LowQualFinSF"]
X = Ames[features]
y = Ames["SalePrice"]
```

---
## Step 3 — Initialize a k-fold cross-validation

```python
kf = KFold(n_splits=5, shuffle=True, random_state=1)
```

---
## Step 4 — Prepare to collect results

```python
results = {}

for alpha in [1, 2]:  # Loop through both alpha values
    coefficients = []
    cv_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
```

---
## Step 5 — Scale features

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler()
        # 拟合并转换数据（一步完成） / Fit and transform data (one step)
        X_train_scaled = scaler.fit_transform(X_train)
        # 用已拟合的模型转换数据 / Transform data with fitted model
        X_test_scaled = scaler.transform(X_test)
```

---
## Step 6 — Initialize and fit the Lasso regression model

```python
lasso_model = Lasso(alpha=alpha, max_iter=20000)
        # 训练模型 / Train the model
        lasso_model.fit(X_train_scaled, y_train)
        # 添加元素到列表末尾 / Append element to list end
        coefficients.append(lasso_model.coef_)
```

---
## Step 7 — Calculate R^2 score using the model's score method

```python
score = lasso_model.score(X_test_scaled, y_test)
        # 添加元素到列表末尾 / Append element to list end
        cv_scores.append(score)

    results[alpha] = (coefficients, cv_scores)
```

---
## Step 8 — Plotting the results

```python
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
alphas = [1, 2]

# 同时获取索引和值 / Get both index and value
for i, alpha in enumerate(alphas):
    coefficients, cv_scores = results[alpha]
```

---
## Step 9 — Plotting the coefficients

```python
# 创建NumPy数组 / Create NumPy array
axes[i, 0].boxplot(np.array(coefficients), labels=features)
    axes[i, 0].set_title(f"Box Plot of Coefficients (Lasso with alpha={alpha})")
    axes[i, 0].set_xlabel("Features")
    axes[i, 0].set_ylabel("Coefficient Value")
    axes[i, 0].grid(True)
```

---
## Step 10 — Plotting the CV scores

```python
# 生成整数序列 / Generate integer sequence
axes[i, 1].plot(range(1, 6), cv_scores, marker="o", linestyle="-")
    axes[i, 1].set_title(f"Cross-Validation R^2 Scores (Lasso with alpha={alpha})")
    axes[i, 1].set_xlabel("Fold")
    # 生成整数序列 / Generate integer sequence
    axes[i, 1].set_xticks(range(1, 6))
    axes[i, 1].set_ylabel("R^2 Score")
    axes[i, 1].set_ylim(min(cv_scores) - 0.05, max(cv_scores) + 0.05)
    axes[i, 1].grid(True)
    # 计算均值 / Calculate mean
    mean_r2 = np.mean(cv_scores)
    axes[i, 1].annotate(f"Mean CV R^2: {mean_r2:.3f}", xy=(1.25, 0.65),
                        color="red", fontsize=12)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
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
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.mean` | 计算均值 | Calculate mean |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lasso / 05 Lasso
# Complete Code / 完整代码
# ===============================

# Import necessary libraries
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Lasso
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# Load the data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv("Ames.csv")
features = ["GrLivArea", "1stFlrSF", "2ndFlrSF", "LowQualFinSF"]
X = Ames[features]
y = Ames["SalePrice"]

# Initialize a k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Prepare to collect results
results = {}

for alpha in [1, 2]:  # Loop through both alpha values
    coefficients = []
    cv_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale features
        # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
        scaler = StandardScaler()
        # 拟合并转换数据（一步完成） / Fit and transform data (one step)
        X_train_scaled = scaler.fit_transform(X_train)
        # 用已拟合的模型转换数据 / Transform data with fitted model
        X_test_scaled = scaler.transform(X_test)

        # Initialize and fit the Lasso regression model
        lasso_model = Lasso(alpha=alpha, max_iter=20000)
        # 训练模型 / Train the model
        lasso_model.fit(X_train_scaled, y_train)
        # 添加元素到列表末尾 / Append element to list end
        coefficients.append(lasso_model.coef_)

        # Calculate R^2 score using the model's score method
        score = lasso_model.score(X_test_scaled, y_test)
        # 添加元素到列表末尾 / Append element to list end
        cv_scores.append(score)

    results[alpha] = (coefficients, cv_scores)

# Plotting the results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
alphas = [1, 2]

# 同时获取索引和值 / Get both index and value
for i, alpha in enumerate(alphas):
    coefficients, cv_scores = results[alpha]

    # Plotting the coefficients
    # 创建NumPy数组 / Create NumPy array
    axes[i, 0].boxplot(np.array(coefficients), labels=features)
    axes[i, 0].set_title(f"Box Plot of Coefficients (Lasso with alpha={alpha})")
    axes[i, 0].set_xlabel("Features")
    axes[i, 0].set_ylabel("Coefficient Value")
    axes[i, 0].grid(True)

    # Plotting the CV scores
    # 生成整数序列 / Generate integer sequence
    axes[i, 1].plot(range(1, 6), cv_scores, marker="o", linestyle="-")
    axes[i, 1].set_title(f"Cross-Validation R^2 Scores (Lasso with alpha={alpha})")
    axes[i, 1].set_xlabel("Fold")
    # 生成整数序列 / Generate integer sequence
    axes[i, 1].set_xticks(range(1, 6))
    axes[i, 1].set_ylabel("R^2 Score")
    axes[i, 1].set_ylim(min(cv_scores) - 0.05, max(cv_scores) + 0.05)
    axes[i, 1].grid(True)
    # 计算均值 / Calculate mean
    mean_r2 = np.mean(cv_scores)
    axes[i, 1].annotate(f"Mean CV R^2: {mean_r2:.3f}", xy=(1.25, 0.65),
                        color="red", fontsize=12)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Refined

# 06 — Refined / 06 Refined

**Chapter 09 — File 6 of 6 / 第09章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Load the data**.

本脚本演示 **Load the data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
```

---
## Step 2 — Load the data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
features = ['GrLivArea', '1stFlrSF', 'LowQualFinSF']  # Remove '2ndFlrSF' after Lasso
X = Ames[features]
y = Ames['SalePrice']
```

---
## Step 3 — Initialize a K-Fold cross-validation

```python
kf = KFold(n_splits=5, shuffle=True, random_state=1)
```

---
## Step 4 — Collect coefficients and CV scores

```python
coefficients = []
cv_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
```

---
## Step 5 — Initialize and fit the linear regression model

```python
model = LinearRegression()
    # 训练模型 / Train the model
    model.fit(X_train, y_train)
    # 添加元素到列表末尾 / Append element to list end
    coefficients.append(model.coef_)
```

---
## Step 6 — Calculate R^2 score using the model's score method

```python
score = model.score(X_test, y_test)
```

---
## Step 7 — print(score)

```python
# 添加元素到列表末尾 / Append element to list end
cv_scores.append(score)
```

---
## Step 8 — Plotting the coefficients

```python
# 创建画布 / Create figure canvas
plt.figure(figsize=(12, 6))
# 创建子图 / Create subplot
plt.subplot(1, 2, 1)
# 创建NumPy数组 / Create NumPy array
plt.boxplot(np.array(coefficients), labels=features)
# 设置图表标题 / Set chart title
plt.title('Box Plot of Coefficients Across Folds (MLR)')
# 设置X轴标签 / Set X-axis label
plt.xlabel('Features')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Coefficient Value')
plt.grid(True)
```

---
## Step 9 — Plotting the CV scores

```python
# 创建子图 / Create subplot
plt.subplot(1, 2, 2)
# 绘制折线图 / Draw line plot
plt.plot(range(1, 6), cv_scores, marker='o', linestyle='-')  # make x-axis to start from 1
# 设置图表标题 / Set chart title
plt.title('Cross-Validation R^2 Scores (MLR)')
# 设置X轴标签 / Set X-axis label
plt.xlabel('Fold')
# 生成整数序列 / Generate integer sequence
plt.xticks(range(1, 6))  # Set x-ticks to match fold numbers
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('R^2 Score')
plt.ylim(min(cv_scores) - 0.05, max(cv_scores) + 0.05)  # Dynamically adjust y-axis limits
plt.grid(True)
```

---
## Step 10 — Annotate mean R^2 score

```python
# 计算均值 / Calculate mean
mean_r2 = np.mean(cv_scores)
plt.annotate(f'Mean CV R^2: {mean_r2:.3f}', xy=(1.25, 0.65), color='red', fontsize=14)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load the data 是机器学习中的常用技术。  
  *Load the data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.mean` | 计算均值 | Calculate mean |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Refined / 06 Refined
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold

# Load the data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
features = ['GrLivArea', '1stFlrSF', 'LowQualFinSF']  # Remove '2ndFlrSF' after Lasso
X = Ames[features]
y = Ames['SalePrice']

# Initialize a K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Collect coefficients and CV scores
coefficients = []
cv_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Initialize and fit the linear regression model
    model = LinearRegression()
    # 训练模型 / Train the model
    model.fit(X_train, y_train)
    # 添加元素到列表末尾 / Append element to list end
    coefficients.append(model.coef_)

    # Calculate R^2 score using the model's score method
    score = model.score(X_test, y_test)
    # print(score)
    # 添加元素到列表末尾 / Append element to list end
    cv_scores.append(score)

# Plotting the coefficients
# 创建画布 / Create figure canvas
plt.figure(figsize=(12, 6))
# 创建子图 / Create subplot
plt.subplot(1, 2, 1)
# 创建NumPy数组 / Create NumPy array
plt.boxplot(np.array(coefficients), labels=features)
# 设置图表标题 / Set chart title
plt.title('Box Plot of Coefficients Across Folds (MLR)')
# 设置X轴标签 / Set X-axis label
plt.xlabel('Features')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Coefficient Value')
plt.grid(True)

# Plotting the CV scores
# 创建子图 / Create subplot
plt.subplot(1, 2, 2)
# 绘制折线图 / Draw line plot
plt.plot(range(1, 6), cv_scores, marker='o', linestyle='-')  # make x-axis to start from 1
# 设置图表标题 / Set chart title
plt.title('Cross-Validation R^2 Scores (MLR)')
# 设置X轴标签 / Set X-axis label
plt.xlabel('Fold')
# 生成整数序列 / Generate integer sequence
plt.xticks(range(1, 6))  # Set x-ticks to match fold numbers
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('R^2 Score')
plt.ylim(min(cv_scores) - 0.05, max(cv_scores) + 0.05)  # Dynamically adjust y-axis limits
plt.grid(True)

# Annotate mean R^2 score
# 计算均值 / Calculate mean
mean_r2 = np.mean(cv_scores)
plt.annotate(f'Mean CV R^2: {mean_r2:.3f}', xy=(1.25, 0.65), color='red', fontsize=14)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 09 Summary / 第09章总结

## Theme / 主题: Chapter 09 / Chapter 09

This chapter contains **6 code files** demonstrating chapter 09.

本章包含 **6 个代码文件**，演示Chapter 09。

---
## Evolution / 演化路线

  1. `01_matrix.ipynb` — Matrix
  2. `02_redundant.ipynb` — Redundant
  3. `03_verify.ipynb` — Verify
  4. `04_plot.ipynb` — Plot
  5. `05_lasso.ipynb` — Lasso
  6. `06_refined.ipynb` — Refined

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 09) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 09）是机器学习流水线中的基础构建块。

---
