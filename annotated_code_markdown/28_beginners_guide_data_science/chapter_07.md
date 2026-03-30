# 数据科学入门
## Chapter 07

---

### Correl

# 01 — Correl / 01 Correl

**Chapter 07 — File 1 of 3 / 第07章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Load the Dataset**.

本脚本演示 **Load the Dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Load the Dataset

```python
import pandas as pd
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Calculate the correlation of all features with 'SalePrice'
Set numeric_only=True to limit the output to numeric columns

```python
correlations = Ames.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)
```

---
## Step 3 — Display the top 10 features most correlated with 'SalePrice'

```python
top_correlations = correlations[1:11]
print(top_correlations)
```

---
## Learning Notes / 学习笔记

- **概念**: Load the Dataset 是机器学习中的常用技术。  
  *Load the Dataset is a common technique in machine learning.*

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
# Correl / 01 Correl
# Complete Code / 完整代码
# ===============================

# Load the Dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Calculate the correlation of all features with 'SalePrice'
# Set numeric_only=True to limit the output to numeric columns
correlations = Ames.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)

# Display the top 10 features most correlated with 'SalePrice'
top_correlations = correlations[1:11]
print(top_correlations)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Heatmap

# 03 — Heatmap / 03 Heatmap

**Chapter 07 — File 2 of 3 / 第07章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Load the Dataset**.

本脚本演示 **Load the Dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

---
## Step 2 — Load the Dataset

```python
Ames = pd.read_csv('Ames.csv')
```

---
## Step 3 — Calculate the top 10 features most correlated with 'SalePrice'

```python
correlations = Ames.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)
top_correlations = correlations[1:11]
```

---
## Step 4 — Select the top correlated features including SalePrice

```python
selected_features = list(top_correlations.index) + ['SalePrice']
```

---
## Step 5 — Compute the correlations for the selected features

```python
correlation_matrix = Ames[selected_features].corr()
```

---
## Step 6 — Set up the matplotlib figure

```python
plt.figure(figsize=(12, 8))
```

---
## Step 7 — Generate a heatmap

```python
sns.heatmap(correlation_matrix, annot=True,
            cmap="coolwarm", linewidths=.5, fmt=".2f", vmin=-1, vmax=1)
```

---
## Step 8 — Title

```python
plt.title("Heatmap of Correlations among Top Features with SalePrice", fontsize=16)
```

---
## Step 9 — Show the heatmap

```python
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load the Dataset 是机器学习中的常用技术。  
  *Load the Dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.show` | 显示图表 | Display plot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Heatmap / 03 Heatmap
# Complete Code / 完整代码
# ===============================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Dataset
Ames = pd.read_csv('Ames.csv')

# Calculate the top 10 features most correlated with 'SalePrice'
correlations = Ames.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)
top_correlations = correlations[1:11]

# Select the top correlated features including SalePrice
selected_features = list(top_correlations.index) + ['SalePrice']

# Compute the correlations for the selected features
correlation_matrix = Ames[selected_features].corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Generate a heatmap
sns.heatmap(correlation_matrix, annot=True,
            cmap="coolwarm", linewidths=.5, fmt=".2f", vmin=-1, vmax=1)

# Title
plt.title("Heatmap of Correlations among Top Features with SalePrice", fontsize=16)

# Show the heatmap
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Scatter

# 04 — Scatter / 04 Scatter

**Chapter 07 — File 3 of 3 / 第07章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Setting up the figure and axes**.

本脚本演示 **Setting up the figure and axes**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Setting up the figure and axes

```python
fig, ax = plt.subplots(2, 2, figsize=(15, 12))
```

---
## Step 3 — Scatter plot for SalePrice vs. OverallQual

```python
sns.scatterplot(x=Ames['OverallQual'], y=Ames['SalePrice'], ax=ax[0, 0],
                color='blue', alpha=0.6)
ax[0, 0].set_title('House Prices vs. Overall Quality')
ax[0, 0].set_ylabel('House Prices')
ax[0, 0].set_xlabel('Overall Quality')
```

---
## Step 4 — Scatter plot for SalePrice vs. GrLivArea

```python
sns.scatterplot(x=Ames['GrLivArea'], y=Ames['SalePrice'], ax=ax[0, 1],
                color='red', alpha=0.6)
ax[0, 1].set_title('House Prices vs. Ground Living Area')
ax[0, 1].set_ylabel('House Prices')
ax[0, 1].set_xlabel('Above Ground Living Area (sq. ft.)')
```

---
## Step 5 — Scatter plot for SalePrice vs. TotalBsmtSF

```python
sns.scatterplot(x=Ames['TotalBsmtSF'], y=Ames['SalePrice'], ax=ax[1, 0],
                color='green', alpha=0.6)
ax[1, 0].set_title('House Prices vs. Total Basement Area')
ax[1, 0].set_ylabel('House Prices')
ax[1, 0].set_xlabel('Total Basement Area (sq. ft.)')
```

---
## Step 6 — Scatter plot for SalePrice vs. 1stFlrSF

```python
sns.scatterplot(x=Ames['1stFlrSF'], y=Ames['SalePrice'], ax=ax[1, 1],
                color='purple', alpha=0.6)
ax[1, 1].set_title('House Prices vs. First Floor Area')
ax[1, 1].set_ylabel('House Prices')
ax[1, 1].set_xlabel('First Floor Area (sq. ft.)')
```

---
## Step 7 — Adjust layout

```python
plt.tight_layout(pad=3.0)
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Setting up the figure and axes 是机器学习中的常用技术。  
  *Setting up the figure and axes is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Scatter / 04 Scatter
# Complete Code / 完整代码
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Ames = pd.read_csv('Ames.csv')

# Setting up the figure and axes
fig, ax = plt.subplots(2, 2, figsize=(15, 12))

# Scatter plot for SalePrice vs. OverallQual
sns.scatterplot(x=Ames['OverallQual'], y=Ames['SalePrice'], ax=ax[0, 0],
                color='blue', alpha=0.6)
ax[0, 0].set_title('House Prices vs. Overall Quality')
ax[0, 0].set_ylabel('House Prices')
ax[0, 0].set_xlabel('Overall Quality')

# Scatter plot for SalePrice vs. GrLivArea
sns.scatterplot(x=Ames['GrLivArea'], y=Ames['SalePrice'], ax=ax[0, 1],
                color='red', alpha=0.6)
ax[0, 1].set_title('House Prices vs. Ground Living Area')
ax[0, 1].set_ylabel('House Prices')
ax[0, 1].set_xlabel('Above Ground Living Area (sq. ft.)')

# Scatter plot for SalePrice vs. TotalBsmtSF
sns.scatterplot(x=Ames['TotalBsmtSF'], y=Ames['SalePrice'], ax=ax[1, 0],
                color='green', alpha=0.6)
ax[1, 0].set_title('House Prices vs. Total Basement Area')
ax[1, 0].set_ylabel('House Prices')
ax[1, 0].set_xlabel('Total Basement Area (sq. ft.)')

# Scatter plot for SalePrice vs. 1stFlrSF
sns.scatterplot(x=Ames['1stFlrSF'], y=Ames['SalePrice'], ax=ax[1, 1],
                color='purple', alpha=0.6)
ax[1, 1].set_title('House Prices vs. First Floor Area')
ax[1, 1].set_ylabel('House Prices')
ax[1, 1].set_xlabel('First Floor Area (sq. ft.)')

# Adjust layout
plt.tight_layout(pad=3.0)
plt.show()
```

---

### Chapter Summary

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **3 code files** demonstrating chapter 07.

本章包含 **3 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `01_correl.ipynb` — Correl
  2. `03_heatmap.ipynb` — Heatmap
  3. `04_scatter.ipynb` — Scatter

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---
