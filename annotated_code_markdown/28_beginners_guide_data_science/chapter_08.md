# 数据科学入门 / Beginner's Guide to Data Science
## Chapter 08

---

### Pairplot

# 01 — Pairplot / 01 Pairplot

**Chapter 08 — File 1 of 2 / 第08章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Load the dataset**.

本脚本演示 **Load the dataset**。

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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import seaborn as sns
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
```

---
## Step 2 — Load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 3 — Calculate the correlation of all features with 'SalePrice'

```python
correlations = Ames.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)
```

---
## Step 4 — Top 5 features most correlated with 'SalePrice' (excluding 'SalePrice' itself)

```python
top_5_features = correlations.index[1:6]
```

---
## Step 5 — Creating the pair plot for these features and 'SalePrice'
Adjust the size by setting height and aspect

```python
sns.pairplot(Ames, vars=['SalePrice'] + list(top_5_features), height=1.35, aspect=1.85)
```

---
## Step 6 — Displaying the plot

```python
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load the dataset 是机器学习中的常用技术。  
  *Load the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.show` | 显示图表 | Display plot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pairplot / 01 Pairplot
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import seaborn as sns
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# Load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Calculate the correlation of all features with 'SalePrice'
correlations = Ames.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)

# Top 5 features most correlated with 'SalePrice' (excluding 'SalePrice' itself)
top_5_features = correlations.index[1:6]

# Creating the pair plot for these features and 'SalePrice'
# Adjust the size by setting height and aspect
sns.pairplot(Ames, vars=['SalePrice'] + list(top_5_features), height=1.35, aspect=1.85)

# Displaying the plot
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Colorplot

# 02 — Colorplot / 02 Colorplot

**Chapter 08 — File 2 of 2 / 第08章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Load the dataset**.

本脚本演示 **Load the dataset**。

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
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import seaborn as sns
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
```

---
## Step 2 — Load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 3 — Convert 'LotShape' to a binary feature: 'Regular' and 'Irregular'

```python
Ames['LotShape_Binary'] = \
    Ames['LotShape'].apply(lambda x: 'Regular' if x == 'Reg' else 'Irregular')
```

---
## Step 4 — Creating the pair plot, color-coded by 'LotShape_Binary'

```python
sns.pairplot(Ames, vars=['SalePrice', 'OverallQual', 'GrLivArea'], hue='LotShape_Binary',
             palette='Set1', height=2.5, aspect=1.75)
```

---
## Step 5 — Display the plot

```python
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Load the dataset 是机器学习中的常用技术。  
  *Load the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.show` | 显示图表 | Display plot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Colorplot / 02 Colorplot
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import seaborn as sns
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# Load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Convert 'LotShape' to a binary feature: 'Regular' and 'Irregular'
Ames['LotShape_Binary'] = \
    Ames['LotShape'].apply(lambda x: 'Regular' if x == 'Reg' else 'Irregular')

# Creating the pair plot, color-coded by 'LotShape_Binary'
sns.pairplot(Ames, vars=['SalePrice', 'OverallQual', 'GrLivArea'], hue='LotShape_Binary',
             palette='Set1', height=2.5, aspect=1.75)

# Display the plot
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 08 Summary / 第08章总结

## Theme / 主题: Chapter 08 / Chapter 08

This chapter contains **2 code files** demonstrating chapter 08.

本章包含 **2 个代码文件**，演示Chapter 08。

---
## Evolution / 演化路线

  1. `01_pairplot.ipynb` — Pairplot
  2. `02_colorplot.ipynb` — Colorplot

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 08) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 08）是机器学习流水线中的基础构建块。

---
