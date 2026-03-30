# 数据科学入门
## Chapter 03

---

### Filter

# 01 — Filter / 01 Filter

**Chapter 03 — File 1 of 9 / 第03章 — 第1个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Load the dataset**.

本脚本演示 **Load the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
import pandas as pd
```

---
## Step 2 — Load the dataset

```python
Ames = pd.read_csv('Ames.csv')
```

---
## Step 3 — Simple querying: Select houses priced above $600,000

```python
high_value_houses = Ames.query('SalePrice > 600000')
print(high_value_houses)
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
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Filter / 01 Filter
# Complete Code / 完整代码
# ===============================

import pandas as pd

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Simple querying: Select houses priced above $600,000
high_value_houses = Ames.query('SalePrice > 600000')
print(high_value_houses)
```

---

➡️ **Next / 下一步**: File 2 of 9

---

### Compound

# 02 — Compound / 02 Compound

**Chapter 03 — File 2 of 9 / 第03章 — 第2个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Advanced querying: Select houses with more than 3 bedrooms and priced below $300,000**.

本脚本演示 **Advanced querying: Select houses with more than 3 bedrooms and priced below $300,000**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
import pandas as pd

Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Advanced querying: Select houses with more than 3 bedrooms and priced below $300,000

```python
specific_houses = Ames.query('BedroomAbvGr > 3 & SalePrice < 300000')
print(specific_houses)
```

---
## Learning Notes / 学习笔记

- **概念**: Advanced querying: Select houses with more than 3 bedrooms and priced below $300,000 是机器学习中的常用技术。  
  *Advanced querying: Select houses with more than 3 bedrooms and priced below $300,000 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Compound / 02 Compound
# Complete Code / 完整代码
# ===============================

import pandas as pd

Ames = pd.read_csv('Ames.csv')

# Advanced querying: Select houses with more than 3 bedrooms and priced below $300,000
specific_houses = Ames.query('BedroomAbvGr > 3 & SalePrice < 300000')
print(specific_houses)
```

---

➡️ **Next / 下一步**: File 3 of 9

---

### Scatterplot

# 03 — Scatterplot / 03 Scatterplot

**Chapter 03 — File 3 of 9 / 第03章 — 第3个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Visualizing the advanced query results**.

本脚本演示 **Visualizing the advanced query results**。

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

Ames = pd.read_csv('Ames.csv')
specific_houses = Ames.query('BedroomAbvGr > 3 & SalePrice < 300000')
```

---
## Step 2 — Visualizing the advanced query results

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='BedroomAbvGr',
                data=specific_houses, palette='viridis')
plt.title('Sales Price vs. Ground Living Area')
plt.xlabel('Ground Living Area (sqft)')
plt.ylabel('Sales Price ($)')
plt.legend(title='Bedrooms Above Ground')
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Visualizing the advanced query results 是机器学习中的常用技术。  
  *Visualizing the advanced query results is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
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
# Scatterplot / 03 Scatterplot
# Complete Code / 完整代码
# ===============================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Ames = pd.read_csv('Ames.csv')
specific_houses = Ames.query('BedroomAbvGr > 3 & SalePrice < 300000')

# Visualizing the advanced query results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='BedroomAbvGr',
                data=specific_houses, palette='viridis')
plt.title('Sales Price vs. Ground Living Area')
plt.xlabel('Ground Living Area (sqft)')
plt.ylabel('Sales Price ($)')
plt.legend(title='Bedrooms Above Ground')
plt.show()
```

---

➡️ **Next / 下一步**: File 4 of 9

---

### Pivot

# 08 — Pivot / 08 Pivot

**Chapter 03 — File 8 of 9 / 第03章 — 第8个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Filter for houses priced below $300,000 and with at least 1 bedroom above grade**.

本脚本演示 **Filter for houses priced below $300,000 and with at least 1 bedroom above grade**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
import pandas as pd
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Filter for houses priced below $300,000 and with at least 1 bedroom above grade

```python
affordable_houses = Ames.query('SalePrice < 300000 & BedroomAbvGr > 0')
```

---
## Step 3 — Create pivot table to analyze average sale price by neighborhood and number of bedrooms

```python
pivot_table = affordable_houses.pivot_table(values='SalePrice',
                                            index='Neighborhood',
                                            columns='BedroomAbvGr',
                                            aggfunc='mean').round(2)
```

---
## Step 4 — Fill missing values (combination not exist) with 0 to avoid seeing NaN

```python
pivot_table = pivot_table.fillna(0)
```

---
## Step 5 — Adjust pandas display options to ensure all columns are shown

```python
pd.set_option('display.max_columns', None)
print(pivot_table)
```

---
## Learning Notes / 学习笔记

- **概念**: Filter for houses priced below $300,000 and with at least 1 bedroom above grade 是机器学习中的常用技术。  
  *Filter for houses priced below $300,000 and with at least 1 bedroom above grade is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `fillna` | 填充缺失值 | Fill missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pivot / 08 Pivot
# Complete Code / 完整代码
# ===============================

import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Filter for houses priced below $300,000 and with at least 1 bedroom above grade
affordable_houses = Ames.query('SalePrice < 300000 & BedroomAbvGr > 0')

# Create pivot table to analyze average sale price by neighborhood and number of bedrooms
pivot_table = affordable_houses.pivot_table(values='SalePrice',
                                            index='Neighborhood',
                                            columns='BedroomAbvGr',
                                            aggfunc='mean').round(2)

# Fill missing values (combination not exist) with 0 to avoid seeing NaN
pivot_table = pivot_table.fillna(0)

# Adjust pandas display options to ensure all columns are shown
pd.set_option('display.max_columns', None)
print(pivot_table)
```

---

➡️ **Next / 下一步**: File 9 of 9

---

### Heatmap

# 09 — Heatmap / 09 Heatmap

**Chapter 03 — File 9 of 9 / 第03章 — 第9个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Create a custom color map**.

本脚本演示 **Create a custom color map**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns

Ames = pd.read_csv('Ames.csv')
affordable_houses = Ames.query('SalePrice < 300000 & BedroomAbvGr > 0')
pivot_table = affordable_houses \
              .pivot_table(values='SalePrice', index='Neighborhood',
                           columns='BedroomAbvGr', aggfunc='mean') \
              .round(2) \
              .fillna(0)
```

---
## Step 2 — Create a custom color map

```python
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
```

---
## Step 3 — Mask for "zero" values to be colored with a different shade

```python
mask = pivot_table == 0
```

---
## Step 4 — Set the size of the plot

```python
plt.figure(figsize=(14, 10))
```

---
## Step 5 — Create a heatmap with the mask

```python
sns.heatmap(pivot_table,
            cmap=cmap,
            annot=True,
            fmt=".0f",
            linewidths=.5,
            mask=mask,
            cbar_kws={'label': 'Average Sales Price ($)'})
```

---
## Step 6 — Adding title and labels for clarity

```python
plt.title('Average Sales Price by Neighborhood and Number of Bedrooms', fontsize=16)
plt.xlabel('Number of Bedrooms Above Grade', fontsize=12)
plt.ylabel('Neighborhood', fontsize=12)
```

---
## Step 7 — Display the heatmap

```python
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Create a custom color map 是机器学习中的常用技术。  
  *Create a custom color map is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `fillna` | 填充缺失值 | Fill missing values |
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
# Heatmap / 09 Heatmap
# Complete Code / 完整代码
# ===============================

import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns

Ames = pd.read_csv('Ames.csv')
affordable_houses = Ames.query('SalePrice < 300000 & BedroomAbvGr > 0')
pivot_table = affordable_houses \
              .pivot_table(values='SalePrice', index='Neighborhood',
                           columns='BedroomAbvGr', aggfunc='mean') \
              .round(2) \
              .fillna(0)

# Create a custom color map
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

# Mask for "zero" values to be colored with a different shade
mask = pivot_table == 0

# Set the size of the plot
plt.figure(figsize=(14, 10))

# Create a heatmap with the mask
sns.heatmap(pivot_table,
            cmap=cmap,
            annot=True,
            fmt=".0f",
            linewidths=.5,
            mask=mask,
            cbar_kws={'label': 'Average Sales Price ($)'})

# Adding title and labels for clarity
plt.title('Average Sales Price by Neighborhood and Number of Bedrooms', fontsize=16)
plt.xlabel('Number of Bedrooms Above Grade', fontsize=12)
plt.ylabel('Neighborhood', fontsize=12)

# Display the heatmap
plt.show()
```

---

### Chapter Summary

# Chapter 03 Summary / 第03章总结

## Theme / 主题: Chapter 03 / Chapter 03

This chapter contains **9 code files** demonstrating chapter 03.

本章包含 **9 个代码文件**，演示Chapter 03。

---
## Evolution / 演化路线

  1. `01_filter.ipynb` — Filter
  2. `02_compound.ipynb` — Compound
  3. `03_scatterplot.ipynb` — Scatterplot
  4. `04_grouped.ipynb` — Grouped
  5. `05_barplot.ipynb` — Barplot
  6. `06_loc.ipynb` — Loc
  7. `07_iloc.ipynb` — Iloc
  8. `08_pivot.ipynb` — Pivot
  9. `09_heatmap.ipynb` — Heatmap

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 03) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 03）是机器学习流水线中的基础构建块。

---
