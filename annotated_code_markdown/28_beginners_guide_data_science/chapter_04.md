# 数据科学入门
## Chapter 04

---

### Pivot

# 05 — Pivot / 05 Pivot

**Chapter 04 — File 5 of 8 / 第04章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Pivot**.

本脚本演示 **05 Pivot**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
import pandas as pd

def categorize_by_price(row):
    if row['SalePrice'] <= quantiles.iloc[0]:
        return 'Low'
    elif row['SalePrice'] <= quantiles.iloc[1]:
        return 'Medium'
    elif row['SalePrice'] <= quantiles.iloc[2]:
        return 'High'
    else:
        return 'Premium'

Ames = pd.read_csv('Ames.csv')
quantiles = Ames['SalePrice'].quantile([0.25, 0.5, 0.75])
Ames['Price_Category'] = Ames.apply(categorize_by_price, axis=1)

pivot = Ames.pivot_table(index="Fireplaces",
                         columns="Price_Category",
                         aggfunc={'GrLivArea':'mean', 'Fireplaces':'count'})
print(pivot)
```

---
## Learning Notes / 学习笔记

- **概念**: Pivot 是机器学习中的常用技术。  
  *Pivot is a common technique in machine learning.*

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
# Pivot / 05 Pivot
# Complete Code / 完整代码
# ===============================

import pandas as pd

def categorize_by_price(row):
    if row['SalePrice'] <= quantiles.iloc[0]:
        return 'Low'
    elif row['SalePrice'] <= quantiles.iloc[1]:
        return 'Medium'
    elif row['SalePrice'] <= quantiles.iloc[2]:
        return 'High'
    else:
        return 'Premium'

Ames = pd.read_csv('Ames.csv')
quantiles = Ames['SalePrice'].quantile([0.25, 0.5, 0.75])
Ames['Price_Category'] = Ames.apply(categorize_by_price, axis=1)

pivot = Ames.pivot_table(index="Fireplaces",
                         columns="Price_Category",
                         aggfunc={'GrLivArea':'mean', 'Fireplaces':'count'})
print(pivot)
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Outerjoin

# 06 — Outerjoin / 06 Outerjoin

**Chapter 04 — File 6 of 8 / 第04章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Split original dataset into 4 DataFrames by Price Category**.

本脚本演示 **Split original dataset into 4 DataFrames by Price Category**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
import pandas as pd

def categorize_by_price(row):
    if row['SalePrice'] <= quantiles.iloc[0]:
        return 'Low'
    elif row['SalePrice'] <= quantiles.iloc[1]:
        return 'Medium'
    elif row['SalePrice'] <= quantiles.iloc[2]:
        return 'High'
    else:
        return 'Premium'

Ames = pd.read_csv('Ames.csv')
quantiles = Ames['SalePrice'].quantile([0.25, 0.5, 0.75])
Ames['Price_Category'] = Ames.apply(categorize_by_price, axis=1)
```

---
## Step 2 — Split original dataset into 4 DataFrames by Price Category

```python
low_priced_homes = Ames.query('Price_Category == "Low"')
medium_priced_homes = Ames.query('Price_Category == "Medium"')
high_priced_homes = Ames.query('Price_Category == "High"')
premium_priced_homes = Ames.query('Price_Category == "Premium"')
```

---
## Step 3 — Stacking Low and Medium categories into an "affordable_homes" DataFrame

```python
affordable_homes = pd.concat([low_priced_homes, medium_priced_homes])
```

---
## Step 4 — Stacking High and Premium categories into a "luxury_homes" DataFrame

```python
luxury_homes = pd.concat([high_priced_homes, premium_priced_homes])
```

---
## Step 5 — Creating pivot tables with both mean living area and home count

```python
aggfunc = {'GrLivArea': 'mean', 'Fireplaces': 'count'}
pivot_affordable = affordable_homes.pivot_table(index='Fireplaces', aggfunc=aggfunc)
pivot_luxury = luxury_homes.pivot_table(index='Fireplaces', aggfunc=aggfunc)
```

---
## Step 6 — Renaming columns and index labels separately

```python
rename_rules = {'GrLivArea': 'AvLivArea', 'Fireplaces': 'HmCount'}

pivot_affordable.rename(columns=rename_rules, inplace=True)
pivot_affordable.index.name = 'Fire'

pivot_luxury.rename(columns=rename_rules, inplace=True)
pivot_luxury.index.name = 'Fire'

pivot_outer_join = pd.merge(pivot_affordable, pivot_luxury, on='Fire', how='outer',
                            suffixes=('_aff', '_lux')).fillna(0)
print(pivot_outer_join)
```

---
## Learning Notes / 学习笔记

- **概念**: Split original dataset into 4 DataFrames by Price Category 是机器学习中的常用技术。  
  *Split original dataset into 4 DataFrames by Price Category is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `fillna` | 填充缺失值 | Fill missing values |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Outerjoin / 06 Outerjoin
# Complete Code / 完整代码
# ===============================

import pandas as pd

def categorize_by_price(row):
    if row['SalePrice'] <= quantiles.iloc[0]:
        return 'Low'
    elif row['SalePrice'] <= quantiles.iloc[1]:
        return 'Medium'
    elif row['SalePrice'] <= quantiles.iloc[2]:
        return 'High'
    else:
        return 'Premium'

Ames = pd.read_csv('Ames.csv')
quantiles = Ames['SalePrice'].quantile([0.25, 0.5, 0.75])
Ames['Price_Category'] = Ames.apply(categorize_by_price, axis=1)

# Split original dataset into 4 DataFrames by Price Category
low_priced_homes = Ames.query('Price_Category == "Low"')
medium_priced_homes = Ames.query('Price_Category == "Medium"')
high_priced_homes = Ames.query('Price_Category == "High"')
premium_priced_homes = Ames.query('Price_Category == "Premium"')

# Stacking Low and Medium categories into an "affordable_homes" DataFrame
affordable_homes = pd.concat([low_priced_homes, medium_priced_homes])

# Stacking High and Premium categories into a "luxury_homes" DataFrame
luxury_homes = pd.concat([high_priced_homes, premium_priced_homes])

# Creating pivot tables with both mean living area and home count
aggfunc = {'GrLivArea': 'mean', 'Fireplaces': 'count'}
pivot_affordable = affordable_homes.pivot_table(index='Fireplaces', aggfunc=aggfunc)
pivot_luxury = luxury_homes.pivot_table(index='Fireplaces', aggfunc=aggfunc)

# Renaming columns and index labels separately
rename_rules = {'GrLivArea': 'AvLivArea', 'Fireplaces': 'HmCount'}

pivot_affordable.rename(columns=rename_rules, inplace=True)
pivot_affordable.index.name = 'Fire'

pivot_luxury.rename(columns=rename_rules, inplace=True)
pivot_luxury.index.name = 'Fire'

pivot_outer_join = pd.merge(pivot_affordable, pivot_luxury, on='Fire', how='outer',
                            suffixes=('_aff', '_lux')).fillna(0)
print(pivot_outer_join)
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Innerjoin

# 07 — Innerjoin / 07 Innerjoin

**Chapter 04 — File 7 of 8 / 第04章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Split original dataset into 4 DataFrames by Price Category**.

本脚本演示 **Split original dataset into 4 DataFrames by Price Category**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
import pandas as pd

def categorize_by_price(row):
    if row['SalePrice'] <= quantiles.iloc[0]:
        return 'Low'
    elif row['SalePrice'] <= quantiles.iloc[1]:
        return 'Medium'
    elif row['SalePrice'] <= quantiles.iloc[2]:
        return 'High'
    else:
        return 'Premium'

Ames = pd.read_csv('Ames.csv')
quantiles = Ames['SalePrice'].quantile([0.25, 0.5, 0.75])
Ames['Price_Category'] = Ames.apply(categorize_by_price, axis=1)
```

---
## Step 2 — Split original dataset into 4 DataFrames by Price Category

```python
low_priced_homes = Ames.query('Price_Category == "Low"')
medium_priced_homes = Ames.query('Price_Category == "Medium"')
high_priced_homes = Ames.query('Price_Category == "High"')
premium_priced_homes = Ames.query('Price_Category == "Premium"')
```

---
## Step 3 — Stacking Low and Medium categories into an "affordable_homes" DataFrame

```python
affordable_homes = pd.concat([low_priced_homes, medium_priced_homes])
```

---
## Step 4 — Stacking High and Premium categories into a "luxury_homes" DataFrame

```python
luxury_homes = pd.concat([high_priced_homes, premium_priced_homes])
```

---
## Step 5 — Creating pivot tables with both mean living area and home count

```python
aggfunc = {'GrLivArea': 'mean', 'Fireplaces': 'count'}
pivot_affordable = affordable_homes.pivot_table(index='Fireplaces', aggfunc=aggfunc)
pivot_luxury = luxury_homes.pivot_table(index='Fireplaces', aggfunc=aggfunc)
```

---
## Step 6 — Renaming columns and index labels separately

```python
rename_rules = {'GrLivArea': 'AvLivArea', 'Fireplaces': 'HmCount'}

pivot_affordable.rename(columns=rename_rules, inplace=True)
pivot_affordable.index.name = 'Fire'

pivot_luxury.rename(columns=rename_rules, inplace=True)
pivot_luxury.index.name = 'Fire'

pivot_inner_join = pd.merge(pivot_affordable, pivot_luxury, on='Fire', how='inner',
                            suffixes=('_aff', '_lux'))
print(pivot_inner_join)
```

---
## Learning Notes / 学习笔记

- **概念**: Split original dataset into 4 DataFrames by Price Category 是机器学习中的常用技术。  
  *Split original dataset into 4 DataFrames by Price Category is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Innerjoin / 07 Innerjoin
# Complete Code / 完整代码
# ===============================

import pandas as pd

def categorize_by_price(row):
    if row['SalePrice'] <= quantiles.iloc[0]:
        return 'Low'
    elif row['SalePrice'] <= quantiles.iloc[1]:
        return 'Medium'
    elif row['SalePrice'] <= quantiles.iloc[2]:
        return 'High'
    else:
        return 'Premium'

Ames = pd.read_csv('Ames.csv')
quantiles = Ames['SalePrice'].quantile([0.25, 0.5, 0.75])
Ames['Price_Category'] = Ames.apply(categorize_by_price, axis=1)

# Split original dataset into 4 DataFrames by Price Category
low_priced_homes = Ames.query('Price_Category == "Low"')
medium_priced_homes = Ames.query('Price_Category == "Medium"')
high_priced_homes = Ames.query('Price_Category == "High"')
premium_priced_homes = Ames.query('Price_Category == "Premium"')

# Stacking Low and Medium categories into an "affordable_homes" DataFrame
affordable_homes = pd.concat([low_priced_homes, medium_priced_homes])

# Stacking High and Premium categories into a "luxury_homes" DataFrame
luxury_homes = pd.concat([high_priced_homes, premium_priced_homes])

# Creating pivot tables with both mean living area and home count
aggfunc = {'GrLivArea': 'mean', 'Fireplaces': 'count'}
pivot_affordable = affordable_homes.pivot_table(index='Fireplaces', aggfunc=aggfunc)
pivot_luxury = luxury_homes.pivot_table(index='Fireplaces', aggfunc=aggfunc)

# Renaming columns and index labels separately
rename_rules = {'GrLivArea': 'AvLivArea', 'Fireplaces': 'HmCount'}

pivot_affordable.rename(columns=rename_rules, inplace=True)
pivot_affordable.index.name = 'Fire'

pivot_luxury.rename(columns=rename_rules, inplace=True)
pivot_luxury.index.name = 'Fire'

pivot_inner_join = pd.merge(pivot_affordable, pivot_luxury, on='Fire', how='inner',
                            suffixes=('_aff', '_lux'))
print(pivot_inner_join)
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Chapter Summary

# Chapter 04 Summary / 第04章总结

## Theme / 主题: Chapter 04 / Chapter 04

This chapter contains **8 code files** demonstrating chapter 04.

本章包含 **8 个代码文件**，演示Chapter 04。

---
## Evolution / 演化路线

  1. `01_category.ipynb` — Category
  2. `02_ecdf.ipynb` — Ecdf
  3. `03_stacking.ipynb` — Stacking
  4. `04_pivot.ipynb` — Pivot
  5. `05_pivot.ipynb` — Pivot
  6. `06_outerjoin.ipynb` — Outerjoin
  7. `07_innerjoin.ipynb` — Innerjoin
  8. `08_crossjoin.ipynb` — Crossjoin

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 04) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 04）是机器学习流水线中的基础构建块。

---
