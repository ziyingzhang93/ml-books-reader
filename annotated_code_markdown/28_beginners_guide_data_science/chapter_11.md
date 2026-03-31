# 数据科学入门 / Beginner's Guide to Data Science
## Chapter 11

---

### Chisq

# 01 — Chisq / 01 Chisq

**Chapter 11 — File 1 of 1 / 第11章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Load the dataset**.

本脚本演示 **Load the dataset**。

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
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
from scipy.stats import chi2_contingency
```

---
## Step 2 — Load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 3 — Extracting the relevant columns

```python
data = Ames[['ExterQual', 'GarageFinish']].copy()
```

---
## Step 4 — Filling missing values in the 'GarageFinish' column with 'No Garage'

```python
# 填充缺失值 / Fill missing values
data['GarageFinish'] = data['GarageFinish'].fillna('No Garage')
```

---
## Step 5 — Grouping 'GarageFinish' into 'With Garage' and 'No Garage'

```python
data['Garage Group'] \
    = data['GarageFinish'] \
      .apply(lambda x: 'With Garage' if x != 'No Garage' else 'No Garage')
```

---
## Step 6 — Grouping 'ExterQual' into 'Great' and 'Average'

```python
data['Quality Group'] \
    = data['ExterQual'].apply(lambda x: 'Great' if x in ['Ex', 'Gd'] else 'Average')
```

---
## Step 7 — Constructing the simplified contingency table

```python
simplified_contingency_table = pd.crosstab(data['Quality Group'], data['Garage Group'])
```

---
## Step 8 — Printing the Observed Frequency

```python
# 打印输出 / Print output
print("Observed Frequencies:")
observed_df = pd.DataFrame(simplified_contingency_table,
                           index=["Average", "Great"],
                           columns=["No Garage", "With Garage"])
# 打印输出 / Print output
print(observed_df)
# 打印输出 / Print output
print()
```

---
## Step 9 — Performing the chi-squared test

```python
chi2_stat, p_value, _, expected_freq = chi2_contingency(simplified_contingency_table)
```

---
## Step 10 — Printing the Expected Frequencies

```python
# 打印输出 / Print output
print("Expected Frequencies:")
# 打印输出 / Print output
print(pd.DataFrame(expected_freq,
                   index=["Average", "Great"],
                   columns=["No Garage", "With Garage"]).round(1))
# 打印输出 / Print output
print()
```

---
## Step 11 — Printing the results of the test

```python
# 打印输出 / Print output
print(f"Chi-squared Statistic: {chi2_stat:.4f}")
# 打印输出 / Print output
print(f"p-value: {p_value:.4e}")
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
# Chisq / 01 Chisq
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Extracting the relevant columns
data = Ames[['ExterQual', 'GarageFinish']].copy()

# Filling missing values in the 'GarageFinish' column with 'No Garage'
# 填充缺失值 / Fill missing values
data['GarageFinish'] = data['GarageFinish'].fillna('No Garage')

# Grouping 'GarageFinish' into 'With Garage' and 'No Garage'
data['Garage Group'] \
    = data['GarageFinish'] \
      .apply(lambda x: 'With Garage' if x != 'No Garage' else 'No Garage')

# Grouping 'ExterQual' into 'Great' and 'Average'
data['Quality Group'] \
    = data['ExterQual'].apply(lambda x: 'Great' if x in ['Ex', 'Gd'] else 'Average')

# Constructing the simplified contingency table
simplified_contingency_table = pd.crosstab(data['Quality Group'], data['Garage Group'])

#Printing the Observed Frequency
# 打印输出 / Print output
print("Observed Frequencies:")
observed_df = pd.DataFrame(simplified_contingency_table,
                           index=["Average", "Great"],
                           columns=["No Garage", "With Garage"])
# 打印输出 / Print output
print(observed_df)
# 打印输出 / Print output
print()

# Performing the chi-squared test
chi2_stat, p_value, _, expected_freq = chi2_contingency(simplified_contingency_table)

# Printing the Expected Frequencies
# 打印输出 / Print output
print("Expected Frequencies:")
# 打印输出 / Print output
print(pd.DataFrame(expected_freq,
                   index=["Average", "Great"],
                   columns=["No Garage", "With Garage"]).round(1))
# 打印输出 / Print output
print()

# Printing the results of the test
# 打印输出 / Print output
print(f"Chi-squared Statistic: {chi2_stat:.4f}")
# 打印输出 / Print output
print(f"p-value: {p_value:.4e}")
```

---

### Chapter Summary / 章节总结

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **1 code files** demonstrating chapter 11.

本章包含 **1 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `01_chisq.ipynb` — Chisq

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---
