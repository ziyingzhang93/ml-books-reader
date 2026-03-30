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
## Step 1 — Step 1

```python
import pandas as pd
from scipy.stats import chi2_contingency
```

---
## Step 2 — Load the dataset

```python
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
print("Observed Frequencies:")
observed_df = pd.DataFrame(simplified_contingency_table,
                           index=["Average", "Great"],
                           columns=["No Garage", "With Garage"])
print(observed_df)
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
print("Expected Frequencies:")
print(pd.DataFrame(expected_freq,
                   index=["Average", "Great"],
                   columns=["No Garage", "With Garage"]).round(1))
print()
```

---
## Step 11 — Printing the results of the test

```python
print(f"Chi-squared Statistic: {chi2_stat:.4f}")
print(f"p-value: {p_value:.4e}")
```

---
## Learning Notes / 学习笔记

- **概念**: Load the dataset 是机器学习中的常用技术。  
  *Load the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Chisq / 01 Chisq
# Complete Code / 完整代码
# ===============================

import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Extracting the relevant columns
data = Ames[['ExterQual', 'GarageFinish']].copy()

# Filling missing values in the 'GarageFinish' column with 'No Garage'
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
print("Observed Frequencies:")
observed_df = pd.DataFrame(simplified_contingency_table,
                           index=["Average", "Great"],
                           columns=["No Garage", "With Garage"])
print(observed_df)
print()

# Performing the chi-squared test
chi2_stat, p_value, _, expected_freq = chi2_contingency(simplified_contingency_table)

# Printing the Expected Frequencies
print("Expected Frequencies:")
print(pd.DataFrame(expected_freq,
                   index=["Average", "Great"],
                   columns=["No Garage", "With Garage"]).round(1))
print()

# Printing the results of the test
print(f"Chi-squared Statistic: {chi2_stat:.4f}")
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
