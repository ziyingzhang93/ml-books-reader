# 数据科学入门
## Chapter 02

---

### Investigate

# 06 — Investigate / 06 Investigate

**Chapter 02 — File 1 of 6 / 第02章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Load the Ames dataset**.

本脚本演示 **Load the Ames dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Load the Ames dataset

```python
import pandas as pd
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Using select_dtypes()

```python
numerical_features = Ames.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = Ames.select_dtypes(include=['object', 'category']).columns.tolist()
print("Numerical features (int64 and float64):", numerical_features)
print("Categorical features (object and category):", categorical_features)
```

---
## Step 3 — Using describe() to automatically extract numerical features

```python
numerical_features = Ames.describe().columns.tolist()
print("Numerical features from describe():", numerical_features)
```

---
## Step 4 — Data dictionary and domain knowledge could be useful in setting the threshold

```python
threshold = 10
categorical_features = Ames.columns[Ames.nunique() <= threshold].tolist()
print("Categorical features based on unique values:", categorical_features)
```

---
## Step 5 — Using value_counts() on each column or feature

```python
print("Value counts:")
for column in Ames.columns:
    print(Ames[column].value_counts())
```

---
## Step 6 — Using info() on the Ames Dataset

```python
print("info():")
Ames.info()
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
| `describe()` | 统计摘要信息 | Statistical summary |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Investigate / 06 Investigate
# Complete Code / 完整代码
# ===============================

# Load the Ames dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Using select_dtypes()
numerical_features = Ames.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = Ames.select_dtypes(include=['object', 'category']).columns.tolist()
print("Numerical features (int64 and float64):", numerical_features)
print("Categorical features (object and category):", categorical_features)

# Using describe() to automatically extract numerical features
numerical_features = Ames.describe().columns.tolist()
print("Numerical features from describe():", numerical_features)

# Data dictionary and domain knowledge could be useful in setting the threshold
threshold = 10
categorical_features = Ames.columns[Ames.nunique() <= threshold].tolist()
print("Categorical features based on unique values:", categorical_features)

# Using value_counts() on each column or feature
print("Value counts:")
for column in Ames.columns:
    print(Ames[column].value_counts())

# Using info() on the Ames Dataset
print("info():")
Ames.info()
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Count

# 09 — Count / 09 Count

**Chapter 02 — File 2 of 6 / 第02章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Load the Ames dataset**.

本脚本演示 **Load the Ames dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Load the Ames dataset

```python
import pandas as pd
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Reassign data type

```python
Ames['MSSubClass'] = Ames['MSSubClass'].astype('object')
Ames['YrSold'] = Ames['YrSold'].astype('object')
Ames['MoSold'] = Ames['MoSold'].astype('object')
```

---
## Step 3 — Determine the data type for each feature after conversion

```python
data_types = Ames.dtypes
```

---
## Step 4 — Tally the total by data type

```python
type_counts = data_types.value_counts()

print(type_counts)
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
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Count / 09 Count
# Complete Code / 完整代码
# ===============================

# Load the Ames dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Reassign data type
Ames['MSSubClass'] = Ames['MSSubClass'].astype('object')
Ames['YrSold'] = Ames['YrSold'].astype('object')
Ames['MoSold'] = Ames['MoSold'].astype('object')

# Determine the data type for each feature after conversion
data_types = Ames.dtypes

# Tally the total by data type
type_counts = data_types.value_counts()

print(type_counts)
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Imputation

# 17 — Imputation / 17 Imputation

**Chapter 02 — File 6 of 6 / 第02章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Load the Ames dataset**.

本脚本演示 **Load the Ames dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Step 1 — Load the Ames dataset

```python
import pandas as pd
import numpy as np
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Calculating the percentage of missing values for each column

```python
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100
data_type = Ames.dtypes
```

---
## Step 3 — Combining the counts and percentages into a DataFrame for better visualization

```python
missing_info = pd.DataFrame({'Missing Values': missing_data,
                             'Percentage': missing_percentage,
                             'Data Type': data_type})
```

---
## Step 4 — Sorting the DataFrame by the percentage of missing values in descending order

```python
missing_info = missing_info.sort_values(by='Percentage', ascending=False)
```

---
## Step 5 — Display columns with missing values of numeric data type

```python
print(missing_info[(missing_info['Missing Values'] > 0)
                   & (missing_info['Data Type'] == np.number)])
```

---
## Step 6 — Initialize a DataFrame to store the concise information

```python
concise_info = pd.DataFrame(columns=['Feature',
                                     'Missing Values After Imputation',
                                     'Mean Value Used to Impute'])
```

---
## Step 7 — Identify and impute missing numerical values, and store the related concise information

```python
missing_numeric_df = missing_info[(missing_info['Missing Values'] > 0) &
                                  (missing_info['Data Type'] == np.number)]

for item in missing_numeric_df.index.tolist():
    mean_value = Ames[item].mean(skipna=True)
    Ames[item].fillna(mean_value, inplace=True)
```

---
## Step 8 — Append the concise information to the concise_info DataFrame

```python
concise_info.loc[len(concise_info)] = pd.Series({
        'Feature': item,
        'Missing Values After Imputation': Ames[item].isnull().sum(),
```

---
## Step 9 — This should be 0 as you are imputing all missing values

```python
'Mean Value Used to Impute': mean_value
    })
```

---
## Step 10 — Display the concise_info DataFrame

```python
print(concise_info)

missing_values_count = Ames.isnull().sum().sum()
print(f'The DataFrame has a total of {missing_values_count} missing values.')
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
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `fillna` | 填充缺失值 | Fill missing values |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Imputation / 17 Imputation
# Complete Code / 完整代码
# ===============================

# Load the Ames dataset
import pandas as pd
import numpy as np
Ames = pd.read_csv('Ames.csv')

# Calculating the percentage of missing values for each column
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100
data_type = Ames.dtypes

# Combining the counts and percentages into a DataFrame for better visualization
missing_info = pd.DataFrame({'Missing Values': missing_data,
                             'Percentage': missing_percentage,
                             'Data Type': data_type})

# Sorting the DataFrame by the percentage of missing values in descending order
missing_info = missing_info.sort_values(by='Percentage', ascending=False)

# Display columns with missing values of numeric data type
print(missing_info[(missing_info['Missing Values'] > 0)
                   & (missing_info['Data Type'] == np.number)])

# Initialize a DataFrame to store the concise information
concise_info = pd.DataFrame(columns=['Feature',
                                     'Missing Values After Imputation',
                                     'Mean Value Used to Impute'])

# Identify and impute missing numerical values, and store the related concise information
missing_numeric_df = missing_info[(missing_info['Missing Values'] > 0) &
                                  (missing_info['Data Type'] == np.number)]

for item in missing_numeric_df.index.tolist():
    mean_value = Ames[item].mean(skipna=True)
    Ames[item].fillna(mean_value, inplace=True)

    # Append the concise information to the concise_info DataFrame
    concise_info.loc[len(concise_info)] = pd.Series({
        'Feature': item,
        'Missing Values After Imputation': Ames[item].isnull().sum(),
        # This should be 0 as you are imputing all missing values
        'Mean Value Used to Impute': mean_value
    })

# Display the concise_info DataFrame
print(concise_info)

missing_values_count = Ames.isnull().sum().sum()
print(f'The DataFrame has a total of {missing_values_count} missing values.')
```

---

### Chapter Summary

# Chapter 02 Summary / 第02章总结

## Theme / 主题: Chapter 02 / Chapter 02

This chapter contains **6 code files** demonstrating chapter 02.

本章包含 **6 个代码文件**，演示Chapter 02。

---
## Evolution / 演化路线

  1. `06_investigate.ipynb` — Investigate
  2. `09_count.ipynb` — Count
  3. `10_findmissing.ipynb` — Findmissing
  4. `13_imputation.ipynb` — Imputation
  5. `14_numerical.ipynb` — Numerical
  6. `17_imputation.ipynb` — Imputation

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 02) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 02）是机器学习流水线中的基础构建块。

---
