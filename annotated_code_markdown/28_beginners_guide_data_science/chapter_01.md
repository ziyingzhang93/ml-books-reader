# 数据科学入门 / Beginner's Guide to Data Science
## Chapter 01

---

### Load

# 01 — Load / 01 Load

**Chapter 01 — File 1 of 8 / 第01章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Load the Ames dataset**.

本脚本演示 **Load the Ames dataset**。

---
## Step 1 — Load the Ames dataset

```python
import pandas as pd
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Dataset shape

```python
print(Ames.shape)

rows, columns = Ames.shape
print(f"The dataset comprises {rows} properties described across {columns} attributes.")
```

---
## Learning Notes / 学习笔记

- **概念**: Load the Ames dataset 是机器学习中的常用技术。  
  *Load the Ames dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load / 01 Load
# Complete Code / 完整代码
# ===============================

# Load the Ames dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Dataset shape
print(Ames.shape)

rows, columns = Ames.shape
print(f"The dataset comprises {rows} properties described across {columns} attributes.")
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Datatype

# 02 — Datatype / 02 Datatype

**Chapter 01 — File 2 of 8 / 第01章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Determine the data type for each feature**.

本脚本演示 **Determine the data type for each feature**。

---
## Step 1 — Step 1

```python
import pandas as pd
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Determine the data type for each feature

```python
data_types = Ames.dtypes
```

---
## Step 3 — Tally the total by data type

```python
type_counts = data_types.value_counts()
print(type_counts)
```

---
## Learning Notes / 学习笔记

- **概念**: Determine the data type for each feature 是机器学习中的常用技术。  
  *Determine the data type for each feature is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Datatype / 02 Datatype
# Complete Code / 完整代码
# ===============================

import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Determine the data type for each feature
data_types = Ames.dtypes

# Tally the total by data type
type_counts = data_types.value_counts()
print(type_counts)
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Datatype

# 03 — Datatype / 03 Datatype

**Chapter 01 — File 3 of 8 / 第01章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Determine the data type for each feature**.

本脚本演示 **Determine the data type for each feature**。

---
## Step 1 — Step 1

```python
import pandas as pd
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Determine the data type for each feature

```python
data_types = Ames.dtypes
```

---
## Step 3 — View a few datatypes from the dataset (first and last 5 features)

```python
print(data_types)
```

---
## Learning Notes / 学习笔记

- **概念**: Determine the data type for each feature 是机器学习中的常用技术。  
  *Determine the data type for each feature is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Datatype / 03 Datatype
# Complete Code / 完整代码
# ===============================

import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Determine the data type for each feature
data_types = Ames.dtypes

# View a few datatypes from the dataset (first and last 5 features)
print(data_types)
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Nan

# 04 — Nan / 04 Nan

**Chapter 01 — File 4 of 8 / 第01章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Create a DataFrame with various types of missing values**.

本脚本演示 **Create a DataFrame with various types of missing values**。

---
## Step 1 — Step 1

```python
import pandas as pd
import numpy as np
```

---
## Step 2 — Create a DataFrame with various types of missing values

```python
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': ['a', 'b', None, 'd', 'e'],
    'C': [np.nan, np.nan, np.nan, np.nan, np.nan],
    'D': [1, 2, 3, 4, 5]
})
```

---
## Step 3 — Use isnull() to identify missing values

```python
missing_data = df.isnull().sum()

print(df)
print()
print(missing_data)
```

---
## Learning Notes / 学习笔记

- **概念**: Create a DataFrame with various types of missing values 是机器学习中的常用技术。  
  *Create a DataFrame with various types of missing values is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nan / 04 Nan
# Complete Code / 完整代码
# ===============================

import pandas as pd
import numpy as np

# Create a DataFrame with various types of missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': ['a', 'b', None, 'd', 'e'],
    'C': [np.nan, np.nan, np.nan, np.nan, np.nan],
    'D': [1, 2, 3, 4, 5]
})

# Use isnull() to identify missing values
missing_data = df.isnull().sum()

print(df)
print()
print(missing_data)
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Missingvalue

# 05 — Missingvalue / 缺失值处理

**Chapter 01 — File 5 of 8 / 第01章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Calculating the percentage of missing values for each column**.

本脚本演示 **Calculating the percentage of missing values for each column**。

---
## Step 1 — Step 1

```python
import pandas as pd
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Calculating the percentage of missing values for each column

```python
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100
```

---
## Step 3 — Combining the counts and percentages into a DataFrame for better visualization

```python
missing_info = pd.DataFrame({'Missing Values': missing_data,
                             'Percentage': missing_percentage})
```

---
## Step 4 — Sorting the DataFrame by the percentage of missing values in descending order

```python
missing_info = missing_info.sort_values(by='Percentage', ascending=False)
```

---
## Step 5 — Display columns with missing values

```python
print(missing_info[missing_info['Missing Values'] > 0])
```

---
## Learning Notes / 学习笔记

- **概念**: Calculating the percentage of missing values for each column 是机器学习中的常用技术。  
  *Calculating the percentage of missing values for each column is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Missingvalue / 缺失值处理
# Complete Code / 完整代码
# ===============================

import pandas as pd
Ames = pd.read_csv('Ames.csv')

# Calculating the percentage of missing values for each column
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100

# Combining the counts and percentages into a DataFrame for better visualization
missing_info = pd.DataFrame({'Missing Values': missing_data,
                             'Percentage': missing_percentage})

# Sorting the DataFrame by the percentage of missing values in descending order
missing_info = missing_info.sort_values(by='Percentage', ascending=False)

# Display columns with missing values
print(missing_info[missing_info['Missing Values'] > 0])
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Missingno

# 06 — Missingno / 缺失值处理

**Chapter 01 — File 6 of 8 / 第01章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Missingno**.

本脚本演示 **缺失值处理**。

---
## Step 1 — Step 1

```python
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

Ames = pd.read_csv('Ames.csv')
msno.matrix(Ames, sparkline=False, fontsize=20)
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Missingno 是机器学习中的常用技术。  
  *Missingno is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Missingno / 缺失值处理
# Complete Code / 完整代码
# ===============================

import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

Ames = pd.read_csv('Ames.csv')
msno.matrix(Ames, sparkline=False, fontsize=20)
plt.show()
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Barchart

# 07 — Barchart / 07 Barchart

**Chapter 01 — File 7 of 8 / 第01章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Calculating the percentage of missing values for each column**.

本脚本演示 **Calculating the percentage of missing values for each column**。

---
## Step 1 — Step 1

```python
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Calculating the percentage of missing values for each column

```python
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100
```

---
## Step 3 — Combining the counts and percentages into a DataFrame for better visualization

```python
missing_info = pd.DataFrame({'Missing Values': missing_data,
                             'Percentage': missing_percentage})
```

---
## Step 4 — Sort the DataFrame columns by the percentage of missing values

```python
sorted_df = Ames[missing_info.sort_values(by='Percentage', ascending=False).index]
```

---
## Step 5 — Select the top 15 columns with the most missing values

```python
top_15_missing = sorted_df.iloc[:, :15]
```

---
## Step 6 — Visual with missingno

```python
msno.bar(top_15_missing)
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Calculating the percentage of missing values for each column 是机器学习中的常用技术。  
  *Calculating the percentage of missing values for each column is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Barchart / 07 Barchart
# Complete Code / 完整代码
# ===============================

import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
Ames = pd.read_csv('Ames.csv')

# Calculating the percentage of missing values for each column
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100

# Combining the counts and percentages into a DataFrame for better visualization
missing_info = pd.DataFrame({'Missing Values': missing_data,
                             'Percentage': missing_percentage})

# Sort the DataFrame columns by the percentage of missing values
sorted_df = Ames[missing_info.sort_values(by='Percentage', ascending=False).index]

# Select the top 15 columns with the most missing values
top_15_missing = sorted_df.iloc[:, :15]

#Visual with missingno
msno.bar(top_15_missing)
plt.show()
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Topmissing

# 08 — Topmissing / 缺失值处理

**Chapter 01 — File 8 of 8 / 第01章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Filter to show only the top 15 columns with the most missing values**.

本脚本演示 **Filter to show only the top 15 columns with the most missing values**。

---
## Step 1 — Step 1

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Ames = pd.read_csv('Ames.csv')
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100
missing_info = pd.DataFrame({'Missing Values': missing_data,
                             'Percentage': missing_percentage})
```

---
## Step 2 — Filter to show only the top 15 columns with the most missing values

```python
top_15_missing_info = missing_info.nlargest(15, 'Percentage').reset_index()
print(top_15_missing_info)
```

---
## Step 3 — Create the horizontal bar plot using seaborn

```python
plt.figure(figsize=(12, 8))
sns.barplot(x='Percentage', y="index", hue="index", data=top_15_missing_info, orient='h')
plt.title('Top 15 Features with Missing Percentages', fontsize=20)
plt.xlabel('Percentage of Missing Values', fontsize=16)
plt.ylabel('Features', fontsize=16)
```

---
## Step 4 — plt.yticks(fontsize=11)

```python
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Filter to show only the top 15 columns with the most missing values 是机器学习中的常用技术。  
  *Filter to show only the top 15 columns with the most missing values is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Topmissing / 缺失值处理
# Complete Code / 完整代码
# ===============================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Ames = pd.read_csv('Ames.csv')
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100
missing_info = pd.DataFrame({'Missing Values': missing_data,
                             'Percentage': missing_percentage})

# Filter to show only the top 15 columns with the most missing values
top_15_missing_info = missing_info.nlargest(15, 'Percentage').reset_index()
print(top_15_missing_info)

# Create the horizontal bar plot using seaborn
plt.figure(figsize=(12, 8))
sns.barplot(x='Percentage', y="index", hue="index", data=top_15_missing_info, orient='h')
plt.title('Top 15 Features with Missing Percentages', fontsize=20)
plt.xlabel('Percentage of Missing Values', fontsize=16)
plt.ylabel('Features', fontsize=16)
#plt.yticks(fontsize=11)
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 01 Summary / 第01章总结

## Theme / 主题: Chapter 01 / Chapter 01

This chapter contains **8 code files** demonstrating chapter 01.

本章包含 **8 个代码文件**，演示Chapter 01。

---
## Evolution / 演化路线

  1. `01_load.ipynb` — Load
  2. `02_datatype.ipynb` — Datatype
  3. `03_datatype.ipynb` — Datatype
  4. `04_nan.ipynb` — Nan
  5. `05_missingvalue.ipynb` — Missingvalue
  6. `06_missingno.ipynb` — Missingno
  7. `07_barchart.ipynb` — Barchart
  8. `08_topmissing.ipynb` — Topmissing

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 01) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 01）是机器学习流水线中的基础构建块。

---
