# 数据科学入门 / Beginner's Guide to Data Science
## Chapter 01

---

### Load



---

### Datatype

# 02 — Datatype / 02 Datatype

**Chapter 01 — File 2 of 8 / 第01章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Determine the data type for each feature**.

本脚本演示 **Determine the data type for each feature**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
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
# 打印输出 / Print output
print(type_counts)
```

---
## Learning Notes / 学习笔记

- **概念**: Determine the data type for each feature 是机器学习中的常用技术。  
  *Determine the data type for each feature is a common technique in machine learning.*

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
# Datatype / 02 Datatype
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Determine the data type for each feature
data_types = Ames.dtypes

# Tally the total by data type
type_counts = data_types.value_counts()
# 打印输出 / Print output
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
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
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
# 打印输出 / Print output
print(data_types)
```

---
## Learning Notes / 学习笔记

- **概念**: Determine the data type for each feature 是机器学习中的常用技术。  
  *Determine the data type for each feature is a common technique in machine learning.*

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
# Datatype / 03 Datatype
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Determine the data type for each feature
data_types = Ames.dtypes

# View a few datatypes from the dataset (first and last 5 features)
# 打印输出 / Print output
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
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
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

# 打印输出 / Print output
print(df)
# 打印输出 / Print output
print()
# 打印输出 / Print output
print(missing_data)
```

---
## Learning Notes / 学习笔记

- **概念**: Create a DataFrame with various types of missing values 是机器学习中的常用技术。  
  *Create a DataFrame with various types of missing values is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nan / 04 Nan
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
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

# 打印输出 / Print output
print(df)
# 打印输出 / Print output
print()
# 打印输出 / Print output
print(missing_data)
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Missingvalue



---

### Missingno

# 06 — Missingno / 缺失值处理

**Chapter 01 — File 6 of 8 / 第01章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **Missingno**.

本脚本演示 **缺失值处理**。

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
import missingno as msno
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
msno.matrix(Ames, sparkline=False, fontsize=20)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Missingno 是机器学习中的常用技术。  
  *Missingno is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.show` | 显示图表 | Display plot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Missingno / 缺失值处理
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import missingno as msno
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
msno.matrix(Ames, sparkline=False, fontsize=20)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Barchart



---

### Topmissing



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
