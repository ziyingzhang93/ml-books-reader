# 数据科学入门 / Beginner's Guide to Data Science
## Chapter 13

---

### Boxplot

# 01 — Boxplot / 01 Boxplot

**Chapter 13 — File 1 of 4 / 第13章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Define feature names in full form for titles and axis**.

本脚本演示 **Define feature names in full form for titles and axis**。

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

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Define feature names in full form for titles and axis

```python
feature_names_full = {
    'LotArea': 'Lot Area (sq ft)',
    'SalePrice': 'Sales Price (US$)',
    'TotRmsAbvGrd': 'Total Rooms Above Ground'
}

# 创建画布 / Create figure canvas
plt.figure(figsize=(18, 6))
features = ['LotArea', 'SalePrice', 'TotRmsAbvGrd']

# 同时获取索引和值 / Get both index and value
for i, feature in enumerate(features, 1):
    # 创建子图 / Create subplot
    plt.subplot(1, 3, i)
    sns.boxplot(y=Ames[feature], color="lightblue")
    # 设置图表标题 / Set chart title
    plt.title(feature_names_full[feature], fontsize=16)
    # 设置Y轴标签 / Set Y-axis label
    plt.ylabel(feature_names_full[feature], fontsize=14)
    # 设置X轴标签 / Set X-axis label
    plt.xlabel('')  # Removing the x-axis label as it's not needed

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Define feature names in full form for titles and axis 是机器学习中的常用技术。  
  *Define feature names in full form for titles and axis is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Boxplot / 01 Boxplot
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import seaborn as sns
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Define feature names in full form for titles and axis
feature_names_full = {
    'LotArea': 'Lot Area (sq ft)',
    'SalePrice': 'Sales Price (US$)',
    'TotRmsAbvGrd': 'Total Rooms Above Ground'
}

# 创建画布 / Create figure canvas
plt.figure(figsize=(18, 6))
features = ['LotArea', 'SalePrice', 'TotRmsAbvGrd']

# 同时获取索引和值 / Get both index and value
for i, feature in enumerate(features, 1):
    # 创建子图 / Create subplot
    plt.subplot(1, 3, i)
    sns.boxplot(y=Ames[feature], color="lightblue")
    # 设置图表标题 / Set chart title
    plt.title(feature_names_full[feature], fontsize=16)
    # 设置Y轴标签 / Set Y-axis label
    plt.ylabel(feature_names_full[feature], fontsize=14)
    # 设置X轴标签 / Set X-axis label
    plt.xlabel('')  # Removing the x-axis label as it's not needed

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Iqr

# 02 — Iqr / 02 Iqr

**Chapter 13 — File 2 of 4 / 第13章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Iqr**.

本脚本演示 **02 Iqr**。

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
features = ['LotArea', 'SalePrice', 'TotRmsAbvGrd']

def detect_outliers_iqr_summary(dataframe, features):
    outliers_summary = {}

    for feature in features:
        data = dataframe[feature]
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        # 获取长度 / Get length
        outliers_summary[feature] = len(outliers)

    return outliers_summary

outliers_summary = detect_outliers_iqr_summary(Ames, features)
# 打印输出 / Print output
print(outliers_summary)
```

---
## Learning Notes / 学习笔记

- **概念**: Iqr 是机器学习中的常用技术。  
  *Iqr is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Iqr / 02 Iqr
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
features = ['LotArea', 'SalePrice', 'TotRmsAbvGrd']

def detect_outliers_iqr_summary(dataframe, features):
    outliers_summary = {}

    for feature in features:
        data = dataframe[feature]
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        # 获取长度 / Get length
        outliers_summary[feature] = len(outliers)

    return outliers_summary

outliers_summary = detect_outliers_iqr_summary(Ames, features)
# 打印输出 / Print output
print(outliers_summary)
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Outlier

# 03 — Outlier / 异常值检测

**Chapter 13 — File 3 of 4 / 第13章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Define a function to detect outliers using the Gaussian model**.

本脚本演示 **Define a function to detect outliers using the Gaussian model**。

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

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
feature_names_full = {
    'LotArea': 'Lot Area (sq ft)',
    'SalePrice': 'Sales Price (US$)',
    'TotRmsAbvGrd': 'Total Rooms Above Ground'
}
features = ['LotArea', 'SalePrice', 'TotRmsAbvGrd']
```

---
## Step 2 — Define a function to detect outliers using the Gaussian model

```python
def detect_outliers_gaussian(dataframe, features, threshold=3):
    outliers_summary = {}

    for feature in features:
        data = dataframe[feature]
        mean = data.mean()
        std_dev = data.std()
        outliers = data[(data < mean - threshold * std_dev) |
                        (data > mean + threshold * std_dev)]
        # 获取长度 / Get length
        outliers_summary[feature] = len(outliers)
```

---
## Step 3 — Visualization

```python
# 创建画布 / Create figure canvas
plt.figure(figsize=(12, 6))
        sns.histplot(data, color="lightblue")
        plt.axvline(mean, color='r', linestyle='-', label=f'Mean: {mean:.2f}')
        plt.axvline(mean - threshold * std_dev, color='y', linestyle='--',
                    label=f'—{threshold} std devs')
        plt.axvline(mean + threshold * std_dev, color='g', linestyle='--',
                    label=f'+{threshold} std devs')
```

---
## Step 4 — Annotate upper 3rd std dev value

```python
annotate_text = f'{mean + threshold * std_dev:.2f}'
        plt.annotate(annotate_text, xy=(mean + threshold * std_dev, 0),
                     xytext=(mean + (threshold + 1.45) * std_dev, 50),
                     arrowprops={'facecolor': 'black',
                                 'arrowstyle': 'wedge,tail_width=0.7'},
                     fontsize=12, ha='center')

        # 设置图表标题 / Set chart title
        plt.title(f'Distribution of {feature_names_full[feature]} with Outliers',
                  fontsize=16)
        # 设置X轴标签 / Set X-axis label
        plt.xlabel(feature_names_full[feature], fontsize=14)
        # 设置Y轴标签 / Set Y-axis label
        plt.ylabel('Frequency', fontsize=14)
        # 显示图例 / Show legend
        plt.legend()
        # 显示图表 / Display the plot
        plt.show()

    return outliers_summary

outliers_gaussian_summary = detect_outliers_gaussian(Ames, features)
# 打印输出 / Print output
print(outliers_gaussian_summary)
```

---
## Learning Notes / 学习笔记

- **概念**: Define a function to detect outliers using the Gaussian model 是机器学习中的常用技术。  
  *Define a function to detect outliers using the Gaussian model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
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
# Outlier / 异常值检测
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import seaborn as sns
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
feature_names_full = {
    'LotArea': 'Lot Area (sq ft)',
    'SalePrice': 'Sales Price (US$)',
    'TotRmsAbvGrd': 'Total Rooms Above Ground'
}
features = ['LotArea', 'SalePrice', 'TotRmsAbvGrd']

# Define a function to detect outliers using the Gaussian model
def detect_outliers_gaussian(dataframe, features, threshold=3):
    outliers_summary = {}

    for feature in features:
        data = dataframe[feature]
        mean = data.mean()
        std_dev = data.std()
        outliers = data[(data < mean - threshold * std_dev) |
                        (data > mean + threshold * std_dev)]
        # 获取长度 / Get length
        outliers_summary[feature] = len(outliers)

        # Visualization
        # 创建画布 / Create figure canvas
        plt.figure(figsize=(12, 6))
        sns.histplot(data, color="lightblue")
        plt.axvline(mean, color='r', linestyle='-', label=f'Mean: {mean:.2f}')
        plt.axvline(mean - threshold * std_dev, color='y', linestyle='--',
                    label=f'—{threshold} std devs')
        plt.axvline(mean + threshold * std_dev, color='g', linestyle='--',
                    label=f'+{threshold} std devs')

        # Annotate upper 3rd std dev value
        annotate_text = f'{mean + threshold * std_dev:.2f}'
        plt.annotate(annotate_text, xy=(mean + threshold * std_dev, 0),
                     xytext=(mean + (threshold + 1.45) * std_dev, 50),
                     arrowprops={'facecolor': 'black',
                                 'arrowstyle': 'wedge,tail_width=0.7'},
                     fontsize=12, ha='center')

        # 设置图表标题 / Set chart title
        plt.title(f'Distribution of {feature_names_full[feature]} with Outliers',
                  fontsize=16)
        # 设置X轴标签 / Set X-axis label
        plt.xlabel(feature_names_full[feature], fontsize=14)
        # 设置Y轴标签 / Set Y-axis label
        plt.ylabel('Frequency', fontsize=14)
        # 显示图例 / Show legend
        plt.legend()
        # 显示图表 / Display the plot
        plt.show()

    return outliers_summary

outliers_gaussian_summary = detect_outliers_gaussian(Ames, features)
# 打印输出 / Print output
print(outliers_gaussian_summary)
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Gaussian

# 04 — Gaussian / 04 Gaussian

**Chapter 13 — File 4 of 4 / 第13章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Define a function to tabulate outliers into a DataFrame**.

本脚本演示 **Define a function to tabulate outliers into a DataFrame**。

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
features = ['LotArea', 'SalePrice', 'TotRmsAbvGrd']
```

---
## Step 2 — Define a function to tabulate outliers into a DataFrame

```python
def outliers_dataframes_gaussian(dataframe, features, threshold=3, num_rows=None):
    outliers_dataframes = {}

    for feature in features:
        data = dataframe[feature]
        mean = data.mean()
        std_dev = data.std()
        outliers = data[(data < mean - threshold * std_dev) |
                        (data > mean + threshold * std_dev)]
```

---
## Step 3 — Create a new DataFrame for outliers of the current feature

```python
outliers_df = dataframe.loc[outliers.index, [feature]].copy()
        outliers_df.rename(columns={feature: 'Outlier Value'}, inplace=True)
        outliers_df['Feature'] = feature
        outliers_df.reset_index(inplace=True)
```

---
## Step 4 — Display specified number of rows (default: full dataframe)

```python
if num_rows:
            # 查看前几行数据（快速预览） / View first rows (quick preview)
            outliers_df = outliers_df.head(num_rows)

        outliers_dataframes[feature] = outliers_df

    return outliers_dataframes
```

---
## Step 5 — Example usage with user-defined number of rows = 7

```python
outliers_gaussian_dataframes = outliers_dataframes_gaussian(Ames, features, num_rows=7)
```

---
## Step 6 — Print each DataFrame with the original format and capitalized 'index'

```python
# 获取字典的键值对 / Get dict key-value pairs
for feature, df in outliers_gaussian_dataframes.items():
    df_reset = df.reset_index().rename(columns={'index': 'Index'})
    # 打印输出 / Print output
    print(f"Outliers for {feature}:\n", df_reset[['Index', 'Feature', 'Outlier Value']])
    # 打印输出 / Print output
    print()
```

---
## Learning Notes / 学习笔记

- **概念**: Define a function to tabulate outliers into a DataFrame 是机器学习中的常用技术。  
  *Define a function to tabulate outliers into a DataFrame is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Gaussian / 04 Gaussian
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
features = ['LotArea', 'SalePrice', 'TotRmsAbvGrd']

# Define a function to tabulate outliers into a DataFrame
def outliers_dataframes_gaussian(dataframe, features, threshold=3, num_rows=None):
    outliers_dataframes = {}

    for feature in features:
        data = dataframe[feature]
        mean = data.mean()
        std_dev = data.std()
        outliers = data[(data < mean - threshold * std_dev) |
                        (data > mean + threshold * std_dev)]

        # Create a new DataFrame for outliers of the current feature
        outliers_df = dataframe.loc[outliers.index, [feature]].copy()
        outliers_df.rename(columns={feature: 'Outlier Value'}, inplace=True)
        outliers_df['Feature'] = feature
        outliers_df.reset_index(inplace=True)

        # Display specified number of rows (default: full dataframe)
        if num_rows:
            # 查看前几行数据（快速预览） / View first rows (quick preview)
            outliers_df = outliers_df.head(num_rows)

        outliers_dataframes[feature] = outliers_df

    return outliers_dataframes

# Example usage with user-defined number of rows = 7
outliers_gaussian_dataframes = outliers_dataframes_gaussian(Ames, features, num_rows=7)

# Print each DataFrame with the original format and capitalized 'index'
# 获取字典的键值对 / Get dict key-value pairs
for feature, df in outliers_gaussian_dataframes.items():
    df_reset = df.reset_index().rename(columns={'index': 'Index'})
    # 打印输出 / Print output
    print(f"Outliers for {feature}:\n", df_reset[['Index', 'Feature', 'Outlier Value']])
    # 打印输出 / Print output
    print()
```

---

### Chapter Summary / 章节总结

# Chapter 13 Summary / 第13章总结

## Theme / 主题: Chapter 13 / Chapter 13

This chapter contains **4 code files** demonstrating chapter 13.

本章包含 **4 个代码文件**，演示Chapter 13。

---
## Evolution / 演化路线

  1. `01_boxplot.ipynb` — Boxplot
  2. `02_iqr.ipynb` — Iqr
  3. `03_outlier.ipynb` — Outlier
  4. `04_gaussian.ipynb` — Gaussian

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 13) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 13）是机器学习流水线中的基础构建块。

---
