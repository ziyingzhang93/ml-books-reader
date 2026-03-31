# 数据科学入门 / Beginner's Guide to Data Science
## Chapter 05

---

### Describe

# 01 — Describe / 01 Describe

**Chapter 05 — File 1 of 6 / 第05章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Importing libraries and loading the dataset**.

本脚本演示 **Importing libraries and loading the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Importing libraries and loading the dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Descriptive Statistics of Sales Price

```python
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
sales_price_description = Ames['SalePrice'].describe()
# 打印输出 / Print output
print(sales_price_description)
```

---
## Learning Notes / 学习笔记

- **概念**: Importing libraries and loading the dataset 是机器学习中的常用技术。  
  *Importing libraries and loading the dataset is a common technique in machine learning.*

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
# Describe / 01 Describe
# Complete Code / 完整代码
# ===============================

# Importing libraries and loading the dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Descriptive Statistics of Sales Price
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
sales_price_description = Ames['SalePrice'].describe()
# 打印输出 / Print output
print(sales_price_description)
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Medianmode

# 02 — Medianmode / 02 Medianmode

**Chapter 05 — File 2 of 6 / 第05章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Medianmode**.

本脚本演示 **02 Medianmode**。

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

median_saleprice = Ames['SalePrice'].median()
# 打印输出 / Print output
print("Median Sale Price:", median_saleprice)

# 转换为NumPy数组 / Convert to NumPy array
mode_saleprice = Ames['SalePrice'].mode().values[0]
# 打印输出 / Print output
print("Mode Sale Price:", mode_saleprice)
```

---
## Learning Notes / 学习笔记

- **概念**: Medianmode 是机器学习中的常用技术。  
  *Medianmode is a common technique in machine learning.*

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
# Medianmode / 02 Medianmode
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

median_saleprice = Ames['SalePrice'].median()
# 打印输出 / Print output
print("Median Sale Price:", median_saleprice)

# 转换为NumPy数组 / Convert to NumPy array
mode_saleprice = Ames['SalePrice'].mode().values[0]
# 打印输出 / Print output
print("Mode Sale Price:", mode_saleprice)
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Dispersion

# 03 — Dispersion / 03 Dispersion

**Chapter 05 — File 3 of 6 / 第05章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Dispersion**.

本脚本演示 **03 Dispersion**。

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

range_saleprice = Ames['SalePrice'].max() - Ames['SalePrice'].min()
# 打印输出 / Print output
print("Range of Sale Price:", range_saleprice)

variance_saleprice = Ames['SalePrice'].var()
# 打印输出 / Print output
print("Variance of Sale Price:", variance_saleprice)

std_dev_saleprice = Ames['SalePrice'].std()
# 打印输出 / Print output
print("Standard Deviation of Sale Price:", std_dev_saleprice)

iqr_saleprice = Ames['SalePrice'].quantile(0.75) - Ames['SalePrice'].quantile(0.25)
# 打印输出 / Print output
print("IQR of Sale Price:", iqr_saleprice)
```

---
## Learning Notes / 学习笔记

- **概念**: Dispersion 是机器学习中的常用技术。  
  *Dispersion is a common technique in machine learning.*

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
# Dispersion / 03 Dispersion
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

range_saleprice = Ames['SalePrice'].max() - Ames['SalePrice'].min()
# 打印输出 / Print output
print("Range of Sale Price:", range_saleprice)

variance_saleprice = Ames['SalePrice'].var()
# 打印输出 / Print output
print("Variance of Sale Price:", variance_saleprice)

std_dev_saleprice = Ames['SalePrice'].std()
# 打印输出 / Print output
print("Standard Deviation of Sale Price:", std_dev_saleprice)

iqr_saleprice = Ames['SalePrice'].quantile(0.75) - Ames['SalePrice'].quantile(0.25)
# 打印输出 / Print output
print("IQR of Sale Price:", iqr_saleprice)
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Percentile

# 04 — Percentile / 04 Percentile

**Chapter 05 — File 4 of 6 / 第05章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Percentile**.

本脚本演示 **04 Percentile**。

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

skewness_saleprice = Ames['SalePrice'].skew()
# 打印输出 / Print output
print("Skewness of Sale Price:", skewness_saleprice)

kurtosis_saleprice = Ames['SalePrice'].kurt()
# 打印输出 / Print output
print("Kurtosis of Sale Price:", kurtosis_saleprice)

tenth_percentile = Ames['SalePrice'].quantile(0.10)
ninetieth_percentile = Ames['SalePrice'].quantile(0.90)
# 打印输出 / Print output
print("10th Percentile:", tenth_percentile)
# 打印输出 / Print output
print("90th Percentile:", ninetieth_percentile)

q1_saleprice = Ames['SalePrice'].quantile(0.25)
q2_saleprice = Ames['SalePrice'].quantile(0.50)
q3_saleprice = Ames['SalePrice'].quantile(0.75)
# 打印输出 / Print output
print("Q1 (25th Percentile):", q1_saleprice)
# 打印输出 / Print output
print("Q2 (Median/50th Percentile):", q2_saleprice)
# 打印输出 / Print output
print("Q3 (75th Percentile):", q3_saleprice)
```

---
## Learning Notes / 学习笔记

- **概念**: Percentile 是机器学习中的常用技术。  
  *Percentile is a common technique in machine learning.*

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
# Percentile / 04 Percentile
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

skewness_saleprice = Ames['SalePrice'].skew()
# 打印输出 / Print output
print("Skewness of Sale Price:", skewness_saleprice)

kurtosis_saleprice = Ames['SalePrice'].kurt()
# 打印输出 / Print output
print("Kurtosis of Sale Price:", kurtosis_saleprice)

tenth_percentile = Ames['SalePrice'].quantile(0.10)
ninetieth_percentile = Ames['SalePrice'].quantile(0.90)
# 打印输出 / Print output
print("10th Percentile:", tenth_percentile)
# 打印输出 / Print output
print("90th Percentile:", ninetieth_percentile)

q1_saleprice = Ames['SalePrice'].quantile(0.25)
q2_saleprice = Ames['SalePrice'].quantile(0.50)
q3_saleprice = Ames['SalePrice'].quantile(0.75)
# 打印输出 / Print output
print("Q1 (25th Percentile):", q1_saleprice)
# 打印输出 / Print output
print("Q2 (Median/50th Percentile):", q2_saleprice)
# 打印输出 / Print output
print("Q3 (75th Percentile):", q3_saleprice)
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Histogram

# 05 — Histogram / 05 Histogram

**Chapter 05 — File 5 of 6 / 第05章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Importing visualization libraries**.

本脚本演示 **Importing visualization libraries**。

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
```

---
## Step 2 — Importing visualization libraries

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import seaborn as sns

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 3 — Setting up the style

```python
sns.set_style("whitegrid")
```

---
## Step 4 — Calculate Mean, Median, Mode for SalePrice

```python
mean_saleprice = Ames['SalePrice'].mean()
median_saleprice = Ames['SalePrice'].median()
# 转换为NumPy数组 / Convert to NumPy array
mode_saleprice = Ames['SalePrice'].mode().values[0]
```

---
## Step 5 — Plotting the histogram

```python
# 创建画布 / Create figure canvas
plt.figure(figsize=(14, 7))
sns.histplot(x=Ames['SalePrice'], bins=30, kde=True, color="skyblue")
plt.axvline(mean_saleprice, color='r', linestyle='--',
            label=f"Mean: ${mean_saleprice:.2f}")
plt.axvline(median_saleprice, color='g', linestyle='-',
            label=f"Median: ${median_saleprice:.2f}")
plt.axvline(mode_saleprice, color='b', linestyle='-.',
            label=f"Mode: ${mode_saleprice:.2f}")
```

---
## Step 6 — Calculating skewness and kurtosis for SalePrice

```python
skewness_saleprice = Ames['SalePrice'].skew()
kurtosis_saleprice = Ames['SalePrice'].kurt()
```

---
## Step 7 — Annotations for skewness and kurtosis

```python
text = 'Skewness: {:.2f}\nKurtosis: {:.2f}' \
        .format(Ames['SalePrice'].skew(), Ames['SalePrice'].kurt())
plt.annotate(text, xy=(500000, 100), fontsize=14,
             bbox={"boxstyle": "round,pad=0.3",
                   "edgecolor": "black",
                   "facecolor": "aliceblue"})
# 设置图表标题 / Set chart title
plt.title('Histogram of Ames\' Housing Prices with KDE and Reference Lines')
# 设置X轴标签 / Set X-axis label
plt.xlabel('Housing Prices')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Frequency')
# 显示图例 / Show legend
plt.legend()
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Importing visualization libraries 是机器学习中的常用技术。  
  *Importing visualization libraries is a common technique in machine learning.*

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
# Histogram / 05 Histogram
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# Importing visualization libraries
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import seaborn as sns

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Setting up the style
sns.set_style("whitegrid")

# Calculate Mean, Median, Mode for SalePrice
mean_saleprice = Ames['SalePrice'].mean()
median_saleprice = Ames['SalePrice'].median()
# 转换为NumPy数组 / Convert to NumPy array
mode_saleprice = Ames['SalePrice'].mode().values[0]

# Plotting the histogram
# 创建画布 / Create figure canvas
plt.figure(figsize=(14, 7))
sns.histplot(x=Ames['SalePrice'], bins=30, kde=True, color="skyblue")
plt.axvline(mean_saleprice, color='r', linestyle='--',
            label=f"Mean: ${mean_saleprice:.2f}")
plt.axvline(median_saleprice, color='g', linestyle='-',
            label=f"Median: ${median_saleprice:.2f}")
plt.axvline(mode_saleprice, color='b', linestyle='-.',
            label=f"Mode: ${mode_saleprice:.2f}")

# Calculating skewness and kurtosis for SalePrice
skewness_saleprice = Ames['SalePrice'].skew()
kurtosis_saleprice = Ames['SalePrice'].kurt()

# Annotations for skewness and kurtosis
text = 'Skewness: {:.2f}\nKurtosis: {:.2f}' \
        .format(Ames['SalePrice'].skew(), Ames['SalePrice'].kurt())
plt.annotate(text, xy=(500000, 100), fontsize=14,
             bbox={"boxstyle": "round,pad=0.3",
                   "edgecolor": "black",
                   "facecolor": "aliceblue"})
# 设置图表标题 / Set chart title
plt.title('Histogram of Ames\' Housing Prices with KDE and Reference Lines')
# 设置X轴标签 / Set X-axis label
plt.xlabel('Housing Prices')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Frequency')
# 显示图例 / Show legend
plt.legend()
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Boxplot

# 06 — Boxplot / 06 Boxplot

**Chapter 05 — File 6 of 6 / 第05章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Horizontal box plot with annotations**.

本脚本演示 **Horizontal box plot with annotations**。

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
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import seaborn as sns
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.lines import Line2D
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Horizontal box plot with annotations

```python
# 创建画布 / Create figure canvas
plt.figure(figsize=(12, 8))
```

---
## Step 3 — Plotting the box plot with specified color and style

```python
sns.boxplot(x=Ames['SalePrice'], color='skyblue', showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "red",
                       "markeredgecolor": "red", "markersize":10})
```

---
## Step 4 — Plotting arrows for Q1, Median and Q3

```python
q1_saleprice = Ames['SalePrice'].quantile(0.25)
q2_saleprice = Ames['SalePrice'].quantile(0.50)
q3_saleprice = Ames['SalePrice'].quantile(0.75)
plt.annotate('Q1', xy=(q1_saleprice, 0.30), xytext=(q1_saleprice - 70000, 0.45),
             arrowprops={'edgecolor': 'black', 'arrowstyle': '->'}, fontsize=14)
plt.annotate('Q3', xy=(q3_saleprice, 0.30), xytext=(q3_saleprice + 20000, 0.45),
             arrowprops={'edgecolor': 'black', 'arrowstyle': '->'}, fontsize=14)
plt.annotate('Median', xy=(q2_saleprice, 0.20), xytext=(q2_saleprice - 90000, 0.05),
             arrowprops={'edgecolor': 'black', 'arrowstyle': '->'}, fontsize=14)
```

---
## Step 5 — Titles, labels, and legends

```python
# 设置图表标题 / Set chart title
plt.title('Box Plot Ames\' Housing Prices', fontsize=16)
# 设置X轴标签 / Set X-axis label
plt.xlabel('Housing Prices', fontsize=14)
plt.yticks([])  # Hide y-axis tick labels
# 显示图例 / Show legend
plt.legend(handles=[Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
           markersize=10, label='Mean')], loc='upper left', fontsize=14)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Horizontal box plot with annotations 是机器学习中的常用技术。  
  *Horizontal box plot with annotations is a common technique in machine learning.*

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
# Boxplot / 06 Boxplot
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import seaborn as sns
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib.lines import Line2D
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Horizontal box plot with annotations
# 创建画布 / Create figure canvas
plt.figure(figsize=(12, 8))

# Plotting the box plot with specified color and style
sns.boxplot(x=Ames['SalePrice'], color='skyblue', showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "red",
                       "markeredgecolor": "red", "markersize":10})

# Plotting arrows for Q1, Median and Q3
q1_saleprice = Ames['SalePrice'].quantile(0.25)
q2_saleprice = Ames['SalePrice'].quantile(0.50)
q3_saleprice = Ames['SalePrice'].quantile(0.75)
plt.annotate('Q1', xy=(q1_saleprice, 0.30), xytext=(q1_saleprice - 70000, 0.45),
             arrowprops={'edgecolor': 'black', 'arrowstyle': '->'}, fontsize=14)
plt.annotate('Q3', xy=(q3_saleprice, 0.30), xytext=(q3_saleprice + 20000, 0.45),
             arrowprops={'edgecolor': 'black', 'arrowstyle': '->'}, fontsize=14)
plt.annotate('Median', xy=(q2_saleprice, 0.20), xytext=(q2_saleprice - 90000, 0.05),
             arrowprops={'edgecolor': 'black', 'arrowstyle': '->'}, fontsize=14)

# Titles, labels, and legends
# 设置图表标题 / Set chart title
plt.title('Box Plot Ames\' Housing Prices', fontsize=16)
# 设置X轴标签 / Set X-axis label
plt.xlabel('Housing Prices', fontsize=14)
plt.yticks([])  # Hide y-axis tick labels
# 显示图例 / Show legend
plt.legend(handles=[Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
           markersize=10, label='Mean')], loc='upper left', fontsize=14)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 05 Summary / 第05章总结

## Theme / 主题: Chapter 05 / Chapter 05

This chapter contains **6 code files** demonstrating chapter 05.

本章包含 **6 个代码文件**，演示Chapter 05。

---
## Evolution / 演化路线

  1. `01_describe.ipynb` — Describe
  2. `02_medianmode.ipynb` — Medianmode
  3. `03_dispersion.ipynb` — Dispersion
  4. `04_percentile.ipynb` — Percentile
  5. `05_histogram.ipynb` — Histogram
  6. `06_boxplot.ipynb` — Boxplot

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 05) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 05）是机器学习流水线中的基础构建块。

---
