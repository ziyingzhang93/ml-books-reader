# 数据科学入门 / Beginner's Guide to Data Science
## Chapter 12

---

### Boxplot

# 01 — Boxplot / 01 Boxplot

**Chapter 12 — File 1 of 6 / 第12章 — 第1个文件（共6个）**

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
## Step 3 — Convert 'YrSold' to a categorical variable

```python
# 转换数据类型 / Convert data type
Ames['YrSold'] = Ames['YrSold'].astype('category')

# 创建画布 / Create figure canvas
plt.figure(figsize=(10, 6))
sns.boxplot(x=Ames['YrSold'], y=Ames['SalePrice'], hue=Ames['YrSold'])
# 设置图表标题 / Set chart title
plt.title('Boxplot of Sales Prices by Year', fontsize=18)
# 设置X轴标签 / Set X-axis label
plt.xlabel('Year Sold', fontsize=15)
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Sales Price (US$)', fontsize=15)
# 显示图例 / Show legend
plt.legend('')
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
| `plt.figure` | 创建画布 | Create figure |
| `plt.show` | 显示图表 | Display plot |
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

# Load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Convert 'YrSold' to a categorical variable
# 转换数据类型 / Convert data type
Ames['YrSold'] = Ames['YrSold'].astype('category')

# 创建画布 / Create figure canvas
plt.figure(figsize=(10, 6))
sns.boxplot(x=Ames['YrSold'], y=Ames['SalePrice'], hue=Ames['YrSold'])
# 设置图表标题 / Set chart title
plt.title('Boxplot of Sales Prices by Year', fontsize=18)
# 设置X轴标签 / Set X-axis label
plt.xlabel('Year Sold', fontsize=15)
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Sales Price (US$)', fontsize=15)
# 显示图例 / Show legend
plt.legend('')
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Meanmedian

# 02 — Meanmedian / 02 Meanmedian

**Chapter 12 — File 2 of 6 / 第12章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Calculating mean and median sales price by year**.

本脚本演示 **Calculating mean and median sales price by year**。

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
## Step 2 — Calculating mean and median sales price by year

```python
summary_table = Ames.groupby('YrSold')['SalePrice'].agg(['mean', 'median'])
```

---
## Step 3 — Rounding the values for better presentation

```python
summary_table = summary_table.round(2)
# 打印输出 / Print output
print(summary_table)
```

---
## Learning Notes / 学习笔记

- **概念**: Calculating mean and median sales price by year 是机器学习中的常用技术。  
  *Calculating mean and median sales price by year is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `groupby` | 分组聚合 | Group and aggregate |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Meanmedian / 02 Meanmedian
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Calculating mean and median sales price by year
summary_table = Ames.groupby('YrSold')['SalePrice'].agg(['mean', 'median'])

# Rounding the values for better presentation
summary_table = summary_table.round(2)
# 打印输出 / Print output
print(summary_table)
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Anova

# 03 — Anova / 03 Anova

**Chapter 12 — File 3 of 6 / 第12章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Perform the ANOVA**.

本脚本演示 **Perform the ANOVA**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import scipy.stats as stats

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Perform the ANOVA

```python
f_value, p_value = stats.f_oneway(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                    for year in Ames['YrSold'].unique()])
# 打印输出 / Print output
print(f_value, p_value)
```

---
## Learning Notes / 学习笔记

- **概念**: Perform the ANOVA 是机器学习中的常用技术。  
  *Perform the ANOVA is a common technique in machine learning.*

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
# Anova / 03 Anova
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import scipy.stats as stats

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Perform the ANOVA
f_value, p_value = stats.f_oneway(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                    for year in Ames['YrSold'].unique()])
# 打印输出 / Print output
print(f_value, p_value)
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Qqplot

# 04 — Qqplot / 04 Qqplot

**Chapter 12 — File 4 of 6 / 第12章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Fit an ordinary least squares model and get residuals**.

本脚本演示 **Fit an ordinary least squares model and get residuals**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 训练模型 / Train the model
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
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
import statsmodels.api as sm
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Fit an ordinary least squares model and get residuals

```python
# 转换数据类型 / Convert data type
model = sm.OLS(Ames['SalePrice'], Ames['YrSold'].astype('int')).fit()
residuals = model.resid
```

---
## Step 3 — Plot QQ plot

```python
sm.qqplot(residuals, line='s')
# 设置图表标题 / Set chart title
plt.title('Normality Assessment of Residuals via QQ Plot', fontsize=18)
# 设置X轴标签 / Set X-axis label
plt.xlabel('Theoretical Quantiles', fontsize=15)
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Sample Residual Quantiles', fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Fit an ordinary least squares model and get residuals 是机器学习中的常用技术。  
  *Fit an ordinary least squares model and get residuals is a common technique in machine learning.*

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
# Qqplot / 04 Qqplot
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import statsmodels.api as sm
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Fit an ordinary least squares model and get residuals
# 转换数据类型 / Convert data type
model = sm.OLS(Ames['SalePrice'], Ames['YrSold'].astype('int')).fit()
residuals = model.resid

# Plot QQ plot
sm.qqplot(residuals, line='s')
# 设置图表标题 / Set chart title
plt.title('Normality Assessment of Residuals via QQ Plot', fontsize=18)
# 设置X轴标签 / Set X-axis label
plt.xlabel('Theoretical Quantiles', fontsize=15)
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Sample Residual Quantiles', fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Tests

# 07 — Tests / 07 Tests

**Chapter 12 — File 5 of 6 / 第12章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Load the dataset**.

本脚本演示 **Load the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 训练模型 / Train the model
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
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
import scipy.stats as stats
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro
```

---
## Step 2 — Load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 3 — Perform the ANOVA

```python
f_value, p_value = stats.f_oneway(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                    for year in Ames['YrSold'].unique()])
# 打印输出 / Print output
print("F-value:", f_value)
# 打印输出 / Print output
print("p-value:", p_value)
```

---
## Step 4 — Fit an ordinary least squares model and get residuals

```python
# 转换数据类型 / Convert data type
model = sm.OLS(Ames['SalePrice'], Ames['YrSold'].astype('int')).fit()
residuals = model.resid
```

---
## Step 5 — Plot QQ plot

```python
sm.qqplot(residuals, line='s')
# 设置图表标题 / Set chart title
plt.title('Normality Assessment of Residuals via QQ Plot', fontsize=18)
# 设置X轴标签 / Set X-axis label
plt.xlabel('Theoretical Quantiles', fontsize=15)
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Sample Residual Quantiles', fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# 显示图表 / Display the plot
plt.show()
```

---
## Step 6 — Shapiro-Wilk Test

```python
shapiro_stat, shapiro_p = shapiro(residuals)
# 打印输出 / Print output
print(f"Shapiro-Wilk Test Statistic: {shapiro_stat}")
# 打印输出 / Print output
print(f"P-value: {shapiro_p}")
```

---
## Step 7 — Check for equal variances using Levene's test

```python
levene_stat, levene_p = stats.levene(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                      for year in Ames['YrSold'].unique()])
# 打印输出 / Print output
print(f"Levene's Test Statistic: {levene_stat}")
# 打印输出 / Print output
print(f"P-value: {levene_p}")
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
# Tests / 07 Tests
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import scipy.stats as stats
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro

# Load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Perform the ANOVA
f_value, p_value = stats.f_oneway(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                    for year in Ames['YrSold'].unique()])
# 打印输出 / Print output
print("F-value:", f_value)
# 打印输出 / Print output
print("p-value:", p_value)

# Fit an ordinary least squares model and get residuals
# 转换数据类型 / Convert data type
model = sm.OLS(Ames['SalePrice'], Ames['YrSold'].astype('int')).fit()
residuals = model.resid

# Plot QQ plot
sm.qqplot(residuals, line='s')
# 设置图表标题 / Set chart title
plt.title('Normality Assessment of Residuals via QQ Plot', fontsize=18)
# 设置X轴标签 / Set X-axis label
plt.xlabel('Theoretical Quantiles', fontsize=15)
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Sample Residual Quantiles', fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# 显示图表 / Display the plot
plt.show()

# Shapiro-Wilk Test
shapiro_stat, shapiro_p = shapiro(residuals)
# 打印输出 / Print output
print(f"Shapiro-Wilk Test Statistic: {shapiro_stat}")
# 打印输出 / Print output
print(f"P-value: {shapiro_p}")

# Check for equal variances using Levene's test
levene_stat, levene_p = stats.levene(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                      for year in Ames['YrSold'].unique()])
# 打印输出 / Print output
print(f"Levene's Test Statistic: {levene_stat}")
# 打印输出 / Print output
print(f"P-value: {levene_p}")
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Nonparametric

# 11 — Nonparametric / 11 Nonparametric

**Chapter 12 — File 6 of 6 / 第12章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Perform the Kruskal-Wallis H-test**.

本脚本演示 **Perform the Kruskal-Wallis H-test**。

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
import scipy.stats as stats
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Perform the Kruskal-Wallis H-test

```python
H_statistic, kruskal_p_value = stats.kruskal(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                               for year in Ames['YrSold'].unique()])
# 打印输出 / Print output
print(H_statistic, kruskal_p_value)
```

---
## Step 3 — Plot histograms of Sales Price for each year

```python
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 8), sharex=True)

# 同时获取索引和值 / Get both index and value
for idx, year in enumerate(sorted(Ames['YrSold'].unique())):
    sns.histplot(Ames[Ames['YrSold'] == year]['SalePrice'], kde=True, ax=axes[idx],
                 color='skyblue')
    axes[idx].set_title(f'Distribution of Sales Prices for Year {year}', fontsize=16)
    axes[idx].set_ylabel('Frequency', fontsize=14)
    if idx == 4:
        axes[idx].set_xlabel('Sales Price', fontsize=15)
    else:
        axes[idx].set_xlabel('')

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---
## Step 4 — Run KS Test from scipy.stats

```python
results = {}
# 同时获取索引和值 / Get both index and value
for i, year1 in enumerate(sorted(Ames['YrSold'].unique())):
    # 同时获取索引和值 / Get both index and value
    for j, year2 in enumerate(sorted(Ames['YrSold'].unique())):
        if i < j:
            ks_stat, ks_p = ks_2samp(Ames[Ames['YrSold'] == year1]['SalePrice'],
                                     Ames[Ames['YrSold'] == year2]['SalePrice'])
            results[f"{year1} vs {year2}"] = (ks_stat, ks_p)
```

---
## Step 5 — Convert the results into a DataFrame for tabular representation

```python
ks_df = pd.DataFrame(results).transpose()
# 获取列名 / Get column names
ks_df.columns = ['KS Statistic', 'P-value']
ks_df.reset_index(inplace=True)
ks_df.rename(columns={'index': 'Years Compared'}, inplace=True)
# 打印输出 / Print output
print(ks_df)
```

---
## Learning Notes / 学习笔记

- **概念**: Perform the Kruskal-Wallis H-test 是机器学习中的常用技术。  
  *Perform the Kruskal-Wallis H-test is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Nonparametric / 11 Nonparametric
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import seaborn as sns
import scipy.stats as stats
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Perform the Kruskal-Wallis H-test
H_statistic, kruskal_p_value = stats.kruskal(*[Ames['SalePrice'][Ames['YrSold'] == year]
                                               for year in Ames['YrSold'].unique()])
# 打印输出 / Print output
print(H_statistic, kruskal_p_value)

# Plot histograms of Sales Price for each year
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 8), sharex=True)

# 同时获取索引和值 / Get both index and value
for idx, year in enumerate(sorted(Ames['YrSold'].unique())):
    sns.histplot(Ames[Ames['YrSold'] == year]['SalePrice'], kde=True, ax=axes[idx],
                 color='skyblue')
    axes[idx].set_title(f'Distribution of Sales Prices for Year {year}', fontsize=16)
    axes[idx].set_ylabel('Frequency', fontsize=14)
    if idx == 4:
        axes[idx].set_xlabel('Sales Price', fontsize=15)
    else:
        axes[idx].set_xlabel('')

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()

# Run KS Test from scipy.stats
results = {}
# 同时获取索引和值 / Get both index and value
for i, year1 in enumerate(sorted(Ames['YrSold'].unique())):
    # 同时获取索引和值 / Get both index and value
    for j, year2 in enumerate(sorted(Ames['YrSold'].unique())):
        if i < j:
            ks_stat, ks_p = ks_2samp(Ames[Ames['YrSold'] == year1]['SalePrice'],
                                     Ames[Ames['YrSold'] == year2]['SalePrice'])
            results[f"{year1} vs {year2}"] = (ks_stat, ks_p)

# Convert the results into a DataFrame for tabular representation
ks_df = pd.DataFrame(results).transpose()
# 获取列名 / Get column names
ks_df.columns = ['KS Statistic', 'P-value']
ks_df.reset_index(inplace=True)
ks_df.rename(columns={'index': 'Years Compared'}, inplace=True)
# 打印输出 / Print output
print(ks_df)
```

---

### Chapter Summary / 章节总结

# Chapter 12 Summary / 第12章总结

## Theme / 主题: Chapter 12 / Chapter 12

This chapter contains **6 code files** demonstrating chapter 12.

本章包含 **6 个代码文件**，演示Chapter 12。

---
## Evolution / 演化路线

  1. `01_boxplot.ipynb` — Boxplot
  2. `02_meanmedian.ipynb` — Meanmedian
  3. `03_anova.ipynb` — Anova
  4. `04_qqplot.ipynb` — Qqplot
  5. `07_tests.ipynb` — Tests
  6. `11_nonparametric.ipynb` — Nonparametric

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 12) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 12）是机器学习流水线中的基础构建块。

---
