# 数据科学入门 / Beginner's Guide to Data Science
## Chapter 10

---

### Histogram

# 01 — Histogram / 01 Histogram

**Chapter 10 — File 1 of 3 / 第10章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Data separation**.

本脚本演示 **Data separation**。

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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Data separation

```python
ac_prices = Ames[Ames['CentralAir'] == 'Y']['SalePrice']
no_ac_prices = Ames[Ames['CentralAir'] == 'N']['SalePrice']
```

---
## Step 3 — Setting up the visualization

```python
# 创建画布 / Create figure canvas
plt.figure(figsize=(10, 6))
```

---
## Step 4 — Histograms for sale prices based on air conditioning
Plotting 'With AC' first for the desired order in the legend

```python
# 绘制直方图 / Draw histogram
plt.hist(ac_prices, bins=30, alpha=0.7, color='blue', edgecolor='blue', lw=0.5,
         label='Sales Prices With AC')
# 计算均值 / Calculate mean
mean_ac = np.mean(ac_prices)
plt.axvline(mean_ac, color='blue', linestyle='dashed', linewidth=1.5,
            label=f'Mean (With AC): ${mean_ac:.2f}')

# 绘制直方图 / Draw histogram
plt.hist(no_ac_prices, bins=30, alpha=0.7, color='red', edgecolor='red', lw=0.5,
         label='Sales Prices Without AC')
# 计算均值 / Calculate mean
mean_no_ac = np.mean(no_ac_prices)
plt.axvline(mean_no_ac, color='red', linestyle='dashed', linewidth=1.5,
            label=f'Mean (Without AC): ${mean_no_ac:.2f}')

# 设置图表标题 / Set chart title
plt.title('Distribution of Sales Prices based on Presence of Air Conditioning',
          fontsize=18)
# 设置X轴标签 / Set X-axis label
plt.xlabel('Sales Price', fontsize=15)
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Number of Houses', fontsize=15)
# 显示图例 / Show legend
plt.legend(loc='upper right')
plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Data separation 是机器学习中的常用技术。  
  *Data separation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.mean` | 计算均值 | Calculate mean |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.hist` | 绘制直方图 | Draw histogram |
| `plt.show` | 显示图表 | Display plot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Histogram / 01 Histogram
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')

# Data separation
ac_prices = Ames[Ames['CentralAir'] == 'Y']['SalePrice']
no_ac_prices = Ames[Ames['CentralAir'] == 'N']['SalePrice']

# Setting up the visualization
# 创建画布 / Create figure canvas
plt.figure(figsize=(10, 6))

# Histograms for sale prices based on air conditioning
# Plotting 'With AC' first for the desired order in the legend
# 绘制直方图 / Draw histogram
plt.hist(ac_prices, bins=30, alpha=0.7, color='blue', edgecolor='blue', lw=0.5,
         label='Sales Prices With AC')
# 计算均值 / Calculate mean
mean_ac = np.mean(ac_prices)
plt.axvline(mean_ac, color='blue', linestyle='dashed', linewidth=1.5,
            label=f'Mean (With AC): ${mean_ac:.2f}')

# 绘制直方图 / Draw histogram
plt.hist(no_ac_prices, bins=30, alpha=0.7, color='red', edgecolor='red', lw=0.5,
         label='Sales Prices Without AC')
# 计算均值 / Calculate mean
mean_no_ac = np.mean(no_ac_prices)
plt.axvline(mean_no_ac, color='red', linestyle='dashed', linewidth=1.5,
            label=f'Mean (Without AC): ${mean_no_ac:.2f}')

# 设置图表标题 / Set chart title
plt.title('Distribution of Sales Prices based on Presence of Air Conditioning',
          fontsize=18)
# 设置X轴标签 / Set X-axis label
plt.xlabel('Sales Price', fontsize=15)
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Number of Houses', fontsize=15)
# 显示图例 / Show legend
plt.legend(loc='upper right')
plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Hypothesis

# 02 — Hypothesis / 02 Hypothesis

**Chapter 10 — File 2 of 3 / 第10章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Performing a two-sample t-test**.

本脚本演示 **Performing a two-sample t-test**。

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
ac_prices = Ames[Ames['CentralAir'] == 'Y']['SalePrice']
no_ac_prices = Ames[Ames['CentralAir'] == 'N']['SalePrice']
```

---
## Step 2 — Performing a two-sample t-test

```python
t_stat, p_value = stats.ttest_ind(ac_prices, no_ac_prices, equal_var=False)
```

---
## Step 3 — Printing the results

```python
if p_value < 0.05:
    result = "reject the null hypothesis"
else:
    result = "fail to reject the null hypothesis"
# 打印输出 / Print output
print(f"With a p-value of {p_value:.5f}, we {result}.")
```

---
## Learning Notes / 学习笔记

- **概念**: Performing a two-sample t-test 是机器学习中的常用技术。  
  *Performing a two-sample t-test is a common technique in machine learning.*

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
# Hypothesis / 02 Hypothesis
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import scipy.stats as stats

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
ac_prices = Ames[Ames['CentralAir'] == 'Y']['SalePrice']
no_ac_prices = Ames[Ames['CentralAir'] == 'N']['SalePrice']

# Performing a two-sample t-test
t_stat, p_value = stats.ttest_ind(ac_prices, no_ac_prices, equal_var=False)

# Printing the results
if p_value < 0.05:
    result = "reject the null hypothesis"
else:
    result = "fail to reject the null hypothesis"
# 打印输出 / Print output
print(f"With a p-value of {p_value:.5f}, we {result}.")
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Onesided

# 03 — Onesided / 03 Onesided

**Chapter 10 — File 3 of 3 / 第10章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Performing a two-sample t-test**.

本脚本演示 **Performing a two-sample t-test**。

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
ac_prices = Ames[Ames['CentralAir'] == 'Y']['SalePrice']
no_ac_prices = Ames[Ames['CentralAir'] == 'N']['SalePrice']
```

---
## Step 2 — Performing a two-sample t-test

```python
t_stat, p_value = stats.ttest_ind(ac_prices, no_ac_prices, equal_var=False,
                                  alternative="greater")
```

---
## Step 3 — Printing the results

```python
if p_value < 0.05:
    result = "reject the null hypothesis"
else:
    result = "fail to reject the null hypothesis"
# 打印输出 / Print output
print(f"With a p-value of {p_value:.5f}, we {result}.")
```

---
## Learning Notes / 学习笔记

- **概念**: Performing a two-sample t-test 是机器学习中的常用技术。  
  *Performing a two-sample t-test is a common technique in machine learning.*

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
# Onesided / 03 Onesided
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
import scipy.stats as stats

# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
Ames = pd.read_csv('Ames.csv')
ac_prices = Ames[Ames['CentralAir'] == 'Y']['SalePrice']
no_ac_prices = Ames[Ames['CentralAir'] == 'N']['SalePrice']

# Performing a two-sample t-test
t_stat, p_value = stats.ttest_ind(ac_prices, no_ac_prices, equal_var=False,
                                  alternative="greater")

# Printing the results
if p_value < 0.05:
    result = "reject the null hypothesis"
else:
    result = "fail to reject the null hypothesis"
# 打印输出 / Print output
print(f"With a p-value of {p_value:.5f}, we {result}.")
```

---

### Chapter Summary / 章节总结

# Chapter 10 Summary / 第10章总结

## Theme / 主题: Chapter 10 / Chapter 10

This chapter contains **3 code files** demonstrating chapter 10.

本章包含 **3 个代码文件**，演示Chapter 10。

---
## Evolution / 演化路线

  1. `01_histogram.ipynb` — Histogram
  2. `02_hypothesis.ipynb` — Hypothesis
  3. `03_onesided.ipynb` — Onesided

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 10) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 10）是机器学习流水线中的基础构建块。

---
