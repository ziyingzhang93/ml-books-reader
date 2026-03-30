# 数据科学入门
## Chapter 09

---

### Tinterval

# 01 — Tinterval / 01 Tinterval

**Chapter 09 — File 1 of 3 / 第09章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Define the confidence level and degrees of freedom**.

本脚本演示 **Define the confidence level and degrees of freedom**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
import scipy.stats as stats
import pandas as pd
Ames = pd.read_csv('Ames.csv')
```

---
## Step 2 — Define the confidence level and degrees of freedom

```python
confidence_level = 0.95
degrees_freedom = Ames['SalePrice'].count() - 1
```

---
## Step 3 — Calculate the confidence interval for 'SalePrice'

```python
confidence_interval = stats.t.interval(confidence_level, degrees_freedom,
                                       loc=Ames['SalePrice'].mean(),
                                       scale=Ames['SalePrice'].sem())
```

---
## Step 4 — Print out the sentence with the confidence interval figures

```python
print(f"The 95% confidence interval for the true mean sales price of all houses in Ames "
      f"is between ${confidence_interval[0]:.2f} and ${confidence_interval[1]:.2f}.")
```

---
## Learning Notes / 学习笔记

- **概念**: Define the confidence level and degrees of freedom 是机器学习中的常用技术。  
  *Define the confidence level and degrees of freedom is a common technique in machine learning.*

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
# Tinterval / 01 Tinterval
# Complete Code / 完整代码
# ===============================

import scipy.stats as stats
import pandas as pd
Ames = pd.read_csv('Ames.csv')

#Define the confidence level and degrees of freedom
confidence_level = 0.95
degrees_freedom = Ames['SalePrice'].count() - 1

#Calculate the confidence interval for 'SalePrice'
confidence_interval = stats.t.interval(confidence_level, degrees_freedom,
                                       loc=Ames['SalePrice'].mean(),
                                       scale=Ames['SalePrice'].sem())

# Print out the sentence with the confidence interval figures
print(f"The 95% confidence interval for the true mean sales price of all houses in Ames "
      f"is between ${confidence_interval[0]:.2f} and ${confidence_interval[1]:.2f}.")
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Distribution

# 02 — Distribution / 02 Distribution

**Chapter 09 — File 2 of 3 / 第09章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Plot the main histogram**.

本脚本演示 **Plot the main histogram**。

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
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
Ames = pd.read_csv('Ames.csv')

confidence_level = 0.95
degrees_freedom = Ames['SalePrice'].count() - 1
confidence_interval = stats.t.interval(confidence_level, degrees_freedom,
                                       loc=Ames['SalePrice'].mean(),
                                       scale=Ames['SalePrice'].sem())
```

---
## Step 2 — Plot the main histogram

```python
plt.figure(figsize=(10, 7))
plt.hist(Ames['SalePrice'], bins=30, color='lightblue', edgecolor='black', alpha=0.5,
         label='Sales Prices Distribution')
```

---
## Step 3 — Vertical lines for sample mean and confidence interval with adjusted styles

```python
plt.axvline(Ames['SalePrice'].mean(), color='blue', linestyle='-',
            label=f'Mean: ${Ames["SalePrice"].mean():,.2f}')
plt.axvline(confidence_interval[0], color='red', linestyle='--',
            label=f'Lower 95% CI: ${confidence_interval[0]:,.2f}')
plt.axvline(confidence_interval[1], color='green', linestyle='--',
            label=f'Upper 95% CI: ${confidence_interval[1]:,.2f}')
```

---
## Step 4 — Annotations and labels

```python
plt.title('Distribution of Sales Prices with Confidence Interval', fontsize=20)
plt.xlabel('Sales Price', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xlim([min(Ames['SalePrice']) - 5000, max(Ames['SalePrice']) + 5000])
plt.legend()
plt.grid(axis='y')
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Plot the main histogram 是机器学习中的常用技术。  
  *Plot the main histogram is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
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
# Distribution / 02 Distribution
# Complete Code / 完整代码
# ===============================

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
Ames = pd.read_csv('Ames.csv')

confidence_level = 0.95
degrees_freedom = Ames['SalePrice'].count() - 1
confidence_interval = stats.t.interval(confidence_level, degrees_freedom,
                                       loc=Ames['SalePrice'].mean(),
                                       scale=Ames['SalePrice'].sem())

# Plot the main histogram
plt.figure(figsize=(10, 7))
plt.hist(Ames['SalePrice'], bins=30, color='lightblue', edgecolor='black', alpha=0.5,
         label='Sales Prices Distribution')

# Vertical lines for sample mean and confidence interval with adjusted styles
plt.axvline(Ames['SalePrice'].mean(), color='blue', linestyle='-',
            label=f'Mean: ${Ames["SalePrice"].mean():,.2f}')
plt.axvline(confidence_interval[0], color='red', linestyle='--',
            label=f'Lower 95% CI: ${confidence_interval[0]:,.2f}')
plt.axvline(confidence_interval[1], color='green', linestyle='--',
            label=f'Upper 95% CI: ${confidence_interval[1]:,.2f}')

# Annotations and labels
plt.title('Distribution of Sales Prices with Confidence Interval', fontsize=20)
plt.xlabel('Sales Price', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xlim([min(Ames['SalePrice']) - 5000, max(Ames['SalePrice']) + 5000])
plt.legend()
plt.grid(axis='y')
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Zoomin

# 03 — Zoomin / 03 Zoomin

**Chapter 09 — File 3 of 3 / 第09章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Creating a second plot focused on the mean and confidence intervals**.

本脚本演示 **Creating a second plot focused on the mean and confidence intervals**。

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
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
Ames = pd.read_csv('Ames.csv')

confidence_level = 0.95
degrees_freedom = Ames['SalePrice'].count() - 1
confidence_interval = stats.t.interval(confidence_level, degrees_freedom,
                                       loc=Ames['SalePrice'].mean(),
                                       scale=Ames['SalePrice'].sem())
```

---
## Step 2 — Creating a second plot focused on the mean and confidence intervals

```python
plt.figure(figsize=(10, 7))
plt.hist(Ames['SalePrice'], bins=30, color='lightblue', edgecolor='black', alpha=0.5,
         label='Sales Prices')
```

---
## Step 3 — Zooming in around the mean and confidence intervals

```python
plt.xlim([150000, 200000])
```

---
## Step 4 — Vertical lines for sample mean and confidence interval with adjusted styles

```python
plt.axvline(Ames['SalePrice'].mean(), color='blue', linestyle='-',
            label=f'Mean: ${Ames["SalePrice"].mean():,.2f}')
plt.axvline(confidence_interval[0], color='red', linestyle='--',
            label=f'Lower 95% CI: ${confidence_interval[0]:,.2f}')
plt.axvline(confidence_interval[1], color='green', linestyle='--',
            label=f'Upper 95% CI: ${confidence_interval[1]:,.2f}')
```

---
## Step 5 — Annotations and labels for the zoomed-in plot

```python
plt.title('Zoomed-in View of Mean and Confidence Intervals', fontsize=20)
plt.xlabel('Sales Price', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.legend()
plt.grid(axis='y')
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Creating a second plot focused on the mean and confidence intervals 是机器学习中的常用技术。  
  *Creating a second plot focused on the mean and confidence intervals is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
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
# Zoomin / 03 Zoomin
# Complete Code / 完整代码
# ===============================

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
Ames = pd.read_csv('Ames.csv')

confidence_level = 0.95
degrees_freedom = Ames['SalePrice'].count() - 1
confidence_interval = stats.t.interval(confidence_level, degrees_freedom,
                                       loc=Ames['SalePrice'].mean(),
                                       scale=Ames['SalePrice'].sem())

# Creating a second plot focused on the mean and confidence intervals
plt.figure(figsize=(10, 7))
plt.hist(Ames['SalePrice'], bins=30, color='lightblue', edgecolor='black', alpha=0.5,
         label='Sales Prices')

# Zooming in around the mean and confidence intervals
plt.xlim([150000, 200000])

# Vertical lines for sample mean and confidence interval with adjusted styles
plt.axvline(Ames['SalePrice'].mean(), color='blue', linestyle='-',
            label=f'Mean: ${Ames["SalePrice"].mean():,.2f}')
plt.axvline(confidence_interval[0], color='red', linestyle='--',
            label=f'Lower 95% CI: ${confidence_interval[0]:,.2f}')
plt.axvline(confidence_interval[1], color='green', linestyle='--',
            label=f'Upper 95% CI: ${confidence_interval[1]:,.2f}')

# Annotations and labels for the zoomed-in plot
plt.title('Zoomed-in View of Mean and Confidence Intervals', fontsize=20)
plt.xlabel('Sales Price', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.legend()
plt.grid(axis='y')
plt.show()
```

---

### Chapter Summary

# Chapter 09 Summary / 第09章总结

## Theme / 主题: Chapter 09 / Chapter 09

This chapter contains **3 code files** demonstrating chapter 09.

本章包含 **3 个代码文件**，演示Chapter 09。

---
## Evolution / 演化路线

  1. `01_tinterval.ipynb` — Tinterval
  2. `02_distribution.ipynb` — Distribution
  3. `03_zoomin.ipynb` — Zoomin

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 09) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 09）是机器学习流水线中的基础构建块。

---
