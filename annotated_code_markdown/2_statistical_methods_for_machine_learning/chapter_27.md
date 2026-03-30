# 统计方法与机器学习
## Chapter 27

---

### Dataset

# 27 — Nonparametric Correlation / 非参数相关性

**Chapter 27 — File 1 of 3**

## Summary / 摘要

Nonparametric correlation measures the strength and direction of association between variables without assuming a linear relationship or normal distribution. This section explores generating monotonically related (but non-linear) data and visualizing it to understand how Spearman's rank correlation and Kendall's tau respond to such relationships.

非参数相关性测度了变量之间的关联强度和方向，无需假设线性关系或正态分布。本章探索生成单调相关(但非线性)的数据，并可视化以理解Spearman秩相关和Kendall的tau如何响应这样的关系。

### Key Concept / 关键概念

Nonparametric correlations rank the data rather than using raw values, making them robust to monotonic transformations and outliers.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
from numpy.random import rand, seed
from matplotlib import pyplot

# Visualization library for plotting scatter plots / 用于绘制散点图的可视化库
import numpy as np
```

## Step 2 — Generate Monotonically Related Data / 生成单调相关数据

```python
# Set random seed for reproducibility / 设置随机种子以保证可重现性
seed(1)

# Generate first variable: uniform random [0, 20) / 生成第一个变量: 均匀分布随机数[0, 20)
data1 = rand(1000) * 20

# Generate second variable: monotonic relationship with noise
# data2 = data1 + noise (preserves ordering even with added noise)
# 生成第二个变量: 与data1的单调关系加上噪声
# data2 = data1 + 噪声 (即使加入噪声也保留排序)
data2 = data1 + (rand(1000) * 10)
```

## Step 3 — Visualize the Relationship / 可视化关系

```python
# Create scatter plot to show the monotonic (but non-linear) relationship
# 创建散点图以显示单调(但非线性)的关系
pyplot.scatter(data1, data2)
pyplot.xlabel('Variable 1 / 变量 1')
pyplot.ylabel('Variable 2 / 变量 2')
pyplot.title('Monotonically Related Variables with Noise / 单调相关变量加噪声')
pyplot.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept**: Monotonic relationships are associations where one variable increases (or decreases) consistently as the other variable increases, but the rate of change is not constant. Nonparametric methods capture these relationships effectively without assuming linearity.

  **统计概念**: 单调关系是指一个变量随着另一个变量增加而持续增加(或减少)的关联，但变化率不恒定。非参数方法能有效捕捉这些关系，无需假设线性性。

- **ML Application**: In feature analysis and exploratory data analysis, detecting monotonic relationships is crucial for understanding how features interact. Nonparametric correlation works well with skewed distributions, categorical ordinal data, and when outliers are present.

  **ML应用**: 在特征分析和探索性数据分析中，检测单调关系对于理解特征如何交互至关重要。非参数相关性对于偏斜分布、序数分类数据和存在离群值的情况效果很好。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `02_spearmans.ipynb`

## Complete Code / 完整代码一览

```python
from numpy.random import rand, seed
from matplotlib import pyplot

seed(1)
data1 = rand(1000) * 20
data2 = data1 + (rand(1000) * 10)

pyplot.scatter(data1, data2)
pyplot.xlabel('Variable 1 / 变量 1')
pyplot.ylabel('Variable 2 / 变量 2')
pyplot.title('Monotonically Related Variables with Noise / 单调相关变量加噪声')
pyplot.show()
```

---

### Spearmans

# 27 — Spearman's Rank Correlation / Spearman秩相关性

**Chapter 27 — File 2 of 3**

## Summary / 摘要

Spearman's rank correlation coefficient ($\rho_s$) measures the monotonic relationship between two continuous or ordinal variables. Instead of using raw values, it ranks each variable and computes Pearson correlation on the ranks. This makes it resistant to outliers and applicable to non-normally distributed data.

Spearman秩相关系数($\rho_s$)测度两个连续或序数变量之间的单调关系。它对每个变量进行排序而不是使用原始值，然后在秩上计算Pearson相关。这使其对离群值有抵抗力，适用于非正态分布的数据。

### Key Formula / 关键公式

$$\rho_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

where $d_i$ is the difference between ranks and $n$ is the number of observations.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Generate Data / 生成数据

```python
from numpy.random import rand, seed

# Set random seed for reproducibility / 设置随机种子以保证可重现性
seed(1)

# Generate two variables with monotonic relationship / 生成具有单调关系的两个变量
data1 = rand(1000) * 20
data2 = data1 + (rand(1000) * 10)
```

## Step 2 — Calculate Spearman's Correlation / 计算Spearman相关

```python
from scipy.stats import spearmanr

# Calculate Spearman's correlation coefficient and p-value
# spearmanr returns: (correlation coefficient, p-value)
# 计算Spearman相关系数和p值
# spearmanr返回: (相关系数, p值)
coef, p = spearmanr(data1, data2)

print('Spearmans correlation coefficient: %.3f' % coef)
```

## Step 3 — Interpret Statistical Significance / 解释统计显著性

```python
# Interpret the p-value using significance level alpha = 0.05
# H0: No monotonic correlation (rho = 0)
# H1: Monotonic correlation exists (rho ≠ 0)
# 使用显著性水平alpha = 0.05解释p值
# H0: 无单调相关(rho = 0)
# H1: 存在单调相关(rho ≠ 0)
alpha = 0.05

if p > alpha:
    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
    print('Samples are correlated (reject H0) p=%.3f' % p)
```

## Learning Notes / 学习笔记

- **Statistical Concept**: Spearman's correlation uses rank-based computation, making it invariant to monotonic transformations. The coefficient ranges from -1 (perfect negative monotonic relationship) to +1 (perfect positive), with 0 indicating no monotonic association. The p-value tests if the observed correlation is statistically significant.

  **统计概念**: Spearman相关使用基于排序的计算，对单调变换是不变的。系数范围从-1(完美负单调关系)到+1(完美正)，0表示无单调关联。p值检验观察到的相关是否统计显著。

- **ML Application**: In feature engineering, Spearman's correlation helps identify monotonic feature relationships without assuming normality, useful for detecting redundant features or feature dependencies. It's preferred over Pearson when dealing with ranked data, discrete ordinal variables, or when the relationship is non-linear but monotonic.

  **ML应用**: 在特征工程中，Spearman相关帮助识别单调特征关系而无需假设正态性，对检测冗余特征或特征依赖很有用。在处理排序数据、离散序数变量或关系是非线性但单调的情况时，它优于Pearson。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `03_kendalls.ipynb`

## Complete Code / 完整代码一览

```python
from numpy.random import rand, seed
from scipy.stats import spearmanr

seed(1)
data1 = rand(1000) * 20
data2 = data1 + (rand(1000) * 10)

coef, p = spearmanr(data1, data2)
print('Spearmans correlation coefficient: %.3f' % coef)

alpha = 0.05
if p > alpha:
    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
    print('Samples are correlated (reject H0) p=%.3f' % p)
```

---
