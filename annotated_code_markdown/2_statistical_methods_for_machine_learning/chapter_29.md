# 统计方法与机器学习
## Chapter 29

---

### Chi Squared

# 29 — Chi-Squared Test of Independence / 卡方独立性检验

**Chapter 29 — File 1 of 1**

## Summary / 摘要

The chi-squared test of independence evaluates whether two categorical variables are associated or independent. Given a contingency table with observed frequencies, the test compares them to expected frequencies under the assumption of independence. A significant chi-squared statistic indicates that the two variables are dependent (associated).

卡方独立性检验评估两个分类变量是否相关或独立。给定具有观察频率的列联表，该检验将其与假设独立下的期望频率进行比较。显著的卡方统计量表示两个变量是相关的(关联的)。

### Key Formula / 关键公式

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

where $O$ is observed frequency and $E$ is expected frequency under independence.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create Contingency Table / 创建列联表

```python
# Define a 2x3 contingency table
# Rows: Category A (2 levels)
# Columns: Category B (3 levels)
# 定义一个2x3列联表
# 行: 类别A(2个水平)
# 列: 类别B(3个水平)
table = [[10, 20, 30],
         [6,  9,  17]]

print('Observed Frequency Table / 观察频率表:')
print(table)
```

## Step 2 — Compute Chi-Squared Statistic / 计算卡方统计量

```python
from scipy.stats import chi2_contingency
from scipy.stats import chi2

# Perform chi-squared test of independence
# chi2_contingency returns:
# stat: chi-squared test statistic
# p: p-value (for contingency test)
# dof: degrees of freedom = (rows - 1) * (cols - 1)
# expected: expected frequencies under independence
# 执行卡方独立性检验
# chi2_contingency返回:
# stat: 卡方检验统计量
# p: p值(用于列联检验)
# dof: 自由度 = (行 - 1) * (列 - 1)
# expected: 独立下的期望频率
stat, p, dof, expected = chi2_contingency(table)

print('Degrees of freedom: %d' % dof)
print('\nExpected Frequency Table / 期望频率表:')
print(expected)
```

## Step 3 — Interpret Using Critical Value / 使用临界值解释

```python
# Approach 1: Compare test statistic to critical value
# At significance level alpha = 0.05, the critical value is at the 95th percentile
# 方法1: 将检验统计量与临界值比较
# 在显著性水平alpha = 0.05时，临界值在第95百分位
prob = 0.95
critical = chi2.ppf(prob, dof)

print('Critical Value Approach / 临界值方法:')
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:
    print('Dependent (reject H0) - Variables are associated')
else:
    print('Independent (fail to reject H0) - No association between variables')
```

## Step 4 — Interpret Using P-Value / 使用P值解释

```python
# Approach 2: Compare p-value to significance level
# H0: The two variables are independent
# H1: The two variables are dependent (associated)
# 方法2: 将p值与显著性水平比较
# H0: 两个变量是独立的
# H1: 两个变量是相关的(关联的)
alpha = 1.0 - prob  # alpha = 0.05

print('\nP-Value Approach / P值方法:')
print('significance=%.3f, p=%.3f' % (alpha, p))

if p <= alpha:
    print('Dependent (reject H0) - Variables are associated')
else:
    print('Independent (fail to reject H0) - No association between variables')
```

## Learning Notes / 学习笔记

- **Statistical Concept**: The chi-squared test compares observed cell frequencies with expected frequencies under independence. The test statistic $\chi^2$ measures the total deviation of observed from expected values. The p-value provides the probability of observing data at least as extreme as the observed data under the null hypothesis of independence. Larger $\chi^2$ values and smaller p-values indicate stronger evidence of association.

  **统计概念**: 卡方检验将观察的单元频率与独立下的期望频率进行比较。检验统计量$\chi^2$测度观察值与期望值的总偏差。p值提供了在零假设(独立)下观察到至少与观察数据极端的数据的概率。更大的$\chi^2$值和更小的p值表示更强的关联证据。

- **ML Application**: In feature selection and categorical data analysis, chi-squared tests determine if categorical features are associated with class labels or outcomes. It's essential for understanding feature-target relationships in classification tasks, identifying significant interactions between categorical features, and preprocessing validation to ensure features aren't spuriously independent or dependent due to data quality issues.

  **ML应用**: 在特征选择和分类数据分析中，卡方检验确定分类特征是否与类标签或结果相关联。它对于理解分类任务中的特征-目标关系至关重要，识别分类特征间的重要交互，以及预处理验证以确保特征不会因数据质量问题而虚假独立或相关。

➡️ **Next**: `../appendix_02/1_versions.ipynb`

## Complete Code / 完整代码一览

```python
from scipy.stats import chi2_contingency, chi2

table = [[10, 20, 30],
         [6,  9,  17]]

stat, p, dof, expected = chi2_contingency(table)

print('dof=%d' % dof)
print(expected)

# Critical value approach
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')

# P-value approach
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))

if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
```

---
