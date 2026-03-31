# 统计方法与机器学习 / Statistical Methods for Machine Learning
## Chapter 28

---

### Dataset

# 28 — Nonparametric Hypothesis Tests / 非参数假设检验

**Chapter 28 — File 1 of 5**

## Summary / 摘要

This chapter covers nonparametric hypothesis tests that do not assume normal distributions. These tests are essential when data violates normality assumptions, contains ordinal measurements, or has small sample sizes. We begin by generating sample datasets that represent independent uniform samples with slightly different bases.

本章涵盖不假设正态分布的非参数假设检验。当数据违反正态性假设、包含序数测量或样本量小时，这些检验至关重要。我们首先生成代表具有略微不同基数的独立均匀样本的样本数据集。

### Tests Covered / 涵盖的检验

- **Mann-Whitney U Test**: For comparing two independent groups
- **Wilcoxon Signed-Rank Test**: For comparing paired/matched samples
- **Kruskal-Wallis H Test**: For comparing 3+ groups (nonparametric ANOVA)
- **Friedman Test**: For repeated measures/within-subject designs

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed, rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# Libraries for nonparametric tests / 非参数检验库
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, friedmanchisquare
```

## Step 2 — Generate Sample Data / 生成样本数据

```python
# Set random seed for reproducibility / 设置随机种子以保证可重现性
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)

# Generate two independent uniform samples with base 50 and 51
# Both samples are uniformly distributed over a range of 10 units
# 生成两个基数为50和51的独立均匀样本
# 两个样本都在10个单位范围内均匀分布
data1 = 50 + (rand(100) * 10)
data2 = 51 + (rand(100) * 10)
```

## Step 3 — Inspect the Data / 检查数据

```python
# Display summary statistics of the generated samples
# 显示生成样本的摘要统计
# 打印输出 / Print output
print('data1: min=%.3f max=%.3f' % (min(data1), max(data1)))
# 打印输出 / Print output
print('data2: min=%.3f max=%.3f' % (min(data2), max(data2)))
# 计算均值 / Calculate mean
print('data1 mean: %.3f' % np.mean(data1))
# 计算均值 / Calculate mean
print('data2 mean: %.3f' % np.mean(data2))
```

## Learning Notes / 学习笔记

- **Statistical Concept**: Nonparametric tests do not assume normal distribution or homogeneity of variance. Instead of using raw data values, they operate on ranks or counts, making them robust to outliers and skewed distributions. These tests have fewer assumptions but may have lower statistical power than parametric alternatives when normality holds.

  **统计概念**: 非参数检验不假设正态分布或方差同质性。它们不是使用原始数据值，而是操作排序或计数，使其对离群值和偏斜分布很鲁棒。这些检验假设更少，但当正态性成立时，统计功效可能比参数替代品低。

- **ML Application**: In real-world ML pipelines, data often violates normality assumptions due to natural variation, measurement error, or skewness. Nonparametric hypothesis tests are crucial for validating model assumptions, comparing model performance across groups without normality constraints, and testing ordinal or categorical feature effects.

  **ML应用**: 在实际的ML管道中，数据常常由于自然变化、测量误差或偏斜而违反正态性假设。非参数假设检验对于验证模型假设、比较模型在各组之间的表现而不受正态性约束、测试序数或分类特征效应至关重要。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `np.mean` | 计算均值 | Calculate mean |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `02_mann_whitney.ipynb`

## Complete Code / 完整代码一览

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed, rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
data1 = 50 + (rand(100) * 10)
data2 = 51 + (rand(100) * 10)

# 打印输出 / Print output
print('data1: min=%.3f max=%.3f' % (min(data1), max(data1)))
# 打印输出 / Print output
print('data2: min=%.3f max=%.3f' % (min(data2), max(data2)))
# 计算均值 / Calculate mean
print('data1 mean: %.3f' % np.mean(data1))
# 计算均值 / Calculate mean
print('data2 mean: %.3f' % np.mean(data2))
```

---

### Mann Whitney

# 28 — Mann-Whitney U Test / Mann-Whitney U检验

**Chapter 28 — File 2 of 5**

## Summary / 摘要

The Mann-Whitney U test (also known as Wilcoxon rank-sum test) is a nonparametric test for comparing two independent samples. Unlike the t-test, it does not assume normal distributions or equal variances. It tests whether the two samples come from the same distribution by comparing the ranks of observations rather than their values.

Mann-Whitney U检验(也称为Wilcoxon秩和检验)是用于比较两个独立样本的非参数检验。与t检验不同，它不假设正态分布或方差相等。它通过比较观察的排序而不是它们的值来检验两个样本是否来自同一分布。

### Key Formula / 关键公式

$$U = n_1 n_2 + \frac{n_1(n_1 + 1)}{2} - R_1$$

where $n_1, n_2$ are sample sizes and $R_1$ is the sum of ranks in the first sample.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Generate Data / 生成数据

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed, rand

# Set random seed for reproducibility / 设置随机种子以保证可重现性
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)

# Generate two independent samples for comparison
# 生成两个独立的样本进行比较
data1 = 50 + (rand(100) * 10)
data2 = 51 + (rand(100) * 10)
```

## Step 2 — Perform Mann-Whitney U Test / 执行Mann-Whitney U检验

```python
from scipy.stats import mannwhitneyu

# Perform Mann-Whitney U test
# Returns: (test statistic U, p-value)
# 执行Mann-Whitney U检验
# 返回: (检验统计量U, p值)
stat, p = mannwhitneyu(data1, data2)

# 打印输出 / Print output
print('Statistics=%.3f, p=%.3f' % (stat, p))
```

## Step 3 — Interpret Results / 解释结果

```python
# Interpret using significance level alpha = 0.05
# H0: Both samples come from the same distribution
# H1: Samples come from different distributions
# 使用显著性水平alpha = 0.05解释
# H0: 两个样本来自同一分布
# H1: 样本来自不同分布
alpha = 0.05

if p > alpha:
    # 打印输出 / Print output
    print('Same distribution (fail to reject H0)')
else:
    # 打印输出 / Print output
    print('Different distribution (reject H0)')
```

## Learning Notes / 学习笔记

- **Statistical Concept**: The Mann-Whitney U test converts raw data to ranks, then compares whether one sample tends to have larger values than the other. It tests stochastic dominance without assuming normality. The test statistic $U$ represents the number of times values from group 1 exceed values from group 2, normalized by sample sizes.

  **统计概念**: Mann-Whitney U检验将原始数据转换为排序，然后比较一个样本的值是否倾向于比另一个样本大。它测试随机优势而不假设正态性。检验统计量$U$表示第1组的值超过第2组值的次数，按样本量标准化。

- **ML Application**: In model evaluation and A/B testing within ML pipelines, the Mann-Whitney U test is used to determine if two groups (e.g., control vs. treatment, before vs. after) have different distributions. It's applicable when comparing algorithmic performance metrics that are non-normally distributed or when validating ranking-based model outputs.

  **ML应用**: 在ML管道中的模型评估和A/B测试中，Mann-Whitney U检验用于确定两个组(例如对照组vs.处理组、之前vs.之后)是否有不同分布。它适用于比较非正态分布的算法性能指标或验证基于排序的模型输出。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `03_wilcoxon_signed_rank.ipynb`

## Complete Code / 完整代码一览

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed, rand
from scipy.stats import mannwhitneyu

# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
data1 = 50 + (rand(100) * 10)
data2 = 51 + (rand(100) * 10)

stat, p = mannwhitneyu(data1, data2)
# 打印输出 / Print output
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    # 打印输出 / Print output
    print('Same distribution (fail to reject H0)')
else:
    # 打印输出 / Print output
    print('Different distribution (reject H0)')
```

---

### Wilcoxon Signed Rank

# 28 — Wilcoxon Signed-Rank Test / Wilcoxon符号秩检验

**Chapter 28 — File 3 of 5**

## Summary / 摘要

The Wilcoxon signed-rank test is a nonparametric test for comparing two paired (dependent) samples. Unlike the paired t-test, it does not assume normality and is robust to outliers. It tests whether the median differences between paired observations differ from zero by computing the ranks of absolute differences.

Wilcoxon符号秩检验是用于比较两个成对(相关)样本的非参数检验。与配对t检验不同，它不假设正态性且对离群值很鲁棒。它通过计算绝对差的排序来检验成对观察间的中位数差是否与零不同。

### Key Characteristic / 关键特征

Tests whether paired differences are symmetric around zero, without assuming normality of the differences.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Generate Paired Data / 生成配对数据

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed, rand

# Set random seed for reproducibility / 设置随机种子以保证可重现性
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)

# Generate two paired samples (same subjects measured twice or matched pairs)
# 生成两个配对样本(相同主体测量两次或匹配对)
data1 = 50 + (rand(100) * 10)
data2 = 51 + (rand(100) * 10)
```

## Step 2 — Perform Wilcoxon Signed-Rank Test / 执行Wilcoxon符号秩检验

```python
from scipy.stats import wilcoxon

# Perform Wilcoxon signed-rank test
# Returns: (test statistic W, p-value)
# W = sum of positive ranks (or can be sum of smaller ranks depending on convention)
# 执行Wilcoxon符号秩检验
# 返回: (检验统计量W, p值)
# W = 正排序的和(或根据约定可以是较小排序的和)
stat, p = wilcoxon(data1, data2)

# 打印输出 / Print output
print('Statistics=%.3f, p=%.3f' % (stat, p))
```

## Step 3 — Interpret Results / 解释结果

```python
# Interpret using significance level alpha = 0.05
# H0: The paired samples come from the same distribution
# H1: The paired samples come from different distributions
# 使用显著性水平alpha = 0.05解释
# H0: 配对样本来自同一分布
# H1: 配对样本来自不同分布
alpha = 0.05

if p > alpha:
    # 打印输出 / Print output
    print('Same distribution (fail to reject H0)')
else:
    # 打印输出 / Print output
    print('Different distribution (reject H0)')
```

## Learning Notes / 学习笔记

- **Statistical Concept**: The Wilcoxon signed-rank test ranks the absolute differences between paired observations and uses only the ranks of positive differences (or the more extreme sign). It tests the null hypothesis that the median difference is zero without assuming symmetric distribution around zero. This makes it more robust than the paired t-test when data is non-normal.

  **统计概念**: Wilcoxon符号秩检验对配对观察之间的绝对差进行排序，仅使用正差的排序(或更极端的符号)。它测试中位数差为零的零假设，无需假设围绕零对称分布。当数据非正态时，这使其比配对t检验更鲁棒。

- **ML Application**: In model improvement validation and before/after analysis, Wilcoxon signed-rank test is used to verify if a model update improves predictions on the same test instances. It's also useful in hyperparameter tuning evaluation, feature importance studies on the same samples, and cross-validation paired comparisons where differences may not be normally distributed.

  **ML应用**: 在模型改进验证和前/后分析中，Wilcoxon符号秩检验用于验证模型更新是否改进了相同测试实例上的预测。它也在超参数调优评估、相同样本上的特征重要性研究和交叉验证配对比较中很有用，其中差可能不是正态分布的。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `04_kruskal_wallis.ipynb`

## Complete Code / 完整代码一览

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed, rand
from scipy.stats import wilcoxon

# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
data1 = 50 + (rand(100) * 10)
data2 = 51 + (rand(100) * 10)

stat, p = wilcoxon(data1, data2)
# 打印输出 / Print output
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    # 打印输出 / Print output
    print('Same distribution (fail to reject H0)')
else:
    # 打印输出 / Print output
    print('Different distribution (reject H0)')
```

---

### Kruskal Wallis

# 28 — Kruskal-Wallis H Test / Kruskal-Wallis H检验

**Chapter 28 — File 4 of 5**

## Summary / 摘要

The Kruskal-Wallis H test is the nonparametric alternative to one-way ANOVA for comparing three or more independent groups. It does not assume normal distributions or equal variances. The test determines whether samples originate from the same distribution by ranking all observations across groups and comparing rank sums.

Kruskal-Wallis H检验是用于比较三个或更多独立组的单因素ANOVA的非参数替代。它不假设正态分布或方差相等。该检验通过对所有组中的观察进行排序并比较排序和来确定样本是否来自同一分布。

### Key Formula / 关键公式

$$H = \frac{12}{n(n+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(n+1)$$

where $R_i$ is the sum of ranks in group $i$, $n_i$ is the group size, and $n$ is the total sample size.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Generate Data from Multiple Groups / 生成多组数据

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed, rand

# Set random seed for reproducibility / 设置随机种子以保证可重现性
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)

# Generate three independent samples with slightly different bases
# 生成三个基数略有不同的独立样本
data1 = 50 + (rand(100) * 10)
data2 = 51 + (rand(100) * 10)
data3 = 52 + (rand(100) * 10)
```

## Step 2 — Perform Kruskal-Wallis Test / 执行Kruskal-Wallis检验

```python
from scipy.stats import kruskal

# Perform Kruskal-Wallis H test
# Can compare 2+ groups; here we compare 3
# Returns: (H statistic, p-value)
# 执行Kruskal-Wallis H检验
# 可以比较2个或更多组；这里比较3个
# 返回: (H统计量, p值)
stat, p = kruskal(data1, data2, data3)

# 打印输出 / Print output
print('Statistics=%.3f, p=%.3f' % (stat, p))
```

## Step 3 — Interpret Results / 解释结果

```python
# Interpret using significance level alpha = 0.05
# H0: All groups come from the same distribution
# H1: At least one group differs from the others
# 使用显著性水平alpha = 0.05解释
# H0: 所有组来自同一分布
# H1: 至少有一个组与其他不同
alpha = 0.05

if p > alpha:
    # 打印输出 / Print output
    print('Same distributions (fail to reject H0)')
else:
    # 打印输出 / Print output
    print('Different distributions (reject H0)')
```

## Learning Notes / 学习笔记

- **Statistical Concept**: The Kruskal-Wallis test extends the Mann-Whitney U test to 3+ groups. It ranks all observations across all groups, then tests whether the rank distributions are similar. The H statistic approximates a chi-square distribution with k-1 degrees of freedom (k = number of groups). It's particularly useful when ANOVA assumptions (normality, homogeneity of variance) are violated.

  **统计概念**: Kruskal-Wallis检验将Mann-Whitney U检验扩展到3个或更多组。它对所有组中的所有观察进行排序，然后测试排序分布是否相似。H统计量近似于自由度为k-1的卡方分布(k =组数)。当ANOVA假设(正态性、方差同质性)被违反时特别有用。

- **ML Application**: In multiclass model comparison and feature effect analysis across groups, Kruskal-Wallis test validates whether a feature or model performance differs significantly across three or more categories. It's used in cross-group A/B/C testing, evaluating treatment effects across multiple conditions, and assessing algorithm performance across multiple datasets or parameter settings without normality constraints.

  **ML应用**: 在多类模型比较和跨组特征效应分析中，Kruskal-Wallis检验验证特征或模型性能是否在三个或更多类别中显著不同。它用于跨组A/B/C测试、评估多个条件下的处理效应，以及在无正态性约束下评估多个数据集或参数设置中的算法性能。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `05_friedman.ipynb`

## Complete Code / 完整代码一览

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed, rand
from scipy.stats import kruskal

# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
data1 = 50 + (rand(100) * 10)
data2 = 51 + (rand(100) * 10)
data3 = 52 + (rand(100) * 10)

stat, p = kruskal(data1, data2, data3)
# 打印输出 / Print output
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    # 打印输出 / Print output
    print('Same distributions (fail to reject H0)')
else:
    # 打印输出 / Print output
    print('Different distributions (reject H0)')
```

---

### Friedman

# 28 — Friedman Test / Friedman检验

**Chapter 28 — File 5 of 5**

## Summary / 摘要

The Friedman test is a nonparametric test for comparing repeated measures (within-subject designs) across three or more conditions. It is the nonparametric counterpart to repeated-measures ANOVA. The test uses ranks of observations within each subject/block and tests whether the rank distributions differ significantly across conditions.

Friedman检验是用于在三个或更多条件下比较重复测量(受试者内设计)的非参数检验。它是重复测量ANOVA的非参数对应物。该检验使用每个被试/块内的观察排序，并测试排序分布是否在条件间显著不同。

### Key Formula / 关键公式

$$\chi_r^2 = \frac{12}{n \cdot k(k+1)} \sum_{j=1}^{k} R_j^2 - 3n(k+1)$$

where $R_j$ is the sum of ranks for condition $j$, $n$ is number of subjects, and $k$ is number of conditions.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Generate Repeated Measures Data / 生成重复测量数据

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed, rand

# Set random seed for reproducibility / 设置随机种子以保证可重现性
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)

# Generate three repeated measures (conditions) for the same 100 subjects
# data1, data2, data3 represent measurements at different time points or conditions
# 生成相同100个被试的三个重复测量(条件)
# data1, data2, data3表示不同时间点或条件下的测量
data1 = 50 + (rand(100) * 10)
data2 = 51 + (rand(100) * 10)
data3 = 52 + (rand(100) * 10)
```

## Step 2 — Perform Friedman Test / 执行Friedman检验

```python
from scipy.stats import friedmanchisquare

# Perform Friedman test for repeated measures
# Can compare 2+ conditions; here we compare 3 conditions
# Returns: (test statistic, p-value)
# 执行重复测量的Friedman检验
# 可以比较2个或更多条件；这里比较3个条件
# 返回: (检验统计量, p值)
stat, p = friedmanchisquare(data1, data2, data3)

# 打印输出 / Print output
print('Statistics=%.3f, p=%.3f' % (stat, p))
```

## Step 3 — Interpret Results / 解释结果

```python
# Interpret using significance level alpha = 0.05
# H0: All conditions have the same distribution (no differences across conditions)
# H1: At least one condition differs from the others
# 使用显著性水平alpha = 0.05解释
# H0: 所有条件有相同分布(条件间无差异)
# H1: 至少有一个条件与其他不同
alpha = 0.05

if p > alpha:
    # 打印输出 / Print output
    print('Same distributions (fail to reject H0)')
else:
    # 打印输出 / Print output
    print('Different distributions (reject H0)')
```

## Learning Notes / 学习笔记

- **Statistical Concept**: The Friedman test is designed for matched/paired designs where the same subjects are measured under multiple conditions. It ranks observations within each subject and tests whether the rank distributions are consistent across conditions. The test statistic follows approximately a chi-square distribution with k-1 degrees of freedom. It's robust to outliers and doesn't assume normality.

  **统计概念**: Friedman检验为匹配/配对设计而设计，其中相同的被试在多个条件下被测量。它对每个被试内的观察进行排序，并测试排序分布是否在条件间一致。检验统计量近似遵循k-1个自由度的卡方分布。它对离群值很鲁棒，不假设正态性。

- **ML Application**: In model validation with repeated measures, Friedman test evaluates whether different algorithms perform significantly differently across multiple datasets or cross-validation folds. It's used for comparing multiple feature engineering strategies on the same data, validating hyperparameter tuning effects across repeated experiments, and time-series prediction models evaluated on sliding windows without requiring normality assumptions.

  **ML应用**: 在带有重复测量的模型验证中，Friedman检验评估不同算法是否在多个数据集或交叉验证折上表现显著不同。它用于在相同数据上比较多个特征工程策略、验证重复实验中的超参数调优效应，以及在滑动窗口上评估的时间序列预测模型而无需要求正态性假设。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `../chapter_29/01_chi_squared.ipynb`

## Complete Code / 完整代码一览

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed, rand
from scipy.stats import friedmanchisquare

# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
data1 = 50 + (rand(100) * 10)
data2 = 51 + (rand(100) * 10)
data3 = 52 + (rand(100) * 10)

stat, p = friedmanchisquare(data1, data2, data3)
# 打印输出 / Print output
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    # 打印输出 / Print output
    print('Same distributions (fail to reject H0)')
else:
    # 打印输出 / Print output
    print('Different distributions (reject H0)')
```

---

### Chapter Summary / 章节总结

# Chapter 28: Nonparametric Hypothesis Tests
# 第28章：非参数假设检验

## Theme | 主题
Rank-based alternatives to parametric tests: robust when normality fails or samples are small.
参数检验的基于等级的替代方案：当正态性失败或样本很小时鲁棒。

## Evolution Roadmap | 演变路线图
```
INDEPENDENT SAMPLES:
  Parametric: Independent t-test
  └─ Nonparametric: Mann-Whitney U test (or Wilcoxon rank-sum)

PAIRED SAMPLES:
  Parametric: Paired t-test
  └─ Nonparametric: Wilcoxon signed-rank test

3+ INDEPENDENT GROUPS:
  Parametric: One-way ANOVA
  └─ Nonparametric: Kruskal-Wallis H test

REPEATED MEASURES (3+ groups, same subjects):
  Parametric: Repeated-measures ANOVA
  └─ Nonparametric: Friedman test
```

## Progression Logic | 进度逻辑

### Stage 1: Mann-Whitney U Test (Mann-Whitney U检验)
**English:** Nonparametric alternative to independent t-test. Combines and ranks data from both groups, computes U-statistic (# of times x < y). H0: distribution of x equals distribution of y.
**中文:** 独立t检验的非参数替代方案。结合并排名来自两个组的数据，计算U统计(x < y的次数)。H0：x的分布等于y的分布。

### Stage 2: Wilcoxon Signed-Rank Test (Wilcoxon有符号秩检验)
**English:** Nonparametric alternative to paired t-test. Computes differences, ranks absolute differences, sums signed ranks. H0: distribution of differences is symmetric around 0.
**中文:** 配对t检验的非参数替代方案。计算差异、等级绝对差异、有符号等级求和。H0：差异分布关于0对称。

### Stage 3: Kruskal-Wallis H Test (Kruskal-Wallis H检验)
**English:** Nonparametric alternative to one-way ANOVA. Combines all groups, ranks, computes H-statistic. H0: all groups have same distribution.
**中文:** 单因素方差分析的非参数替代方案。结合所有组、排名、计算H统计。H0：所有组具有相同分布。

### Stage 4: Friedman Test (Friedman检验)
**English:** Nonparametric alternative to repeated-measures ANOVA. Each subject gets ranks within their own observations. Computes rank-based statistic. H0: no difference across repeated measures.
**中文:** 重复测量方差分析的非参数替代方案。每个受试者在自己的观察中获得等级。计算基于等级的统计。H0：重复测量中无差异。

## ML Relevance | ML相关性

1. **Robustness Without Normality (无需正态的鲁棒性)**: Nonparametric tests work when data violates normality assumption.
2. **Small Samples (小样本)**: No need for large n to rely on CLT; tests are valid even with tiny samples.
3. **Ordinal Data (序数数据)**: Naturally handle rankings, survey ratings, and ordinal scales.
4. **Outlier Protection (异常值保护)**: Ranks suppress influence of extreme values.


---
