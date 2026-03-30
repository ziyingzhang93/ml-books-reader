# 概率论与机器学习
## Chapter 09

---

### Normal Sample

# 01 — Normal Distribution Sample / 正态分布样本

**Chapter 09 — File 1 of 7**

## Summary

We sample from a normal (Gaussian) distribution with mean 50 and standard deviation 5 using numpy.random.normal(). The normal distribution is the most fundamental continuous distribution in probability and statistics.

我们使用numpy.random.normal()从均值为50、标准差为5的正态（高斯）分布中抽样。正态分布是概率和统计中最基础的连续分布。

**Formula:**

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import necessary libraries / 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
```

## Step 2 — Set Parameters / 设置参数

```python
# Define normal distribution parameters / 定义正态分布参数
mu = 50              # Mean / 均值
sigma = 5            # Standard deviation / 标准差
num_samples = 10000  # Number of samples / 样本数
```

## Step 3 — Generate Samples / 生成样本

```python
# Sample from normal distribution / 从正态分布中抽样
# numpy.random.normal(loc=mu, scale=sigma, size=num_samples)
samples = np.random.normal(loc=mu, scale=sigma, size=num_samples)

print(f'Normal Distribution Sample / 正态分布样本')
print(f'Parameters: μ={mu}, σ={sigma}')
print(f'样本数: {num_samples}')
print(f'\nSample statistics / 样本统计:')
print(f'Mean / 均值: {np.mean(samples):.4f}')
print(f'Std Dev / 标准差: {np.std(samples):.4f}')
print(f'Min / 最小值: {np.min(samples):.4f}')
print(f'Max / 最大值: {np.max(samples):.4f}')
```

## Step 4 — Visualize Samples / 可视化样本

```python
# Create visualization / 创建可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram / 直方图
axes[0].hist(samples, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[0].axvline(np.mean(samples), color='r', linestyle='--', linewidth=2, 
                 label=f'Sample Mean: {np.mean(samples):.2f}')
axes[0].axvline(mu, color='g', linestyle='--', linewidth=2, 
                 label=f'Theoretical μ: {mu}')
axes[0].set_xlabel('Value / 值')
axes[0].set_ylabel('Density / 密度')
axes[0].set_title(f'Histogram of Normal Samples (μ={mu}, σ={sigma})')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Q-Q Plot / Q-Q图
from scipy import stats
stats.probplot(samples, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot: Normal Distribution')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Step 5 — Analyze Distribution Properties / 分析分布性质

```python
# Calculate percentiles and confidence intervals / 计算百分位数和置信区间
print('\nDistribution Properties / 分布性质:')
print('=' * 60)

# Calculate percentiles / 计算百分位数
percentiles = [2.5, 5, 25, 50, 75, 95, 97.5]
print(f'\n{"Percentile":>12s} {"Value":>12s}')
print('-' * 26)
for p in percentiles:
    value = np.percentile(samples, p)
    print(f'{p:11.1f}% {value:12.4f}')

# Confidence intervals / 置信区间
mean = np.mean(samples)
std = np.std(samples)
print(f'\nConfidence Intervals (based on sample): / 置信区间（基于样本）:')
print(f'68% (μ ± 1σ):  [{mean - std:.2f}, {mean + std:.2f}]')
print(f'95% (μ ± 2σ):  [{mean - 2*std:.2f}, {mean + 2*std:.2f}]')
print(f'99% (μ ± 3σ):  [{mean - 3*std:.2f}, {mean + 3*std:.2f}]')
```

## Learning Notes / 学习笔记

- **Concept**: The normal distribution (Gaussian) is the most important continuous probability distribution. It appears naturally in many phenomena due to the Central Limit Theorem, which states that the sum of many independent random variables approaches a normal distribution.
  
  **概念**：正态分布（高斯分布）是最重要的连续概率分布。由于中心极限定理，它自然出现在许多现象中，该定理指出许多独立随机变量的和趋向于正态分布。

- **ML Application**: Normal distribution is fundamental for regression models, probabilistic classifiers, anomaly detection, and understanding model outputs. Many machine learning algorithms assume normally distributed features or residuals.
  
  **机器学习应用**：正态分布对于回归模型、概率分类器、异常检测和理解模型输出至关重要。许多机器学习算法假设特征或残差正态分布。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.mean` | 计算均值 | Calculate mean |
| `np.random` | 随机数生成 | Random number generation |
| `np.std` | 计算标准差 | Calculate standard deviation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

## Next / 下一步

➡️ **Next**: `02_normal_plot.ipynb`

## Complete Code / 完整代码一览

```python
# Complete Normal Distribution Sampling / 完整正态分布抽样

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Define normal distribution parameters / 定义正态分布参数
mu = 50              # Mean / 均值
sigma = 5            # Standard deviation / 标准差
num_samples = 10000  # Number of samples / 样本数

# Sample from normal distribution / 从正态分布中抽样
samples = np.random.normal(loc=mu, scale=sigma, size=num_samples)

print(f'Normal Distribution Sample')
print(f'Parameters: μ={mu}, σ={sigma}')
print(f'Sample count: {num_samples}')
print(f'\nSample statistics:')
print(f'Mean: {np.mean(samples):.4f}')
print(f'Std Dev: {np.std(samples):.4f}')
print(f'Min: {np.min(samples):.4f}')
print(f'Max: {np.max(samples):.4f}')

# Create visualization / 创建可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram / 直方图
axes[0].hist(samples, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[0].axvline(np.mean(samples), color='r', linestyle='--', linewidth=2, 
                 label=f'Sample Mean: {np.mean(samples):.2f}')
axes[0].axvline(mu, color='g', linestyle='--', linewidth=2, 
                 label=f'Theoretical μ: {mu}')
axes[0].set_xlabel('Value / 值')
axes[0].set_ylabel('Density / 密度')
axes[0].set_title(f'Histogram of Normal Samples (μ={mu}, σ={sigma})')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Q-Q Plot / Q-Q图
stats.probplot(samples, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot: Normal Distribution')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate percentiles and confidence intervals / 计算百分位数和置信区间
print('\nDistribution Properties:')
print('=' * 60)

percentiles = [2.5, 5, 25, 50, 75, 95, 97.5]
print(f'\n{"Percentile":>12s} {"Value":>12s}')
print('-' * 26)
for p in percentiles:
    value = np.percentile(samples, p)
    print(f'{p:11.1f}% {value:12.4f}')

# Confidence intervals / 置信区间
mean = np.mean(samples)
std = np.std(samples)
print(f'\nConfidence Intervals:')
print(f'68% (μ ± 1σ):  [{mean - std:.2f}, {mean + std:.2f}]')
print(f'95% (μ ± 2σ):  [{mean - 2*std:.2f}, {mean + 2*std:.2f}]')
print(f'99% (μ ± 3σ):  [{mean - 3*std:.2f}, {mean + 3*std:.2f}]')
```

---

### Normal Middle

# 03 — Normal Middle Range / 正态分布中间范围

**Chapter 09 — File 3 of 7**

## Summary

We find the middle 95% range of a normal distribution using the percent point function (PPF), which is the inverse of the CDF. We calculate the values at the 2.5th and 97.5th percentiles using norm.ppf().

我们使用百分点函数（PPF）（CDF的反函数）找到正态分布的中间95%范围。我们使用norm.ppf()计算第2.5和97.5百分位数的值。

**Formula:**

$$P(\mu + z_{0.025} \sigma \leq X \leq \mu + z_{0.975} \sigma) = 0.95$$

where $z_{0.025} = -1.96$ and $z_{0.975} = 1.96$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import necessary libraries / 导入必要的库
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
```

## Step 2 — Define Parameters / 定义参数

```python
# Define normal distribution parameters / 定义正态分布参数
mu = 50      # Mean / 均值
sigma = 5    # Standard deviation / 标准差
```

## Step 3 — Calculate 95% Confidence Interval / 计算95%置信区间

```python
# Using norm.ppf() - Percent Point Function (inverse of CDF)
# norm.ppf(q) returns x such that P(X <= x) = q
# 使用norm.ppf() - 百分点函数（CDF的反函数）
# norm.ppf(q)返回x使得P(X <= x) = q

# Find the 2.5th and 97.5th percentiles / 找到第2.5和97.5百分位数
# These bound the middle 95% of the distribution / 这些限定了分布的中间95%
lower_percentile = 0.025
upper_percentile = 0.975

# Calculate z-scores / 计算z分数
z_lower = norm.ppf(lower_percentile)
z_upper = norm.ppf(upper_percentile)

print(f'95% Confidence Interval Calculation / 95%置信区间计算')
print(f'=' * 70)
print(f'Distribution: N(μ={mu}, σ={sigma})')
print(f'\nZ-scores (standardized):/')
print(f'Z_{lower_percentile} = {z_lower:.4f}')
print(f'Z_{upper_percentile} = {z_upper:.4f}')

# Convert to original scale / 转换到原始尺度
lower_bound = mu + z_lower * sigma
upper_bound = mu + z_upper * sigma

print(f'\nOriginal Scale:/')
print(f'Lower Bound: μ + z_lower × σ = {mu} + {z_lower:.4f} × {sigma} = {lower_bound:.4f}')
print(f'Upper Bound: μ + z_upper × σ = {mu} + {z_upper:.4f} × {sigma} = {upper_bound:.4f}')
print(f'\nMiddle 95% Range: [{lower_bound:.4f}, {upper_bound:.4f}]')
print(f'中间95%范围: [{lower_bound:.4f}, {upper_bound:.4f}]')
```

## Step 4 — Verify with CDF / 使用CDF验证

```python
# Verify by calculating CDF at these points / 通过在这些点计算CDF来验证
prob_below_upper = norm.cdf(upper_bound, loc=mu, scale=sigma)
prob_below_lower = norm.cdf(lower_bound, loc=mu, scale=sigma)
prob_in_range = prob_below_upper - prob_below_lower

print(f'\nVerification using CDF / 使用CDF验证:')
print(f'P(X <= {lower_bound:.4f}) = {prob_below_lower:.6f}')
print(f'P(X <= {upper_bound:.4f}) = {prob_below_upper:.6f}')
print(f'P({lower_bound:.4f} <= X <= {upper_bound:.4f}) = {prob_in_range:.6f}')
print(f'\nExpected: 0.95, Got: {prob_in_range:.6f}')
```

## Step 5 — Calculate Other Confidence Levels / 计算其他置信水平

```python
# Calculate confidence intervals for different levels / 计算不同水平的置信区间
confidence_levels = [0.68, 0.90, 0.95, 0.99]

print(f'\nConfidence Intervals at Different Levels / 不同水平的置信区间:')
print(f'=' * 70)
print(f'{"Confidence":>12s} {"Lower Bound":>15s} {"Upper Bound":>15s} {"Width":>12s}')
print('-' * 70)

for conf in confidence_levels:
    # Calculate tail probabilities / 计算尾部概率
    alpha = 1 - conf
    lower_p = alpha / 2
    upper_p = 1 - alpha / 2
    
    # Find bounds / 找到边界
    z_l = norm.ppf(lower_p)
    z_u = norm.ppf(upper_p)
    
    lower = mu + z_l * sigma
    upper = mu + z_u * sigma
    width = upper - lower
    
    print(f'{conf*100:10.0f}% [{lower:13.4f}, {upper:13.4f}] {width:12.4f}')
```

## Step 6 — Visualize Confidence Intervals / 可视化置信区间

```python
# Create visualization / 创建可视化
fig, ax = plt.subplots(figsize=(12, 6))

# Plot PDF / 绘制PDF
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = norm.pdf(x, loc=mu, scale=sigma)
ax.plot(x, pdf, 'b-', linewidth=2, label='PDF')

# Highlight 95% interval / 高亮95%区间
x_95 = np.linspace(lower_bound, upper_bound, 100)
pdf_95 = norm.pdf(x_95, loc=mu, scale=sigma)
ax.fill_between(x_95, pdf_95, alpha=0.6, color='green', label='95% CI')

# Add vertical lines for bounds / 添加边界的竖线
ax.axvline(lower_bound, color='r', linestyle='--', linewidth=2, 
           label=f'Lower: {lower_bound:.2f}')
ax.axvline(upper_bound, color='r', linestyle='--', linewidth=2, 
           label=f'Upper: {upper_bound:.2f}')
ax.axvline(mu, color='black', linestyle='-', linewidth=1.5, 
           label=f'Mean: {mu}')

# Shade the tail regions / 给尾部区域着色
x_left = x[x < lower_bound]
pdf_left = norm.pdf(x_left, loc=mu, scale=sigma)
ax.fill_between(x_left, pdf_left, alpha=0.3, color='red', label='Tails (5%)')

x_right = x[x > upper_bound]
pdf_right = norm.pdf(x_right, loc=mu, scale=sigma)
ax.fill_between(x_right, pdf_right, alpha=0.3, color='red')

ax.set_xlabel('Value / 值')
ax.set_ylabel('Probability Density / 概率密度')
ax.set_title(f'95% Confidence Interval: N(μ={mu}, σ={sigma})')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Learning Notes / 学习笔记

- **Concept**: The percent point function (PPF) is the inverse of the CDF. It allows us to find the value x such that a given probability mass falls below x. This is essential for constructing confidence intervals and understanding quantiles.
  
  **概念**：百分点函数（PPF）是CDF的反函数。它允许我们找到值x，使得给定的概率质量低于x。这对于构造置信区间和理解分位数至关重要。

- **ML Application**: Confidence intervals are fundamental for model evaluation, hypothesis testing, and quantifying uncertainty in predictions. Understanding how to compute them is essential for responsible machine learning.
  
  **机器学习应用**：置信区间对于模型评估、假设检验和量化预测中的不确定性至关重要。理解如何计算它们对于负责任的机器学习至关重要。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

## Next / 下一步

➡️ **Next**: `04_exponential_sample.ipynb`

## Complete Code / 完整代码一览

```python
# Complete Normal Middle Range Analysis / 完整正态分布中间范围分析

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# Define normal distribution parameters / 定义正态分布参数
mu = 50      # Mean / 均值
sigma = 5    # Standard deviation / 标准差

# Find the 2.5th and 97.5th percentiles / 找到第2.5和97.5百分位数
lower_percentile = 0.025
upper_percentile = 0.975

# Calculate z-scores / 计算z分数
z_lower = norm.ppf(lower_percentile)
z_upper = norm.ppf(upper_percentile)

print(f'95% Confidence Interval Calculation')
print(f'Distribution: N(μ={mu}, σ={sigma})')
print(f'\nZ-scores (standardized):')
print(f'Z_{lower_percentile} = {z_lower:.4f}')
print(f'Z_{upper_percentile} = {z_upper:.4f}')

# Convert to original scale / 转换到原始尺度
lower_bound = mu + z_lower * sigma
upper_bound = mu + z_upper * sigma

print(f'\nOriginal Scale:')
print(f'Lower Bound: μ + z_lower × σ = {mu} + {z_lower:.4f} × {sigma} = {lower_bound:.4f}')
print(f'Upper Bound: μ + z_upper × σ = {mu} + {z_upper:.4f} × {sigma} = {upper_bound:.4f}')
print(f'\nMiddle 95% Range: [{lower_bound:.4f}, {upper_bound:.4f}]')

# Verify by calculating CDF at these points / 通过在这些点计算CDF来验证
prob_below_upper = norm.cdf(upper_bound, loc=mu, scale=sigma)
prob_below_lower = norm.cdf(lower_bound, loc=mu, scale=sigma)
prob_in_range = prob_below_upper - prob_below_lower

print(f'\nVerification using CDF:')
print(f'P(X <= {lower_bound:.4f}) = {prob_below_lower:.6f}')
print(f'P(X <= {upper_bound:.4f}) = {prob_below_upper:.6f}')
print(f'P({lower_bound:.4f} <= X <= {upper_bound:.4f}) = {prob_in_range:.6f}')

# Calculate confidence intervals for different levels / 计算不同水平的置信区间
confidence_levels = [0.68, 0.90, 0.95, 0.99]

print(f'\nConfidence Intervals at Different Levels:')
print(f'=' * 70)
print(f'{"Confidence":>12s} {"Lower Bound":>15s} {"Upper Bound":>15s} {"Width":>12s}')
print('-' * 70)

for conf in confidence_levels:
    alpha = 1 - conf
    lower_p = alpha / 2
    upper_p = 1 - alpha / 2
    
    z_l = norm.ppf(lower_p)
    z_u = norm.ppf(upper_p)
    
    lower = mu + z_l * sigma
    upper = mu + z_u * sigma
    width = upper - lower
    
    print(f'{conf*100:10.0f}% [{lower:13.4f}, {upper:13.4f}] {width:12.4f}')

# Create visualization / 创建可视化
fig, ax = plt.subplots(figsize=(12, 6))

x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = norm.pdf(x, loc=mu, scale=sigma)
ax.plot(x, pdf, 'b-', linewidth=2, label='PDF')

# Highlight 95% interval / 高亮95%区间
x_95 = np.linspace(lower_bound, upper_bound, 100)
pdf_95 = norm.pdf(x_95, loc=mu, scale=sigma)
ax.fill_between(x_95, pdf_95, alpha=0.6, color='green', label='95% CI')

# Add vertical lines for bounds / 添加边界的竖线
ax.axvline(lower_bound, color='r', linestyle='--', linewidth=2, 
           label=f'Lower: {lower_bound:.2f}')
ax.axvline(upper_bound, color='r', linestyle='--', linewidth=2, 
           label=f'Upper: {upper_bound:.2f}')
ax.axvline(mu, color='black', linestyle='-', linewidth=1.5, 
           label=f'Mean: {mu}')

# Shade the tail regions / 给尾部区域着色
x_left = x[x < lower_bound]
pdf_left = norm.pdf(x_left, loc=mu, scale=sigma)
ax.fill_between(x_left, pdf_left, alpha=0.3, color='red', label='Tails (5%)')

x_right = x[x > upper_bound]
pdf_right = norm.pdf(x_right, loc=mu, scale=sigma)
ax.fill_between(x_right, pdf_right, alpha=0.3, color='red')

ax.set_xlabel('Value / 值')
ax.set_ylabel('Probability Density / 概率密度')
ax.set_title(f'95% Confidence Interval: N(μ={mu}, σ={sigma})')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### Chapter Summary

# Chapter 9: Continuous Distributions

## Overview
This chapter explores **continuous probability distributions**, moving from the most common (Normal) to specialized distributions (Exponential, Pareto). Each distribution is motivated by real-world scenarios.

## Key Concepts
- **Probability Density Function (PDF)**: Height of the probability curve (not probability itself)
- **Cumulative Distribution Function (CDF)**: Probability up to a given value
- **Normal Distribution**: Bell curve, most common in nature and measurement
- **Exponential Distribution**: Time between rare events (memoryless process)
- **Pareto Distribution**: Heavy-tailed, "80-20 rule" distribution

## Evolution of Examples

### Normal Distribution (Most Common)
1. **01_normal_sample.py**: Generate samples from Normal distribution
2. **02_normal_plot.py**: Plot PDF and CDF
3. **03_normal_middle.py**: Calculate middle 95% (±2σ from mean)

### Exponential Distribution (Waiting Times)
4. **04_exponential_sample.py**: Generate samples from Exponential
5. **05_exponential_plot.py**: Plot PDF and CDF

### Pareto Distribution (Heavy Tails)
6. **06_pareto_sample.py**: Generate samples from Pareto
7. **07_pareto_plot.py**: Plot PDF and CDF

## Logic Flow
**Most Common (Normal) → Waiting Times (Exponential) → Heavy Tails (Pareto)**

## Key Characteristics

### Normal Distribution
- Symmetric bell curve
- Central Limit Theorem: averages of samples approach Normal
- 68% within ±1σ, 95% within ±2σ, 99.7% within ±3σ

### Exponential Distribution
- Right-skewed, memoryless property
- Models: lifetime of components, waiting times for rare events
- Single parameter λ controls both mean and variance

### Pareto Distribution
- Heavy right tail, power-law decay
- Models: wealth distribution, city sizes, web traffic
- "80-20 rule": 20% of people own 80% of wealth

## Key Takeaways
1. Different distributions model different real-world phenomena
2. PDF and CDF are complementary views of the same distribution
3. Heavy-tailed distributions violate many assumptions of Normal-based methods
4. Choosing the right distribution is crucial for modeling and inference

---
