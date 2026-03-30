# 统计方法与机器学习 / Statistical Methods for Machine Learning
## Chapter 20

---

### Tolerance

# 20 — Tolerance Intervals / 容差区间

**Chapter 20 — File 1 of 2**

## Summary / 摘要

A tolerance interval is a statistical range that contains a specified proportion of the population with a given confidence level. Unlike confidence intervals (which estimate parameters) and prediction intervals (which estimate single future values), tolerance intervals bound the range where a proportion p of the population is expected to fall with confidence level (1-alpha). For normally distributed data with 95% coverage and 99% confidence, the interval uses chi-square and normal critical values.

容差区间是一个统计范围，它以给定的置信水平包含总体的指定比例。与置信区间（估计参数）和预测区间（估计单个未来值）不同，容差区间界定了其中总体比例p预计以置信水平(1-alpha)落在的范围。对于正态分布的数据，具有95%覆盖率和99%置信度，区间使用卡方和正态临界值。

## Step 1 — Import Libraries / 导入库

```python
# Import required libraries
# 导入所需库
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Set random seed for reproducibility
# 设置随机种子以保证可重复性
np.random.seed(42)
```

## Step 2 — Generate Normal Distribution Data / 生成正态分布数据

```python
# Generate data from normal distribution
# 从正态分布生成数据
mu = 100  # Mean / 均值
sigma = 15  # Standard deviation / 标准差
n = 50  # Sample size / 样本大小

data = np.random.normal(mu, sigma, n)

# Calculate sample statistics
# 计算样本统计量
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)  # Use unbiased estimator / 使用无偏估计

# Display data properties
# 显示数据属性
print(f"Sample size: {n}")
print(f"Sample mean: {sample_mean:.2f}")
print(f"Sample std: {sample_std:.2f}")
print(f"Min value: {np.min(data):.2f}")
print(f"Max value: {np.max(data):.2f}")
```

## Step 3 — Define Tolerance Interval Parameters / 定义容差区间参数

```python
# Tolerance interval parameters
# 容差区间参数
coverage = 0.95  # Proportion of population to cover / 要覆盖的总体比例
confidence = 0.99  # Confidence level / 置信水平
alpha = 1 - confidence  # Significance level / 显著性水平

# Compute critical values
# 计算临界值
# z_coverage: normal quantile for coverage level
# z_coverage: 覆盖水平的正态分位数
z_coverage = stats.norm.ppf((1 + coverage) / 2)

# chi2_val: chi-square quantile for confidence level
# chi2_val: 置信水平的卡方分位数
chi2_val = stats.chi2.ppf(1 - alpha, df=n-1)

# k_factor: correction factor combining both values
# k_factor: 组合两个值的修正因子
k_factor = np.sqrt((n - 1) * (1 + 1/n) * chi2_val / stats.chi2.ppf(coverage, df=1))

print(f"Coverage: {coverage*100:.1f}%")
print(f"Confidence: {confidence*100:.1f}%")
print(f"z_coverage: {z_coverage:.4f}")
print(f"chi2_val: {chi2_val:.4f}")
print(f"k_factor: {k_factor:.4f}")
```

## Step 4 — Calculate Tolerance Interval / 计算容差区间

```python
# Calculate margin of error using k-factor and sample standard deviation
# 使用k因子和样本标准差计算误差范围
margin = z_coverage * sample_std * np.sqrt(1 + 1/n)

# Calculate tolerance interval bounds
# 计算容差区间边界
lower_bound = sample_mean - margin
upper_bound = sample_mean + margin

# Display tolerance interval
# 显示容差区间
print(f"\nTolerance Interval:")
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")
print(f"Interval width: {upper_bound - lower_bound:.2f}")
print(f"\nInterpretation: We are {confidence*100:.0f}% confident that ")
print(f"{coverage*100:.0f}% of the population lies between {lower_bound:.2f} and {upper_bound:.2f}")
```

## Step 5 — Visualize Data and Interval / 可视化数据和区间

```python
# Create visualization
# 创建可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Plot 1: Histogram with tolerance interval
# 图1: 直方图与容差区间
axes[0].hist(data, bins=15, edgecolor='black', alpha=0.7, density=True)

# Overlay normal distribution curve
# 覆盖正态分布曲线
x = np.linspace(data.min() - 10, data.max() + 10, 100)
axes[0].plot(x, stats.norm.pdf(x, sample_mean, sample_std), 'r-', linewidth=2, label='Normal fit')

# Add tolerance interval lines
# 添加容差区间线
axes[0].axvline(lower_bound, color='green', linestyle='--', linewidth=2, label='Tolerance interval')
axes[0].axvline(upper_bound, color='green', linestyle='--', linewidth=2)
axes[0].axvline(sample_mean, color='blue', linestyle='-', linewidth=2, label='Sample mean')

axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')
axes[0].set_title(f'Data Distribution with {coverage*100:.0f}% / {confidence*100:.0f}% Tolerance Interval')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Box plot with tolerance interval
# 图2: 箱线图与容差区间
axes[1].boxplot([data], vert=True, labels=['Data'])
axes[1].axhline(lower_bound, color='green', linestyle='--', linewidth=2, label='Tolerance interval')
axes[1].axhline(upper_bound, color='green', linestyle='--', linewidth=2)
axes[1].axhline(sample_mean, color='blue', linestyle='-', linewidth=2, label='Sample mean')

axes[1].set_ylabel('Value')
axes[1].set_title('Box Plot with Tolerance Interval')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept**: Tolerance intervals are wider than confidence intervals because they account for both estimation uncertainty (from sample variability) and the natural spread of the population. The k-factor multiplies the standard deviation to capture a specified population proportion at a chosen confidence level, combining chi-square and normal distributions.
  
  **统计概念**: 容差区间比置信区间更宽，因为它们考虑估计不确定性（来自样本变异性）和总体的自然分布。k因子将标准差相乘以在选定的置信水平处捕获指定的总体比例，结合了卡方和正态分布。

- **ML Application**: Tolerance intervals are critical in quality control, manufacturing process monitoring, and risk assessment. They define specification limits in production systems, ensure product compliance with standards, and help identify when processes drift outside acceptable ranges. Essential for setting operational thresholds in production ML systems.
  
  **ML应用**: 容差区间在质量控制、制造过程监控和风险评估中至关重要。它们定义了生产系统中的规格限制，确保产品符合标准，并帮助识别流程何时超出可接受范围。对于在生产ML系统中设置操作阈值至关重要。

➡️ **Next**: `02_tolerance_to_sample_size.ipynb`

## Complete Code / 完整代码一览

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)
mu, sigma, n = 100, 15, 50
data = np.random.normal(mu, sigma, n)

sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)

coverage, confidence = 0.95, 0.99
alpha = 1 - confidence

z_coverage = stats.norm.ppf((1 + coverage) / 2)
chi2_val = stats.chi2.ppf(1 - alpha, df=n-1)

margin = z_coverage * sample_std * np.sqrt(1 + 1/n)
lower_bound = sample_mean - margin
upper_bound = sample_mean + margin

print(f"Tolerance Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Width: {upper_bound - lower_bound:.2f}")
```

---

### Tolerance To Sample Size

# 20 — Tolerance Interval vs. Sample Size / 容差区间与样本大小

**Chapter 20 — File 2 of 2**

## Summary / 摘要

The width of a tolerance interval decreases as sample size increases, following an inverse square root relationship. With larger samples, the estimation uncertainty diminishes, resulting in tighter bounds. This notebook demonstrates how the tolerance interval width changes when varying sample sizes from 5 to 14 observations, showing the practical trade-off between data collection cost and desired precision.

容差区间的宽度随着样本大小的增加而减小，遵循倒平方根关系。样本越大，估计不确定性越小，导致边界越紧。此笔记本演示了当样本大小从5到14个观察值变化时，容差区间宽度如何变化，显示了数据收集成本和所需精度之间的实际权衡。

## Step 1 — Import Libraries / 导入库

```python
# Import required libraries
# 导入所需库
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Set random seed for reproducibility
# 设置随机种子以保证可重复性
np.random.seed(42)
```

## Step 2 — Define Parameters and Sample Sizes / 定义参数和样本大小

```python
# Population parameters
# 总体参数
mu = 100
sigma = 15

# Tolerance interval parameters
# 容差区间参数
coverage = 0.95  # 95% coverage / 95%覆盖
confidence = 0.99  # 99% confidence / 99%置信
alpha = 1 - confidence

# Critical values (same for all sample sizes)
# 临界值（对所有样本大小相同）
z_coverage = stats.norm.ppf((1 + coverage) / 2)

# Range of sample sizes to test
# 要测试的样本大小范围
sample_sizes = np.arange(5, 15)  # n=5 to n=14 / n=5到n=14

print(f"Population mean: {mu}")
print(f"Population std: {sigma}")
print(f"Coverage: {coverage*100:.0f}%, Confidence: {confidence*100:.0f}%")
print(f"Sample sizes to evaluate: {sample_sizes}")
```

## Step 3 — Calculate Tolerance Intervals for Different Sample Sizes / 为不同样本大小计算容差区间

```python
# Store results
# 存储结果
interval_widths = []
lower_bounds = []
upper_bounds = []

# Calculate tolerance interval for each sample size
# 为每个样本大小计算容差区间
for n in sample_sizes:
    # Generate sample data
    # 生成样本数据
    sample_data = np.random.normal(mu, sigma, n)
    
    # Calculate sample statistics
    # 计算样本统计量
    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)
    
    # Get chi-square critical value
    # 获取卡方临界值
    chi2_val = stats.chi2.ppf(1 - alpha, df=n-1)
    
    # Calculate margin of error
    # 计算误差范围
    margin = z_coverage * sigma * np.sqrt(1 + 1/n)  # Use population sigma for consistency
                                                      # 使用总体标准差以保持一致性
    
    # Calculate bounds
    # 计算边界
    lower = mu - margin
    upper = mu + margin
    width = upper - lower
    
    # Store results
    # 存储结果
    interval_widths.append(width)
    lower_bounds.append(lower)
    upper_bounds.append(upper)
    
    print(f"n={n:2d}: Width={width:.2f}, Bounds=[{lower:.2f}, {upper:.2f}]")
```

```python
# Convert to arrays for plotting
# 转换为数组以进行绘制
interval_widths = np.array(interval_widths)
lower_bounds = np.array(lower_bounds)
upper_bounds = np.array(upper_bounds)
```

## Step 4 — Plot Tolerance Interval Width vs. Sample Size / 绘制容差区间宽度对样本大小

```python
# Create comprehensive visualization
# 创建综合可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Tolerance interval width vs. sample size
# 图1: 容差区间宽度对样本大小
axes[0, 0].plot(sample_sizes, interval_widths, marker='o', linewidth=2, markersize=8, color='blue')
axes[0, 0].set_xlabel('Sample Size (n)')
axes[0, 0].set_ylabel('Interval Width')
axes[0, 0].set_title('Tolerance Interval Width vs. Sample Size')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(sample_sizes)

# Add annotation for trend
# 为趋势添加注释
axes[0, 0].annotate('Decreasing trend\n(Inverse sqrt relationship)', 
                    xy=(sample_sizes[-1], interval_widths[-1]), 
                    xytext=(sample_sizes[-3], interval_widths[-3] + 1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10)

# Plot 2: Log scale to show inverse sqrt relationship
# 图2: 对数刻度显示倒平方根关系
axes[0, 1].loglog(sample_sizes, interval_widths, marker='s', linewidth=2, markersize=8, color='green')
axes[0, 1].set_xlabel('Sample Size (n) - log scale')
axes[0, 1].set_ylabel('Interval Width - log scale')
axes[0, 1].set_title('Log-Log Plot: Inverse Sqrt Relationship')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Tolerance intervals as error bars
# 图3: 容差区间作为误差条
axes[1, 0].errorbar(sample_sizes, np.ones_like(sample_sizes) * mu, 
                    yerr=interval_widths/2,
                    fmt='o', linewidth=2, markersize=8, capsize=5, color='purple')
axes[1, 0].axhline(mu, color='black', linestyle='-', linewidth=1)
axes[1, 0].set_xlabel('Sample Size (n)')
axes[1, 0].set_ylabel('Center (mean)')
axes[1, 0].set_title('Tolerance Intervals at Different Sample Sizes')
axes[1, 0].set_xticks(sample_sizes)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([95, 105])

# Plot 4: Rate of change in width
# 图4: 宽度变化率
width_reduction = (interval_widths[0] - interval_widths) / interval_widths[0] * 100
axes[1, 1].bar(sample_sizes, width_reduction, color='orange', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Sample Size (n)')
axes[1, 1].set_ylabel('Width Reduction (%)')
axes[1, 1].set_title('Cumulative Width Reduction from n=5')
axes[1, 1].set_xticks(sample_sizes)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

```python
# Summary statistics
# 摘要统计
print(f"\nTolerance Interval Width Summary:")
print(f"At n=5:  Width = {interval_widths[0]:.2f}")
print(f"At n=14: Width = {interval_widths[-1]:.2f}")
print(f"Reduction: {interval_widths[0] - interval_widths[-1]:.2f} ({(1 - interval_widths[-1]/interval_widths[0])*100:.1f}% decrease)")
print(f"\nWidth reduction is proportional to 1/sqrt(n):")
print(f"sqrt(5) = {np.sqrt(5):.2f}, sqrt(14) = {np.sqrt(14):.2f}")
print(f"Ratio: {np.sqrt(5)/np.sqrt(14):.2f} ≈ {interval_widths[0]/interval_widths[-1]:.2f}")
```

```python
# Create a table of values
# 创建值表
print(f"\nDetailed Results Table:")
print(f"{'n':>3} | {'Width':>8} | {'Lower':>8} | {'Upper':>8} | {'% Reduction':>12}")
print("-" * 50)
for i, n in enumerate(sample_sizes):
    reduction_pct = (1 - interval_widths[i]/interval_widths[0]) * 100
    print(f"{n:3d} | {interval_widths[i]:8.2f} | {lower_bounds[i]:8.2f} | {upper_bounds[i]:8.2f} | {reduction_pct:11.1f}%")
```

## Learning Notes / 学习笔记

- **Statistical Concept**: Tolerance interval width decreases with sample size following the relationship: width ∝ 1/√n. This inverse square root law means quadrupling the sample size only halves the interval width. The relationship comes from the standard error term √(1+1/n) in the margin calculation, demonstrating fundamental sampling variability principles.
  
  **统计概念**: 容差区间宽度随样本大小减小，遵循以下关系：宽度∝1/√n。这个倒平方根定律意味着将样本大小增加四倍只能将区间宽度减半。该关系来自裕度计算中的标准误差项√(1+1/n)，演示了基本的采样变异性原则。

- **ML Application**: Sample size planning is critical in production systems where tolerance intervals define operational specifications. Understanding the width-sample size tradeoff helps optimize data collection budgets in quality control, sensor calibration, and process validation. Larger samples provide tighter specifications but at higher cost, requiring careful cost-benefit analysis.
  
  **ML应用**: 样本大小规划在容差区间定义操作规格的生产系统中至关重要。理解宽度-样本大小权衡有助于优化质量控制、传感器校准和流程验证中的数据收集预算。较大的样本提供更紧的规格，但成本更高，需要仔细的成本效益分析。

➡️ **Next**: `../chapter_21/01_confidence_interval_50.ipynb`

## Complete Code / 完整代码一览

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)
mu, sigma = 100, 15
coverage, confidence = 0.95, 0.99
alpha = 1 - confidence
z_coverage = stats.norm.ppf((1 + coverage) / 2)

sample_sizes = np.arange(5, 15)
interval_widths = []

for n in sample_sizes:
    margin = z_coverage * sigma * np.sqrt(1 + 1/n)
    width = 2 * margin
    interval_widths.append(width)
    print(f"n={n}: Width={width:.2f}")

plt.plot(sample_sizes, interval_widths, marker='o', linewidth=2)
plt.xlabel('Sample Size (n)')
plt.ylabel('Interval Width')
plt.title('Tolerance Interval Width vs. Sample Size')
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 20: Tolerance Intervals
# 第20章：容差区间

## Theme | 主题
Population-level prediction: what range contains a specified proportion of the population?
总体水平预测：什么范围包含总体的指定比例？

## Evolution Roadmap | 演变路线图
```
Sample Data (observed)
└─ Tolerance Interval Computation
   (e.g., 95% TI: 95% of population lies in this interval with 95% confidence)
   └─ Effect of Sample Size on Interval Width
      (larger n → narrower TI)
```

## Progression Logic | 进度逻辑

### Stage 1: Motivation (动机)
**English:** Unlike CI (about sample mean) and PI (about one future observation), TI is about the population distribution: where do most future values lie?
**中文:** 与CI(关于样本均值)和PI(关于一个未来观察)不同，TI涉及总体分布：大多数未来值位于何处？

### Stage 2: Tolerance Interval Formula (容差区间公式)
**English:** For a normal distribution, TI = mean ± k * stdev, where k depends on sample size n, coverage (e.g., 95%), and confidence level (e.g., 95%).
**中文:** 对于正态分布，TI = 均值 ± k * 标准差，其中k取决于样本量n、覆盖率(例如95%)和置信水平(例如95%)。

### Stage 3: Sample Size Effect (样本量效应)
**English:** Plot TI width vs. sample size. Larger n → k decreases → narrower TI (more precise knowledge of population).
**中文:** 绘制TI宽度与样本量。较大的n → k减少 → TI较窄(更精确的总体知识)。

## ML Relevance | ML相关性

1. **Process Control (过程控制)**: TI defines acceptable range for manufacturing or quality metrics.
2. **Prediction Bounds (预测界)**: TI differs from PI: PI predicts one future value; TI bounds future realizations of the population.
3. **Risk Assessment (风险评估)**: TI quantifies coverage of extreme values, useful for financial and safety-critical applications.
4. **Specification Limits (规格限)**: Compare TI with engineering specifications to assess process capability.


---
