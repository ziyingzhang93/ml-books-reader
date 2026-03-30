# 概率论与机器学习
## Chapter 08

---

### Binomial Moments

# 02 — Binomial Moments / 二项分布矩

**Chapter 08 — File 2 of 6**

## Summary

We calculate the mean and variance (moments) of the binomial distribution using scipy.stats.binom.stats(). The first moment is the mean (expected value), and the second moment is the variance.

我们使用scipy.stats.binom.stats()计算二项分布的均值和方差（矩）。第一矩是均值（期望值），第二矩是方差。

**Formula:**

$$E[X] = np$$

$$\text{Var}[X] = np(1-p)$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import necessary libraries / 导入必要的库
from scipy.stats import binom
import numpy as np
```

## Step 2 — Define Parameters / 定义参数

```python
# Define binomial parameters / 定义二项分布参数
n = 100  # Number of trials / 试验次数
p = 0.3  # Probability of success / 成功概率
```

## Step 3 — Calculate Moments / 计算矩

```python
# Using scipy.stats.binom.stats() / 使用scipy.stats.binom.stats()
# Returns mean, variance, skewness, kurtosis
# 返回均值、方差、偏度、峰度
mean, var, skew, kurt = binom.stats(n=n, p=p, moments='mvsk')

print(f'Binomial Distribution: n={n}, p={p}')
print(f'二项分布: n={n}, p={p}')
print(f'\nMean (First Moment) / 均值（第一矩): {mean:.4f}')
print(f'Expected from formula E[X]=np / 公式预期值E[X]=np: {n*p:.4f}')
print(f'\nVariance (Second Moment) / 方差（第二矩): {var:.4f}')
print(f'Expected from formula Var[X]=np(1-p) / 公式预期值Var[X]=np(1-p): {n*p*(1-p):.4f}')
print(f'\nSkewness / 偏度: {skew:.4f}')
print(f'Kurtosis / 峰度: {kurt:.4f}')
```

## Step 4 — Compare Multiple Distributions / 比较多个分布

```python
# Compare moments for different parameters / 比较不同参数的矩
print('\nComparison of Different Binomial Distributions / 不同二项分布的比较')
print('=' * 80)

# Test different values of n and p / 测试不同的n和p值
test_params = [(10, 0.5), (20, 0.3), (50, 0.5), (100, 0.2), (100, 0.8)]

for n_test, p_test in test_params:
    mean_test, var_test, _, _ = binom.stats(n=n_test, p=p_test, moments='mvsk')
    std_test = np.sqrt(var_test)
    print(f'n={n_test:3d}, p={p_test:.1f}: Mean={mean_test:6.2f}, Var={var_test:6.2f}, Std={std_test:6.2f}')
```

## Step 5 — Standard Deviation Analysis / 标准差分析

```python
# Calculate standard deviation / 计算标准差
std = np.sqrt(var)

print(f'\nStandard Deviation / 标准差')
print(f'Std Dev = sqrt(Var) = sqrt({var:.4f}) = {std:.4f}')
print(f'标准差 = sqrt(方差) = sqrt({var:.4f}) = {std:.4f}')
print(f'\nFor Confidence Intervals / 对于置信区间:')
print(f'68% within μ ± 1σ: [{mean-std:.2f}, {mean+std:.2f}]')
print(f'95% within μ ± 2σ: [{mean-2*std:.2f}, {mean+2*std:.2f}]')
```

## Learning Notes / 学习笔记

- **Concept**: The moments of a distribution (mean, variance, skewness, kurtosis) fully characterize its shape and behavior. For binomial distribution, mean grows linearly with n, and variance grows with both n and p(1-p).
  
  **概念**：分布的矩（均值、方差、偏度、峰度）完全刻画其形状和行为。对于二项分布，均值随n线性增长，方差随n和p(1-p)增长。

- **ML Application**: Understanding moments is crucial for parameter estimation, constructing confidence intervals, and predicting the reliability of estimators in machine learning models.
  
  **机器学习应用**：理解矩对于参数估计、构造置信区间和预测机器学习模型中估计器的可靠性至关重要。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

## Next / 下一步

➡️ **Next**: `03_binomial_mass.ipynb`

## Complete Code / 完整代码一览

```python
# Complete Binomial Moments Analysis / 完整二项分布矩分析

from scipy.stats import binom
import numpy as np

# Define binomial parameters / 定义二项分布参数
n = 100  # Number of trials / 试验次数
p = 0.3  # Probability of success / 成功概率

# Using scipy.stats.binom.stats() / 使用scipy.stats.binom.stats()
mean, var, skew, kurt = binom.stats(n=n, p=p, moments='mvsk')

print(f'Binomial Distribution: n={n}, p={p}')
print(f'二项分布: n={n}, p={p}')
print(f'\nMean (First Moment) / 均值（第一矩): {mean:.4f}')
print(f'Expected from formula E[X]=np / 公式预期值E[X]=np: {n*p:.4f}')
print(f'\nVariance (Second Moment) / 方差（第二矩): {var:.4f}')
print(f'Expected from formula Var[X]=np(1-p) / 公式预期值Var[X]=np(1-p): {n*p*(1-p):.4f}')
print(f'\nSkewness / 偏度: {skew:.4f}')
print(f'Kurtosis / 峰度: {kurt:.4f}')

# Compare moments for different parameters / 比较不同参数的矩
print('\nComparison of Different Binomial Distributions / 不同二项分布的比较')
print('=' * 80)

test_params = [(10, 0.5), (20, 0.3), (50, 0.5), (100, 0.2), (100, 0.8)]

for n_test, p_test in test_params:
    mean_test, var_test, _, _ = binom.stats(n=n_test, p=p_test, moments='mvsk')
    std_test = np.sqrt(var_test)
    print(f'n={n_test:3d}, p={p_test:.1f}: Mean={mean_test:6.2f}, Var={var_test:6.2f}, Std={std_test:6.2f}')

# Calculate standard deviation / 计算标准差
std = np.sqrt(var)

print(f'\nStandard Deviation / 标准差')
print(f'Std Dev = sqrt(Var) = sqrt({var:.4f}) = {std:.4f}')
print(f'\nFor Confidence Intervals / 对于置信区间:')
print(f'68% within μ ± 1σ: [{mean-std:.2f}, {mean+std:.2f}]')
print(f'95% within μ ± 2σ: [{mean-2*std:.2f}, {mean+2*std:.2f}]')
```

---

### Binomial Mass

# 03 — Binomial Mass Function / 二项分布质量函数

**Chapter 08 — File 3 of 6**

## Summary

We calculate the Probability Mass Function (PMF) of the binomial distribution using scipy.stats.binom.pmf(). We demonstrate how the PMF changes as we vary the number of trials n, showing the shift in distribution shape.

我们使用scipy.stats.binom.pmf()计算二项分布的概率质量函数（PMF）。我们演示改变试验次数n时PMF如何变化，展示分布形状的移动。

**Formula:**

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import necessary libraries / 导入必要的库
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
```

## Step 2 — Calculate PMF for Single Distribution / 计算单一分布的PMF

```python
# Define parameters / 定义参数
n = 10
p = 0.5

# Calculate PMF for all possible values / 计算所有可能值的PMF
# PMF(k) = P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
# PMF(k) = P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
x = np.arange(0, n+1)
pmf = binom.pmf(k=x, n=n, p=p)

print(f'Binomial PMF: n={n}, p={p}')
print(f'二项PMF: n={n}, p={p}')
print(f'\n{"k":>3s} {"P(X=k)":>10s}')
print('=' * 15)
for k, prob in zip(x, pmf):
    print(f'{k:3d} {prob:10.6f}')
```

## Step 3 — Visualize PMF for Different n Values / 可视化不同n值的PMF

```python
# Plot PMF for different values of n / 绘制不同n值的PMF
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
n_values = [10, 50, 100]
p = 0.5

for idx, n in enumerate(n_values):
    # Calculate PMF / 计算PMF
    x = np.arange(0, n+1)
    pmf = binom.pmf(k=x, n=n, p=p)
    
    # Plot bar chart / 绘制条形图
    ax = axes[idx]
    ax.bar(x, pmf, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Successes / 成功次数')
    ax.set_ylabel('Probability / 概率')
    ax.set_title(f'Binomial PMF (n={n}, p={p})')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Step 4 — Calculate Probability of Events / 计算事件概率

```python
# Calculate probabilities for specific events / 计算特定事件的概率
n = 20
p = 0.3

# P(X = 5) / P(X = 5)
prob_exact_5 = binom.pmf(k=5, n=n, p=p)
print(f'P(X = 5) with n={n}, p={p}: {prob_exact_5:.6f}')
print(f'n={n}, p={p}时，P(X = 5): {prob_exact_5:.6f}')

# P(X <= 5) / P(X <= 5)
prob_le_5 = sum([binom.pmf(k=k, n=n, p=p) for k in range(0, 6)])
print(f'\nP(X <= 5) with n={n}, p={p}: {prob_le_5:.6f}')
print(f'n={n}, p={p}时，P(X <= 5): {prob_le_5:.6f}')

# P(X > 5) / P(X > 5)
prob_gt_5 = 1 - prob_le_5
print(f'\nP(X > 5) with n={n}, p={p}: {prob_gt_5:.6f}')
print(f'n={n}, p={p}时，P(X > 5): {prob_gt_5:.6f}')
```

## Learning Notes / 学习笔记

- **Concept**: The PMF gives the exact probability of each outcome in a discrete distribution. As n increases, the binomial distribution becomes more concentrated around its mean and approximates a normal distribution (Central Limit Theorem).
  
  **概念**：PMF给出离散分布每个结果的精确概率。当n增加时，二项分布更加集中在其均值周围，并逼近正态分布（中心极限定理）。

- **ML Application**: PMF is essential for computing exact probabilities in classification models, evaluating model predictions, and understanding decision boundaries in probabilistic classifiers.
  
  **机器学习应用**：PMF对于计算分类模型中的精确概率、评估模型预测和理解概率分类器中的决策边界至关重要。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

## Next / 下一步

➡️ **Next**: `04_binomial_cumulative.ipynb`

## Complete Code / 完整代码一览

```python
# Complete Binomial PMF Analysis / 完整二项PMF分析

from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

# Define parameters / 定义参数
n = 10
p = 0.5

# Calculate PMF for all possible values / 计算所有可能值的PMF
x = np.arange(0, n+1)
pmf = binom.pmf(k=x, n=n, p=p)

print(f'Binomial PMF: n={n}, p={p}')
print(f'{"k":>3s} {"P(X=k)":>10s}')
print('=' * 15)
for k, prob in zip(x, pmf):
    print(f'{k:3d} {prob:10.6f}')

# Plot PMF for different values of n / 绘制不同n值的PMF
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
n_values = [10, 50, 100]
p = 0.5

for idx, n in enumerate(n_values):
    # Calculate PMF / 计算PMF
    x = np.arange(0, n+1)
    pmf = binom.pmf(k=x, n=n, p=p)
    
    # Plot bar chart / 绘制条形图
    ax = axes[idx]
    ax.bar(x, pmf, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Successes / 成功次数')
    ax.set_ylabel('Probability / 概率')
    ax.set_title(f'Binomial PMF (n={n}, p={p})')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate probabilities for specific events / 计算特定事件的概率
n = 20
p = 0.3

prob_exact_5 = binom.pmf(k=5, n=n, p=p)
print(f'\nP(X = 5) with n={n}, p={p}: {prob_exact_5:.6f}')

prob_le_5 = sum([binom.pmf(k=k, n=n, p=p) for k in range(0, 6)])
print(f'P(X <= 5) with n={n}, p={p}: {prob_le_5:.6f}')

prob_gt_5 = 1 - prob_le_5
print(f'P(X > 5) with n={n}, p={p}: {prob_gt_5:.6f}')
```

---

### Multinomial Mass

# 06 — Multinomial Mass Function / 多项分布质量函数

**Chapter 08 — File 6 of 6**

## Summary

We calculate the Probability Mass Function (PMF) of the multinomial distribution for a specific outcome [33, 33, 34]. The multinomial PMF gives the probability of observing exactly this combination of counts across the three categories.

我们计算特定结果[33, 33, 34]的多项分布的概率质量函数（PMF）。多项PMF给出在三个类别中观察到这种计数组合的精确概率。

**Formula:**

$$P(X_1 = x_1, X_2 = x_2, X_3 = x_3) = \frac{n!}{x_1! x_2! x_3!} p_1^{x_1} p_2^{x_2} p_3^{x_3}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import necessary libraries / 导入必要的库
import numpy as np
from scipy.special import factorial, multinomial
import itertools
```

## Step 2 — Implement Multinomial PMF / 实现多项PMF

```python
# Define multinomial PMF function / 定义多项PMF函数
def multinomial_pmf(x, p):
    """
    Calculate multinomial PMF: P(X = x)
    计算多项PMF: P(X = x)
    
    Args:
        x: array of counts (e.g., [33, 33, 34]) / 计数数组
        p: array of probabilities (e.g., [1/3, 1/3, 1/3]) / 概率数组
    
    Returns:
        Probability P(X_1 = x_1, X_2 = x_2, ...) / 概率
    """
    # n! / (x_1! * x_2! * ... * x_k!) / n! / (x_1! * x_2! * ... * x_k!)
    n = int(np.sum(x))
    multinomial_coefficient = factorial(n) / np.prod([factorial(xi) for xi in x])
    
    # p_1^x_1 * p_2^x_2 * ... * p_k^x_k / p_1^x_1 * p_2^x_2 * ... * p_k^x_k
    prob_term = np.prod([p[i]**x[i] for i in range(len(x))])
    
    return multinomial_coefficient * prob_term

print('Multinomial PMF Function Defined')
print('多项PMF函数已定义')
```

## Step 3 — Calculate PMF for Specific Outcome / 计算特定结果的PMF

```python
# Define parameters / 定义参数
n = 100                  # Total number of trials / 总试验次数
x = [33, 33, 34]         # Specific outcome / 特定结果
p = [1/3, 1/3, 1/3]      # Equal probabilities / 相等概率

# Calculate probability / 计算概率
prob = multinomial_pmf(x, p)

print(f'Multinomial PMF Calculation / 多项PMF计算')
print(f'=' * 60)
print(f'n = {n}, x = {x}, p = {p}')
print(f'\nP(X_1 = {x[0]}, X_2 = {x[1]}, X_3 = {x[2]}) = {prob:.10f}')
print(f'\nIn scientific notation / 科学记数法: {prob:.6e}')
```

## Step 4 — Calculate PMF for Multiple Outcomes / 计算多个结果的PMF

```python
# Calculate PMF for nearby outcomes / 计算相邻结果的PMF
outcomes = [
    [30, 30, 40],
    [32, 33, 35],
    [33, 33, 34],
    [34, 33, 33],
    [35, 35, 30]
]

print('\nMultiple Outcomes / 多个结果:')
print('=' * 70)
print(f'{"Outcome":>20s} {"Probability":>20s} {"Log Probability":>20s}')
print('-' * 70)

for outcome in outcomes:
    prob = multinomial_pmf(outcome, p)
    log_prob = np.log(prob) if prob > 0 else -np.inf
    print(f'{str(outcome):>20s} {prob:20.10f} {log_prob:20.6f}')
```

## Step 5 — Compare with Simulation / 与模拟比较

```python
# Generate multinomial samples and compare / 生成多项样本并比较
num_samples = 100000
samples = np.random.multinomial(n=n, pvals=p, size=num_samples)

# Count occurrences of [33, 33, 34] / 计算[33, 33, 34]的出现次数
target = [33, 33, 34]
matches = np.sum(np.all(samples == target, axis=1))
empirical_prob = matches / num_samples

# Theoretical probability / 理论概率
theoretical_prob = multinomial_pmf(target, p)

print(f'\nComparison: Simulation vs Theory / 比较：模拟 vs 理论')
print(f'=' * 60)
print(f'Target outcome: {target}')
print(f'Empirical probability (from {num_samples:,} samples): {empirical_prob:.10f}')
print(f'Theoretical probability: {theoretical_prob:.10f}')
print(f'Difference: {abs(empirical_prob - theoretical_prob):.10f}')
```

## Learning Notes / 学习笔记

- **Concept**: The multinomial PMF extends the binomial to multiple categories. Unlike the binomial (which sums to 1 over all single-category outcomes), the multinomial PMF defines probabilities over all possible combinations of counts that sum to n.
  
  **概念**：多项PMF将二项分布扩展到多个类别。与二项分布（对所有单一类别结果求和为1）不同，多项PMF定义了所有计数组合（和为n）的概率。

- **ML Application**: Multinomial PMF is used in multinomial logistic regression, naive Bayes classifiers (for categorical features), and generative models that need to compute exact probabilities of observed data compositions.
  
  **机器学习应用**：多项PMF用于多项逻辑回归、朴素贝叶斯分类器（对于分类特征）和需要计算观测数据组成精确概率的生成模型。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |

## Complete Code / 完整代码一览

```python
# Complete Multinomial PMF Analysis / 完整多项PMF分析

import numpy as np
from scipy.special import factorial

# Define multinomial PMF function / 定义多项PMF函数
def multinomial_pmf(x, p):
    """
    Calculate multinomial PMF: P(X = x)
    
    Args:
        x: array of counts / 计数数组
        p: array of probabilities / 概率数组
    
    Returns:
        Probability P(X_1 = x_1, X_2 = x_2, ...) / 概率
    """
    n = int(np.sum(x))
    multinomial_coefficient = factorial(n) / np.prod([factorial(xi) for xi in x])
    prob_term = np.prod([p[i]**x[i] for i in range(len(x))])
    return multinomial_coefficient * prob_term

# Define parameters / 定义参数
n = 100                  # Total number of trials
x = [33, 33, 34]         # Specific outcome
p = [1/3, 1/3, 1/3]      # Equal probabilities

# Calculate probability / 计算概率
prob = multinomial_pmf(x, p)

print(f'Multinomial PMF Calculation')
print(f'n = {n}, x = {x}, p = {p}')
print(f'\nP(X_1 = {x[0]}, X_2 = {x[1]}, X_3 = {x[2]}) = {prob:.10f}')
print(f'In scientific notation: {prob:.6e}')

# Calculate PMF for multiple outcomes / 计算多个结果的PMF
outcomes = [
    [30, 30, 40],
    [32, 33, 35],
    [33, 33, 34],
    [34, 33, 33],
    [35, 35, 30]
]

print('\nMultiple Outcomes:')
print('=' * 70)
print(f'{"Outcome":>20s} {"Probability":>20s} {"Log Probability":>20s}')
print('-' * 70)

for outcome in outcomes:
    prob = multinomial_pmf(outcome, p)
    log_prob = np.log(prob) if prob > 0 else -np.inf
    print(f'{str(outcome):>20s} {prob:20.10f} {log_prob:20.6f}')

# Compare with simulation / 与模拟比较
num_samples = 100000
samples = np.random.multinomial(n=n, pvals=p, size=num_samples)

target = [33, 33, 34]
matches = np.sum(np.all(samples == target, axis=1))
empirical_prob = matches / num_samples
theoretical_prob = multinomial_pmf(target, p)

print(f'\nComparison: Simulation vs Theory')
print(f'Target outcome: {target}')
print(f'Empirical probability (from {num_samples:,} samples): {empirical_prob:.10f}')
print(f'Theoretical probability: {theoretical_prob:.10f}')
print(f'Difference: {abs(empirical_prob - theoretical_prob):.10f}')
```

---
