# 统计方法与机器学习 / Statistical Methods for Machine Learning
## Chapter 11

---

### Gaussian

# 11.1 — Gaussian Critical Values / 高斯临界值



**Chapter 11 — File 1 of 3**



## Summary / 摘要



Critical values are essential for hypothesis testing and confidence intervals. The Percent Point Function (PPF), also called the inverse CDF or quantile function, computes the value at which a certain percentage of the distribution lies below.



临界值对于假设检验和置信区间至关重要。百分位点函数（PPF），也称为反向 CDF 或分位数函数，计算分布中某个百分比位于其下方的值。



**Mathematical Relationship / 数学关系:**



$$\text{PPF}(p) = F^{-1}(p) \text{ where } F(x) = P(X \leq x)$$



If $F(x_0) = p$, then $\text{PPF}(p) = x_0$.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
from scipy.stats import norm



# PPF 与 CDF 互为反函数

# PPF and CDF are inverse functions of each other
```

## Step 2 — Compute Gaussian Critical Values / 计算高斯临界值

```python
# 对于 α = 0.05 的显著性水平，单尾检验

# For significance level α = 0.05, one-tailed test

alpha = 0.05



# 计算 95% 分位数（单尾）

# Compute 95% quantile (one-tailed)

critical_value = norm.ppf(1 - alpha)



# 打印输出 / Print output
print(f"Significance level (α): {alpha}")

# 打印输出 / Print output
print(f"Confidence level: {1 - alpha}")

# 打印输出 / Print output
print(f"Critical value (one-tailed, Gaussian): {critical_value:.6f}")
```

## Step 3 — Verify with CDF / 用CDF验证

```python
# 使用 CDF 验证：PPF 和 CDF 互为反函数

# Verify using CDF: PPF and CDF are inverses

verification_cdf = norm.cdf(critical_value)



# 打印输出 / Print output
print(f"\nVerification / 验证:")

# 打印输出 / Print output
print(f"PPF({1 - alpha}) = {critical_value:.6f}")

# 打印输出 / Print output
print(f"CDF({critical_value:.6f}) = {verification_cdf:.6f}")

# 打印输出 / Print output
print(f"Difference: {abs(verification_cdf - (1 - alpha)):.10f}")

# 打印输出 / Print output
print("\nThe CDF value should equal 1 - α = 0.95")

```

## Step 4 — Common Critical Values / 常见临界值

## Step 4 — Common Critical Values / 常见临界值

```python
# 常见的显著性水平和置信水平

# Common significance levels and confidence levels

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd



# 创建关键值的表格

# Create table of critical values

data = []



for alpha in [0.10, 0.05, 0.01, 0.001]:

    # 单尾检验

    # One-tailed test

    one_tail = norm.ppf(1 - alpha)

    

    # 双尾检验

    # Two-tailed test

    alpha_half = alpha / 2

    two_tail = norm.ppf(1 - alpha_half)

    

    # 添加元素到列表末尾 / Append element to list end
    data.append({

        'α': alpha,

        'Confidence': f"{(1-alpha)*100:.1f}%",

        'One-tailed': f"{one_tail:.6f}",

        'Two-tailed': f"{two_tail:.6f}"

    })



df = pd.DataFrame(data)

# 打印输出 / Print output
print("\nGaussian Critical Values / 高斯临界值:")

# 打印输出 / Print output
print(df.to_string(index=False))
```

## Step 5 — Visualize PPF / 可视化PPF

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange, linspace

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot



# 绘制 PPF 函数

# Plot the PPF function

probabilities = linspace(0.001, 0.999, 500)

ppf_values = norm.ppf(probabilities)



pyplot.figure(figsize=(12, 5))



# 左图：PPF 函数

# Left plot: PPF function

pyplot.subplot(1, 2, 1)

pyplot.plot(probabilities, ppf_values, linewidth=2)

pyplot.axhline(y=norm.ppf(0.95), color='red', linestyle='--', alpha=0.7, label='95% quantile')

pyplot.axvline(x=0.95, color='red', linestyle='--', alpha=0.7)

pyplot.xlabel('Probability / 概率')

pyplot.ylabel('Value / 值')

pyplot.title('Gaussian PPF (Percent Point Function) / 高斯PPF')

pyplot.grid(True, alpha=0.3)

pyplot.legend()



# 右图：PPF 和 CDF 的关系

# Right plot: Relationship between PPF and CDF

sample_space = linspace(-4, 4, 300)

cdf = norm.cdf(sample_space)



pyplot.subplot(1, 2, 2)

pyplot.plot(sample_space, cdf, linewidth=2, label='CDF')

critical_val = norm.ppf(0.95)

pyplot.axvline(x=critical_val, color='red', linestyle='--', alpha=0.7, label='PPF(0.95)')

pyplot.axhline(y=0.95, color='red', linestyle='--', alpha=0.7)

pyplot.scatter([critical_val], [0.95], color='red', s=100, zorder=5)

pyplot.xlabel('Value / 值')

pyplot.ylabel('Cumulative Probability / 累积概率')

pyplot.title('Gaussian CDF (Inverse Relationship) / 高斯CDF')

pyplot.grid(True, alpha=0.3)

pyplot.legend()



pyplot.tight_layout()

pyplot.show()
```

## Learning Notes / 学习笔记



- **Statistical Concept / 统计学概念**: The PPF (Percent Point Function) is the inverse of the CDF. It answers the question: "What value has p% of the distribution below it?" Critical values in hypothesis testing are derived from the PPF for a given significance level. / PPF（百分位点函数）是 CDF 的倒数。它回答问题："什么值下有 p% 的分布？" 假设检验中的临界值是针对给定的显著性水平从 PPF 导出的。



- **ML Application / 机器学习应用**: PPF is used to construct confidence intervals, compute prediction intervals, and determine threshold values in anomaly detection. It's essential in calibration of probabilistic models and computing quantile regression. / PPF 用于构造置信区间、计算预测区间以及确定异常检测中的阈值。它对于概率模型的校准和计算分位数回归至关重要。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

➡️ **Next**: `02_t_distribution.ipynb`

## Complete Code / 完整代码一览

```python
from scipy.stats import norm

# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import linspace

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd



# Compute critical value

alpha = 0.05

critical_value = norm.ppf(1 - alpha)

# 打印输出 / Print output
print(f"Critical value (α={alpha}, one-tailed): {critical_value:.6f}")



# Verify with CDF

# 打印输出 / Print output
print(f"Verification: CDF({critical_value:.6f}) = {norm.cdf(critical_value):.6f}")



# Create table of critical values

data = []

for alpha in [0.10, 0.05, 0.01, 0.001]:

    one_tail = norm.ppf(1 - alpha)

    two_tail = norm.ppf(1 - alpha/2)

    # 添加元素到列表末尾 / Append element to list end
    data.append({'α': alpha, 'One-tailed': f"{one_tail:.4f}", 'Two-tailed': f"{two_tail:.4f}"})

# 打印输出 / Print output
print("\nCritical Values:")

# 打印输出 / Print output
print(pd.DataFrame(data).to_string(index=False))



# Plot PPF

probabilities = linspace(0.001, 0.999, 500)

ppf_values = norm.ppf(probabilities)

pyplot.plot(probabilities, ppf_values, linewidth=2)

pyplot.xlabel('Probability')

pyplot.ylabel('Value')

pyplot.title('Gaussian PPF')

pyplot.grid(True, alpha=0.3)

pyplot.show()
```

---

### T Distribution

# 11.2 — Student's t Critical Values / Student t 临界值



**Chapter 11 — File 2 of 3**



## Summary / 摘要



Student's t critical values are used when the population standard deviation is unknown. The critical values depend on both the significance level and the degrees of freedom. As degrees of freedom increase, t critical values approach Gaussian critical values.



当总体标准差未知时，使用 Student t 临界值。临界值取决于显著性水平和自由度。随着自由度增加，t 临界值接近高斯临界值。



**Key Relationship / 关键关系:**



$$t_{critical}(\alpha, df) = t.\text{ppf}(1 - \alpha/2, df) \text{ (two-tailed)}$$



As $df \to \infty$, $t_{critical} \to z_{critical}$ (Gaussian value)

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
from scipy.stats import t, norm

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd



# t 分布 PPF 取决于自由度

# t-distribution PPF depends on degrees of freedom
```

## Step 2 — Compute t Critical Values / 计算 t 临界值

```python
# 对于 α = 0.05，df = 10

# For α = 0.05, df = 10

alpha = 0.05

df = 10



# 计算 t 临界值（双尾）

# Compute t critical value (two-tailed)

alpha_half = alpha / 2

t_critical = t.ppf(1 - alpha_half, df)



# 为了比较，计算高斯临界值

# For comparison, compute Gaussian critical value

z_critical = norm.ppf(1 - alpha_half)



# 打印输出 / Print output
print(f"Significance level (α): {alpha}")

# 打印输出 / Print output
print(f"Degrees of freedom: {df}")

# 打印输出 / Print output
print(f"\nt-critical value (two-tailed): {t_critical:.6f}")

# 打印输出 / Print output
print(f"z-critical value (Gaussian): {z_critical:.6f}")

# 打印输出 / Print output
print(f"Difference: {abs(t_critical - z_critical):.6f}")

# 打印输出 / Print output
print("\nNote: t is more extreme (further from 0) due to heavier tails")
```

## Step 3 — Verify with CDF / 用CDF验证

```python
# 使用 CDF 验证

# Verify using CDF

verification = t.cdf(t_critical, df)



# 打印输出 / Print output
print(f"\nVerification / 验证:")

# 打印输出 / Print output
print(f"PPF({1 - alpha_half}, df={df}) = {t_critical:.6f}")

# 打印输出 / Print output
print(f"CDF({t_critical:.6f}, df={df}) = {verification:.6f}")

# 打印输出 / Print output
print(f"Expected: {1 - alpha_half:.6f}")

# 打印输出 / Print output
print(f"Difference: {abs(verification - (1 - alpha_half)):.10f}")
```

## Step 4 — Critical Values for Different Degrees of Freedom / 不同自由度的临界值

```python
# 常见的自由度

# Common degrees of freedom

dof_values = [1, 2, 5, 10, 30, 60, 120, float('inf')]

alpha = 0.05

alpha_half = alpha / 2



data = []

for df_val in dof_values:

    if df_val == float('inf'):

        t_crit = norm.ppf(1 - alpha_half)

        df_display = '∞ (Gaussian)'

    else:

        t_crit = t.ppf(1 - alpha_half, df_val)

        df_display = str(int(df_val))

    

    # 添加元素到列表末尾 / Append element to list end
    data.append({

        'Degrees of Freedom': df_display,

        'Critical Value': f"{t_crit:.6f}",

        'Difference from Gaussian': f"{abs(t_crit - norm.ppf(1 - alpha_half)):.6f}" if df_val != float('inf') else 'N/A'

    })



df_table = pd.DataFrame(data)

# 打印输出 / Print output
print(f"\nt-distribution Critical Values (α = {alpha}, two-tailed) / t分布临界值:")

# 打印输出 / Print output
print(df_table.to_string(index=False))
```

## Step 5 — Visualize Convergence to Gaussian / 可视化收敛到高斯

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import linspace

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot



# 绘制随自由度变化的临界值

# Plot critical values as function of degrees of freedom

dof_range = linspace(1, 100, 200)

t_criticals = [t.ppf(0.975, df_val) for df_val in dof_range]  # α=0.05, two-tailed

z_critical_val = norm.ppf(0.975)



pyplot.figure(figsize=(10, 6))

pyplot.plot(dof_range, t_criticals, linewidth=2, label='t-distribution critical values')

pyplot.axhline(y=z_critical_val, color='red', linestyle='--', linewidth=2, label='Gaussian critical value (limit)')

pyplot.xlabel('Degrees of Freedom / 自由度')

pyplot.ylabel('Critical Value / 临界值')

pyplot.title('Convergence of t Critical Values to Gaussian (α=0.05, two-tailed) / t临界值向高斯的收敛')

pyplot.legend()

pyplot.grid(True, alpha=0.3)

pyplot.show()



# 打印输出 / Print output
print(f"\nAs df increases, t critical values approach {z_critical_val:.6f}")
```

## Learning Notes / 学习笔记



- **Statistical Concept / 统计学概念**: t critical values are larger (more extreme) than Gaussian critical values because the t-distribution has heavier tails, accounting for uncertainty from estimating the standard deviation. This conservatism decreases as sample size increases. / t 临界值比高斯临界值更极端，因为 t 分布有更厚的尾部，考虑了从估计标准差的不确定性。随着样本量增加，这种保守性会降低。



- **ML Application / 机器学习应用**: t-tests compare group means when population variance is unknown. Construction of confidence intervals for regression coefficients, model parameters, and effect sizes relies on t critical values. t-tests are fundamental in A/B testing and experimental design validation. / t 检验在总体方差未知时比较组均值。回归系数、模型参数和效应量的置信区间的构造依赖于 t 临界值。t 检验在 A/B 检验和实验设计验证中是基础。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

➡️ **Next**: `03_chi_squared.ipynb`

## Complete Code / 完整代码一览

```python
from scipy.stats import t, norm

# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import linspace

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd



# Compute critical values

alpha = 0.05

df = 10

alpha_half = alpha / 2



t_critical = t.ppf(1 - alpha_half, df)

z_critical = norm.ppf(1 - alpha_half)



# 打印输出 / Print output
print(f"t-critical (df={df}): {t_critical:.6f}")

# 打印输出 / Print output
print(f"z-critical: {z_critical:.6f}")

# 打印输出 / Print output
print(f"Difference: {abs(t_critical - z_critical):.6f}")



# Create table

dof_values = [1, 2, 5, 10, 30, 120, float('inf')]

data = []

for df_val in dof_values:

    if df_val == float('inf'):

        crit = norm.ppf(1 - alpha_half)

        label = '∞'

    else:

        crit = t.ppf(1 - alpha_half, df_val)

        label = str(int(df_val))

    # 添加元素到列表末尾 / Append element to list end
    data.append({'df': label, 'Critical Value': f"{crit:.6f}"})



# 打印输出 / Print output
print("\nCritical Values Table:")

# 打印输出 / Print output
print(pd.DataFrame(data).to_string(index=False))



# Plot convergence

dof_range = linspace(1, 100, 200)

t_criticals = [t.ppf(0.975, d) for d in dof_range]

pyplot.plot(dof_range, t_criticals, linewidth=2, label='t-distribution')

pyplot.axhline(y=z_critical, color='red', linestyle='--', label='Gaussian')

pyplot.xlabel('Degrees of Freedom')

pyplot.ylabel('Critical Value')

pyplot.title('Convergence to Gaussian')

pyplot.legend()

pyplot.grid(True, alpha=0.3)

pyplot.show()
```

---

### Chi Squared

# 11.3 — Chi-Squared Critical Values / 卡方临界值



**Chapter 11 — File 3 of 3**



## Summary / 摘要



Chi-squared critical values are used in goodness-of-fit tests, tests of independence, and variance tests. Unlike the Gaussian and t-distributions, the chi-squared distribution is always positive and its critical values increase with degrees of freedom.



卡方临界值用于拟合优度检验、独立性检验和方差检验。与高斯分布和 t 分布不同，卡方分布始终为正，其临界值随自由度增加。



**Key Formula / 关键公式:**



$$\chi^2_{critical}(\alpha, k) = \chi^2.\text{ppf}(1 - \alpha, k)$$



where $k$ is the degrees of freedom.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
from scipy.stats import chi2

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd



# 卡方分布 PPF 只处理正值

# Chi-squared distribution PPF only handles positive values
```

## Step 2 — Compute Chi-Squared Critical Values / 计算卡方临界值

```python
# 对于 α = 0.05，df = 10

# For α = 0.05, df = 10

alpha = 0.05

df = 10



# 计算卡方临界值

# Compute chi-squared critical value

chi2_critical = chi2.ppf(1 - alpha, df)



# 打印输出 / Print output
print(f"Significance level (α): {alpha}")

# 打印输出 / Print output
print(f"Degrees of freedom: {df}")

# 打印输出 / Print output
print(f"\nChi-squared critical value: {chi2_critical:.6f}")

# 打印输出 / Print output
print(f"\nInterpretation / 解释:")

# 打印输出 / Print output
print(f"Reject H0 if observed χ² > {chi2_critical:.4f}")
```

## Step 3 — Verify with CDF / 用CDF验证

```python
# 使用 CDF 验证

# Verify using CDF

verification = chi2.cdf(chi2_critical, df)



# 打印输出 / Print output
print(f"\nVerification / 验证:")

# 打印输出 / Print output
print(f"PPF({1 - alpha}, df={df}) = {chi2_critical:.6f}")

# 打印输出 / Print output
print(f"CDF({chi2_critical:.6f}, df={df}) = {verification:.6f}")

# 打印输出 / Print output
print(f"Expected: {1 - alpha:.6f}")

# 打印输出 / Print output
print(f"Difference: {abs(verification - (1 - alpha)):.10f}")
```

## Step 4 — Critical Values for Different Degrees of Freedom / 不同自由度的临界值

```python
# 常见的显著性水平

# Common significance levels

alpha_levels = [0.10, 0.05, 0.01]

dof_values = [1, 2, 5, 10, 20, 30]



# 创建临界值的表格

# Create table of critical values

for alpha in alpha_levels:

    # 打印输出 / Print output
    print(f"\nCritical Values for α = {alpha}:")

    # 打印输出 / Print output
    print("-" * 50)

    data = []

    for df_val in dof_values:

        crit = chi2.ppf(1 - alpha, df_val)

        # 添加元素到列表末尾 / Append element to list end
        data.append({'Degrees of Freedom': df_val, 'Critical Value': f"{crit:.4f}"})

    # 打印输出 / Print output
    print(pd.DataFrame(data).to_string(index=False))
```

## Step 5 — Effect of Degrees of Freedom / 自由度的影响

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import linspace

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot



# 绘制临界值随自由度的变化

# Plot how critical values change with degrees of freedom

dof_range = linspace(1, 50, 200)

alpha_vals = [0.10, 0.05, 0.01]



pyplot.figure(figsize=(10, 6))



for alpha in alpha_vals:

    criticals = [chi2.ppf(1 - alpha, df_val) for df_val in dof_range]

    pyplot.plot(dof_range, criticals, linewidth=2, label=f'α = {alpha}')



pyplot.xlabel('Degrees of Freedom / 自由度')

pyplot.ylabel('Critical Value / 临界值')

pyplot.title('Chi-Squared Critical Values / 卡方临界值')

pyplot.legend()

pyplot.grid(True, alpha=0.3)

pyplot.show()
```

## Step 6 — Practical Example: Goodness-of-Fit Test / 实际例子：拟合优度检验

```python
# 实际例子：卡方检验

# Practical example: Chi-squared test

# 假设我们有 6 面骰子的数据

# Suppose we have data from rolling a 6-sided die



observed = [10, 12, 8, 15, 14, 11]  # 观察到的频数

expected = [10, 10, 10, 10, 10, 10]  # 期望频数（均匀分布）



# 计算卡方统计量

# Compute chi-squared statistic

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 创建NumPy数组 / Create NumPy array
chi2_stat = sum((np.array(observed) - np.array(expected))**2 / np.array(expected))



# 自由度 = 类别数 - 1

# Degrees of freedom = number of categories - 1

# 获取长度 / Get length
df_test = len(observed) - 1



# 获取临界值

# Get critical values for different α levels

alpha_05 = 0.05

critical_05 = chi2.ppf(1 - alpha_05, df_test)



# 计算 p 值

# Compute p-value

p_value = 1 - chi2.cdf(chi2_stat, df_test)



# 打印输出 / Print output
print("Chi-Squared Goodness-of-Fit Test / 卡方拟合优度检验:")

# 打印输出 / Print output
print("="*50)

# 打印输出 / Print output
print(f"Observed frequencies: {observed}")

# 打印输出 / Print output
print(f"Expected frequencies: {expected}")

# 打印输出 / Print output
print(f"\nχ² statistic: {chi2_stat:.4f}")

# 打印输出 / Print output
print(f"Degrees of freedom: {df_test}")

# 打印输出 / Print output
print(f"p-value: {p_value:.6f}")

# 打印输出 / Print output
print(f"\nCritical value (α = {alpha_05}): {critical_05:.4f}")

# 打印输出 / Print output
print()



if chi2_stat > critical_05:

    # 打印输出 / Print output
    print(f"Decision: REJECT H0 (χ² = {chi2_stat:.4f} > {critical_05:.4f})")

    # 打印输出 / Print output
    print("The die does not appear to be fair.")

else:

    # 打印输出 / Print output
    print(f"Decision: FAIL TO REJECT H0 (χ² = {chi2_stat:.4f} ≤ {critical_05:.4f})")

    # 打印输出 / Print output
    print("The die appears to be fair.")
```

## Learning Notes / 学习笔记



- **Statistical Concept / 统计学概念**: Chi-squared critical values increase with degrees of freedom because the distribution shifts rightward. Critical values are always positive. The chi-squared test assumes expected frequencies are sufficiently large (usually ≥5) for validity. / 卡方临界值随自由度增加而增加，因为分布向右移动。临界值始终为正。卡方检验假设期望频数足够大（通常 ≥5）以确保有效性。



- **ML Application / 机器学习应用**: Chi-squared tests are crucial for feature selection in categorical data (selecting features with significant relationship to target). Used in goodness-of-fit tests to validate that data matches expected distributions. Essential in contingency table analysis for detecting associations between categorical variables. / 卡方检验对分类数据的特征选择至关重要（选择与目标有显著关系的特征）。用于拟合优度检验以验证数据是否与期望分布匹配。在列联表分析中对检测分类变量之间的关联至关重要。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `matplotlib` | 绑图库 | Plotting library |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

➡️ **Next**: `../chapter_12/01_test_dataset.ipynb`

## Complete Code / 完整代码一览

```python
from scipy.stats import chi2

# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import linspace

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd



# Compute critical values

alpha = 0.05

df = 10

chi2_critical = chi2.ppf(1 - alpha, df)

# 打印输出 / Print output
print(f"Critical value (α={alpha}, df={df}): {chi2_critical:.6f}")



# Verify with CDF

# 打印输出 / Print output
print(f"Verification: CDF({chi2_critical:.4f}, df={df}) = {chi2.cdf(chi2_critical, df):.6f}")



# Plot critical values

dof_range = linspace(1, 50, 200)

for alpha in [0.10, 0.05, 0.01]:

    criticals = [chi2.ppf(1 - alpha, d) for d in dof_range]

    pyplot.plot(dof_range, criticals, linewidth=2, label=f'α = {alpha}')

pyplot.xlabel('Degrees of Freedom')

pyplot.ylabel('Critical Value')

pyplot.title('Chi-Squared Critical Values')

pyplot.legend()

pyplot.grid(True, alpha=0.3)

pyplot.show()



# Example: Goodness-of-fit test

observed = [10, 12, 8, 15, 14, 11]

expected = [10, 10, 10, 10, 10, 10]

# 创建NumPy数组 / Create NumPy array
chi2_stat = sum((np.array(observed) - np.array(expected))**2 / np.array(expected))

# 获取长度 / Get length
df_test = len(observed) - 1

p_value = 1 - chi2.cdf(chi2_stat, df_test)

# 打印输出 / Print output
print(f"\nχ² statistic: {chi2_stat:.4f}, p-value: {p_value:.6f}")
```

---

### Chapter Summary / 章节总结

# Chapter 11: Critical Values
# 第11章：临界值

## Theme | 主题
Inverse CDF (Percent Point Function): from probabilities to distribution quantiles for hypothesis testing.
逆CDF(百分点函数)：从概率到分布分位数，用于假设检验。

## Evolution Roadmap | 演变路线图
```
GAUSSIAN PPF (Percent Point Function):
  PPF(α) → critical value z_α

STUDENT'S T PPF:
  PPF(α, df) → critical value t_α,df

CHI-SQUARED PPF:
  PPF(α, df) → critical value χ²_α,df
```

## Progression Logic | 进度逻辑

### Stage 1: Gaussian Quantiles (高斯分位数)
**English:** For significance level α (e.g., 0.05), compute z-critical using Gaussian PPF. Example: PPF(0.975) ≈ 1.96 (two-tailed test at α=0.05).
**中文:** 对于显著性水平α(例如0.05)，使用高斯PPF计算z临界值。例如：PPF(0.975) ≈ 1.96(双尾检验，α=0.05)。

### Stage 2: t-Distribution Quantiles (t分布分位数)
**English:** For df degrees of freedom, compute t_critical. Fatter tails mean larger critical value than Gaussian. As df → ∞, t_critical → z_critical.
**中文:** 对于df自由度，计算t临界值。尾部较厚意味着比高斯更大的临界值。当df → ∞时，t临界值 → z临界值。

### Stage 3: Chi-Squared Quantiles (卡方分位数)
**English:** For df degrees of freedom, compute χ²_critical. Always one-tailed because χ² ≥ 0. Used in goodness-of-fit and contingency table tests.
**中文:** 对于df自由度，计算χ²临界值。总是单尾的，因为χ² ≥ 0。用于拟合优度和列联表测试。

## ML Relevance | ML相关性

1. **Decision Boundaries (决策边界)**: Critical values determine when to reject null hypotheses in statistical tests.
2. **Hypothesis Testing (假设检验)**: t-test and χ² test rely on PPF to compute p-value thresholds.
3. **Confidence Intervals (置信区间)**: PPF determines the multiplier (e.g., 1.96 for 95% CI) applied to standard error.
4. **Model Validation (模型验证)**: Critical values inform whether observed differences are statistically significant or due to chance.


---
