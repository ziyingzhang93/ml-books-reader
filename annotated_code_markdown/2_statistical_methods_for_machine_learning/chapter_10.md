# 统计方法与机器学习 / Statistical Methods for Machine Learning
## Chapter 10

---

### Gaussian Pdf

# 10.1 — Gaussian Probability Density Function / 高斯概率密度函数

**Chapter 10 — File 1 of 6**

## Summary / 摘要

The Gaussian (Normal) distribution is the most important probability distribution in statistics and machine learning. This notebook demonstrates how to compute and visualize the PDF using scipy.stats.norm.pdf().

高斯（正态）分布是统计学和机器学习中最重要的概率分布。本笔记本演示如何使用 scipy.stats.norm.pdf() 计算和可视化 PDF。

**Mathematical Formula / 数学公式:**

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

where $\mu$ is the mean and $\sigma$ is the standard deviation.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from scipy.stats import norm

# 创建样本空间：从-5到5，步长0.001
# Create sample space: from -5 to 5 with step 0.001
# 生成整数序列 / Generate integer sequence
sample_space = arange(-5, 5, 0.001)
```

## Step 2 — Compute Gaussian PDF / 计算高斯PDF

```python
# 使用标准高斯分布：mean=0.0, std=1.0
# Use standard Gaussian: mean=0.0, standard deviation=1.0
pdf = norm.pdf(sample_space, 0.0, 1.0)

# 打印前几个值检查
# Print first few values for inspection
# 打印输出 / Print output
print("First 5 PDF values:")
# 打印输出 / Print output
print(pdf[:5])
```

## Step 3 — Plot the PDF / 绘制PDF

```python
# 绘制概率密度函数
# Plot the probability density function
pyplot.figure(figsize=(10, 6))
pyplot.plot(sample_space, pdf, linewidth=2)
pyplot.xlabel('Value / 值')
pyplot.ylabel('Probability Density / 概率密度')
pyplot.title('Gaussian PDF (μ=0, σ=1) / 高斯PDF')
pyplot.grid(True, alpha=0.3)
pyplot.show()

# 高斯分布的关键性质：
# Key properties of Gaussian distribution:
# 1. 对称性 (Symmetry)：关于均值对称 (symmetric around mean)
# 2. 峰值 (Peak)：在均值处达到最大值 (maximum at mean)
# 3. 尾部 (Tails)：渐近于零 (asymptotic to zero)
# 打印输出 / Print output
print(f"Maximum PDF value: {pdf.max():.6f}")
# 打印输出 / Print output
print(f"Location of maximum: x = {sample_space[pdf.argmax()]:.3f}")
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念**: The Gaussian PDF describes the probability density at each point. It integrates to 1 over the entire range, and the maximum occurs at the mean. / 高斯PDF描述每个点的概率密度。它在整个范围内积分为1，最大值出现在均值处。

- **ML Application / 机器学习应用**: Gaussian distributions are fundamental to many ML algorithms (linear regression, Gaussian Naive Bayes, Gaussian Process). Assuming normally distributed errors is common in supervised learning. / 高斯分布对许多ML算法至关重要（线性回归、高斯朴素贝叶斯、高斯过程）。在监督学习中，假设误差正态分布是常见的。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `02_gaussian_cdf.ipynb`

## Complete Code / 完整代码一览

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from scipy.stats import norm

# Create sample space from -5 to 5 with step 0.001
# 生成整数序列 / Generate integer sequence
sample_space = arange(-5, 5, 0.001)

# Compute Gaussian PDF with mean=0.0, std=1.0
pdf = norm.pdf(sample_space, 0.0, 1.0)

# Plot the probability density function
pyplot.figure(figsize=(10, 6))
pyplot.plot(sample_space, pdf, linewidth=2)
pyplot.xlabel('Value')
pyplot.ylabel('Probability Density')
pyplot.title('Gaussian PDF (μ=0, σ=1)')
pyplot.grid(True, alpha=0.3)
pyplot.show()

# 打印输出 / Print output
print(f"Maximum PDF value: {pdf.max():.6f}")
# 打印输出 / Print output
print(f"Location of maximum: x = {sample_space[pdf.argmax()]:.3f}")
```

---

### Gaussian Cdf

# 10.2 — Gaussian Cumulative Distribution Function / 高斯累积分布函数

**Chapter 10 — File 2 of 6**

## Summary / 摘要

The Cumulative Distribution Function (CDF) gives the probability that a random variable is less than or equal to a value. This notebook demonstrates how to compute and visualize the CDF using scipy.stats.norm.cdf().

累积分布函数（CDF）给出随机变量小于等于某个值的概率。本笔记本演示如何使用 scipy.stats.norm.cdf() 计算和可视化 CDF。

**Mathematical Formula / 数学公式:**

$$F(x) = P(X \leq x) = \int_{-\infty}^{x} \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(t-\mu)^2}{2\sigma^2}\right) dt$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from scipy.stats import norm

# 创建样本空间：从-5到5，步长0.001
# Create sample space: from -5 to 5 with step 0.001
# 生成整数序列 / Generate integer sequence
sample_space = arange(-5, 5, 0.001)
```

## Step 2 — Compute Gaussian CDF / 计算高斯CDF

```python
# 计算标准高斯分布的CDF
# Compute CDF for standard Gaussian distribution
cdf = norm.cdf(sample_space)

# CDF的关键性质：
# Key properties of CDF:
# 打印输出 / Print output
print(f"CDF at x=-5: {cdf[0]:.6f}")
# 打印输出 / Print output
print(f"CDF at x=0 (mean): {norm.cdf(0):.6f}")
# 打印输出 / Print output
print(f"CDF at x=5: {cdf[-1]:.6f}")

# CDF单调递增，从0到1
# CDF is monotonically increasing from 0 to 1
# 打印输出 / Print output
print(f"CDF is monotonic: {(cdf == sorted(cdf)).all()}")
```

## Step 3 — Plot the CDF / 绘制CDF

```python
# 绘制累积分布函数
# Plot the cumulative distribution function
pyplot.figure(figsize=(10, 6))
pyplot.plot(sample_space, cdf, linewidth=2)
pyplot.xlabel('Value / 值')
pyplot.ylabel('Cumulative Probability / 累积概率')
pyplot.title('Gaussian CDF (μ=0, σ=1) / 高斯CDF')
pyplot.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% probability')
pyplot.axvline(x=0, color='r', linestyle='--', alpha=0.5)
pyplot.grid(True, alpha=0.3)
pyplot.legend()
pyplot.show()
```

## Step 4 — Interpret CDF Values / 解释CDF值

```python
# CDF可用于计算概率
# CDF can be used to compute probabilities
prob_less_than_0 = norm.cdf(0)
prob_less_than_1 = norm.cdf(1)
prob_between_0_and_1 = prob_less_than_1 - prob_less_than_0

# 打印输出 / Print output
print(f"P(X ≤ 0) = {prob_less_than_0:.4f}")
# 打印输出 / Print output
print(f"P(X ≤ 1) = {prob_less_than_1:.4f}")
# 打印输出 / Print output
print(f"P(0 < X ≤ 1) = {prob_between_0_and_1:.4f}")

# 68-95-99.7规则（经验法则）
# 68-95-99.7 rule (empirical rule)
within_1_sigma = norm.cdf(1) - norm.cdf(-1)
within_2_sigma = norm.cdf(2) - norm.cdf(-2)
within_3_sigma = norm.cdf(3) - norm.cdf(-3)

# 打印输出 / Print output
print(f"\nEmpirical Rule / 经验法则:")
# 打印输出 / Print output
print(f"Within 1σ: {within_1_sigma:.4f} (≈68%)")
# 打印输出 / Print output
print(f"Within 2σ: {within_2_sigma:.4f} (≈95%)")
# 打印输出 / Print output
print(f"Within 3σ: {within_3_sigma:.4f} (≈99.7%)")
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念**: The CDF is the integral of the PDF. It provides cumulative probabilities and is always non-decreasing. F(x)=0.5 at the mean for a symmetric distribution. / CDF 是 PDF 的积分。它提供累积概率并始终非递减。对于对称分布，F(x)=0.5 在均值处。

- **ML Application / 机器学习应用**: CDF is used in hypothesis testing, confidence interval construction, and statistical inference. It helps determine p-values and critical regions for significance tests. / CDF 用于假设检验、置信区间构造和统计推断。它有助于确定 p 值和显著性检验的临界区域。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `03_students_t_pdf.ipynb`

## Complete Code / 完整代码一览

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
from scipy.stats import norm

# Create sample space from -5 to 5
# 生成整数序列 / Generate integer sequence
sample_space = arange(-5, 5, 0.001)

# Compute Gaussian CDF
cdf = norm.cdf(sample_space)

# Plot the CDF
pyplot.figure(figsize=(10, 6))
pyplot.plot(sample_space, cdf, linewidth=2)
pyplot.xlabel('Value')
pyplot.ylabel('Cumulative Probability')
pyplot.title('Gaussian CDF (μ=0, σ=1)')
pyplot.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
pyplot.axvline(x=0, color='r', linestyle='--', alpha=0.5)
pyplot.grid(True, alpha=0.3)
pyplot.show()

# Compute probabilities
prob_between_0_and_1 = norm.cdf(1) - norm.cdf(0)
within_1_sigma = norm.cdf(1) - norm.cdf(-1)
# 打印输出 / Print output
print(f"P(0 < X ≤ 1) = {prob_between_0_and_1:.4f}")
# 打印输出 / Print output
print(f"Within 1σ: {within_1_sigma:.4f}")
```

---

### Students T Pdf

# 10.3 — Student's t Probability Density Function / Student t 概率密度函数



**Chapter 10 — File 3 of 6**



## Summary / 摘要



Student's t-distribution is used when the population standard deviation is unknown and must be estimated from sample data. It is commonly used in hypothesis testing and confidence intervals. This notebook visualizes how the t-distribution approaches the Gaussian as degrees of freedom increase.



Student t 分布在总体标准差未知且必须从样本数据估计时使用。它在假设检验和置信区间中常用。本笔记本可视化 t 分布如何随自由度增加而接近高斯分布。



**Mathematical Formula / 数学公式:**



$$f(t; \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\pi\nu}\Gamma\left(\frac{\nu}{2}\right)}\left(1+\frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$



where $\nu$ is the degrees of freedom.


---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

from scipy.stats import t, norm



# 创建样本空间

# Create sample space

# 生成整数序列 / Generate integer sequence
sample_space = arange(-5, 5, 0.001)
```

## Step 2 — Compute Student's t PDF / 计算 Student t PDF

```python
# 使用自由度 = 样本空间长度 - 1

# Use degrees of freedom = len(sample_space) - 1

# 获取长度 / Get length
dof = len(sample_space) - 1

# 打印输出 / Print output
print(f"Degrees of freedom: {dof}")



# 计算 t 分布的 PDF

# Compute t-distribution PDF

pdf_t = t.pdf(sample_space, dof)



# 为了比较，也计算标准高斯分布

# For comparison, also compute standard Gaussian PDF

pdf_gaussian = norm.pdf(sample_space)
```

## Step 3 — Plot and Compare / 绘制并比较

```python
# 绘制 t 分布 PDF

# Plot t-distribution PDF

pyplot.figure(figsize=(10, 6))

pyplot.plot(sample_space, pdf_t, linewidth=2, label=f"t-distribution (df={dof})")

pyplot.plot(sample_space, pdf_gaussian, linewidth=2, linestyle='--', label="Gaussian (Normal)")

pyplot.xlabel('Value / 值')

pyplot.ylabel('Probability Density / 概率密度')

pyplot.title("Student's t vs Gaussian Distribution / Student t 与高斯分布")

pyplot.legend()

pyplot.grid(True, alpha=0.3)

pyplot.show()



# 打印输出 / Print output
print(f"Max t-PDF: {pdf_t.max():.6f}")

# 打印输出 / Print output
print(f"Max Gaussian PDF: {pdf_gaussian.max():.6f}")
```

## Step 4 — Effect of Degrees of Freedom / 自由度的影响

```python
# t 分布的一个重要性质：随着自由度增加，它逼近高斯分布

# Important property: as df increases, t-distribution approaches Gaussian

pyplot.figure(figsize=(12, 6))



# 绘制不同自由度的 t 分布

# Plot t-distributions with different degrees of freedom

for dof_val in [1, 5, 10, 30, 100]:

    pdf = t.pdf(sample_space, dof_val)

    pyplot.plot(sample_space, pdf, linewidth=2, label=f"df={dof_val}")



# 添加高斯分布作为参考

# Add Gaussian as reference

pyplot.plot(sample_space, norm.pdf(sample_space), linewidth=2, linestyle='--', 

            color='black', label="Gaussian (limit)")



pyplot.xlabel('Value / 值')

pyplot.ylabel('Probability Density / 概率密度')

pyplot.title("Effect of Degrees of Freedom on t-Distribution / 自由度对 t 分布的影响")

pyplot.legend()

pyplot.grid(True, alpha=0.3)

pyplot.xlim(-5, 5)

pyplot.show()



# 打印输出 / Print output
print("Notice: t 分布有更厚的尾部（heavier tails）")

# 打印输出 / Print output
print("Notice: t-distribution has heavier tails than Gaussian")
```

## Learning Notes / 学习笔记



- **Statistical Concept / 统计学概念**: The t-distribution has heavier tails than the Gaussian, accounting for additional uncertainty when estimating from small samples. As n→∞, t converges to Gaussian. This is used when σ is unknown and estimated from sample. / t 分布的尾部比高斯分布更厚，考虑了从小样本估计时的额外不确定性。当 n→∞ 时，t 收敛到高斯分布。当 σ 未知且从样本估计时使用。



- **ML Application / 机器学习应用**: Student's t-test is fundamental for comparing means between groups. Welch's t-test assumes unequal variances. t-distributions are important in Bayesian methods and robust regression. / Student t 检验是比较组间均值的基础。Welch's t-test 假设方差不等。t 分布在贝叶斯方法和稳健回归中很重要。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `04_students_t_cdf.ipynb`

## Complete Code / 完整代码一览

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

from scipy.stats import t, norm



# 生成整数序列 / Generate integer sequence
sample_space = arange(-5, 5, 0.001)

# 获取长度 / Get length
dof = len(sample_space) - 1



# Compute t-distribution and Gaussian PDFs

pdf_t = t.pdf(sample_space, dof)

pdf_gaussian = norm.pdf(sample_space)



# Plot comparison

pyplot.figure(figsize=(10, 6))

pyplot.plot(sample_space, pdf_t, linewidth=2, label=f"t-distribution (df={dof})")

pyplot.plot(sample_space, pdf_gaussian, linewidth=2, linestyle='--', label="Gaussian")

pyplot.xlabel('Value')

pyplot.ylabel('Probability Density')

pyplot.title("Student's t vs Gaussian Distribution")

pyplot.legend()

pyplot.grid(True, alpha=0.3)

pyplot.show()



# Show effect of degrees of freedom

pyplot.figure(figsize=(12, 6))

for dof_val in [1, 5, 10, 30, 100]:

    pdf = t.pdf(sample_space, dof_val)

    pyplot.plot(sample_space, pdf, linewidth=2, label=f"df={dof_val}")

pyplot.plot(sample_space, norm.pdf(sample_space), linewidth=2, linestyle='--', 

            color='black', label="Gaussian")

pyplot.xlabel('Value')

pyplot.ylabel('Probability Density')

pyplot.title("Effect of Degrees of Freedom on t-Distribution")

pyplot.legend()

pyplot.grid(True, alpha=0.3)

pyplot.show()
```

---

### Students T Cdf



---

### Chi Squared Pdf

# 10.5 — Chi-Squared Probability Density Function / 卡方概率密度函数



**Chapter 10 — File 5 of 6**



## Summary / 摘要



The chi-squared distribution is used extensively in statistical testing for variance, goodness-of-fit tests, and tests of independence in contingency tables. It is always positive and right-skewed. This notebook demonstrates how to compute and visualize the chi-squared PDF.



卡方分布广泛用于方差统计检验、拟合优度检验和列联表独立性检验。它始终为正且右偏。本笔记本演示如何计算和可视化卡方 PDF。



**Mathematical Formula / 数学公式:**



$$f(x; k) = \frac{1}{2^{k/2}\Gamma(k/2)} x^{k/2-1} e^{-x/2}$$



where $k$ is the degrees of freedom (positive integer).

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

from scipy.stats import chi2



# 创建样本空间：从0到50（卡方分布只有正值）

# Create sample space: from 0 to 50 (chi-squared only has positive values)

# 生成整数序列 / Generate integer sequence
sample_space = arange(0, 50, 0.01)
```

## Step 2 — Compute Chi-Squared PDF / 计算卡方 PDF

```python
# 使用自由度 = 20

# Use degrees of freedom = 20

dof = 20



# 计算卡方分布的 PDF

# Compute chi-squared distribution PDF

pdf = chi2.pdf(sample_space, dof)



# 打印输出 / Print output
print(f"Degrees of freedom: {dof}")

# 打印输出 / Print output
print(f"Mean of chi-squared: {dof}")

# 打印输出 / Print output
print(f"Variance of chi-squared: {2*dof}")

# 打印输出 / Print output
print(f"Max PDF value: {pdf.max():.6f}")
```

## Step 3 — Plot the Chi-Squared PDF / 绘制卡方 PDF

```python
# 绘制卡方分布 PDF

# Plot chi-squared distribution PDF

pyplot.figure(figsize=(10, 6))

pyplot.plot(sample_space, pdf, linewidth=2)

pyplot.xlabel('Value / 值')

pyplot.ylabel('Probability Density / 概率密度')

pyplot.title(f'Chi-Squared PDF (df={dof}) / 卡方 PDF')

pyplot.axvline(x=dof, color='red', linestyle='--', label=f'Mean = {dof}')

pyplot.grid(True, alpha=0.3)

pyplot.legend()

pyplot.show()
```

## Step 4 — Compare Different Degrees of Freedom / 比较不同自由度

```python
# 卡方分布的形状随自由度变化

# Chi-squared distribution shape varies with degrees of freedom

pyplot.figure(figsize=(12, 6))



# 绘制不同自由度的卡方分布

# Plot chi-squared distributions with different degrees of freedom

for dof_val in [1, 2, 5, 10, 20, 30]:

    pdf = chi2.pdf(sample_space, dof_val)

    pyplot.plot(sample_space, pdf, linewidth=2, label=f"df={dof_val}")



pyplot.xlabel('Value / 值')

pyplot.ylabel('Probability Density / 概率密度')

pyplot.title('Chi-Squared Distribution for Different Degrees of Freedom / 不同自由度的卡方分布')

pyplot.legend()

pyplot.grid(True, alpha=0.3)

pyplot.xlim(0, 50)

pyplot.show()



# 打印输出 / Print output
print("Key observations / 关键观察:")

# 打印输出 / Print output
print("1. 小自由度：分布更陡峭（Small df: distribution more skewed）")

# 打印输出 / Print output
print("2. 大自由度：分布更对称（Large df: distribution more symmetric）")

# 打印输出 / Print output
print("3. 均值 = 自由度 (Mean = df)")

# 打印输出 / Print output
print("4. 总是右偏（Always right-skewed）")
```

## Step 5 — Properties of Chi-Squared / 卡方分布的性质

```python
# 卡方分布的统计性质

# Statistical properties of chi-squared distribution

from scipy.stats import chi2

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np



dof_val = 20



# 计算均值、方差、偏度和峰度

# Compute mean, variance, skewness, and kurtosis

mean_val = chi2.mean(dof_val)

var_val = chi2.var(dof_val)

skew_val = chi2.stats(dof_val, moments='s')

kurt_val = chi2.stats(dof_val, moments='k')



# 打印输出 / Print output
print(f"Chi-Squared Distribution (df={dof_val}):")

# 打印输出 / Print output
print(f"Mean / 均值: {mean_val:.4f}")

# 打印输出 / Print output
print(f"Variance / 方差: {var_val:.4f}")

# 打印输出 / Print output
print(f"Standard Deviation / 标准差: {np.sqrt(var_val):.4f}")

# 打印输出 / Print output
print(f"Skewness / 偏度: {skew_val:.4f}")

# 打印输出 / Print output
print(f"Excess Kurtosis / 超额峰度: {kurt_val:.4f}")



# 打印输出 / Print output
print(f"\n68% of data within [μ-σ, μ+σ] / 68% 数据在 [μ-σ, μ+σ] 内")

lower = mean_val - np.sqrt(var_val)

upper = mean_val + np.sqrt(var_val)

prob = chi2.cdf(upper, dof_val) - chi2.cdf(max(0, lower), dof_val)

# 打印输出 / Print output
print(f"Actual probability / 实际概率: {prob:.4f}")
```

## Learning Notes / 学习笔记



- **Statistical Concept / 统计学概念**: The chi-squared distribution is the sum of squared standard normal variables. It appears in hypothesis tests for variance (chi-squared test), goodness-of-fit tests, and tests of independence. It always takes non-negative values. / 卡方分布是平方标准正态变量的和。它出现在方差假设检验（卡方检验）、拟合优度检验和独立性检验中。它总是非负值。



- **ML Application / 机器学习应用**: Chi-squared tests are used for feature selection, model comparison, and testing distributional assumptions. In anomaly detection, chi-squared distance measures dissimilarity between distributions. Essential in testing independence between categorical variables. / 卡方检验用于特征选择、模型比较和测试分布假设。在异常检测中，卡方距离测量分布之间的差异。在测试分类变量之间的独立性时至关重要。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `06_chi_squared_cdf.ipynb`

## Complete Code / 完整代码一览

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

from scipy.stats import chi2

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np



# 生成整数序列 / Generate integer sequence
sample_space = arange(0, 50, 0.01)

dof = 20



# Compute chi-squared PDF

pdf = chi2.pdf(sample_space, dof)



# Plot chi-squared PDF

pyplot.figure(figsize=(10, 6))

pyplot.plot(sample_space, pdf, linewidth=2)

pyplot.xlabel('Value')

pyplot.ylabel('Probability Density')

pyplot.title(f'Chi-Squared PDF (df={dof})')

pyplot.axvline(x=dof, color='red', linestyle='--', label=f'Mean = {dof}')

pyplot.grid(True, alpha=0.3)

pyplot.legend()

pyplot.show()



# Compare different degrees of freedom

pyplot.figure(figsize=(12, 6))

for dof_val in [1, 2, 5, 10, 20, 30]:

    pdf = chi2.pdf(sample_space, dof_val)

    pyplot.plot(sample_space, pdf, linewidth=2, label=f"df={dof_val}")

pyplot.xlabel('Value')

pyplot.ylabel('Probability Density')

pyplot.title('Chi-Squared Distribution for Different Degrees of Freedom')

pyplot.legend()

pyplot.grid(True, alpha=0.3)

pyplot.xlim(0, 50)

pyplot.show()
```

---

### Chi Squared Cdf

# 10.6 — Chi-Squared Cumulative Distribution Function / 卡方累积分布函数



**Chapter 10 — File 6 of 6**



## Summary / 摘要



The chi-squared CDF is used to compute p-values and critical values for chi-squared tests. This notebook demonstrates how to compute and interpret the chi-squared CDF for hypothesis testing.



卡方 CDF 用于计算卡方检验的 p 值和临界值。本笔记本演示如何为假设检验计算和解释卡方 CDF。



**Key Applications / 关键应用:**



- Goodness-of-fit tests (拟合优度检验)

- Tests of independence in contingency tables (列联表独立性检验)

- Variance hypothesis tests (方差假设检验)

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

from scipy.stats import chi2



# 创建样本空间

# Create sample space

# 生成整数序列 / Generate integer sequence
sample_space = arange(0, 50, 0.01)
```

## Step 2 — Compute Chi-Squared CDF / 计算卡方 CDF

```python
# 使用自由度 = 20

# Use degrees of freedom = 20

dof = 20



# 计算卡方分布的 CDF

# Compute chi-squared distribution CDF

cdf = chi2.cdf(sample_space, dof)



# 打印输出 / Print output
print(f"Degrees of freedom: {dof}")

# 打印输出 / Print output
print(f"CDF at x={dof}: {chi2.cdf(dof, dof):.4f}")

# 打印输出 / Print output
print(f"This means ~{chi2.cdf(dof, dof)*100:.1f}% of values are below the mean")
```

## Step 3 — Plot the Chi-Squared CDF / 绘制卡方 CDF

```python
# 绘制卡方分布 CDF

# Plot chi-squared distribution CDF

pyplot.figure(figsize=(10, 6))

pyplot.plot(sample_space, cdf, linewidth=2)

pyplot.xlabel('Value / 值')

pyplot.ylabel('Cumulative Probability / 累积概率')

pyplot.title(f'Chi-Squared CDF (df={dof}) / 卡方 CDF')

pyplot.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95% (Critical value)')

pyplot.axvline(x=dof, color='green', linestyle='--', alpha=0.5, label=f'Mean = {dof}')

pyplot.grid(True, alpha=0.3)

pyplot.legend()

pyplot.show()
```

## Step 4 — Critical Values for Hypothesis Testing / 假设检验的临界值

```python
# 对于常见的显著性水平，计算临界值

# For common significance levels, compute critical values

alpha_levels = [0.10, 0.05, 0.01]



# 打印输出 / Print output
print(f"Critical values for χ² distribution (df={dof}):")

# 打印输出 / Print output
print("="*50)



for alpha in alpha_levels:

    critical_value = chi2.ppf(1 - alpha, dof)

    # 打印输出 / Print output
    print(f"α = {alpha} (significance level):")

    # 打印输出 / Print output
    print(f"  Critical value: {critical_value:.4f}")

    # 打印输出 / Print output
    print(f"  Reject H0 if χ² > {critical_value:.4f}")

    # 打印输出 / Print output
    print()
```

## Step 5 — Different Degrees of Freedom / 不同自由度

```python
# 比较不同自由度的卡方 CDF

# Compare chi-squared CDFs for different degrees of freedom

pyplot.figure(figsize=(12, 6))



for dof_val in [1, 5, 10, 20, 30]:

    cdf = chi2.cdf(sample_space, dof_val)

    pyplot.plot(sample_space, cdf, linewidth=2, label=f"df={dof_val}")



# 标记α = 0.05临界值

pyplot.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='α = 0.05 (95%)')



pyplot.xlabel('Value / 值')

pyplot.ylabel('Cumulative Probability / 累积概率')

pyplot.title('Chi-Squared CDF for Different Degrees of Freedom / 不同自由度的卡方 CDF')

pyplot.legend()

pyplot.grid(True, alpha=0.3)

pyplot.xlim(0, 50)

pyplot.show()
```

## Step 6 — Example: Goodness-of-Fit Test / 示例：拟合优度检验

```python
# 实际例子：拟合优度检验

# Example: Goodness-of-fit test

# 假设我们有一个观察到的卡方统计量

# Suppose we have an observed chi-squared statistic



observed_chi2 = 32.5

dof_test = 10



# 计算 p 值

# Compute p-value

p_value = 1 - chi2.cdf(observed_chi2, dof_test)



# 打印输出 / Print output
print(f"Hypothesis Test / 假设检验:")

# 打印输出 / Print output
print(f"Null Hypothesis (H0): Data follows expected distribution")

# 打印输出 / Print output
print(f"Alternative Hypothesis (H1): Data does not follow expected distribution")

# 打印输出 / Print output
print()

# 打印输出 / Print output
print(f"Observed χ² statistic: {observed_chi2:.4f}")

# 打印输出 / Print output
print(f"Degrees of freedom: {dof_test}")

# 打印输出 / Print output
print(f"p-value: {p_value:.6f}")

# 打印输出 / Print output
print()



# 在α = 0.05水平下做决策

# Make decision at α = 0.05 level

alpha_test = 0.05

if p_value < alpha_test:

    # 打印输出 / Print output
    print(f"Result / 结果: REJECT H0 (p-value = {p_value:.6f} < α = {alpha_test})")

    # 打印输出 / Print output
    print("Data significantly differs from expected distribution")

else:

    # 打印输出 / Print output
    print(f"Result / 结果: FAIL TO REJECT H0 (p-value = {p_value:.6f} >= α = {alpha_test})")

    # 打印输出 / Print output
    print("Data is consistent with expected distribution")
```

## Learning Notes / 学习笔记



- **Statistical Concept / 统计学概念**: The chi-squared CDF is essential for computing p-values in goodness-of-fit and independence tests. A test statistic is compared to critical values derived from the CDF to make decisions about hypothesis rejection. / 卡方 CDF 对于在拟合优度和独立性检验中计算 p 值至关重要。将检验统计量与 CDF 导出的临界值进行比较以决定假设拒绝。



- **ML Application / 机器学习应用**: Chi-squared tests are used for feature selection (selecting features with significant association to target), model comparison, and validating categorical distributions. In anomaly detection, chi-squared statistics measure deviation from normal behavior. / 卡方检验用于特征选择（选择与目标有显著关联的特征）、模型比较和验证分类分布。在异常检测中，卡方统计量测量与正常行为的偏差。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `../chapter_11/01_gaussian.ipynb`

## Complete Code / 完整代码一览

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

from scipy.stats import chi2



# 生成整数序列 / Generate integer sequence
sample_space = arange(0, 50, 0.01)

dof = 20



# Compute chi-squared CDF

cdf = chi2.cdf(sample_space, dof)



# Plot CDF

pyplot.figure(figsize=(10, 6))

pyplot.plot(sample_space, cdf, linewidth=2)

pyplot.xlabel('Value')

pyplot.ylabel('Cumulative Probability')

pyplot.title(f'Chi-Squared CDF (df={dof})')

pyplot.axhline(y=0.95, color='red', linestyle='--', alpha=0.5)

pyplot.grid(True, alpha=0.3)

pyplot.show()



# Critical values

# 打印输出 / Print output
print(f"Critical values for χ² (df={dof}):")

for alpha in [0.10, 0.05, 0.01]:

    critical = chi2.ppf(1 - alpha, dof)

    # 打印输出 / Print output
    print(f"α = {alpha}: {critical:.4f}")



# Example hypothesis test

observed_chi2 = 32.5

dof_test = 10

p_value = 1 - chi2.cdf(observed_chi2, dof_test)

# 打印输出 / Print output
print(f"\nObserved χ² = {observed_chi2}, p-value = {p_value:.6f}")
```

---

### Chapter Summary / 章节总结

# Chapter 10: Distribution Functions
# 第10章：分布函数

## Theme | 主题
Exploring three critical distributions via probability density (PDF) and cumulative (CDF) functions.
通过概率密度(PDF)和累积(CDF)函数探索三个关键分布。

## Evolution Roadmap | 演变路线图
```
GAUSSIAN (Normal Distribution):
  PDF (density curve)
  └─ CDF (cumulative probability)

STUDENT'S T-DISTRIBUTION:
  PDF (wider tails than Gaussian)
  └─ CDF (cumulative probability)

CHI-SQUARED (χ²) DISTRIBUTION:
  PDF (right-skewed)
  └─ CDF (cumulative probability)
```

## Progression Logic | 进度逻辑

### Stage 1: Gaussian PDF (高斯PDF)
**English:** Plot the symmetric, bell-shaped Gaussian PDF. Parameters: mean (μ), standard deviation (σ).
**中文:** 绘制对称的钟形高斯PDF。参数：均值(μ)、标准差(σ)。

### Stage 2: Gaussian CDF (高斯CDF)
**English:** Plot the cumulative probability: P(X ≤ x). S-shaped curve asymptotes to 0 and 1.
**中文:** 绘制累积概率：P(X ≤ x)。S形曲线渐近于0和1。

### Stage 3: t-Distribution PDF (t分布PDF)
**English:** Plot for various degrees of freedom (df). Higher df → narrower tails (approach Gaussian). Lower df → fatter tails (more extreme values).
**中文:** 绘制不同自由度(df)的曲线。较高的df → 尾部较窄(接近高斯)。较低的df → 尾部较宽(更极端的值)。

### Stage 4: t-Distribution CDF (t分布CDF)
**English:** Cumulative probability for t-distribution. Used in confidence intervals and hypothesis tests on small samples.
**中文:** t分布的累积概率。用于小样本的置信区间和假设检验。

### Stage 5: Chi-Squared PDF (卡方PDF)
**English:** Right-skewed distribution with one parameter (df). Used for goodness-of-fit and independence tests.
**中文:** 右偏分布，有一个参数(df)。用于拟合优度和独立性测试。

### Stage 6: Chi-Squared CDF (卡方CDF)
**English:** Cumulative probability for χ². Critical values for hypothesis testing are read from CDF inverse (PPF).
**中文:** χ²的累积概率。假设检验的临界值从CDF逆(PPF)读取。

## ML Relevance | ML相关性

1. **Statistical Testing (统计检验)**: PDFs define probability densities; CDFs are used to compute p-values.
2. **Hypothesis Testing (假设检验)**: t-test uses t-distribution CDF; χ² test uses χ² distribution CDF.
3. **Confidence Intervals (置信区间)**: Inverse CDF (PPF, quantile function) determines critical values for CI construction.
4. **Model Diagnostics (模型诊断)**: Comparing observed vs. theoretical CDFs (Kolmogorov-Smirnov test) assesses goodness of fit.


---
